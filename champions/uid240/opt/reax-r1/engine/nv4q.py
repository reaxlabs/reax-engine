"""nv4q.py (rgh lane Q): NVFP4 activation quantisation with a DYNAMIC power-of-two per-tensor global scale in 3 launches
(zero a 2-float buffer, one atomic-max pass over x, the quant pass). The per-block numerics are the owner's
fp4_triton._quant16 (reference image) unchanged: block scale e4m3 = RN(bamax / (6 g)) clamped to 448, values RN onto
the e2m1 grid with fp32 IEEE division; g = 2^ceil(log2(amax / 2688)) (power of two: a block's e4m3 rounding then does not
depend on which other rows share the batch). The quant pass also writes alpha = g * g_w for the GEMM epilogue.

lane QX additions (flag-gated, default path unchanged): x46=True selects per 16-block between the RTN scale (block amax ->
6) and the "4/6" scale (block amax -> 4, Four Over Six 2025) by the smaller squared error of the dequantised block (ties ->
the 6 scale); perm (int32 [K]) gathers x[:, perm] while quantising (K-permutation of the GEMM; the weight's columns are
permuted the same way at start -> exact in real arithmetic).
"""
import torch
import triton
import triton.language as tl

from fp4_triton import _quant16, _store, _nb  # owner's per-block quantiser + swizzled store (reference image)


@triton.jit
def _amax_kernel(x_ptr, n, buf_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
    tl.atomic_max(buf_ptr, tl.max(tl.abs(v), axis=0))


@triton.jit
def _quant_dyn_kernel(x_ptr, buf_ptr, wg_ptr, out_ptr, sc_ptr, K, CB, NB: tl.constexpr):
    row, pk = tl.program_id(0).to(tl.int64), tl.program_id(1)
    am = tl.load(buf_ptr)
    g = tl.exp2(tl.ceil(tl.log2(tl.maximum(am / 2688.0, 1e-30))))
    if row == 0:
        if pk == 0:
            tl.store(buf_ptr + 1, g)
            tl.store(buf_ptr + 2, g * tl.load(wg_ptr))
    blk0 = pk * NB
    offs = (blk0 + tl.arange(0, NB))[:, None] * 16 + tl.arange(0, 16)[None, :]
    v = tl.load(x_ptr + row * K + offs).to(tl.float32)
    code, sb = _quant16(v, g, NB)
    _store(code, sb, out_ptr, sc_ptr, row, blk0, K, CB, NB)


@triton.jit
def _deq16(code, sf):
    """codes int32 (sign bit 3) -> dequantised fp32 values (e2m1 grid 0 .5 1 1.5 2 3 4 6) * sf[:, None]"""
    m = code & 7
    v = tl.where(m < 4, m.to(tl.float32) * 0.5, tl.where(m == 7, 6.0, (m - 2).to(tl.float32)))
    v = tl.where((code & 8) != 0, -v, v)
    return v * sf[:, None]


@triton.jit
def _quant16_c(v, g, C: tl.constexpr):
    """_quant16 with the block amax mapped to C (6.0 = the owner's RTN) -> (codes, scale bytes, sf, sse)"""
    bamax = tl.max(tl.abs(v), axis=1)
    s8 = tl.minimum(tl.math.div_rn(bamax, C * g), 448.0).to(tl.float8e4nv)
    sf = tl.maximum(s8.to(tl.float32) * g, 1e-30)
    q = tl.minimum(tl.maximum(tl.math.div_rn(v, sf[:, None]), -6.0), 6.0)
    a = tl.abs(q)
    idx = ((a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32) + (a > 1.25).to(tl.int32) + (a >= 1.75).to(tl.int32)
           + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32) + (a > 5.0).to(tl.int32))
    code = idx | ((q < 0).to(tl.int32) << 3)
    d = _deq16(code, sf) - v
    return code, s8.to(tl.uint8, bitcast=True), tl.sum(d * d, axis=1)


@triton.jit
def _quant_dyn46_kernel(x_ptr, buf_ptr, wg_ptr, out_ptr, sc_ptr, perm_ptr, K, CB, NB: tl.constexpr, X46: tl.constexpr,
                        PERM: tl.constexpr):
    row, pk = tl.program_id(0).to(tl.int64), tl.program_id(1)
    am = tl.load(buf_ptr)
    g = tl.exp2(tl.ceil(tl.log2(tl.maximum(am / 2688.0, 1e-30))))
    if row == 0:
        if pk == 0:
            tl.store(buf_ptr + 1, g)
            tl.store(buf_ptr + 2, g * tl.load(wg_ptr))
    blk0 = pk * NB
    offs = (blk0 + tl.arange(0, NB))[:, None] * 16 + tl.arange(0, 16)[None, :]
    if PERM:
        offs = tl.load(perm_ptr + offs)
    v = tl.load(x_ptr + row * K + offs).to(tl.float32)
    if X46:
        c6, s6, e6 = _quant16_c(v, g, 6.0)
        c4, s4, e4 = _quant16_c(v, g, 4.0)
        use4 = e4 < e6
        code = tl.where(use4[:, None], c4, c6)
        sb = tl.where(use4, s4, s6)
    else:
        code, sb = _quant16(v, g, NB)
    _store(code, sb, out_ptr, sc_ptr, row, blk0, K, CB, NB)


def quant_dyn(x, wg, x46=False, perm=None):
    """x bf16 [M, K] contiguous, wg fp32 [1] (weight global scale) -> (packed uint8 [M, K/2], swizzled scales uint8,
    buf fp32 [3] = (amax, g, alpha))"""
    M, K = x.shape
    buf = torch.zeros(3, device=x.device, dtype=torch.float32)
    n = M * K
    BLOCK = 8192
    _amax_kernel[(triton.cdiv(n, BLOCK),)](x, n, buf, BLOCK=BLOCK, num_warps=8)
    CB = -(-(K // 16) // 4)
    sc = torch.zeros(-(-M // 128) * CB * 512, device=x.device, dtype=torch.uint8)
    out = torch.empty(M, K // 2, device=x.device, dtype=torch.uint8)
    nb = _nb(K)
    if x46 or perm is not None:
        _quant_dyn46_kernel[(M, K // (16 * nb))](x, buf, wg, out, sc, perm if perm is not None else buf, K, CB, NB=nb,
                                                 X46=bool(x46), PERM=perm is not None)
        return out, sc, buf
    _quant_dyn_kernel[(M, K // (16 * nb))](x, buf, wg, out, sc, K, CB, NB=nb)
    return out, sc, buf
