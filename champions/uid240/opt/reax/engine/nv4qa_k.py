"""nv4qa_k.py (kai lever C): Triton kernels of nv4qa (NVFP4 activation scaling for rgh's NV4 projections).

quant(x, wg, AM, GM) = rgh's nv4q.quant_dyn (zero a 3-float buffer, rgh's atomic-max pass, the quant pass writing packed
e2m1 codes + 128x4 swizzled e4m3 scales and (amax, g, alpha = g * g_w)) with
  AM block scales: 0 rtn (the owner's _quant16 numerics), 1 four over six (choice by _err_fast), 2 mse (best of e4m3 codes -2..+6 around amax/6)
  GM global scale: 0 pow2 (rgh's), 1 pow2h (2x rgh's), 2 mse (best of 2x rgh's * 2^(k/4), k = 0..3, by squared error:
     one more pass writes per-(row, chunk) partial errors, a fixed-order torch sum + argmin picks k on the device, no sync)
Torch reference: nv4qa.ref_quant; GPU check: nv4qa.selfcheck.
"""
import torch
import triton
import triton.language as tl

from fp4_triton import _codes16, _store, _nb  # owner's e2m1 rounding + swizzled store (reference image)
from nv4q import _amax_kernel  # rgh's atomic-max pass
from torch._inductor.runtime.triton_helpers import libdevice

NUM_WARPS = 4  # quant pass warps (rgh's launch default)


@triton.jit
def _err_fast(v, sf):
    """squared error of the e2m1 rounding of v [NB,16] at block scales sf [NB] (> 0), for choosing a scale only: the
    reciprocal multiply instead of the IEEE division, the grid rounding by rint per segment (round half to even = the
    even code index on the e2m1 grid); the codes themselves are then made with _codes16 (division) at the chosen scale"""
    r = 1.0 / sf
    a = tl.minimum(tl.abs(v) * r[:, None], 6.0)
    m = tl.where(a < 2.0, libdevice.rint(a * 2.0) * 0.5,
                 tl.where(a < 4.0, libdevice.rint(a), tl.minimum(libdevice.rint(a * 0.5) * 2.0, 6.0)))
    d = m * sf[:, None] - tl.abs(v)
    return tl.sum(d * d, axis=1)


@triton.jit
def _q16e(v, g, NB: tl.constexpr, AM: tl.constexpr):
    """v [NB,16] fp32 -> (codes int32 [NB,16], scale bytes uint8 [NB], squared error fp32 [NB])"""
    bamax = tl.max(tl.abs(v), axis=1)
    raw6 = tl.math.div_rn(bamax, 6.0 * g)
    s8 = tl.minimum(raw6, 448.0).to(tl.float8e4nv)
    sf = tl.maximum(s8.to(tl.float32) * g, 1e-30)
    code, deq = _codes16(v, sf)
    err = tl.sum((deq - v) * (deq - v), axis=1)
    sb = s8.to(tl.uint8, bitcast=True)
    if AM == 1:  # four over six: the scale choice by _err_fast (~10 % less time than two division passes, GPU bench
        # 28 Oct: 101 vs 115 us at M2048 K17408), the codes by one division pass at the chosen scale; equal to the
        # reference except near-ties of the two block errors (selfcheck: <= 1e-5 of the bytes)
        raw4 = tl.math.div_rn(bamax, 4.0 * g)
        s4 = tl.minimum(tl.where(raw4 <= 448.0, raw4, raw6), 448.0).to(tl.float8e4nv)
        sf4 = tl.maximum(s4.to(tl.float32) * g, 1e-30)
        take = _err_fast(v, sf4) < _err_fast(v, sf)
        code, deq = _codes16(v, tl.where(take, sf4, sf))
        sb = tl.where(take, s4.to(tl.uint8, bitcast=True), sb)
        err = tl.sum((deq - v) * (deq - v), axis=1)
    if AM == 2:
        b0 = sb.to(tl.int32)
        for d in tl.static_range(-2, 7):
            if d != 0:
                bc = tl.minimum(tl.maximum(b0 + d, 0), 126).to(tl.uint8)
                sfc = tl.maximum(bc.to(tl.float8e4nv, bitcast=True).to(tl.float32) * g, 1e-30)
                c2, d2 = _codes16(v, sfc)
                e2 = tl.sum((d2 - v) * (d2 - v), axis=1)
                take = e2 < err  # ties keep the earlier candidate (nvfp4_acc order: amax/6, -2, -1, +1 .. +6)
                code = tl.where(take[:, None], c2, code)
                sb = tl.where(take, bc, sb)
                err = tl.minimum(err, e2)
    return code, sb, err


@triton.jit
def _gbase(am, GM: tl.constexpr):
    g = tl.exp2(tl.ceil(tl.log2(tl.maximum(am / 2688.0, 1e-30))))  # rgh's power of two (nv4q._quant_dyn_kernel)
    if GM >= 1:
        g = g * 2.0
    return g


@triton.jit
def _gmul(g, k):
    """g * 2^(k/4) with the fp32 constants of nv4qa.GCAND (g is a power of two: the products are exact)"""
    return tl.where(k == 1, g * 1.189207115002721,
                    tl.where(k == 2, g * 1.4142135623730951, tl.where(k == 3, g * 1.681792830507429, g)))


@triton.jit
def _gerr_kernel(x_ptr, buf_ptr, part_ptr, K, NB: tl.constexpr, AM: tl.constexpr):
    row, pk = tl.program_id(0).to(tl.int64), tl.program_id(1)
    g = _gbase(tl.load(buf_ptr), 2)
    blk0 = pk * NB
    offs = (blk0 + tl.arange(0, NB))[:, None] * 16 + tl.arange(0, 16)[None, :]
    v = tl.load(x_ptr + row * K + offs).to(tl.float32)
    base = (row * tl.num_programs(1) + pk) * 4
    for k in tl.static_range(4):
        _c, _s, e = _q16e(v, _gmul(g, tl.full([], k, tl.int32)), NB, AM)
        tl.store(part_ptr + base + k, tl.sum(e, axis=0))


@triton.jit
def _quant_kernel(x_ptr, buf_ptr, wg_ptr, k_ptr, out_ptr, sc_ptr, K, CB, NB: tl.constexpr, AM: tl.constexpr,
                  GM: tl.constexpr):
    row, pk = tl.program_id(0).to(tl.int64), tl.program_id(1)
    g = _gbase(tl.load(buf_ptr), GM)
    if GM == 2:
        g = _gmul(g, tl.load(k_ptr))
    if row == 0:
        if pk == 0:
            tl.store(buf_ptr + 1, g)
            tl.store(buf_ptr + 2, g * tl.load(wg_ptr))
    blk0 = pk * NB
    offs = (blk0 + tl.arange(0, NB))[:, None] * 16 + tl.arange(0, 16)[None, :]
    v = tl.load(x_ptr + row * K + offs).to(tl.float32)
    code, sb, _e = _q16e(v, g, NB, AM)
    _store(code, sb, out_ptr, sc_ptr, row, blk0, K, CB, NB)


def quant(x, wg, AM: int, GM: int):
    """x bf16 [M, K] contiguous, wg fp32 [1] -> (packed uint8 [M, K/2], swizzled scales uint8, buf fp32 [3] = (amax, g, alpha))"""
    M, K = x.shape
    buf = torch.zeros(3, device=x.device, dtype=torch.float32)
    n = M * K
    BLOCK = 8192
    _amax_kernel[(triton.cdiv(n, BLOCK),)](x, n, buf, BLOCK=BLOCK, num_warps=8)
    CB = -(-(K // 16) // 4)
    sc = torch.zeros(-(-M // 128) * CB * 512, device=x.device, dtype=torch.uint8)
    out = torch.empty(M, K // 2, device=x.device, dtype=torch.uint8)
    nb = _nb(K)
    grid = (M, K // (16 * nb))
    kbest = buf  # unused unless GM == 2
    if GM == 2:
        part = torch.empty(M * grid[1] * 4, device=x.device, dtype=torch.float32)
        _gerr_kernel[grid](x, buf, part, K, NB=nb, AM=AM)
        kbest = part.view(-1, 4).sum(0).argmin().to(torch.int32).reshape(1)
    _quant_kernel[grid](x, buf, wg, kbest, out, sc, K, CB, NB=nb, AM=AM, GM=GM, num_warps=NUM_WARPS)
    return out, sc, buf
