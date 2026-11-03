"""actq.py (kai lever Q, KAI_ACTQ; unset / 0 = off): the NVFP4 activation quantisation of rgh's NV4 projections (nv4.py,
MAM_NV4) fed straight from the FP8 producers the engine already runs. Bit-identical to nv4q.quant_dyn on the same input:
the same global amax (so the same power-of-two scale g), then nv4q's own per-block quantiser (fp4_triton._quant16).

In the base stack an NV4 projection's input arrives as (x8, xs) = FP8 e4m3 rows + per-row fp32 scales from a fused
norm+quant producer (the engine picks FP8 producers because NV4Linear.mode is "fp8"), and nv4q then runs a zero fill, an
atomic-max pass over the whole bf16 input and the quant pass. Parts (KAI_ACTQ = comma list; 1 / all = every part):

  deq   out_proj / o_proj (only (x8, xs) is passed): NV4Linear first dequantises with three eager torch kernels
        (x8.float(), * xs, .to(bf16): ~11 B moved per element). Here: the global amax comes from the row scales alone and
        the quant pass reads x8 + xs and dequantises in registers (bf16(fp32(x8) * xs), the same IEEE ops), so there is
        no bf16 copy, no fp32 temporaries and no pass over the data for the amax.
        Why the amax is exact: a row whose amax a >= 1e-12 gets xs = a / 448, and its largest element quantises to
        e4m3(a / xs) = 448 (a / xs is within a few ulp of 448, clamped to <= 448, and e4m3's next value is 416), so the
        row's largest dequantised magnitude is bf16(fp32(448 * xs)); fp32 multiply and bf16 rounding are monotone, so the
        tensor amax is bf16(fp32(448 * max xs)). A row with a < 1e-12 (xs clamped) gives < 2e-12: if the top value is
        < 2e-12 the kernel reduces the data itself (never seen in practice; exact either way).
  xs    in_proj_qkvz / qkv / gate-up (bf16 x passed together with its (x8, xs)): max|x| = bf16(a*) with a* = the row
        amax the producer divided by 448 (Inductor keeps it in fp32, the Triton producers round first; both give
        max|x| = bf16(a*)). 448 * max xs is within 2^-21 of a* (division <= 2 ulp, multiply 0.5 ulp), so when
        bf16(u (1 - 2^-19)) == bf16(u (1 + 2^-19)) for u = fp32(448 * max xs), that value IS max|x|; otherwise (an a*
        within 2^-19 of a bf16 rounding boundary: ~0.2 % of calls) the same one-program kernel reduces x itself. The
        quant pass is nv4q._quant_dyn_kernel unchanged.
  silu  down_proj (bf16 silu(g) * u from the compiled K.silu_mul, then nv4q): one kernel writes silu(g) * u with the
        compiled kernel's numerics (Inductor's silu decomposition g / (1 + exp(-g)) in fp32, libdevice exp, one bf16
        rounding after * u) and its atomic amax; the quant pass is nv4q's. Bit-equality with K.silu_mul is a start-up
        check (check()), not an assumption.

Kai lever C (nv4qa: MAM_NV4_ASCALE / MAM_NV4_AGLOBAL) changes the quantiser itself: in its non-default modes actq stays out
of the way (rgh_quant), so C and Q compose as "C wins".

check() runs every enabled part on the serving GPU against the reference chain (the engine's own producers, nv4q) at start;
a part whose bytes differ anywhere is switched off with a log line (the reference path stays). tools/actqbench.py is the
GPU microbench (bit-equality incl. adversarial rows + speed per part).
"""
from __future__ import annotations

import os

import torch

PARTS_ALL = ("deq", "xs", "silu")
ENV = "KAI_ACTQ"
PARTS: frozenset = frozenset()  # read at call time (levers.py switches it)
EPS = 2.0 ** -19  # xs: relative bound on |448 * xs - a| / a (true bound ~2^-21)
TINY = 2e-12  # below: a clamped row (amax < 1e-12) cannot be told apart -> reduce the data
AMAX_BLOCK = 1024  # rows per iteration of the scale pass
FB_BLOCK = 4096  # elements per iteration of the (rare) data pass
SILU_BLOCK = 1024

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice  # the compiled kernels' exp

    from fp4_triton import _nb, _quant16, _store  # the owner's per-block quantiser + swizzled store (as nv4q)
except Exception:  # CPU test machines: the math helpers below still import
    triton = tl = libdevice = None


def parse(text) -> frozenset:
    s = ("" if text is None else str(text)).strip().lower()
    if s in ("", "0", "off", "false", "no"):
        return frozenset()
    if s in ("1", "all", "on", "true", "yes"):
        return frozenset(PARTS_ALL)
    parts = [p.strip() for p in s.replace("+", ",").split(",") if p.strip()]
    bad = [p for p in parts if p not in PARTS_ALL]
    if bad:
        raise ValueError(f"{ENV}={text!r}: unknown part(s) {bad} (parts: {', '.join(PARTS_ALL)}; 1 / all = every part)")
    return frozenset(parts)


def resolve(env) -> frozenset:
    return parse(env.get(ENV))


# ------------------------------------------------------------------------------------------------ torch references (CPU)
def bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).float()


def amax_from_scales_ref(s: torch.Tensor, mode: str):
    """the scale pass in torch: fp32 amax tensor (0-dim) or None (= the kernel would reduce the data)"""
    u = s.float().max() * 448.0
    if mode == "deq":
        am = bf16(u)
        return am if am >= TINY else None
    lo, hi = bf16(u * (1.0 - EPS)), bf16(u * (1.0 + EPS))
    return lo if (lo == hi and lo >= TINY) else None


def fp8_rows_ref(x: torch.Tensor):
    """the engine's FP8 row quant (pplx_engine._q): (x8, xs) of x (fp32 or bf16 rows)"""
    s = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12) / 448.0
    return (x.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn), s


def silu_mul_inductor_ref(gu: torch.Tensor, inter: int) -> torch.Tensor:
    """compiled K.silu_mul numerics in torch: fp32 g / (1 + exp(-g)) * u, one bf16 rounding"""
    g, u = gu[:, :inter].float(), gu[:, inter:].float()
    return ((g / (1.0 + torch.exp(-g))) * u).to(torch.bfloat16)


# ------------------------------------------------------------------------------------------------ kernels
if triton is not None:
    @triton.jit
    def _amax_scales_kernel(s_ptr, M, x_ptr, xstride, K, n, buf_ptr, TINY_: tl.constexpr, EPS_: tl.constexpr,
                            MODE: tl.constexpr, BLOCK: tl.constexpr, FB: tl.constexpr):
        """one program. MODE 0 (deq): x_ptr = x8 [M, K] e4m3, buf[0] = max |bf16(fp32(x8) * xs)|. MODE 1 (xs): x_ptr =
        x [M, K] bf16, buf[0] = max |x|. Both from the row scales s [M] when that is exact, else from the data."""
        offs = tl.arange(0, BLOCK)
        acc = tl.zeros([BLOCK], tl.float32)
        for i in range(0, M, BLOCK):
            acc = tl.maximum(acc, tl.load(s_ptr + i + offs, mask=i + offs < M, other=0.0))
        u = tl.max(acc, axis=0) * 448.0
        if MODE == 0:
            am = u.to(tl.bfloat16).to(tl.float32)
            ok = am >= TINY_
        else:
            am = (u * (1.0 - EPS_)).to(tl.bfloat16).to(tl.float32)
            hi = (u * (1.0 + EPS_)).to(tl.bfloat16).to(tl.float32)
            ok = (am == hi) & (am >= TINY_)
        if ok.to(tl.int32) == 0:
            fo = tl.arange(0, FB)
            dacc = tl.zeros([FB], tl.float32)
            for j in range(0, n, FB):
                idx = j + fo
                m = idx < n
                r = idx // K
                c = idx - r * K
                if MODE == 0:
                    v = (tl.load(x_ptr + r * xstride + c, mask=m, other=0.0).to(tl.float32)
                         * tl.load(s_ptr + r, mask=m, other=0.0)).to(tl.bfloat16).to(tl.float32)
                else:
                    v = tl.load(x_ptr + r * xstride + c, mask=m, other=0.0).to(tl.float32)
                dacc = tl.maximum(dacc, tl.abs(v))
            am = tl.max(dacc, axis=0)
        tl.store(buf_ptr, am)

    @triton.jit
    def _quant_deq_kernel(q_ptr, s_ptr, qstride, buf_ptr, wg_ptr, out_ptr, sc_ptr, K, CB, NB: tl.constexpr):
        """nv4q._quant_dyn_kernel on bf16(fp32(x8) * xs) computed in registers"""
        row, pk = tl.program_id(0).to(tl.int64), tl.program_id(1)
        am = tl.load(buf_ptr)
        g = tl.exp2(tl.ceil(tl.log2(tl.maximum(am / 2688.0, 1e-30))))
        if row == 0:
            if pk == 0:
                tl.store(buf_ptr + 1, g)
                tl.store(buf_ptr + 2, g * tl.load(wg_ptr))
        blk0 = pk * NB
        offs = (blk0 + tl.arange(0, NB))[:, None] * 16 + tl.arange(0, 16)[None, :]
        v = (tl.load(q_ptr + row * qstride + offs).to(tl.float32) * tl.load(s_ptr + row)).to(tl.bfloat16).to(tl.float32)
        code, sb = _quant16(v, g, NB)
        _store(code, sb, out_ptr, sc_ptr, row, blk0, K, CB, NB)

    @triton.jit
    def _silu_amax_kernel(gu_ptr, y_ptr, buf_ptr, I, BLOCK: tl.constexpr, EAGER: tl.constexpr):
        """y = silu(g) * u as the compiled K.silu_mul (EAGER=False) or as eager torch (EAGER=True: bf16 after silu and after
        the product); atomic max of |y| into buf[0] (zeroed by the caller)"""
        row = tl.program_id(0).to(tl.int64)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = c < I
        g = tl.load(gu_ptr + row * (2 * I) + c, mask=m, other=0.0).to(tl.float32)
        u = tl.load(gu_ptr + row * (2 * I) + I + c, mask=m, other=0.0).to(tl.float32)
        a = g / (1.0 + libdevice.exp(-g))
        if EAGER:
            a = a.to(tl.bfloat16).to(tl.float32)
        y = (a * u).to(tl.bfloat16)
        tl.store(y_ptr + row * I + c, y, mask=m)
        tl.atomic_max(buf_ptr, tl.max(tl.where(m, tl.abs(y.to(tl.float32)), 0.0), axis=0))


def _scales_out(M, K, dev):
    CB = -(-(K // 16) // 4)
    sc = torch.zeros(-(-M // 128) * CB * 512, device=dev, dtype=torch.uint8)  # = nv4q.quant_dyn (padding rows 0)
    out = torch.empty(M, K // 2, device=dev, dtype=torch.uint8)
    return CB, sc, out


def _amax(s, x, mode: int, buf):
    M, K = x.shape
    _amax_scales_kernel[(1,)](s, M, x, x.stride(0), K, M * K, buf, TINY_=TINY, EPS_=EPS, MODE=mode, BLOCK=AMAX_BLOCK,
                              FB=FB_BLOCK, num_warps=8)


def _row_scales(s, M):
    return s.reshape(M) if s.is_contiguous() else s.reshape(M).contiguous()


def quant_deq(x8, xs, wg):
    """== nv4q.quant_dyn((x8.float() * xs).to(bf16), wg) without the bf16 tensor; x8 [M, K] e4m3 (row stride free,
    columns contiguous), xs [M, 1] fp32"""
    M, K = x8.shape
    s = _row_scales(xs, M)
    buf = torch.empty(3, device=x8.device, dtype=torch.float32)
    _amax(s, x8, 0, buf)
    CB, sc, out = _scales_out(M, K, x8.device)
    nb = _nb(K)
    _quant_deq_kernel[(M, K // (16 * nb))](x8, s, x8.stride(0), buf, wg, out, sc, K, CB, NB=nb)
    return out, sc, buf


def quant_xs(x, xs, wg):
    """== nv4q.quant_dyn(x, wg) for x [M, K] bf16 contiguous whose FP8 row scales are xs [M, 1] (amax from the scales)"""
    import nv4q
    M, K = x.shape
    s = _row_scales(xs, M)
    buf = torch.empty(3, device=x.device, dtype=torch.float32)
    _amax(s, x, 1, buf)
    CB, sc, out = _scales_out(M, K, x.device)
    nb = _nb(K)
    nv4q._quant_dyn_kernel[(M, K // (16 * nb))](x, buf, wg, out, sc, K, CB, NB=nb)
    return out, sc, buf


def silu_y(gu, inter: int, buf, eager: bool = False):
    M = gu.shape[0]
    y = torch.empty(M, inter, device=gu.device, dtype=torch.bfloat16)
    _silu_amax_kernel[(M, triton.cdiv(inter, SILU_BLOCK))](gu, y, buf, inter, BLOCK=SILU_BLOCK, EAGER=eager, num_warps=4)
    return y


SILU_EAGER = False  # set by check() if (only) the eager formula matches K.silu_mul


def quant_silu(gu, inter: int, wg):
    """== nv4q.quant_dyn(K.silu_mul(gu, inter), wg); gu [M, 2 inter] bf16 contiguous"""
    import nv4q
    M = gu.shape[0]
    buf = torch.zeros(3, device=gu.device, dtype=torch.float32)
    y = silu_y(gu, inter, buf, SILU_EAGER)
    CB, sc, out = _scales_out(M, inter, gu.device)
    nb = _nb(inter)
    nv4q._quant_dyn_kernel[(M, inter // (16 * nb))](y, buf, wg, out, sc, inter, CB, NB=nb)
    return out, sc, buf


# ------------------------------------------------------------------------------------------------ routing
def _fp8_pair(q, M=None):
    if not (isinstance(q, tuple) and len(q) == 2):
        return False
    x8, xs = q
    return (isinstance(x8, torch.Tensor) and isinstance(xs, torch.Tensor) and x8.dtype == torch.float8_e4m3fn
            and x8.dim() == 2 and x8.stride(1) == 1 and xs.dtype == torch.float32 and xs.numel() == x8.shape[0]
            and (M is None or x8.shape[0] == M) and x8.is_cuda)


def route(lin, x, q):
    """which part handles NV4Linear(x, q): "deq", "xs" or None (nv4's own path)"""
    P = PARTS
    if not P or not _fp8_pair(q):
        return None
    if x is None:
        return "deq" if "deq" in P else None
    if "xs" in P and x.dtype == torch.bfloat16 and x.dim() == 2 and tuple(x.shape) == tuple(q[0].shape):
        return "xs"
    return None


def rgh_quant(nv4) -> bool:
    """True while NV4Linear quantises with rgh's nv4q.quant_dyn: nv4.quant_x is nv4's own, or kai lever C's router (nv4qa)
    in its default modes (which calls rgh's). Lever C's other modes (and any other router) keep nv4's path."""
    q = getattr(nv4, "quant_x", None)
    mod = getattr(q, "__module__", "")
    if mod == "nv4":
        return True
    if mod == "nv4qa":
        import sys
        qa = sys.modules.get("nv4qa")
        return qa is not None and qa.BLOCK == "rtn" and qa.GLOBAL == "pow2"
    return False


def install(nv4, pp, parts):
    """patch nv4.NV4Linear.__call__ (deq / xs) and PplxEngine._mlp (silu) once; set PARTS. The patches read PARTS per call,
    so PARTS = frozenset() is exactly nv4's path."""
    global PARTS
    NV = nv4.NV4Linear
    if not getattr(NV, "_kai_actq", False):
        orig_call = NV.__call__

        def __call__(self, x, q=None, out=None):
            r = route(self, x, q) if PARTS and rgh_quant(nv4) else None
            if r == "deq":
                return nv4.gemm(*quant_deq(q[0], q[1], self.wg), self, out)
            if r == "xs":
                return nv4.gemm(*quant_xs(x.contiguous(), q[1], self.wg), self, out)
            return orig_call(self, x, q, out)
        NV.__call__ = __call__
        NV._kai_actq_orig = orig_call
        NV._kai_actq = True
    E = pp.PplxEngine
    if not getattr(E, "_kai_actq", False):
        prev = E._mlp

        def _mlp(self, L, x, xq=None):
            gu, dn = L["gu"], L["down"]
            if ("silu" in PARTS and getattr(gu, "fq4", False) and getattr(dn, "fq4", False)
                    and isinstance(dn, NV) and x.is_cuda and rgh_quant(nv4)):
                g = gu(x, xq)
                return nv4.gemm(*quant_silu(g.contiguous(), self.inter, dn.wg), dn, None)
            return prev(self, L, x, xq)
        E._mlp = _mlp
        E._kai_actq_prev_mlp = prev
        E._kai_actq = True
    PARTS = frozenset(parts)
    return PARTS


# ------------------------------------------------------------------------------------------------ start-up check
def _cmp(ref, new):
    """number of differing items over (packed, scales, buf)"""
    return int(sum(int((r.view(torch.uint8) != n.view(torch.uint8)).sum()) if r.dtype == torch.uint8 else
                   int((r != n).sum()) for r, n in zip(ref, new)))


def check(pp, parts, device="cuda", M=(611, 1503), seed=1234, log=print):
    """run each part against the reference chain on synthetic activations made by the engine's own producers (compiled
    K.add_rms_q / K.gate_mul_q / K.silu_mul, the Triton gated-norm quant); returns the parts that are bit-identical
    everywhere (the others stay off)"""
    import nv4q
    global SILU_EAGER
    K = pp.K
    gen = torch.Generator(device=device).manual_seed(seed)
    wg = torch.full((1,), 3.1e-4, device=device, dtype=torch.float32)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, device=device, generator=gen) * scale).to(torch.bfloat16)

    ok, bad = set(), {}
    for p in parts:
        n_bad = 0
        try:
            for m in M:
                if p == "xs":
                    N = 5120
                    _, x, _, xs = K.add_rms_q(rnd(m, N, scale=4), rnd(m, N, scale=2), rnd(N, scale=0.3), 1e-6)
                    n_bad += _cmp(nv4q.quant_dyn(x.contiguous(), wg), quant_xs(x.contiguous(), xs, wg))
                elif p == "deq":
                    o, gt = rnd(m, 6144, scale=3), rnd(m, 6144)
                    o[: m // 7] = 0  # zero rows: clamped scales
                    pairs = [K.gate_mul_q(o, gt)]
                    if getattr(pp, "gated_norm_q_triton", None) is not None:
                        pairs.append(pp.gated_norm_q_triton(o.view(1, m, 48, 128), gt, rnd(128, scale=0.5), 1e-6, m))
                    pairs.append((torch.zeros(5, 256, device=device, dtype=torch.float8_e4m3fn),  # all rows clamped
                                  torch.full((5, 1), 1e-12 / 448.0, device=device)))
                    for a8, as_ in pairs:
                        n_bad += _cmp(nv4q.quant_dyn((a8.float() * as_).to(torch.bfloat16), wg), quant_deq(a8, as_, wg))
                elif p == "silu":
                    inter = 17408
                    gu = rnd(m, 2 * inter, scale=3)
                    yref = K.silu_mul(gu, inter).contiguous()
                    hit = None
                    for eager in (False, True):
                        y = silu_y(gu, inter, torch.zeros(3, device=device), eager)
                        if torch.equal(y.view(torch.int16), yref.view(torch.int16)):
                            hit = eager
                            break
                    if hit is None:
                        n_bad += 1
                        break
                    SILU_EAGER = hit
                    n_bad += _cmp(nv4q.quant_dyn(yref, wg), quant_silu(gu, inter, wg))
        except Exception as exc:  # noqa: BLE001  any failure keeps the reference path
            log(f"kai actq: check {p} raised {type(exc).__name__}: {exc}")
            n_bad = -1
        if n_bad == 0:
            ok.add(p)
        else:
            bad[p] = n_bad
    torch.cuda.synchronize(device)
    if bad:
        log(f"kai actq: check FAILED for {sorted(bad)} ({bad}); those parts keep nv4's path")
    return frozenset(ok)
