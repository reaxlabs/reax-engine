"""NVFP4 accuracy levers, all derived from the weights at load (no shipped weight or calibration files):

  * block Hadamard rotation (QuaRot/SpinQuant style): y = x W^T = (x H)(W H)^T with H = blockdiag(H_b) along K, so the weights
    are rotated once at load and the activations right before their NVFP4 quantisation;
  * calibration-free smoothing (SmoothQuant with a weight-only activation proxy), power-of-two factors folded into the
    post-attention norm weight / gate|up columns (gate|up input) and into the up rows / down columns (down input): exact folds;
  * block-scale choice for the e4m3 block scales: "amax" (amax/6, the hardware recipe and fp4.quantize_nvfp4), "46"
    (per block the better of amax/6 and amax/4, "four over six"), "mse" (per block the best of nine e4m3 scales around amax/6).

Pure torch (CPU testable); fake_nvfp4 is the bit-faithful emulation used by the study tool and the tests.
"""
from __future__ import annotations

import math

import torch

try:  # in the image engine/ is on sys.path (pplx_engine does `from fp4 import ...`); in the tests it is the `engine` package
    from fp4 import E2M1, f32_to_e2m1_codes, to_blocked
except ImportError:  # pragma: no cover - depends on sys.path
    from engine.fp4 import E2M1, f32_to_e2m1_codes, to_blocked

KINDS = ("gu", "down")
SCALE_MODES = ("amax", "46", "mse")
_MSE_STEPS = range(-2, 7)  # e4m3 code offsets searched by "mse": 0.84x .. ~1.7x of amax/6 (covers the /4 mapping)


# ---------------------------------------------------------------- rotation
def _is_pow2(n: int) -> bool:
    return isinstance(n, int) and n >= 1 and n & (n - 1) == 0


def hadamard(n: int, dtype=torch.float32) -> torch.Tensor:
    """Normalised Sylvester Hadamard matrix H_n (symmetric, orthogonal); n a power of two."""
    if not _is_pow2(n):
        raise ValueError(f"hadamard: {n} is not a power of two")
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(n)).to(dtype)


def block_rotate(x: torch.Tensor, b: int) -> torch.Tensor:
    """x [..., K] -> x @ blockdiag(H_b, ..., H_b). Power-of-4 b in bf16: the matrix is exact (entries +-2^-k) and the product
    runs in bf16 with fp32 accumulation; otherwise the product is computed in fp32 (or x's own fp32/fp64)."""
    if not _is_pow2(b):
        raise ValueError(f"rotation block {b} is not a power of two")
    K = x.shape[-1]
    if K % b:
        raise ValueError(f"rotation block {b} does not divide K={K}")
    if x.dtype in (torch.float32, torch.float64) or (x.dtype == torch.bfloat16 and int(math.log2(b)) % 2 == 0):
        cd = x.dtype
    else:
        cd = torch.float32
    h = hadamard(b, cd).to(x.device)
    y = (x.to(cd).reshape(-1, K // b, b) @ h).reshape(x.shape)
    return y.to(x.dtype)


# ---------------------------------------------------------------- NVFP4 emulation with a block-scale choice
def _codes_for(blk, sf):
    q = blk / sf.clamp(min=1e-30)[..., None]
    return f32_to_e2m1_codes(q.clamp(-6, 6))


def _deq(codes, sf):
    e = E2M1.to(codes.device)
    mag = e[(codes & 7).long()]
    return torch.where((codes >> 3).bool(), -mag, mag) * sf[..., None]


def _nvfp4_codes_rows(xf, g, mode):
    M, K = xf.shape
    blk = xf.view(M, K // 16, 16)
    bamax = blk.abs().amax(-1)
    s = (bamax / (6.0 * g)).clamp(max=448.0).to(torch.float8_e4m3fn)
    codes = _codes_for(blk, s.float() * g)
    if mode == "amax":
        return codes, s
    best = ((_deq(codes, s.float() * g) - blk) ** 2).sum(-1)
    if mode == "46":
        raw4 = bamax / (4.0 * g)
        cands = [torch.where(raw4 <= 448.0, raw4, bamax / (6.0 * g)).clamp(max=448.0).to(torch.float8_e4m3fn)]
    elif mode == "mse":
        b0 = s.view(torch.uint8).to(torch.int16)
        cands = [(b0 + d).clamp(0, 0x7E).to(torch.uint8).view(torch.float8_e4m3fn) for d in _MSE_STEPS if d]
    else:
        raise ValueError(f"NVFP4 scale mode {mode!r}: one of {SCALE_MODES}")
    for sc in cands:
        c = _codes_for(blk, sc.float() * g)
        e = ((_deq(c, sc.float() * g) - blk) ** 2).sum(-1)
        take = e < best  # ties keep the earlier (amax/6) choice
        codes = torch.where(take[..., None], c, codes)
        s = torch.where(take, sc.view(torch.uint8), s.view(torch.uint8)).view(torch.float8_e4m3fn)
        best = torch.minimum(best, e)
    return codes, s


def nvfp4_codes(x: torch.Tensor, amax=None, mode: str = "amax", rows: int = 4096, g=None):
    """x [M, K] -> (e2m1 codes uint8 [M, K] (sign<<3 | idx), e4m3 block scales [M, K/16] (row-major, not swizzled),
    fp32 global scale g). mode "amax" is byte-identical to fp4.quantize_nvfp4. Processed in row chunks (memory).
    g: the global scale itself (fp4_triton.global_scale(amax)), overrides amax."""
    if mode not in SCALE_MODES:
        raise ValueError(f"NVFP4 scale mode {mode!r}: one of {SCALE_MODES}")
    M, K = x.shape
    if K % 16:
        raise ValueError(f"K={K} is not a multiple of 16")
    xf = x.float()
    if g is None:
        amax = xf.abs().amax() if amax is None else torch.as_tensor(amax, dtype=torch.float32, device=x.device)
        g = (amax / (448.0 * 6.0)).clamp(min=1e-12)
    else:
        g = torch.as_tensor(g, dtype=torch.float32, device=x.device).reshape(())
    cs, ss = [], []
    for a in range(0, M, rows):
        c, s = _nvfp4_codes_rows(xf[a:a + rows], g, mode)
        cs.append(c.view(-1, K))
        ss.append(s)
    return torch.cat(cs), torch.cat(ss), g


def dequant(codes, s, g):
    M, K = codes.shape
    return _deq(codes.view(M, K // 16, 16), s.float() * g).view(M, K)


def fake_nvfp4(x, amax=None, mode="amax"):
    """Quantise-dequantise x [M, K] through NVFP4 (fp32 result): exactly the operand the FP4 tensor cores multiply."""
    return dequant(*nvfp4_codes(x, amax, mode))


def quantize_weight(w: torch.Tensor, mode: str = "amax"):
    """Like fp4.quantize_nvfp4(w) (packed fp4x2 [N, K/2], swizzled e4m3 scales, global scale) with a block-scale choice."""
    codes, s, g = nvfp4_codes(w, None, mode)
    packed = (codes[:, 1::2] << 4) | codes[:, ::2]
    return packed.view(torch.float4_e2m1fn_x2), to_blocked(s), g


# ---------------------------------------------------------------- calibration-free smoothing
def smooth_factors(act: torch.Tensor, wmax: torch.Tensor, alpha: float, pow2: bool = True) -> torch.Tensor:
    """SmoothQuant factors s_j = act_j^alpha / wmax_j^(1-alpha), geometric mean 1, clamped to 2^[-8, 8]; rounded to powers of
    two by default so that x / s and W * s are exact. The activation is divided by s, the weight column multiplied."""
    a = act.double().abs().clamp(min=1e-8)
    w = wmax.double().abs().clamp(min=1e-8)
    s = a.pow(alpha) / w.pow(1.0 - alpha)
    s = s / torch.exp(torch.log(s).mean())
    s = s.clamp(2.0 ** -8, 2.0 ** 8)
    if pow2:
        s = torch.exp2(torch.round(torch.log2(s)))
    return s.to(act.dtype if act.dtype.is_floating_point else torch.float32)


def gu_act_proxy(post_w: torch.Tensor) -> torch.Tensor:
    """Per-channel scale of the gate|up input that the weights fix: the norm gain |1 + w| (Qwen3.5 zero-centred RMSNorm)."""
    return (1.0 + post_w.float()).abs()


def down_act_proxy(post_w, wg, wu) -> torch.Tensor:
    """Per-channel scale proxy of h = silu(gate) * up: ||W_gate[j] * gain|| * ||W_up[j] * gain|| (unit-variance normed input)."""
    gain = gu_act_proxy(post_w)
    return (wg.float() * gain).norm(dim=1) * (wu.float() * gain).norm(dim=1)


def fold_gu_smooth(post_w, wg, wu, s):
    """x' = x / s folded into the norm gain (returned in fp32: (1 + w') = (1 + w) / s); gate|up columns * s."""
    sf = s.to(torch.float32 if post_w.dtype != torch.float64 else torch.float64)
    pw = (1.0 + post_w.to(sf.dtype)) / sf - 1.0
    return pw, (wg * s.to(wg.dtype)[None, :]), (wu * s.to(wu.dtype)[None, :])


def fold_down_smooth(wu, wd, s):
    """h' = h / s folded into the up rows (h is linear in up); down columns * s."""
    return wu / s.to(wu.dtype)[:, None], wd * s.to(wd.dtype)[None, :]


# ---------------------------------------------------------------- option parsing
def parse_kind_spec(spec, what: str) -> dict:
    """'' -> {}; '<v>' -> both kinds; 'gu=<v>,down=<v>'. what: 'rot' (power of two 16..1024) | 'smooth' (alpha 0..1)."""
    spec = "" if spec is None else str(spec).strip()
    if not spec:
        return {}
    items = [x.strip() for x in spec.split(",") if x.strip()]
    if len(items) == 1 and "=" not in items[0]:
        pairs = [(k, items[0]) for k in KINDS]
    elif all("=" in x for x in items):
        pairs = [tuple(p.strip() for p in x.split("=", 1)) for x in items]
    else:
        raise ValueError(f"nvfp4_{what} {spec!r}: use <v> or gu=<v>,down=<v>")
    out = {}
    for k, v in pairs:
        if k not in KINDS or k in out:
            raise ValueError(f"nvfp4_{what} {spec!r}: unknown or repeated kind {k!r} (kinds {KINDS})")
        try:
            f = float(v)
        except ValueError:
            raise ValueError(f"nvfp4_{what} {spec!r}: {v!r} is not a number") from None
        if what == "rot":
            if f != int(f) or not _is_pow2(int(f)) or not 16 <= int(f) <= 1024:
                raise ValueError(f"nvfp4_rot {spec!r}: block {v!r} must be a power of two in 16..1024")
            out[k] = int(f)
        elif what == "smooth":
            if not (math.isfinite(f) and 0.0 <= f <= 1.0):
                raise ValueError(f"nvfp4_smooth {spec!r}: alpha {v!r} must be in [0, 1]")
            out[k] = f
        else:
            raise ValueError(what)
    return out


def smooth_mlp(post_w, wg, wu, wd, alphas: dict, gu_nv: bool, down_nv: bool):
    """Apply the calibration-free smoothing of one MLP. alphas: {"gu": a, "down": a} (parse_kind_spec). Only NVFP4 sides are
    smoothed (an FP8 side must keep the reference's numerics): gu -> post_w returned in fp32 ((1 + w) / s - 1), gate|up columns
    * s; down -> up rows / s (exact for FP8 per-row weight scales too: power-of-two factors), down columns * s.
    Returns (post_w, wg, wu, wd)."""
    if gu_nv and "gu" in alphas:
        s1 = smooth_factors(gu_act_proxy(post_w), torch.maximum(wg.abs().amax(0), wu.abs().amax(0)).float(), alphas["gu"])
        post_w, wg, wu = fold_gu_smooth(post_w, wg, wu, s1.to(wg.device))
    if down_nv and "down" in alphas:
        s2 = smooth_factors(down_act_proxy(post_w, wg, wu), wd.abs().amax(0).float(), alphas["down"])
        wu, wd = fold_down_smooth(wu, wd, s2.to(wd.device))
    return post_w, wg, wu, wd


def _ref_static(x, g, mode):
    """torch reference of fp4_triton's static quantisers: (packed bytes, swizzled scale bytes)."""
    codes, s, _ = nvfp4_codes(x, mode=mode, g=g)
    return ((codes[:, 1::2] << 4) | codes[:, ::2]), to_blocked(s).view(torch.uint8)


def selfcheck_four6(fp4_triton, device, M: int = 333, K: int = 5120, tol: float = 1e-3, seed: int = 0) -> dict:
    """Start-up check of the FOUR6 Triton quantisers against nvfp4_codes(mode="46"): the fraction of differing packed bytes
    and scale bytes must stay <= tol (only near-ties of the two block errors may differ: summation order). Raises otherwise."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.distributions.StudentT(3.0).sample((M, K)) * torch.rand(1, K, generator=gen) * 4).to(device, torch.bfloat16)
    g = (x.float().abs().amax() * 1.5 / (448.0 * 6.0)).reshape(1).float()
    out = {}
    q, s, _ = fp4_triton.nvfp4_quant_static(x.contiguous(), g, True)
    rq, rs = _ref_static(x, g, "46")
    out["quant"] = _mismatch(q, s, rq, rs)
    h, d = x, (x * 0.5).contiguous()
    w = (torch.rand(K, generator=gen) * 0.5).to(device, torch.bfloat16)
    _, xn, q2, s2 = fp4_triton.add_rms_nvfp4_static(h, d, w, 1e-6, g, True, True)
    rq2, rs2 = _ref_static(xn, g, "46")
    out["add_rms"] = _mismatch(q2, s2, rq2, rs2)
    bad = {k: v for k, v in out.items() if max(v) > tol}
    if bad:
        raise RuntimeError(f"nvfp4 four-over-six self-check failed (packed, scale mismatch fractions): {bad}")
    return out


def _mismatch(q, s, rq, rs):
    q, s = q.view(torch.uint8), s.view(torch.uint8).flatten()
    n = rs.numel()
    return (float((q != rq.to(q.device)).float().mean()), float((s[:n] != rs.to(s.device)).float().mean()))
