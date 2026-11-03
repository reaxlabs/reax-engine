"""nv4qa.py (kai lever C): activation scaling for rgh's NVFP4 W4A4 projections (nv4.py, MAM_NV4).

MAM_NV4_ASCALE  block scales of the activations (16 values per e4m3 scale):
    rtn  (default) rgh's: e4m3(amax_block / 6g), values RN onto e2m1 (fp4_triton._quant16, the owner's quantiser)
    46   per block the better (squared error) of amax/6 and amax/4 ("four over six"; nvfp4_acc mode "46")
    mse  per block the best of the e4m3 scales -2..+6 codes around amax/6 (nvfp4_acc mode "mse")
MAM_NV4_AGLOBAL per-tensor global scale g of the activations (computed per call from the batch amax, as rgh's):
    pow2 (default) rgh's g0 = 2^ceil(log2(amax / 2688)) (a power of two: a block's e4m3 rounding then does not depend on
         which other rows share the batch)
    pow2h 2 g0: still a power of two (batch independent), every block's amax/6 is <= 224, so the larger scales 46 / mse
         pick (amax/4, up to ~1.7x amax/6) are never cut at 448
    mse  the best of 2 g0 * 2^(k/4), k = 0..3, by the tensor's squared error (an extra error pass; NOT batch independent:
         the choice depends on the whole batch)
Both unset = rgh's own quantiser (nv4q.quant_dyn) is called, bit for bit. Otherwise nv4.quant_x is routed here (install)
and the Triton kernels of nv4qa_k.py run (same 128x4 swizzled scale layout and (amax, g, alpha) buffer as rgh's).
The weights are unchanged (rgh's quant_w, method rtn | mse from the plan).

ref_quant is the torch reference (CPU testable, nvfp4_acc numerics); selfcheck compares the kernels against it on the GPU
(run at start when the lever is on; tolerance: near-ties of block errors may differ by summation order).
"""
from __future__ import annotations

import os

import torch

try:  # in the image engine/ is on sys.path; in the tests too (tests insert it)
    import nvfp4_acc as _na
    from fp4 import to_blocked as _to_blocked
except ImportError:  # pragma: no cover - depends on sys.path
    from engine import nvfp4_acc as _na
    from engine.fp4 import to_blocked as _to_blocked

ENV_BLOCK, ENV_GLOBAL = "MAM_NV4_ASCALE", "MAM_NV4_AGLOBAL"
BLOCKS = ("rtn", "46", "mse")       # index = nv4qa_k AM constexpr
GLOBALS = ("pow2", "pow2h", "mse")  # index = nv4qa_k GM constexpr
GCAND = (1.0, 1.189207115002721, 1.4142135623730951, 1.681792830507429)  # 2^(k/4), k = 0..3 (global mse candidates)
_NA_MODE = {"rtn": "amax", "46": "46", "mse": "mse"}


def parse_block(v) -> str:
    s = "" if v is None else str(v).strip()
    s = s or "rtn"
    if s not in BLOCKS:
        raise ValueError(f"{ENV_BLOCK}={v!r}: one of {BLOCKS}")
    return s


def parse_global(v) -> str:
    s = "" if v is None else str(v).strip()
    s = s or "pow2"
    if s not in GLOBALS:
        raise ValueError(f"{ENV_GLOBAL}={v!r}: one of {GLOBALS}")
    return s


def resolve(env) -> tuple:
    """(block mode, global mode) of this environment; ValueError on a bad value"""
    return parse_block(env.get(ENV_BLOCK)), parse_global(env.get(ENV_GLOBAL))


def is_default(modes) -> bool:
    return tuple(modes) == ("rtn", "pow2")


BLOCK, GLOBAL = resolve(os.environ)


def set_modes(block=None, glob=None) -> None:
    """switch at run time (levers.py); validated before anything changes"""
    global BLOCK, GLOBAL
    b = parse_block(block) if block is not None else BLOCK
    g = parse_global(glob) if glob is not None else GLOBAL
    BLOCK, GLOBAL = b, g


def quant_x(x, wg):
    """nv4.quant_x replacement: bf16 [M, K] contiguous -> (packed uint8 [M, K/2], swizzled scales uint8, buf fp32 [3] =
    (amax, g_x, alpha = g_x * g_w)). Default modes: rgh's nv4q.quant_dyn itself."""
    if BLOCK == "rtn" and GLOBAL == "pow2":
        import nv4q
        return nv4q.quant_dyn(x, wg)
    import nv4qa_k
    return nv4qa_k.quant(x, wg, BLOCKS.index(BLOCK), GLOBALS.index(GLOBAL))


def install(nv4) -> None:
    """route every NV4Linear's activation quantisation through quant_x (NV4Linear.mm reads nv4.quant_x per call)"""
    nv4.quant_x = quant_x


# ---------------------------------------------------------------- torch reference
def pow2_scale(amax) -> torch.Tensor:
    """rgh's g0 = 2^ceil(log2(max(amax / 2688, 1e-30))) in fp32"""
    a = torch.as_tensor(amax, dtype=torch.float32)
    return torch.exp2(torch.ceil(torch.log2(torch.clamp(a / 2688.0, min=1e-30)))).reshape(())


def _sse(xf, codes, s, g) -> float:
    return float(((_na.dequant(codes, s, g).double() - xf.double()) ** 2).sum())


def global_candidates(g0, glob):
    if glob == "pow2":
        return [g0]
    gb = g0 * 2.0
    if glob == "pow2h":
        return [gb]
    return [gb * torch.tensor(f, dtype=torch.float32, device=gb.device) for f in GCAND]


def ref_quant(x, block="rtn", glob="pow2", g0=None):
    """x [M, K] -> (e2m1 codes uint8 [M, K], e4m3 block scales [M, K/16] row-major, g fp32 0-d); the global mse choice
    keeps the first candidate on ties"""
    block, glob = parse_block(block), parse_global(glob)
    xf = x.float()
    if g0 is None:
        g0 = pow2_scale(xf.abs().amax())
    best = None
    for g in global_candidates(g0.reshape(()).to(xf.device), glob):
        c, s, _ = _na.nvfp4_codes(x, mode=_NA_MODE[block], g=g)
        if glob != "mse":
            return c, s, g
        e = _sse(xf, c, s, g)
        if best is None or e < best[0]:
            best = (e, c, s, g)
    return best[1], best[2], best[3]


def pack(codes, s):
    """(codes [M, K], scales [M, K/16]) -> (packed uint8 [M, K/2] (low nibble = even column), swizzled scale bytes as
    nv4q.quant_dyn allocates them: ceil(M/128) * ceil(K/64) * 512)"""
    packed = ((codes[:, 1::2] << 4) | codes[:, ::2]).to(torch.uint8)
    return packed, _to_blocked(s.view(torch.uint8)).contiguous()


# ---------------------------------------------------------------- GPU self-check of the kernels against the reference
def check_input(M: int, K: int, seed: int = 0) -> torch.Tensor:
    """the self-check's heavy-tailed fp32 data [M, K] (Student-t, 3 dof, per-column scales, one outlier), drawn from its own
    seed under a forked CPU RNG: the same data at every start, and the engine's global RNG state is left as it was"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        x = torch.distributions.StudentT(3.0).sample((M, K)) * (torch.rand(1, K) * 4 + 0.05)
    x[M // 3, K // 7] = 61.0
    return x


def selfcheck(device, M: int = 333, K: int = 5120, tol: float = 1e-3, seed: int = 0, pairs=None) -> dict:
    """every non-default (block, global) pair of nv4qa_k against ref_quant on heavy-tailed data: fractions of differing
    packed bytes and scale bytes must be <= tol (block-error near-ties), g and alpha must match (a global mse choice may
    differ only on a tie of the tensor errors). pairs: [(block, global)] to check (default: all; the engine start checks
    only its own pair, each pair compiles its own kernels). Returns {"block/global": (packed, scales, g) mismatch};
    raises otherwise."""
    import nv4qa_k
    x = check_input(M, K, seed).to(device=device, dtype=torch.bfloat16).contiguous()
    xf = x.float()
    wg = torch.tensor([0.37], device=device, dtype=torch.float32)
    out, bad = {}, {}
    for bi, b in enumerate(BLOCKS):
        for gi, gl in enumerate(GLOBALS):
            if (b, gl) == ("rtn", "pow2") or (pairs is not None and (b, gl) not in {tuple(p) for p in pairs}):
                continue
            packed, sc, buf = nv4qa_k.quant(x, wg, bi, gi)
            kg = buf[1].reshape(())
            rc, rs, rg = ref_quant(x, b, gl)
            gmis = 0.0
            if float(kg) != float(rg):
                tie = gl == "mse" and any(float(kg) == float(c) for c in global_candidates(pow2_scale(xf.abs().amax()), gl))
                if tie:
                    kc, ks, _ = _na.nvfp4_codes(x, mode=_NA_MODE[b], g=kg)
                    e_k, e_r = _sse(xf, kc, ks, kg), _sse(xf, rc, rs, rg)
                    tie = abs(e_k - e_r) <= 1e-5 * max(e_r, 1e-30)
                    rc, rs, rg = kc, ks, kg
                gmis = 0.0 if tie else 1.0
            if abs(float(buf[2]) - float(rg) * float(wg)) > 1e-6 * abs(float(rg) * float(wg)):
                gmis = 1.0
            rp, rsc = pack(rc, rs)
            n = rsc.numel()
            pm = float((packed.view(torch.uint8) != rp.to(packed.device)).float().mean())
            sm = float((sc.view(torch.uint8).flatten()[:n] != rsc.to(sc.device)).float().mean())
            out[f"{b}/{gl}"] = (pm, sm, gmis)
            if pm > tol or sm > tol or gmis:
                bad[f"{b}/{gl}"] = out[f"{b}/{gl}"]
    if bad:
        raise RuntimeError(f"nv4qa self-check failed (packed, scale, g mismatch): {bad}")
    return out
