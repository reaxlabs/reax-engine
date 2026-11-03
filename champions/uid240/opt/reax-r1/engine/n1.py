"""n1.py (kai lever N1, KAI_N1; unset / 0 = off: this module and libnv4s.so are never imported): SwiGLU and the global
amax of the down projection's NVFP4 quantisation folded into the epilogue of rgh's NVFP4 gate|up GEMM.

Base chain of an MLP layer whose gate|up and down are both rgh NV4 projections (nv4.py, MAM_NV4):
    gate|up GEMM -> bf16 g|u [M, 2I] -> actq's _silu_amax_kernel (or K.silu_mul + nv4q's amax pass) -> y [M, I] + amax
    -> nv4q._quant_dyn_kernel -> down GEMM
With N1:
    gate|up GEMM with the fused epilogue (libnv4s.so, nv4s.cu) -> y [M, I] + amax -> nv4q._quant_dyn_kernel -> down GEMM
The [M, 2I] bf16 tensor is never written and the silu kernel disappears (profile: 82 us x 21 layers per mammoth step).

At install the gate|up weight rows are interleaved in place (row 2j = gate row j, row 2j + 1 = up row j, with their
NVFP4 block scales; the per-tensor global scale is unchanged), so one epilogue thread holds the gate and up value of an
output column. Exact (nv4s.cu's header says why): the start-up check (check(), after the warm-up as actq's) compares
  1. the epilogue's element function (nv4s_silu_probe: the same device functions) with the compiled K.silu_mul on every
     finite bf16 g against 64 u values (12 edge cases + 52 fixed pseudo-random; 65280 x 64 elements);
  2. the whole chain on real layer weights (the original rows rebuilt by the inverse permutation) for several M: y and
     amax vs nv4.gemm (whatever router production runs) + K.silu_mul + nv4q's amax, and the down projection's quantised
     input (packed, scales, buf) vs nv4q.quant_dyn;
any difference switches N1 off and restores the original rows (exact inverse permutation).

KAI_N1 = 1 (the shipped tile table n1_tab.json) | a table path | 0. KAI_N1_TILE = "t[:raster:swz]" forces one fused
configuration for every M (benches). The fused call needs the default (rgh) quantiser on both projections (lever C's
non-default modes keep the base path), MAM_NV4_GEMM=cutlass, and a down projection that is an rgh NV4Linear.
"""
from __future__ import annotations

import ctypes
import json
import os

ENV = "KAI_N1"
_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TAB = os.path.join(_DIR, "n1_tab.json")
TILES = {1: (128, 128, 256, "Coop"), 2: (128, 256, 128, "Coop"), 3: (128, 128, 128, "Ping"), 4: (128, 128, 256, "Ping"),
         8: (256, 128, 128, "Coop")}  # = nv4s.cu / nv4s_dispatch.cu (nv4t.cu's numbers for the same configurations)
RASTERS = (0, 1, 2)
SWIZZLES = (1, 2, 4, 8)
DEFAULT_CFG = (3, 0, 1)  # nv4g's default variant 3 (128x128x128 ping-pong), heuristic raster, swizzle 1
FALLBACK_RC = (1, 2, 3, 6, 7)
ACTIVE = False  # read per call (levers.py switches it)
_lib = None
_TAB: list = []  # [(lo, hi, tile, raster, swz)] sorted
_FORCE = None
_REQ = False  # set by the _mlp patch around the gate|up call: the nv4.gemm hook may answer with a fused result
stats = {"fused": 0, "fallback": 0, "plain": 0}


# ------------------------------------------------------------------------------------------------ switch / table
def resolve(env):
    """KAI_N1 -> table path ("" = built-in default config only) or None (off)"""
    v = (env.get(ENV) or "").strip()
    if v.lower() in ("", "0", "off", "false", "no"):
        return None
    if v.lower() in ("1", "on", "true", "yes"):
        return DEFAULT_TAB if os.path.exists(DEFAULT_TAB) else ""
    if v.endswith(".json") and os.path.exists(v):
        return v
    raise ValueError(f"{ENV}={v!r}: 1 (shipped table), 0 / unset (off) or an existing .json table")


def parse_cfg(text):
    """'t[:raster:swz]' -> (tile, raster, swz); ValueError otherwise"""
    parts = [p for p in str(text).strip().split(":")]
    try:
        t = int(parts[0])
        r = int(parts[1]) if len(parts) > 1 else 0
        s = int(parts[2]) if len(parts) > 2 else 1
    except (ValueError, IndexError):
        raise ValueError(f"N1 config {text!r}: tile[:raster:swizzle]") from None
    if t not in TILES or r not in RASTERS or s not in SWIZZLES or len(parts) > 3:
        raise ValueError(f"N1 config {text!r}: tile in {sorted(TILES)}, raster in {RASTERS}, swizzle in {SWIZZLES}")
    return t, r, s


def parse_table(raw):
    """{"version": 1, "entries": [{"m": [lo, hi], "tile", "raster", "swz"}, ...]} -> sorted [(lo, hi, t, r, s)]"""
    if not isinstance(raw, dict) or raw.get("version") != 1 or not isinstance(raw.get("entries"), list):
        raise ValueError("n1 table: want {'version': 1, 'entries': [...]}")
    out = []
    for e in raw["entries"]:
        try:
            (lo, hi), t, r, s = e["m"], int(e["tile"]), int(e["raster"]), int(e["swz"])
            lo, hi = int(lo), int(hi)
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"n1 table: bad entry {e!r}") from None
        if t not in TILES or r not in RASTERS or s not in SWIZZLES or not 1 <= lo <= hi:
            raise ValueError(f"n1 table: bad entry {e!r}")
        out.append((lo, hi, t, r, s))
    out.sort()
    for a, b in zip(out, out[1:]):
        if b[0] <= a[1]:
            raise ValueError(f"n1 table: overlapping M ranges {a[:2]} {b[:2]}")
    return out


def load_table(path):
    if not path:
        return []
    with open(path) as f:
        return parse_table(json.load(f))


def pick(M, table=None, force=None):
    """the fused configuration (tile, raster, swz) for M rows"""
    if force is not None:
        return force
    for lo, hi, t, r, s in (_TAB if table is None else table):
        if lo <= M <= hi:
            return t, r, s
    return DEFAULT_CFG


# ------------------------------------------------------------------------------------------------ weights
def perm(N, torch):
    """new row r -> old row: [g0, u0, g1, u1, ...] for the [g (I rows); u (I rows)] gate|up weight"""
    if N % 2:
        raise ValueError(f"N1: gate|up rows {N} not even")
    I = N // 2
    p = torch.empty(N, dtype=torch.long)
    p[0::2] = torch.arange(I)
    p[1::2] = torch.arange(I, N)
    return p


def inv_perm(p, torch):
    q = torch.empty_like(p)
    q[p] = torch.arange(p.numel())
    return q


def unblock(flat, rows, cols):
    """inverse of nv4.to_blocked for rows % 128 == 0 and cols % 4 == 0: swizzled [rows * cols] -> [rows, cols]"""
    if rows % 128 or cols % 4:
        raise ValueError(f"N1: scale matrix {rows}x{cols} is padded (rows % 128, cols % 4)")
    rb, cb = rows // 128, cols // 4
    c = flat.reshape(rb * cb, 32, 4, 4).transpose(1, 2)  # = to_blocked's [rb*cb, 4, 32, 4]
    return c.reshape(rb, cb, 128, 4).permute(0, 2, 1, 3).reshape(rows, cols)


def permute_rows(wq, wsc, p, K):
    """NVFP4 weight (packed [N, K/2] uint8, swizzled e4m3 scales as uint8) with its rows reordered by p (new -> old):
    identical to quantising the reordered bf16 weight (per-row blocks, per-tensor global scale)"""
    import nv4
    N = wq.shape[0]
    sc = unblock(wsc, N, K // 16)
    return wq[p.to(wq.device)].contiguous(), nv4.to_blocked(sc[p.to(sc.device)].contiguous()).contiguous()


# ------------------------------------------------------------------------------------------------ library
def open_lib(path=None):
    """libnv4s.so (or another build of it, benches) as the fused library"""
    global _lib
    if _lib is None or path is not None:
        t = ctypes.CDLL(path or os.path.join(_DIR, "libnv4s.so"))
        t.nv4s_gemm.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4 + [ctypes.c_void_p,
                                                                                                 ctypes.c_size_t,
                                                                                                 ctypes.c_void_p]
        t.nv4s_gemm.restype = ctypes.c_int
        t.nv4s_ws.argtypes = [ctypes.c_int] * 6
        t.nv4s_ws.restype = ctypes.c_size_t
        t.nv4s_has.argtypes = [ctypes.c_int]
        t.nv4s_has.restype = ctypes.c_int
        t.nv4s_silu_mode.restype = ctypes.c_int
        t.nv4s_silu_probe.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_long, ctypes.c_void_p]
        t.nv4s_silu_probe.restype = ctypes.c_int
        _lib = t
    return _lib


def fused(xq, xs, buf, lin, cfg=None):
    """the fused gate|up GEMM: (y [M, I] bf16, ybuf fp32 [3] with ybuf[0] = max |y|) or None (configuration refused)"""
    import torch
    import nv4
    wq, wsc = lin.n1w
    M = xq.shape[0]
    N, K = lin.out_features, lin.in_features
    dev = xq.device
    y = torch.empty(M, N // 2, device=dev, dtype=torch.bfloat16)
    ybuf = torch.zeros(3, device=dev, dtype=torch.float32)
    wsb = nv4._ws.get(dev.index)
    if wsb is None:
        wsb = nv4._ws[dev.index] = torch.empty(32 << 20, device=dev, dtype=torch.uint8)
    t, r, s = cfg or pick(M, force=_FORCE)
    rc = open_lib().nv4s_gemm(t, r, s, xq.data_ptr(), wq.data_ptr(), xs.data_ptr(), wsc.data_ptr(), buf.data_ptr() + 8,
                              y.data_ptr(), ybuf.data_ptr(), M, N, K, y.stride(0), wsb.data_ptr(), wsb.numel(),
                              torch._C._cuda_getCurrentRawStream(dev.index))
    if rc in FALLBACK_RC:
        return None
    if rc:
        raise RuntimeError(f"nv4s_gemm tile {t} rc {rc} (M {M} N {N} K {K})")
    return y, ybuf


def quant_y(y, ybuf, wg):
    """the down projection's NVFP4 input from (y, amax): nv4q's quant pass, exactly as actq.quant_silu after its silu"""
    import actq
    import nv4q
    from fp4_triton import _nb
    M, I = y.shape
    CB, sc, out = actq._scales_out(M, I, y.device)
    nb = _nb(I)
    nv4q._quant_dyn_kernel[(M, I // (16 * nb))](y, ybuf, wg, out, sc, I, CB, NB=nb)
    return out, sc, ybuf


class _Interleaved:
    """nv4.gemm view of an N1 gate|up projection running rgh's plain kernel on the interleaved rows"""

    def __init__(self, lin):
        self.lin = lin
        self.wq, self.wsc = lin.n1w
        self.wg = lin.wg
        self.out_features, self.in_features = lin.out_features, lin.in_features

    def variant(self, M):
        return self.lin.variant(M)


class Fused:
    __slots__ = ("y", "buf")

    def __init__(self, y, buf):
        self.y, self.buf = y, buf


def deinterleave(d):
    """[M, 2I] with interleaved columns -> [M, 2I] = [g | u] (contiguous)"""
    import torch
    return torch.cat([d[:, 0::2], d[:, 1::2]], 1)


# ------------------------------------------------------------------------------------------------ install / patches
def _rgh_quant(nv4):
    import actq
    return actq.rgh_quant(nv4)


def install_patches(nv4, pp):
    """wrap nv4.gemm (N1 projections) and PplxEngine._mlp once; both read ACTIVE / the projection's n1w per call"""
    if not getattr(nv4, "_kai_n1", False):
        orig = nv4.gemm

        def gemm(xq, xs, buf, lin, out=None):
            if getattr(lin, "n1w", None) is None:
                return orig(xq, xs, buf, lin, out)
            if _REQ and ACTIVE and out is None and nv4.GEMM == "cutlass":
                r = fused(xq, xs, buf, lin)
                if r is not None:
                    stats["fused"] += 1
                    return Fused(*r)
                stats["fallback"] += 1
            if getattr(lin, "wq", None) is not None:  # original rows kept (benches): rgh's path unchanged
                return orig(xq, xs, buf, lin, out)
            stats["plain"] += 1
            g = deinterleave(orig(xq, xs, buf, _Interleaved(lin), None))  # exact: the same columns, moved
            if out is not None:
                out.copy_(g)
                return out
            return g
        nv4.gemm = gemm
        nv4._kai_n1_orig_gemm = orig
        nv4._kai_n1 = True
    E = pp.PplxEngine
    if not getattr(E, "_kai_n1", False):
        prev = E._mlp
        NV = nv4.NV4Linear

        def _mlp(self, L, x, xq=None):
            global _REQ
            gu, dn = L["gu"], L["down"]
            if not (ACTIVE and getattr(gu, "n1w", None) is not None and isinstance(dn, NV) and getattr(dn, "fq4", False)
                    and x is not None and x.is_cuda and _rgh_quant(nv4)):
                return prev(self, L, x, xq)
            _REQ = True
            try:
                r = gu(x, xq)
            finally:
                _REQ = False
            if isinstance(r, Fused):
                return nv4.gemm(*quant_y(r.y, r.buf, dn.wg), dn, None)
            return _down_from_g(self, nv4, pp, dn, r)
        E._mlp = _mlp
        E._kai_n1_prev_mlp = prev
        E._kai_n1 = True


def _down_from_g(self, nv4, pp, dn, g):
    """the base chain after the gate|up GEMM (a refused fused call): actq's silu part or rgh's K.silu_mul path"""
    import actq
    if "silu" in actq.PARTS and actq.rgh_quant(nv4):
        return nv4.gemm(*actq.quant_silu(g.contiguous(), self.inter, dn.wg), dn, None)
    return dn(pp.K.silu_mul(g, self.inter))


def targets(engine, nv4):
    """[(layer, gate|up NV4Linear)] whose down projection is an rgh NV4Linear too"""
    out = []
    for i, L in enumerate(getattr(engine, "layers", ()) or ()):
        if not isinstance(L, dict):
            continue
        gu, dn = L.get("gu"), L.get("down")
        if isinstance(gu, nv4.NV4Linear) and isinstance(dn, nv4.NV4Linear) and gu.out_features % 256 == 0:
            out.append((i, gu))
    return out


def interleave(lin, keep=False):
    """attach the interleaved rows (lin.n1w); keep=False releases the original rows (lin.wq / lin.wsc = None)"""
    import torch
    p = perm(lin.out_features, torch)
    lin.n1w = permute_rows(lin.wq, lin.wsc, p, lin.in_features)
    if not keep:
        lin.wq = lin.wsc = None


def restore(lin):
    """undo interleave(): the original rows back from n1w (exact inverse permutation); n1w dropped"""
    import torch
    if getattr(lin, "n1w", None) is None:
        return
    if getattr(lin, "wq", None) is None:
        q = inv_perm(perm(lin.out_features, torch), torch)
        lin.wq, lin.wsc = permute_rows(lin.n1w[0], lin.n1w[1], q, lin.in_features)
    lin.n1w = None


def install(nv4, pp, engine, path, keep=False, log=print):
    """KAI_N1 on: interleave every eligible gate|up projection, patch nv4.gemm / _mlp, ACTIVE = True (check() after the
    warm-up decides whether it stays on). Returns the number of projections."""
    global ACTIVE, _TAB, _FORCE
    import torch
    open_lib()
    _TAB = load_table(path)
    f = os.environ.get("KAI_N1_TILE", "").strip()
    _FORCE = parse_cfg(f) if f else None
    if nv4.GEMM != "cutlass":
        log(f"kai n1: off (MAM_NV4_GEMM={nv4.GEMM}: the fused epilogue is the CUTLASS kernel's)")
        return 0
    tg = targets(engine, nv4)
    for _, lin in tg:
        interleave(lin, keep=keep)
    torch.cuda.empty_cache()
    install_patches(nv4, pp)
    ACTIVE = bool(tg)
    rng = f" (layers {tg[0][0]}-{tg[-1][0]})" if tg else ""
    log(f"kai n1: {len(tg)} gate|up projections interleaved{rng}, table "
        f"{os.path.basename(path) if path else '(default config)'}, force {_FORCE}, silu mode {_lib.nv4s_silu_mode()}")
    return len(tg)


def uninstall(engine, nv4, log=print):
    """N1 off for good: original rows back in every projection"""
    global ACTIVE
    ACTIVE = False
    for _, lin in targets(engine, nv4):
        restore(lin)
    import torch
    torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ start-up check
U_FIXED = (1.0, -1.0, 0.0, -0.0, 3.140625, -2.71875, 1e-3, 65280.0, -6.0e-39, 1.1754944e-38, 3.0e38, 0.0078125)


def _u_random(n=52, seed=12345):
    """n fixed pseudo-random u values (sign, exponent 2^-20 .. 2^20, 7-bit mantissa) from a 32-bit LCG: no RNG library"""
    out, x = [], seed
    for _ in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        sign = -1.0 if x & 1 else 1.0
        e = ((x >> 1) % 41) - 20
        mant = 1.0 + ((x >> 8) & 127) / 128.0
        out.append(sign * mant * 2.0 ** e)
    return tuple(out)


U_SET = U_FIXED + _u_random()


def probe_check(K, device="cuda", u_set=U_SET):
    """the epilogue's element function vs the compiled K.silu_mul on every finite bf16 g x u_set; returns the number of
    differing elements (bf16 bits)"""
    import torch
    lib = open_lib()
    g = torch.arange(-32768, 32768, dtype=torch.int32, device=device).to(torch.int16).view(torch.bfloat16)
    g = g[torch.isfinite(g.float())].reshape(-1, 256)  # 65280 finite values = 255 rows of 256
    stream = torch._C._cuda_getCurrentRawStream(g.device.index)
    bad = 0
    for u0 in u_set:
        u = torch.full_like(g, u0)
        ref = K.silu_mul(torch.cat([g, u], 1), 256).contiguous()
        y = torch.empty_like(g)
        if lib.nv4s_silu_probe(g.data_ptr(), u.data_ptr(), y.data_ptr(), None, None, None, g.numel(), stream):
            return -1
        bad += int((y.view(torch.int16) != ref.view(torch.int16)).sum())
        CHECKED["probe"] += y.numel()
    torch.cuda.synchronize(device)
    return bad


def chain_check(nv4, K, engine, device="cuda", Ms=(37, 611, 1503, 2944), seed=4321, layers=None, log=print):
    """the fused chain vs the base chain on real gate|up / down weights (original rows rebuilt) for every M; returns the
    number of differing items (y bits, amax, packed fp4 bytes, scale bytes, buf)"""
    tg = targets(engine, nv4)
    if not tg:
        return -1  # nothing to compare: never a pass
    import torch
    import nv4q
    gen = torch.Generator(device=device).manual_seed(seed)
    pick_layers = layers or sorted({tg[0][0], tg[-1][0]})
    bad = 0
    for li in pick_layers:
        lin = engine.layers[li]["gu"]
        dn = engine.layers[li]["down"]
        orig = _Orig(lin, torch)
        for M in Ms:
            x = (torch.randn(M, lin.in_features, device=device, generator=gen)
                 * torch.exp2(torch.randint(-3, 3, (M, 1), device=device, generator=gen).float())).to(torch.bfloat16)
            xq, xs, buf = nv4q.quant_dyn(x, lin.wg)
            g = nv4._kai_n1_orig_gemm(xq, xs, buf, orig, None)
            yref = K.silu_mul(g, lin.out_features // 2).contiguous()
            qref = nv4q.quant_dyn(yref, dn.wg)
            r = fused(xq, xs, buf, lin)
            if r is None:
                log(f"kai n1: check M {M}: fused configuration {pick(M, force=_FORCE)} refused")
                bad += 1
                continue
            y, ybuf = r
            n_y = int((y.view(torch.int16) != yref.view(torch.int16)).sum())
            n_a = int(ybuf[0].item() != float(yref.float().abs().max().item()))
            q = quant_y(y, ybuf, dn.wg)
            n_q = sum(int((a.view(torch.uint8) != b.view(torch.uint8)).sum()) if a.dtype == torch.uint8 else
                      int((a != b).sum()) for a, b in zip(qref, q))
            if n_y or n_a or n_q:
                log(f"kai n1: check layer {li} M {M}: y {n_y} amax {n_a} quant {n_q} differ")
            bad += n_y + n_a + n_q
            CHECKED["chain_runs"] += 1
            CHECKED["chain_y"] += y.numel()
        del orig
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return bad


class _Orig:
    """the original (non-interleaved) rows of an N1 projection, rebuilt for the check"""

    def __init__(self, lin, torch):
        self.lin = lin
        if getattr(lin, "wq", None) is not None:
            self.wq, self.wsc = lin.wq, lin.wsc
        else:
            q = inv_perm(perm(lin.out_features, torch), torch)
            self.wq, self.wsc = permute_rows(lin.n1w[0], lin.n1w[1], q, lin.in_features)
        self.wg = lin.wg
        self.out_features, self.in_features = lin.out_features, lin.in_features

    def variant(self, M):
        return self.lin.variant(M)


CHECKED = {"probe": 0, "chain_runs": 0, "chain_y": 0}  # what the last check() compared


def check(nv4, pp, engine, device="cuda", log=print):
    """start-up check (after the warm-up); on any difference (or nothing compared) N1 is switched off and the original
    rows restored. Returns True when N1 stays on."""
    global ACTIVE
    if not ACTIVE:
        return False
    CHECKED.update(probe=0, chain_runs=0, chain_y=0)
    try:
        nb = probe_check(pp.K, device)
        nc = chain_check(nv4, pp.K, engine, device, log=log) if nb == 0 else -1
    except Exception as exc:  # noqa: BLE001  any failure keeps the base path
        log(f"kai n1: check raised {type(exc).__name__}: {exc}")
        nb, nc = -1, -1
    if nb == 0 and nc == 0 and CHECKED["chain_runs"] > 0:
        log(f"kai n1: check bit-equal (silu probe 0 differences over {CHECKED['probe']} elements; chain 0 differences "
            f"over {CHECKED['chain_runs']} fused runs, {CHECKED['chain_y']} y elements + amax + quantised input): on")
        return True
    log(f"kai n1: check FAILED (silu probe {nb}, chain {nc}): off, original rows restored")
    uninstall(engine, nv4, log=log)
    return False
