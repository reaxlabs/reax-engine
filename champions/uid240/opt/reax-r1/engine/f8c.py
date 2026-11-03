"""f8c.py (kai lever R5-1, 10-28): ctypes front-end of libf8c.so, CUTLASS 4.5.0 sm120 FP8 GEMMs with f8g's semantics.

Switch MAM_F8CUT=1 (default off; only acts with MAM_F8 / MAM_F8_GU, which load f8g): pplx_engine calls install(f8g), which
wraps f8g.rowwise and f8g.swiglu_q. A call whose (kind, N, K) has a plan for its 64-row M bucket in the per-shape table
runs the CUTLASS kernel; everything else (and any refusal) runs the previous f8g function unchanged.
  rowwise  (kind 0): out = bf16((acc * xs[r]) * ws[n])                           (f8c.cu kind 0, rowwise EVT)
  swiglu_q (kind 2): gu = bf16(acc) [M, 2H] (f8c.cu kind 1), then f8c_swq = rgh's smqs1 + f8_rowquant in one pass per row
The epilogue / SwiGLU / quant arithmetic is f8g's; the mainloop is CUTLASS's (mma.sync kind::f8f6f4 m16n8k32, fp32 accumulate),
with the same in-order 32-wide k chain as f8g, so the answers are f8g's bit for bit (tools/gemmbench_c.py: 9,600 / 9,600
candidates torch.equal on the RTX PRO 6000, engine/KERNELS.md).

  MAM_F8CUT_SHAPES  names (down, out (= o), qkvz, qkv, gu, all) that may use CUTLASS; default all shapes of the table
  MAM_F8CUT_TAB     name=v[:raster[:swizzle]],...  one plan for every M of that shape (overrides the table)
  MAM_F8CUT_MIN_M   smallest M routed to CUTLASS (default f8g.MIN_M; f8g's own gate f8g.ok still applies in pplx_engine)
Variants (f8c.cu): 0 128x128x128 coop, 1 128x256x64 coop, 2 256x128x64 coop, 3 128x128x128 ping-pong, 4 128x128x64 ping,
5 64x128x128 ping, 6 128x64x128 ping, 7 128x128x64 coop. raster 0 heuristic, 1 along M, 2 along N; swizzle 1/2/4/8."""
import ctypes
import os

NAMES = {"down": (0, 5120, 17408), "out": (0, 5120, 6144), "o": (0, 5120, 6144), "qkvz": (0, 16384, 5120),
         "qkv": (0, 14336, 5120), "gu": (2, 34816, 5120)}
VARIANTS = {0: (128, 128, 128, "coop"), 1: (128, 256, 64, "coop"), 2: (256, 128, 64, "coop"), 3: (128, 128, 128, "ping"),
            4: (128, 128, 64, "ping"), 5: (64, 128, 128, "ping"), 6: (128, 64, 128, "ping"), 7: (128, 128, 64, "coop")}
WS_BYTES = 1 << 20

# (kind, N, K): {M64 bucket: (variant, raster, swizzle)}: tools/gemmbench_c.py, two runs on the RTX PRO 6000 (28 Oct,
# /home/me/engfast-phase1/v2/runs/r5-f8cut): adopted only where the SAME plan was > 3 % faster than f8g's c8a plan in both
# runs (min of the two gains in the comment); every other bucket (and c16 sizes, gate/up) stays on f8g
_TAB = {
    (0, 5120, 17408): {1792: (4, 2, 1)},                                       # down  +3.2 %
    (0, 5120, 6144): {1536: (7, 1, 1), 1600: (7, 1, 1), 1664: (4, 1, 1)},     # out/o +3.6 / +3.5 / +3.7 %
    (0, 16384, 5120): {1792: (2, 0, 1)},                                       # qkvz  +3.2 %
    (0, 14336, 5120): {1280: (1, 1, 1)},                                       # qkv   +4.8 %
}


def parse_shapes(s):
    """'down,gu' -> {(kind, N, K), ...}; '' or 'all' -> None (= every shape)"""
    s = (s or "").strip()
    if not s or s == "all":
        return None
    out = set()
    for n in s.split(","):
        n = n.strip()
        if not n:
            continue
        if n not in NAMES:
            raise ValueError(f"MAM_F8CUT_SHAPES: unknown shape {n!r} (known: {', '.join(NAMES)}, all)")
        out.add(NAMES[n])
    return out


def parse_tab(s):
    """'down=1:2:1,gu=0' -> {(kind, N, K): (variant, raster, swizzle)}"""
    out = {}
    for e in (s or "").split(","):
        e = e.strip()
        if not e:
            continue
        name, _, plan = e.partition("=")
        name = name.strip()
        if name not in NAMES or not plan:
            raise ValueError(f"MAM_F8CUT_TAB: bad entry {e!r} (name=v[:raster[:swizzle]], names {', '.join(NAMES)})")
        p = [int(x) for x in plan.split(":")]
        if len(p) > 3:
            raise ValueError(f"MAM_F8CUT_TAB: bad plan {plan!r} (v[:raster[:swizzle]])")
        v, r, w = p + [0, 1][len(p) - 1:]
        if v not in VARIANTS or r not in (0, 1, 2) or w not in (1, 2, 4, 8):
            raise ValueError(f"MAM_F8CUT_TAB: bad plan {plan!r} (variant {sorted(VARIANTS)}, raster 0-2, swizzle 1/2/4/8)")
        out[NAMES[name]] = (v, r, w)
    return out


SHAPES = parse_shapes(os.environ.get("MAM_F8CUT_SHAPES", ""))
FORCE = parse_tab(os.environ.get("MAM_F8CUT_TAB", ""))
_MIN_M = os.environ.get("MAM_F8CUT_MIN_M")


def plan(M, N, K, kind, tab=None, force=None, shapes=None, min_m=256):
    """(variant, raster, swizzle) for one call, or None (= f8g)"""
    key = (kind, N, K)
    if M < min_m or (shapes is not None and key not in shapes):
        return None
    force = FORCE if force is None else force
    if key in force:
        return force[key]
    e = (_TAB if tab is None else tab).get(key, {}).get(-(-M // 64) * 64)
    return tuple(e) if e is not None else None


_lib = None


def load():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libf8c.so"))
        lib.f8c_gemm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        lib.f8c_gemm.restype = ctypes.c_int
        lib.f8c_ws.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.f8c_ws.restype = ctypes.c_size_t
        lib.f8c_stages.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.f8c_stages.restype = ctypes.c_int
        lib.f8c_swq_launch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.f8c_swq_launch.restype = ctypes.c_int
        _lib = lib
    return _lib


_ws = {}


def workspace(dev):
    import torch
    w = _ws.get(dev.index)
    if w is None:
        w = _ws[dev.index] = torch.empty(WS_BYTES, device=dev, dtype=torch.uint8)
    return w


def gemm(kind, p, x8, xs, lin, out, ldo, st):
    """one CUTLASS call: kind 0 rowwise / 1 unscaled; p = (variant, raster, swizzle); returns the library code"""
    M, K = x8.shape
    N = lin.out_features
    ws = workspace(x8.device)
    return load().f8c_gemm(p[0], kind, x8.data_ptr(), x8.stride(0), lin.w8.data_ptr(), K, xs.data_ptr(),
                           lin.ws.data_ptr(), out.data_ptr(), ldo, M, N, K, p[2], p[1], ws.data_ptr(), ws.numel(), st)


def swq(gu, xs, sw, H, st, y=None, quant=True):
    """f8c_swq: (q [M, H] e4m3, s [M, 1] fp32) of gu = bf16(acc) [M, 2H] (rgh's smqs1 + f8_rowquant); y optional"""
    import torch
    M = gu.shape[0]
    q = torch.empty(M, H, device=gu.device, dtype=torch.float8_e4m3fn) if quant else None
    s = torch.empty(M, 1, device=gu.device, dtype=torch.float32) if quant else None
    r = load().f8c_swq_launch(gu.data_ptr(), gu.stride(0), M, H, xs.data_ptr(), sw.data_ptr(),
                              y.data_ptr() if y is not None else None, q.data_ptr() if quant else None,
                              s.data_ptr() if quant else None, st)
    if r:
        raise RuntimeError(f"f8c_swq failed: {r} (M {M} H {H})")
    return q, s


def installed(f8g):
    return getattr(f8g, "_f8c_orig", None) is not None


def uninstall(f8g):
    """restore the f8g functions install() wrapped (levers.py switches F8CUT at run time)"""
    o = getattr(f8g, "_f8c_orig", None)
    if o is not None:
        f8g.rowwise, f8g.swiglu_q = o
        f8g._f8c_orig = None
    return f8g


def install(f8g):
    """wrap f8g.rowwise / f8g.swiglu_q (idempotent)"""
    import torch
    if installed(f8g):
        return f8g
    min_m = int(_MIN_M) if _MIN_M else f8g.MIN_M
    orig_rowwise, orig_swiglu_q = f8g.rowwise, f8g.swiglu_q

    def rowwise(x8, xs, lin):
        M, K = x8.shape
        N = lin.out_features
        p = plan(M, N, K, 0, shapes=SHAPES, min_m=min_m)
        if p is not None:
            out = torch.empty(M, N, device=x8.device, dtype=torch.bfloat16)
            r = gemm(0, p, x8, xs, lin, out, N, f8g._raw(x8.device.index))
            if r == 0:
                return out
            if r not in (2, 3):  # 2 / 3 = CUTLASS cannot implement / workspace too small: f8g
                raise RuntimeError(f"f8c_gemm rowwise failed: {r} (M {M} N {N} K {K} plan {p})")
        return orig_rowwise(x8, xs, lin)

    def swiglu_q(x8, xs, lin, H):
        M, K = x8.shape
        N = lin.out_features
        p = plan(M, N, K, 2, shapes=SHAPES, min_m=min_m)
        if p is not None and not f8g._RQ2:
            st = f8g._raw(x8.device.index)
            gu = torch.empty(M, N, device=x8.device, dtype=torch.bfloat16)
            r = gemm(1, p, x8, xs, lin, gu, N, st)
            if r == 0:
                return swq(gu, xs, lin.ws, H, st)
            if r not in (2, 3):
                raise RuntimeError(f"f8c_gemm gate/up failed: {r} (M {M} N {N} K {K} plan {p})")
        return orig_swiglu_q(x8, xs, lin, H)

    f8g._f8c_orig = (orig_rowwise, orig_swiglu_q)
    f8g.rowwise, f8g.swiglu_q = rowwise, swiglu_q
    return f8g


def lever_parse(v):
    """levers.py F8CUT value: '0' (off), '1' (on, table _TAB) or a MAM_F8CUT_TAB spec (on, those plans for every M)"""
    s = str(v).strip()
    if s.lower() in ("0", "off", "false", "no", ""):
        return "0"
    if s.lower() in ("1", "on", "true", "yes"):
        return "1"
    parse_tab(s)
    return s


def lever_set(f8g, v):
    global FORCE
    v = lever_parse(v)
    if v == "0":
        uninstall(f8g)
        return
    FORCE = {} if v == "1" else parse_tab(v)
    install(f8g)


def lever_get(f8g):
    if not installed(f8g):
        return "0"
    return ",".join(f"{n}={':'.join(map(str, p))}" for n, p in ((n, FORCE[k]) for n, k in NAMES.items()
                                                               if k in FORCE and n != "o")) or "1"
