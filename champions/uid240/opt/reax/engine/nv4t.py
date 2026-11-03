"""nv4t.py (kai lever B, 10-28): exact NVFP4 tile table for rgh's NVFP4 GEMM (libnv4t.so, source nv4t.cu).

Switch MAM_NV4TAB (default unset = off: this module and libnv4t.so are never imported; nv4.py's own variant choice
stays): "1" = the shipped table engine/nv4t_tab.json, or a path to another table file. Only acts when MAM_NV4 installed
NVFP4 projections: backends/qwen35.py calls install(nv4, path) after nv4.install (and before lever K's nv4k, whose
lossy stream-K entries then take precedence for their shapes).

Table (JSON): {"version": 1, "entries": [{"N", "K", "m": [lo, hi], "tile", "raster", "swz"}, ...]}: every nv4_gemm call
of that (N, K) with lo <= M <= hi runs nv4t_gemm(tile, raster, swz) instead of the nv4g variant nv4 picked. Tiles
(nv4t.cu): 0-5 = nv4g variants 0-5, 6-10 new CUTLASS tile / schedule configurations; raster 0 heuristic (nv4g's),
1 along M, 2 along N; swizzle 1 (nv4g's), 2, 4, 8. All of them are bit-identical to nv4g (same collective, same
per-element fp32 k chain; nv4t.cu's header says why); install() checks that on the GPU for every entry at start and
drops any entry that is not (or that CUTLASS refuses). A refusal at run time (rc 1/2/3/6) runs rgh's kernel instead."""
import ctypes
import json
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(_DIR, "nv4t_tab.json")
# tile -> (M, N, K, schedule) = nv4t.cu's NV4T_TILE list
TILES = {0: (128, 128, 128, "MainAuto"), 1: (128, 128, 256, "Coop"), 2: (128, 256, 128, "Coop"),
         3: (128, 128, 128, "Ping"), 4: (128, 128, 256, "Ping"), 5: (128, 64, 256, "Ping"),
         6: (128, 64, 128, "Ping"), 7: (128, 64, 128, "Coop"), 8: (256, 128, 128, "Coop"),
         9: (128, 64, 256, "Coop"), 10: (256, 64, 128, "Coop")}
RASTERS = {0: "heuristic", 1: "along_m", 2: "along_n"}
SWIZZLES = (1, 2, 4, 8)
SHAPES = {"gu": (34816, 5120), "down": (5120, 17408), "qkvz": (16384, 5120), "out": (5120, 6144), "o": (5120, 6144),
          "qkv": (14336, 5120)}
FALLBACK_RC = (1, 2, 3, 6)


def resolve(env):
    """MAM_NV4TAB -> table path or None (off)"""
    v = (env.get("MAM_NV4TAB") or "").strip()
    if v in ("", "0"):
        return None
    if v == "1":
        return DEFAULT
    if v.endswith(".json") and os.path.exists(v):
        return v
    raise ValueError(f"MAM_NV4TAB={v!r}: 1 (shipped table), 0 / unset (off) or an existing .json table")


def parse(raw):
    """table JSON -> {(N, K): [(lo, hi, tile, raster, swz), ...] sorted by lo}; ValueError on anything malformed"""
    if not isinstance(raw, dict) or raw.get("version") != 1 or not isinstance(raw.get("entries"), list):
        raise ValueError("nv4t table: want {'version': 1, 'entries': [...]}")
    out = {}
    for e in raw["entries"]:
        try:
            N, K, (lo, hi), tile, raster, swz = (int(e["N"]), int(e["K"]), e["m"], int(e["tile"]), int(e["raster"]),
                                                 int(e["swz"]))
            lo, hi = int(lo), int(hi)
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"nv4t table: bad entry {e!r}") from None
        if tile not in TILES or raster not in RASTERS or swz not in SWIZZLES or not 1 <= lo <= hi:
            raise ValueError(f"nv4t table: bad entry {e!r}")
        out.setdefault((N, K), []).append((lo, hi, tile, raster, swz))
    for nk, rows in out.items():
        rows.sort()
        for a, b in zip(rows, rows[1:]):
            if b[0] <= a[1]:
                raise ValueError(f"nv4t table: overlapping M ranges for {nk}: {a[:2]} {b[:2]}")
    return out


def load(path):
    with open(path) as f:
        return parse(json.load(f))


def lookup(table, N, K, M):
    for lo, hi, tile, raster, swz in table.get((N, K), ()):
        if lo <= M <= hi:
            return tile, raster, swz
    return None


class Router:
    """stands in for nv4's ctypes library: nv4_gemm / nv4_ws calls hitting the table go to libnv4t.so"""

    def __init__(self, base, tlib, table):
        self.base, self.tlib, self.table = base, tlib, table
        self.fallbacks = self.hits = 0

    def nv4_gemm(self, v, A, B, SFA, SFB, alpha, D, M, N, K, ldd, wsp, wsb, st):
        e = lookup(self.table, N, K, M)
        if e is not None:
            self.hits += 1
            r = self.tlib.nv4t_gemm(e[0], e[1], e[2], A, B, SFA, SFB, alpha, D, M, N, K, ldd, wsp, wsb, st)
            if r not in FALLBACK_RC:
                return r
            self.fallbacks += 1
        return self.base.nv4_gemm(v, A, B, SFA, SFB, alpha, D, M, N, K, ldd, wsp, wsb, st)

    def nv4_ws(self, v, M, N, K):
        e = lookup(self.table, N, K, M)
        if e is not None:
            return self.tlib.nv4t_ws(e[0], e[1], e[2], M, N, K)
        return self.base.nv4_ws(v, M, N, K)

    def drop(self, N, K, lo):
        rows = [r for r in self.table.get((N, K), ()) if r[0] != lo]
        if rows:
            self.table[(N, K)] = rows
        else:
            self.table.pop((N, K), None)

    def __getattr__(self, name):
        return getattr(self.base, name)


def open_lib():
    t = ctypes.CDLL(os.path.join(_DIR, "libnv4t.so"))
    t.nv4t_gemm.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_void_p,
                                                                                             ctypes.c_size_t,
                                                                                             ctypes.c_void_p]
    t.nv4t_gemm.restype = ctypes.c_int
    t.nv4t_ws.argtypes = [ctypes.c_int] * 6
    t.nv4t_ws.restype = ctypes.c_size_t
    t.nv4t_has.argtypes = [ctypes.c_int]
    t.nv4t_has.restype = ctypes.c_int
    return t


def operands(torch, M, N, K, dev, seed=0):
    """random NVFP4 operands (any byte is two valid e2m1 codes) with heavy-tailed block scales (e4m3, log-uniform over
    2^-9 .. 2^8), generated on the device: (A, B, SFA, SFB, alpha) as uint8 / fp32 tensors; SFA / SFB swizzled"""
    import nv4
    g = torch.Generator(device=dev).manual_seed(seed)
    A = torch.randint(0, 256, (M, K // 2), generator=g, device=dev, dtype=torch.uint8)
    B = torch.randint(0, 256, (N, K // 2), generator=g, device=dev, dtype=torch.uint8)

    def scales(rows):
        e = torch.empty(rows, K // 16, device=dev).uniform_(-9, 8, generator=g)
        return nv4.to_blocked(torch.exp2(e).to(torch.float8_e4m3fn)).view(torch.uint8).contiguous()

    alpha = torch.tensor([0.0, 0.0, 2.0 ** -12], device=dev, dtype=torch.float32)
    return A, B, scales(M), scales(N), alpha


def rows(A, SFA, M, K):
    """the first M rows of A and their scales (the swizzled layout stores whole 128-row blocks in order)"""
    return A[:M], SFA[:-(-M // 128) * 128 * (K // 16)]


def ref_variants(engine):
    """(M, N, K) -> the nv4g variant nv4.py runs for that call: NV4Linear.variant of the engine's first NVFP4 projection of
    that (N, K) (rgh's nv4_tab.json bucket), else nv4's default MAM_NV4_VAR (3)"""
    lins = {}
    for L in (getattr(engine, "layers", None) or ()):
        for o in (L.values() if isinstance(L, dict) else ()):
            if getattr(o, "fq4", False) and callable(getattr(o, "variant", None)):
                lins.setdefault((o.out_features, o.in_features), o)

    def ref(M, N, K):
        lin = lins.get((N, K))
        return lin.variant(M) if lin is not None else int(os.environ.get("MAM_NV4_VAR", "3"))
    return ref


def check(router, torch, dev, ref=None):
    """start-up self-check: every table entry at its lowest and highest M vs rgh's nv4_gemm with the variant production
    runs there (ref(M, N, K), default ref_variants(None)): torch.equal on random operands (one set per shape); entries
    that differ or are refused are dropped. Returns (kept, dropped)."""
    ref = ref or ref_variants(None)
    ws = torch.empty(32 << 20, device=dev, dtype=torch.uint8)
    st = torch.cuda.current_stream(dev).cuda_stream
    kept = dropped = 0
    for (N, K), entries in list(router.table.items()):
        mmax = max(e[1] for e in entries)
        A0, B, SFA0, SFB, alpha = operands(torch, mmax, N, K, dev, seed=N + K)
        for lo, hi, tile, raster, swz in list(entries):
            ok = True
            for M in sorted({lo, hi}):
                A, SFA = rows(A0, SFA0, M, K)
                outs = []
                for fn in ("t", "base"):
                    D = torch.full((M, N), float("nan"), device=dev, dtype=torch.bfloat16)
                    p = (A.data_ptr(), B.data_ptr(), SFA.data_ptr(), SFB.data_ptr(), alpha.data_ptr() + 8,
                         D.data_ptr(), M, N, K, N, ws.data_ptr(), ws.numel(), st)
                    rc = (router.tlib.nv4t_gemm(tile, raster, swz, *p) if fn == "t" else
                          router.base.nv4_gemm(ref(M, N, K), *p))
                    outs.append(D if rc == 0 else None)
                torch.cuda.synchronize(dev)
                if outs[0] is None or outs[1] is None or not torch.equal(outs[0], outs[1]):
                    ok = False
                    break
            if ok:
                kept += 1
            else:
                dropped += 1
                router.drop(N, K, lo)
        del A0, B, SFA0, SFB
    torch.cuda.empty_cache()
    return kept, dropped


def install(nv4, path, engine=None):
    import torch
    table = load(path)
    if not table:
        return None
    r = Router(nv4.lib(), open_lib(), table)
    kept, dropped = check(r, torch, torch.device("cuda", torch.cuda.current_device()), ref_variants(engine))
    nv4._lib = r
    print(f"kai nv4t: exact NVFP4 tile table {os.path.basename(path)}: {kept} entries "
          f"({dropped} dropped by the start-up bit-equality check) over {len(r.table)} shapes", flush=True)
    return r
