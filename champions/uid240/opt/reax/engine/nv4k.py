"""nv4k.py (kai lever K, 10-28): CUTLASS stream-K variants of rgh's NVFP4 GEMM (libnv4k.so, source nv4k.cu).

Switch MAM_NV4K=N:K=v,... (default unset = off; only acts when MAM_NV4 installed NVFP4 projections): backends/qwen35.py
calls install(nv4, spec) after nv4.install; every nv4_gemm call of a listed (N, K) runs stream-K variant v instead of
the variant nv4's table picked (6 = 128x128x128, 7 = 128x256x128, 8 = 128x128x256, cooperative, Deterministic
reduction). Shape names are accepted too: down = 5120:17408, out = o = 5120:6144, gu = 34816:5120, qkvz = 16384:5120,
qkv = 14336:5120 (e.g. MAM_NV4K=down=6,out=6). If CUTLASS refuses the problem (rc 2) or the workspace is too small
(rc 3), the call runs the variant nv4 picked (rgh's kernel) instead. LOSSY ORDER: split tiles are summed from fp32 partials
(not bit-identical to nv4g; deterministic for a given M)."""
import ctypes
import os

NAMES = {"down": (5120, 17408), "out": (5120, 6144), "o": (5120, 6144), "gu": (34816, 5120), "qkvz": (16384, 5120),
         "qkv": (14336, 5120)}
VARIANTS = (6, 7, 8)


def parse(spec):
    out = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            key, v = part.rsplit("=", 1)
            v = int(v)
            nk = NAMES[key] if key in NAMES else tuple(int(x) for x in key.split(":"))
            if len(nk) != 2:
                raise ValueError
        except (ValueError, KeyError):
            raise ValueError(f"MAM_NV4K: bad entry {part!r} (want N:K=v or name=v)") from None
        if v not in VARIANTS:
            raise ValueError(f"MAM_NV4K: variant {v} is not a stream-K variant {VARIANTS}")
        out[nk] = v
    return out


class Router:
    """stands in for nv4's ctypes library: nv4_gemm / nv4_ws of the mapped (N, K) go to libnv4k.so"""

    def __init__(self, base, klib, table):
        self.base, self.klib, self.table = base, klib, table
        self.fallbacks = 0

    def nv4_gemm(self, v, A, B, SFA, SFB, alpha, D, M, N, K, ldd, wsp, wsb, st):
        kv = self.table.get((N, K))
        if kv is not None:
            r = self.klib.nv4k_gemm(kv, A, B, SFA, SFB, alpha, D, M, N, K, ldd, wsp, wsb, st)
            if r not in (2, 3):
                return r
            self.fallbacks += 1
        return self.base.nv4_gemm(v, A, B, SFA, SFB, alpha, D, M, N, K, ldd, wsp, wsb, st)

    def nv4_ws(self, v, M, N, K):
        kv = self.table.get((N, K))
        if kv is not None:
            return self.klib.nv4k_ws(kv, M, N, K)
        return self.base.nv4_ws(v, M, N, K)

    def __getattr__(self, name):
        return getattr(self.base, name)


def load():
    k = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libnv4k.so"))
    k.nv4k_gemm.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_void_p,
                                                                                         ctypes.c_size_t,
                                                                                         ctypes.c_void_p]
    k.nv4k_gemm.restype = ctypes.c_int
    k.nv4k_ws.argtypes = [ctypes.c_int] * 4
    k.nv4k_ws.restype = ctypes.c_size_t
    return k


def install(nv4, spec):
    table = parse(spec)
    if not table:
        return None
    r = Router(nv4.lib(), load(), table)
    nv4._lib = r
    print(f"kai nv4k: stream-K NVFP4 variants {table}", flush=True)
    return r
