"""f8g.py (mam lane): ctypes front-end of libf8g.so, the FP8 e4m3 mma.sync GEMMs for sm_120 (f8g.cu).

rowwise(x8, xs, lin)      == torch._scaled_mm(x8, lin.w8, scale_a=xs, scale_b=lin.ws, out_dtype=bf16) (bit-identical to the
                             reference CUTLASS rowwise kernel at every M tested: K-sequential fp32 accumulation per element)
swiglu_q(x8, xs, lin, H)  == silu_mul_q_scaled_triton(scaled_mm(x8, w8, 1, 1), xs, ws, H): the gate/up GEMM with the _smqs math
                             in its epilogue (bf16 y [M, H]) + a per-row e4m3 quant pass
Tile config per (M, N): a wave model of the hetero (128-row big + 64-row small tiles) grid on this GPU's SM count
(the choice never changes the numbers: every config accumulates in the same k order).
rgh c8 (round 3, MAM_F8TAB=1 default): a MEASURED tile table _TAB overrides the wave model for the model's five GEMM
shapes at the step sizes of the validator's c8 / c16 closed loops (M 961..3328, 64-row buckets): (cfg, mbig, raster) per
bucket, timed on this GPU with the weights of 16 different layers (lane mam probe p3, mprobe2.py MP_F8TAB at mid-bucket M);
every entry was checked bit-identical to the wave-model pick, and by construction all of them are (same mma, same k order).
"""
import ctypes
import os

import torch

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libf8g.so"))
_lib.f8_gemm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                         ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
_lib.f8_gemm.restype = ctypes.c_int
_lib.f8_rowquant_launch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p]
_lib.f8_rowquant_launch.restype = ctypes.c_int
# rgh c8 (f8g2 library, MAM_F8RQ2=1 default when the library has it): the gate/up epilogue also publishes each row's max
# |bf16 y| (atomicMax) and f8_rowquant2 quantises with it (one read of y instead of two; the same max of the same bf16
# values -> the same scale and the same e4m3 bits as f8_rowquant)
try:
    _lib.f8_gemm2.argtypes = _lib.f8_gemm.argtypes + [ctypes.c_void_p]
    _lib.f8_gemm2.restype = ctypes.c_int
    _lib.f8_rowquant2_launch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_void_p, ctypes.c_void_p]
    _lib.f8_rowquant2_launch.restype = ctypes.c_int
    _HAS_RQ2 = True
except AttributeError:  # the f8m library (no f8_gemm2)
    _HAS_RQ2 = False
_RQ2 = _HAS_RQ2 and os.environ.get("MAM_F8RQ2", "0") == "1"
_raw = torch._C._cuda_getCurrentRawStream
_NSM = None
MIN_M = int(os.environ.get("MAM_F8_MIN_M", "256"))
_cfg = {}
# n-major CTA order with the small tiles next to their n-tile's big tiles, for weights larger than about half the L2
# (gate/up: 178 MB); smaller weights keep the big-tiles-first order (better tail balance)
RASTER_BYTES = int(os.environ.get("MAM_F8_RASTER_BYTES", str(96 << 20)))


def _ms(tb, ts, ratio, nsm):
    """list-scheduling makespan (in small-tile times) of tb big tiles (ratio small tiles long) then ts small tiles"""
    q, r = divmod(tb, nsm)
    lo, hi = q * ratio, (q + 1) * ratio if r else q * ratio
    cap = (nsm - r) * (hi - lo) if r else 0
    if ts <= cap:
        return hi
    return hi + -(-(ts - cap) // nsm)


def _pick(M, N, kind):
    global _NSM
    k = (M, N, kind)
    c = _cfg.get(k)
    if c is None:
        if _NSM is None:
            _NSM = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        ncols = N // 2 if kind == 2 else N
        best = None
        for cfg, bn, rate in ((0, 256, 1.0), (1, 128, 0.92)):
            bnc = bn // 2 if kind == 2 else bn
            if ncols % bnc:
                continue
            nt = ncols // bnc
            mb = (M // 128) * 128
            tb, ts = (mb // 128) * nt, -(-(M - mb) // 64) * nt
            t = _ms(tb, ts, 2, _NSM) * 64 * bn / rate
            if best is None or t < best[0]:
                best = (t, cfg)
        c = _cfg[k] = best[1] if best else -1
    return c


_TAB = {  # (kind, N, K): {M64: (cfg, mbig, raster)}  mktab2.py f8tab_p3.json (mid-bucket M, min gain 0.010)
    (0, 5120, 6144): {1152: (2, -1, 0), 1216: (2, -1, 0), 1280: (2, -1, 0), 1344: (2, 1248, 0), 1408: (2, 1248, 0), 1472: (1, 1248, 0), 1536: (3, 1248, 0), 1600: (3, 1248, 0), 1664: (3, 1248, 0), 1792: (3, 1248, 0), 1856: (1, 1632, 0), 2304: (2, -1, 0), 2368: (2, -1, 0), 2432: (2, 2272, 0), 2496: (2, -1, 0), 2560: (2, -1, 0), 2624: (2, 2528, 0), 2688: (2, 2528, 0), 2752: (3, 2272, 0), 2816: (3, 2400, 0), 2880: (3, 2400, 0), 2944: (3, 2400, 0), 3072: (0, -1, 1)},
    (0, 5120, 17408): {1088: (2, -1, 0), 1152: (2, -1, 0), 1216: (2, -1, 0), 1344: (2, 1248, 0), 1408: (2, 1248, 0), 1472: (1, 1248, 0), 1536: (1, -1, 1), 1600: (1, -1, 1), 1792: (3, 1248, 0), 2304: (2, -1, 0), 2368: (2, -1, 0), 2432: (2, -1, 0), 2496: (2, -1, 0), 2560: (2, -1, 0), 2624: (2, 2528, 0), 2688: (2, 2528, 0), 2944: (3, 2400, 0), 3072: (0, -1, 1), 3136: (2, -1, 1), 3264: (0, -1, 1)},
    (0, 14336, 5120): {1024: (3, 864, 0), 1088: (0, -1, 1), 1152: (0, -1, 1), 1216: (0, -1, 1), 1280: (3, -1, 0), 1344: (2, -1, 0), 1408: (2, -1, 0), 1472: (2, -1, 0), 1536: (0, -1, 1), 1600: (0, -1, 1), 1664: (2, -1, 0), 1728: (2, -1, 0), 1856: (0, 1760, 0), 1920: (3, 1760, 0), 2048: (2, -1, 0), 2112: (2, -1, 0), 2176: (2, -1, 0), 2240: (2, -1, 0), 2304: (2, -1, 0), 2624: (2, -1, 0), 2752: (0, 2656, 0), 2944: (2, -1, 0), 3136: (0, 3040, 0), 3328: (0, -1, 1)},
    (0, 16384, 5120): {1024: (2, -1, 0), 1088: (2, -1, 0), 1152: (2, -1, 0), 1216: (2, -1, 0), 1280: (2, -1, 0), 1408: (2, -1, 0), 1536: (2, -1, 0), 1600: (0, 1504, 0), 1664: (3, 1504, 0), 1792: (2, -1, 0), 1856: (2, -1, 0), 1920: (2, -1, 0), 1984: (2, -1, 0), 2048: (3, 1888, 0), 2176: (2, -1, 0), 2368: (0, 2272, 0), 2688: (2, -1, 0), 2752: (0, 2656, 0), 2944: (2, -1, 0)},
}
_TABON = os.environ.get("MAM_F8TAB", "0") == "1"
_plans = {}
_plans_tab = None


def _plan(M, N, K, kind):
    """(cfg, mbig, raster) of one f8_gemm call: the measured table entry of M's 64-row bucket, else the wave model"""
    global _plans_tab
    if _plans_tab is not _TAB:  # (a probe may swap _TAB at run time)
        _plans.clear()
        _plans_tab = _TAB
    k = (M, N, K, kind)
    p = _plans.get(k)
    if p is None:
        e = _TAB.get((kind, N, K), {}).get(-(-M // 64) * 64) if _TABON else None
        p = _plans[k] = tuple(e) if e is not None else (_pick(M, N, kind), -1, int(N * K >= RASTER_BYTES))
    return p


def ok(M, N, K):
    return M >= MIN_M and K % 256 == 0 and N % 128 == 0


def rowwise(x8, xs, lin):
    M, K = x8.shape
    N = lin.out_features
    out = torch.empty(M, N, device=x8.device, dtype=torch.bfloat16)
    cfg, mbig, rast = _plan(M, N, K, 0)
    r = _lib.f8_gemm(0, cfg, x8.data_ptr(), x8.stride(0), lin.w8.data_ptr(), K, xs.data_ptr(), lin.ws.data_ptr(),
                     out.data_ptr(), N, M, N, K, mbig, _raw(x8.device.index), rast)
    if r:
        raise RuntimeError(f"f8_gemm rowwise failed: {r} (M {M} N {N} K {K})")
    return out


def swiglu_q(x8, xs, lin, H):
    M, K = x8.shape
    N = lin.out_features
    st = _raw(x8.device.index)
    y = torch.empty(M, H, device=x8.device, dtype=torch.bfloat16)
    cfg, mbig, rast = _plan(M, N, K, 2)
    q = torch.empty(M, H, device=x8.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(M, 1, device=x8.device, dtype=torch.float32)
    if _RQ2:
        rm = torch.zeros(M, device=x8.device, dtype=torch.float32)
        r = _lib.f8_gemm2(2, cfg, x8.data_ptr(), x8.stride(0), lin.w8.data_ptr(), K, xs.data_ptr(), lin.ws.data_ptr(),
                          y.data_ptr(), H, M, N, K, mbig, st, rast, rm.data_ptr())
        if r:
            raise RuntimeError(f"f8_gemm2 swiglu failed: {r} (M {M} N {N} K {K})")
        _lib.f8_rowquant2_launch(y.data_ptr(), M, H, rm.data_ptr(), q.data_ptr(), s.data_ptr(), st)
        return q, s
    r = _lib.f8_gemm(2, cfg, x8.data_ptr(), x8.stride(0), lin.w8.data_ptr(), K, xs.data_ptr(), lin.ws.data_ptr(),
                     y.data_ptr(), H, M, N, K, mbig, st, rast)
    if r:
        raise RuntimeError(f"f8_gemm swiglu failed: {r} (M {M} N {N} K {K})")
    _lib.f8_rowquant_launch(y.data_ptr(), M, H, q.data_ptr(), s.data_ptr(), st)
    return q, s
