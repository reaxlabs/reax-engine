"""f8k.py (kai lever K, 10-28): ctypes front-end of libf8k.so, exact chained stream-K for rgh's f8g FP8 GEMMs.

Switch MAM_F8SK=1 (default off; only acts with MAM_F8 / MAM_F8_GU, which load f8g): pplx_engine calls install(f8g), which
wraps f8g.rowwise and f8g.swiglu_q. A call whose (kind, N, K) is in MAM_F8SK_SHAPES (default "down,out" = the N = 5120
GEMMs; names down, out (= o), qkvz, qkv, gu, or all) and that has at least one big tile per CTA runs the persistent
stream-K kernel (f8k.cu) on G = SM-count CTAs; everything else, and any refusal, runs rgh's f8_gemm unchanged.
Answers are bit-identical to f8_gemm (f8k.cu header, engine/KERNELS.md; GPU check: tools/gemmbench_k.py).
  MAM_F8SK_CFG   N:K=cfg,...  tile config per shape (f8g numbering: 0 128x256, 1 128x128, 2 256x128, 3 128x256 BK32,
                 5 128x128 BK32; SwiGLU 0 1 3); default f8k_sched.DEFAULT_CFG
  MAM_F8SK_MBIG  rows in big tiles (-1 = all whole big tiles; f8g's mbig)
  MAM_F8SK_SPIN_US  consumer wait budget before it computes a missing head itself (default 2000)
Workspace: per (device, stream) G x 128 KiB fp32 slots (24.6 MB at G = 188) + G int32 flags, allocated on first use.
Refused under KAI_E63 (pplx_engine raises at import: E63 calls f8g inside captured graphs, and the first use per
stream would allocate inside a capture)."""
import ctypes
import os

import torch

import f8k_sched as S

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libf8k.so"))
_lib.f8k_gemm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_longlong]
_lib.f8k_gemm.restype = ctypes.c_int
PART = 256 * 128  # floats per CTA slot (f8k.cu PART)

SHAPES = S.parse_shapes(os.environ.get("MAM_F8SK_SHAPES", ""))
CFG = S.parse_cfg(os.environ.get("MAM_F8SK_CFG", ""))
MBIG = int(os.environ.get("MAM_F8SK_MBIG", "-1"))
SPIN_NS = int(os.environ.get("MAM_F8SK_SPIN_US", "2000")) * 1000
_ws = {}


def _nsm(f8g, dev):
    if f8g._NSM is None:
        f8g._NSM = torch.cuda.get_device_properties(dev).multi_processor_count
    return f8g._NSM


def workspace(dev, stream, G):
    k = (dev.index, stream)
    w = _ws.get(k)
    if w is None or w[1].numel() < G:
        w = _ws[k] = (torch.empty(G * PART, device=dev, dtype=torch.float32),
                      torch.zeros(G, device=dev, dtype=torch.int32))
    return w


def cfg_for(kind, N, K):
    return CFG.get((N, K), S.DEFAULT_CFG.get((kind, N, K), 1))


def gemm(kind, cfg, x8, xs, lin, out, ldo, G, st, mbig=MBIG):
    """one f8k_gemm call; returns the library code (0 ok, -5 refused: too few tiles, -2 shape)"""
    M, K = x8.shape
    N = lin.out_features
    part, flag = workspace(x8.device, st, G)
    return _lib.f8k_gemm(kind, cfg, x8.data_ptr(), x8.stride(0), lin.w8.data_ptr(), K, xs.data_ptr(), lin.ws.data_ptr(),
                         out.data_ptr(), ldo, M, N, K, mbig, st, G, part.data_ptr(), flag.data_ptr(), SPIN_NS)


def install(f8g):
    orig_rowwise, orig_swiglu_q = f8g.rowwise, f8g.swiglu_q

    def rowwise(x8, xs, lin):
        M, K = x8.shape
        N = lin.out_features
        if (0, N, K) in SHAPES:
            G = _nsm(f8g, x8.device)
            cfg = cfg_for(0, N, K)
            if S.ok(M, N, K, 0, cfg, G, MBIG):
                out = torch.empty(M, N, device=x8.device, dtype=torch.bfloat16)
                r = gemm(0, cfg, x8, xs, lin, out, N, G, f8g._raw(x8.device.index))
                if r == 0:
                    return out
                if r != -5:
                    raise RuntimeError(f"f8k_gemm rowwise failed: {r} (M {M} N {N} K {K} cfg {cfg})")
        return orig_rowwise(x8, xs, lin)

    def swiglu_q(x8, xs, lin, H):
        M, K = x8.shape
        N = lin.out_features
        if (2, N, K) in SHAPES and not f8g._RQ2:
            G = _nsm(f8g, x8.device)
            cfg = cfg_for(2, N, K)
            if S.ok(M, N, K, 2, cfg, G, MBIG):
                st = f8g._raw(x8.device.index)
                y = torch.empty(M, H, device=x8.device, dtype=torch.bfloat16)
                r = gemm(2, cfg, x8, xs, lin, y, H, G, st)
                if r == 0:
                    q = torch.empty(M, H, device=x8.device, dtype=torch.float8_e4m3fn)
                    s = torch.empty(M, 1, device=x8.device, dtype=torch.float32)
                    f8g._lib.f8_rowquant_launch(y.data_ptr(), M, H, q.data_ptr(), s.data_ptr(), st)
                    return q, s
                if r != -5:
                    raise RuntimeError(f"f8k_gemm swiglu failed: {r} (M {M} N {N} K {K} cfg {cfg})")
        return orig_swiglu_q(x8, xs, lin, H)

    f8g.rowwise, f8g.swiglu_q = rowwise, swiglu_q
    f8g._f8k_installed = True
    return f8g
