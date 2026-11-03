"""nv4.py (rgh lane Q): selective NVFP4 projections for the mammoth f8m stack.

MAM_NV4="<types>@<layers>[:<method>],..." (empty/unset = OFF: the engine is exactly f8m). types: gu, down, qkvz, out,
qkv, o; layers "a-b" ranges joined by '+'; method rtn (default) | mse. Example: MAM_NV4="gu@20-47:mse,down@24-43".

Conversion is DERIVED AT START from the pinned bf16 checkpoint in /model (no weight files in the image, no calibration
data): per projection the weight is read from the safetensors, quantised to NVFP4 (e2m1 values, ue4m3 scale per 16 K
elements, fp32 per-tensor global scale g_w = amax/2688; "mse": per block the best of 8 scale candidates by squared
error) and the FP8 copy is released. Activations are quantised per call: per-tensor global scale = the power of two
>= amax/2688 (a power of two keeps every block's e4m3 rounding independent of the batch), block scales by RTN
(fp4_triton.nvfp4_quant_static, the owner's quantiser shipped in the reference image). GEMM = libnv4g.so (CUTLASS sm120
block-scaled tensor-op kernel, source nv4g.cu / nv4_dispatch.cu next to it) or torch F.scaled_mm (cuBLASLt) when
MAM_NV4_GEMM=cublaslt. Every other projection keeps the f8m FP8 path unchanged.

lane QX numerics (flag-gated by the method string; "rtn" / "mse" are unchanged and bit-identical to lane Q):
method = <weight>[+x46][+perm], weight rtn | mse | gptq.
  gptq  GPTQ (Frantar et al. 2022) error compensation for NVFP4: columns quantised left to right, each block-16 scale
        picked by MSE (the 8 "mse" candidates) from the already error-compensated weights at the block start, rounding
        error spread over the remaining columns with the inverse Hessian of the projection's input. The Hessian
        H = sum x^T x is accumulated AT START from activations of the PUBLIC seed set (0x52454158; FAQ "Quantisation and
        repacking": calibration-free or activations computed at start from the public seed set), see calibrate().
  x46   activation block scale per 16-block = the better (smaller SSE) of amax -> 6 (RTN) and amax -> 4 ("Four Over Six").
  perm  K-permutation of the GEMM by descending calibration channel energy diag(H) (weight columns permuted at start,
        x gathered in the quantiser; exact in real arithmetic; for down_proj it is free in production: permute the
        gate/up rows instead).
"""
import ctypes
import json
import math
import os
import time

import torch
import torch.nn.functional as F

SPEC = os.environ.get("MAM_NV4", "").strip()
GEMM = os.environ.get("MAM_NV4_GEMM", "cutlass")
_DIR = os.path.dirname(os.path.abspath(__file__))
MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
SLOT = {"gu": "gu", "down": "down", "qkvz": "in_qkvz", "out": "out", "qkv": "qkv", "o": "o"}
_lib = None
_ws = {}
_VAR = {}  # (M bucket, N, K) -> CUTLASS variant (MAM_NV4_TAB json or default)


def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(os.path.join(_DIR, "libnv4g.so"))
        _lib.nv4_gemm.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_void_p,
                                                                                              ctypes.c_size_t,
                                                                                              ctypes.c_void_p]
        _lib.nv4_gemm.restype = ctypes.c_int
        _lib.nv4_ws.argtypes = [ctypes.c_int] * 4
        _lib.nv4_ws.restype = ctypes.c_size_t
    return _lib


def parse(spec, n_layers):
    out = {}
    for part in filter(None, spec.split(",")):
        types, _, rest = part.partition("@")
        rng, _, method = rest.partition(":")
        layers = []
        if rng in ("", "all"):
            layers = list(range(n_layers))
        else:
            for r in rng.split("+"):
                a, _, b = r.partition("-")
                layers += list(range(int(a), int(b or a) + 1))
        for t in types.split("+"):
            for i in layers:
                out[(i, t)] = method or "rtn"
    return out


def _codes(q):
    mid = torch.tensor(MID, device=q.device)
    a = q.abs()
    idx = torch.bucketize(a, mid, right=False)
    on_mid = (idx < 7) & (a == mid[idx.clamp(max=6)])
    idx = torch.where(on_mid & (idx % 2 == 1), idx + 1, idx)
    return ((q < 0).to(torch.uint8) << 3) | idx.to(torch.uint8)


def _deq(c):
    v = torch.tensor(E2M1, device=c.device)[(c & 7).long()]
    return torch.where((c & 8) != 0, -v, v)


def to_blocked(m):
    rows, cols = m.shape
    rb, cb = -(-rows // 128), -(-cols // 4)
    pr, pc = rb * 128 - rows, cb * 4 - cols
    if pr or pc:
        m = F.pad(m, (0, pc, 0, pr))
    return m.view(rb, 128, cb, 4).permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def quant_w(w, method="rtn", rows=4096):
    """[N, K] bf16 -> (packed uint8 [N, K/2], swizzled e4m3 scales (uint8), g fp32 [1]); processed in row chunks"""
    N, K = w.shape
    g = (w.float().abs().amax() / 2688.0).clamp(min=1e-12)
    packed = torch.empty(N, K // 2, device=w.device, dtype=torch.uint8)
    sc = torch.empty(N, K // 16, device=w.device, dtype=torch.float8_e4m3fn)
    for r0 in range(0, N, rows):
        blk = w[r0:r0 + rows].float().view(-1, K // 16, 16)
        bamax = blk.abs().amax(-1)
        if method == "mse":
            best_e = best_s = None
            for c in (8.0, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0):
                s = (bamax / (c * g)).clamp(max=448.0).to(torch.float8_e4m3fn)
                sf = (s.float() * g).clamp(min=1e-30)
                e = ((_deq(_codes((blk / sf[..., None]).clamp(-6, 6))) * sf[..., None] - blk) ** 2).sum(-1)
                if best_e is None:
                    best_e, best_s = e, s.view(torch.uint8)
                else:
                    b = e < best_e
                    best_e = torch.where(b, e, best_e)
                    best_s = torch.where(b, s.view(torch.uint8), best_s)
            s = best_s.view(torch.float8_e4m3fn)
        else:
            s = (bamax / (6.0 * g)).clamp(max=448.0).to(torch.float8_e4m3fn)
        sf = (s.float() * g).clamp(min=1e-30)
        c = _codes((blk / sf[..., None]).clamp(-6, 6)).view(-1, K)
        packed[r0:r0 + rows] = (c[:, 1::2] << 4) | c[:, ::2]
        sc[r0:r0 + rows] = s
    return packed, to_blocked(sc).view(torch.uint8), g.reshape(1)


_FT = None


def ft():
    global _FT
    if _FT is None:
        import fp4_triton
        _FT = fp4_triton
    return _FT


def quant_x(x, wg):
    """bf16 [M, K] contiguous -> (packed uint8, swizzled scales uint8, buf fp32 [3] = (amax, g_x, alpha = g_x * g_w));
    g_x = 2^ceil(log2(amax / 2688)) (nv4q.quant_dyn: zero, atomic-max, quant = 3 launches + the scale memset)"""
    import nv4q
    return nv4q.quant_dyn(x, wg)


def gemm(xq, xs, buf, lin, out=None):
    M = xq.shape[0]
    N, K = lin.out_features, lin.in_features
    if out is None:
        out = torch.empty(M, N, device=xq.device, dtype=torch.bfloat16)
    if GEMM == "cublaslt":
        S, R = F.ScalingType, F.SwizzleType
        y = F.scaled_mm(xq.view(torch.float4_e2m1fn_x2), lin.wq.view(torch.float4_e2m1fn_x2).t(),
                        scale_a=[xs.view(torch.float8_e4m3fn), buf[1]],
                        scale_recipe_a=[S.BlockWise1x16, S.TensorWise],
                        scale_b=[lin.wsc.view(torch.float8_e4m3fn), lin.wg.reshape(())],
                        scale_recipe_b=[S.BlockWise1x16, S.TensorWise],
                        swizzle_a=[R.SWIZZLE_32_4_4, R.NO_SWIZZLE], swizzle_b=[R.SWIZZLE_32_4_4, R.NO_SWIZZLE],
                        output_dtype=torch.bfloat16)
        out.copy_(y)
        return out
    dev = xq.device.index
    v = lin.variant(M)
    wsb = _ws.get(dev)
    if wsb is None:
        wsb = _ws[dev] = torch.empty(32 << 20, device=xq.device, dtype=torch.uint8)
    r = lib().nv4_gemm(v, xq.data_ptr(), lin.wq.data_ptr(), xs.data_ptr(), lin.wsc.data_ptr(), buf.data_ptr() + 8,
                       out.data_ptr(), M, N, K, out.stride(0), wsb.data_ptr(), wsb.numel(),
                       torch._C._cuda_getCurrentRawStream(dev))
    if r:
        raise RuntimeError(f"nv4_gemm v{v} rc {r} (M {M} N {N} K {K})")
    return out


def _codes_t(q, mid):
    """_codes with the e2m1 decision boundaries as a device tensor (no host->device copy: capturable, same bits)"""
    a = q.abs()
    idx = torch.bucketize(a, mid, right=False)
    on_mid = (idx < 7) & (a == mid[idx.clamp(max=6)])
    idx = torch.where(on_mid & (idx % 2 == 1), idx + 1, idx)
    return ((q < 0).to(torch.uint8) << 3) | idx.to(torch.uint8)


def _deq_t(c, grid):
    v = grid[(c & 7).long()]
    return torch.where((c & 8) != 0, -v, v)


def _gptq_inner(W1, Hi, E1, C1, S1, g, cands, mid, grid):
    """one GPTQ block (all its columns, left to right) in place: W1 [N, B] fp32 (error-compensated as we go), Hi [B, B]
    = the block of the upper Cholesky factor of H^-1, E1 [N, B] scaled errors, C1 [N, B] uint8 codes, S1 [N, B/16] e4m3
    block scales (picked at each 16-column group start from the current W1 by the 8 MSE candidates, first minimum)"""
    sf = None
    for i in range(W1.shape[1]):
        if i % 16 == 0:
            blk = W1[:, i:i + 16]
            am = blk.abs().amax(-1)
            s8 = (am[None, :] / (cands[:, None] * g)).clamp(max=448.0).to(torch.float8_e4m3fn)  # [8, N]
            sfc = (s8.float() * g).clamp(min=1e-30)
            e = ((_deq_t(_codes_t((blk[None] / sfc[..., None]).clamp(-6, 6), mid), grid) * sfc[..., None] - blk[None]) ** 2).sum(-1)
            j = e.argmin(0)  # first minimum = candidate order of quant_w (strict improvement only)
            s8 = s8.view(torch.uint8).gather(0, j[None]).view(torch.float8_e4m3fn)[0]
            S1[:, i // 16] = s8
            sf = (s8.float() * g).clamp(min=1e-30)
        wc = W1[:, i]
        c = _codes_t((wc / sf).clamp(-6, 6), mid)
        C1[:, i] = c
        err = (wc - _deq_t(c, grid) * sf) / Hi[i, i]
        W1[:, i:] -= err[:, None] * Hi[i, i:][None, :]
        E1[:, i] = err


_GG = {}  # (N, B, device) -> (graph, static buffers) of _gptq_inner (MAM_NV4_GPTQ_GRAPH=1); cleared by gptq_release()


def gptq_release():
    _GG.clear()
    torch.cuda.empty_cache()


def _gptq_graph(N, B, dev, g, cands, mid, grid):
    key = (N, B, dev.index)
    e = _GG.get(key)
    if e is None:
        st = {"W1": torch.zeros(N, B, device=dev), "Hi": torch.eye(B, device=dev), "E1": torch.zeros(N, B, device=dev),
              "C1": torch.zeros(N, B, device=dev, dtype=torch.uint8),
              "S1": torch.zeros(N, B // 16, device=dev, dtype=torch.float8_e4m3fn),
              "g": torch.ones((), device=dev), "cands": cands.clone(), "mid": mid.clone(), "grid": grid.clone()}
        s = torch.cuda.Stream(dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):  # warm-up (lazy kernel init) outside the capture
            _gptq_inner(st["W1"], st["Hi"], st["E1"], st["C1"], st["S1"], st["g"], st["cands"], st["mid"], st["grid"])
        torch.cuda.current_stream(dev).wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            _gptq_inner(st["W1"], st["Hi"], st["E1"], st["C1"], st["S1"], st["g"], st["cands"], st["mid"], st["grid"])
        e = _GG[key] = (gr, st)
    gr, st = e
    st["g"].copy_(g)
    return gr, st


def gptq_quant_w(w, H, damp=0.01, B=128):
    """GPTQ for NVFP4 (see module doc). w [N, K] (bf16/fp32), H [K, K] fp32 = sum x^T x. Returns the quant_w triple
    (packed uint8 [N, K/2], swizzled e4m3 scales uint8, g fp32 [1]). Global scale g = amax(w) / 2688 as in quant_w.
    MAM_NV4_GPTQ_GRAPH=1 (default) replays each 128-column block as one CUDA graph of the same kernels (bit-identical to
    the eager loop, MAM_NV4_GPTQ_GRAPH=0; start-up time only)."""
    N, K = w.shape
    dev = w.device
    W = w.float().clone()
    g = (W.abs().amax() / 2688.0).clamp(min=1e-12)
    H = H.clone()
    dg = torch.diag(H)
    dead = dg == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H.diagonal().add_(damp * torch.mean(torch.diag(H)))
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    del H
    codes = torch.empty(N, K, device=dev, dtype=torch.uint8)
    S = torch.empty(N, K // 16, device=dev, dtype=torch.float8_e4m3fn)
    cands = torch.tensor((8.0, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0), device=dev)
    mid = torch.tensor(MID, device=dev)
    grid = torch.tensor(E2M1, device=dev)
    use_g = os.environ.get("MAM_NV4_GPTQ_GRAPH", "1") == "1" and K % B == 0
    gr = st = None
    if use_g:
        try:
            gr, st = _gptq_graph(N, B, dev, g, cands, mid, grid)
        except Exception as exc:  # noqa: BLE001
            print(f"rgh nv4: GPTQ graph capture failed ({exc!r}); eager loop", flush=True)
            _GG.pop((N, B, dev.index), None)
            gr = None
    for i1 in range(0, K, B):
        i2 = min(i1 + B, K)
        if gr is not None:
            st["W1"].copy_(W[:, i1:i2])
            st["Hi"].copy_(Hinv[i1:i2, i1:i2])
            gr.replay()
            codes[:, i1:i2] = st["C1"]
            S[:, i1 // 16:i2 // 16] = st["S1"]
            E1 = st["E1"]
        else:
            W1 = W[:, i1:i2].clone()
            E1 = torch.empty_like(W1)
            C1 = torch.empty(N, i2 - i1, device=dev, dtype=torch.uint8)
            S1 = torch.empty(N, (i2 - i1) // 16, device=dev, dtype=torch.float8_e4m3fn)
            _gptq_inner(W1, Hinv[i1:i2, i1:i2], E1, C1, S1, g, cands, mid, grid)
            codes[:, i1:i2] = C1
            S[:, i1 // 16:i2 // 16] = S1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    packed = (codes[:, 1::2] << 4) | codes[:, ::2]
    return packed, to_blocked(S).view(torch.uint8), g.reshape(1)


def method_parts(method):
    parts = (method or "rtn").split("+")
    return parts[0], "x46" in parts[1:], "perm" in parts[1:]


def needs_h(method):
    base, _, perm = method_parts(method)
    return base == "gptq" or perm


class NV4Linear:
    """drop-in for an f8m Linear: mode stays 'fp8' so the engine's fused FP8 input paths are unchanged; the input then
    arrives as (x8, xs) and is dequantised to bf16 first (exact FP8 values), bf16 x is used when the engine passes it."""
    fq4 = True
    mode = "fp8"

    def __init__(self, w, method, out_features, in_features, tab=None, H=None):
        self.out_features, self.in_features = out_features, in_features
        base, self.x46, perm = method_parts(method)
        self.perm = None
        if perm:
            p = torch.argsort(torch.diag(H), descending=True, stable=True)
            self.perm = p.to(torch.int32).contiguous()
            w = w[:, p]
            H = H[p][:, p]
        if base == "gptq":
            self.wq, self.wsc, self.wg = gptq_quant_w(w, H)
        else:
            self.wq, self.wsc, self.wg = quant_w(w, base)
        self.method = method
        self.tab = tab or {}

    def variant(self, M):
        b = -(-M // 256) * 256
        return self.tab.get(f"{b}:{self.out_features}:{self.in_features}", int(os.environ.get("MAM_NV4_VAR", "3")))

    def nbytes(self):
        return self.wq.numel() + self.wsc.numel()

    def mm(self, x, out=None):
        if self.x46 or self.perm is not None:
            import nv4q
            xq, xs, buf = nv4q.quant_dyn(x.contiguous(), self.wg, self.x46, self.perm)
            return gemm(xq, xs, buf, self, out)
        xq, xs, buf = quant_x(x.contiguous(), self.wg)
        return gemm(xq, xs, buf, self, out)

    def __call__(self, x, q=None, out=None):
        if x is None:
            x = (q[0].float() * q[1]).to(torch.bfloat16)
        return self.mm(x, out)


def weight(get, i, t):
    p = f"layers.{i}."
    if t == "gu":
        return torch.cat([get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")], 0)
    if t == "down":
        return get(p + "mlp.down_proj.weight")
    if t == "qkvz":
        return torch.cat([get(p + "linear_attn.in_proj_qkv.weight"), get(p + "linear_attn.in_proj_z.weight")], 0)
    if t == "out":
        return get(p + "linear_attn.out_proj.weight")
    if t == "qkv":
        return torch.cat([get(p + "self_attn.q_proj.weight"), get(p + "self_attn.k_proj.weight"),
                          get(p + "self_attn.v_proj.weight")], 0)
    if t == "o":
        return get(p + "self_attn.o_proj.weight")
    raise KeyError(t)


class _HCap:
    """calibration shim: accumulates H = sum x^T x (fp32) of a projection's input, compute stays the wrapped module"""
    fq4 = True
    mode = "fp8"

    def __init__(self, orig):
        self.orig = orig
        self.out_features, self.in_features = orig.out_features, orig.in_features
        self.H = torch.zeros(orig.in_features, orig.in_features, device="cuda", dtype=torch.float32)

    def __getattr__(self, k):
        return getattr(self.__dict__["orig"], k)

    def __call__(self, x, q=None, out=None):
        xi = x if x is not None else (q[0].float() * q[1]).to(torch.bfloat16)
        xf = xi.float()
        self.H.addmm_(xf.t(), xf)
        return self.orig(x, q, out=out) if out is not None else self.orig(x, q)


def calib_encs(be, n):
    """the public seed set rendered by the backend: MAM_NV4_CAL = path of a bodies json (items_for_pool('s1-pro', n,
    seed=0x52454158) rendered like the validator's UDS client) or 'seed' = generate them here (nv4seed.py)"""
    src = os.environ.get("MAM_NV4_CAL", "seed")
    if src == "seed":
        import nv4seed
        bodies = nv4seed.bodies(n)
    else:
        bodies = json.load(open(src))[:n]
    encs = []
    for b in bodies:
        (qid, q), = b["questions"].items()
        encs.append(be.encode(b["state"], q))
    return encs


def calibrate(be, eng, keys, encs, C=8):
    """{(layer, type): H} for the given keys from one pass of encs (c8 batches) through the engine as it is now"""
    caps = {}
    for i, t in keys:
        L = eng.layers[i]
        caps[(i, t)] = L[SLOT[t]] = _HCap(L[SLOT[t]])
    try:
        with torch.no_grad():
            for j in range(0, len(encs), C):
                be.run(encs[j:j + C])
        torch.cuda.synchronize()
    finally:
        for (i, t), c in caps.items():
            eng.layers[i][SLOT[t]] = c.orig
    return {k: c.H for k, c in caps.items()}


def h_chunks(eng, keys, gb):
    """split keys into chunks whose fp32 Hessians fit in gb GiB"""
    out, cur, tot = [], [], 0
    for k in keys:
        K = eng.layers[k[0]][SLOT[k[1]]].in_features
        b = K * K * 4
        if cur and tot + b > gb * (1 << 30):
            out.append(cur)
            cur, tot = [], 0
        cur.append(k)
        tot += b
    if cur:
        out.append(cur)
    return out


def plan_needs_h(n_layers):
    return any(needs_h(m) for m in parse(SPEC, n_layers).values())


def install(eng, model_dir, be=None):
    """replace the planned projections of a built PplxEngine (fp8) by NVFP4 ones; returns the number replaced"""
    plan = parse(SPEC, eng.n_layers)
    if not plan:
        return 0
    from safetensors import safe_open
    t0 = time.time()
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    pre = "language_model." if any(k.startswith("language_model.") for k in idx) else "model.language_model."
    handles = {}

    def get(key):
        f = idx[pre + key]
        h = handles.get(f)
        if h is None:
            h = handles[f] = safe_open(os.path.join(model_dir, f), framework="pt", device="cuda")
        return h.get_tensor(pre + key)

    tab = {}
    tp = os.environ.get("MAM_NV4_TAB", os.path.join(_DIR, "nv4_tab.json"))
    if os.path.exists(tp):
        tab = json.load(open(tp))
    n = 0
    todo = [(k, m) for k, m in sorted(plan.items())
            if SLOT[k[1]] in eng.layers[k[0]] and not (k[0] == eng.n_layers - 1 and k[1] in ("qkv", "o"))]
    hk = [k for k, m in todo if needs_h(m)]
    tc = 0.0
    new = {}
    if hk:
        # calibrated methods first, every Hessian from the UNCHANGED f8m engine (no NVFP4 projection installed yet), in
        # chunks of MAM_NV4_CAL_GB GiB of fp32 Hessians; the new modules are held aside until all chunks are done
        if be is None:
            raise RuntimeError("nv4: calibrated methods (gptq / perm) need the backend (install(..., be=backend))")
        encs = calib_encs(be, int(os.environ.get("MAM_NV4_CAL_N", "256")))
        meth = dict(todo)
        for chunk in h_chunks(eng, hk, float(os.environ.get("MAM_NV4_CAL_GB", "6"))):
            t1 = time.time()
            Hs = calibrate(be, eng, chunk, encs)
            tc += time.time() - t1
            for (i, t) in chunk:
                old = eng.layers[i][SLOT[t]]
                new[(i, t)] = NV4Linear(weight(get, i, t), meth[(i, t)], old.out_features, old.in_features, tab,
                                        H=Hs.pop((i, t)))
            del Hs
            torch.cuda.empty_cache()
        print(f"rgh nv4: calibration {len(hk)} Hessians from {len(encs)} public seed-set items in {tc:.1f} s", flush=True)
    for (i, t), method in todo:
        L = eng.layers[i]
        s = SLOT[t]
        old = L[s]
        L[s] = new.pop((i, t)) if (i, t) in new else NV4Linear(weight(get, i, t), method, old.out_features,
                                                                 old.in_features, tab)
        del old
        n += 1
    handles.clear()
    gptq_release()
    print(f"rgh nv4: {n} projections NVFP4 ({SPEC}; gemm {GEMM}) in {time.time() - t0:.1f} s", flush=True)
    return n


def patch_engine(PP):
    """PplxEngine._mlp: route gate/up and down through NVFP4 when either is an NV4Linear (else the f8m path)"""
    orig_mlp = PP.PplxEngine._mlp
    import f8g
    K_ = PP.K

    def swiglu_y(x8, xs, lin, H):
        M, K = x8.shape
        N = lin.out_features
        y = torch.empty(M, H, device=x8.device, dtype=torch.bfloat16)
        r = f8g._lib.f8_gemm(2, f8g._pick(M, N, 2), x8.data_ptr(), x8.stride(0), lin.w8.data_ptr(), K, xs.data_ptr(),
                             lin.ws.data_ptr(), y.data_ptr(), H, M, N, K, -1, f8g._raw(x8.device.index),
                             int(N * K >= f8g.RASTER_BYTES))
        if r:
            raise RuntimeError(f"f8_gemm swiglu failed: {r}")
        return y

    def _mlp(self, L, x, xq=None):
        gu4, dn4 = getattr(L["gu"], "fq4", False), getattr(L["down"], "fq4", False)
        if not (gu4 or dn4):
            return orig_mlp(self, L, x, xq)
        if gu4:
            g = L["gu"](x, xq)
            if dn4:
                return L["down"](K_.silu_mul(g, self.inter))
            return L["down"](None, K_.silu_mul_q(g, self.inter))
        if xq is not None and f8g.ok(x.shape[0], L["gu"].out_features, L["gu"].in_features):
            y = swiglu_y(xq[0], xq[1], L["gu"], self.inter)
        else:
            y = K_.silu_mul(L["gu"](x, xq), self.inter)
        return L["down"](y)
    PP.PplxEngine._mlp = _mlp
