# SPDX-License-Identifier: Apache-2.0
"""Exact fused element-wise kernels for the Gemma4 VISION tower (eager, batch-invariant numerics), VIS_FUSE=1.

The vision tower runs Hugging Face's Gemma4 modules eagerly; with VLLM_BATCH_INVARIANT=1 every RMSNorm is 9 kernels
(x.float(), pow(2), vLLM's batch-invariant mean, + eps, pow(-0.5) = rsqrt, two muls, weight.float(), the cast back) and
every RoPE / GELU*up a dozen more. These replacements compute the SAME float32 operations in the same order with the
same intermediate roundings (bf16 rounding exactly where an eager op writes a bf16 tensor), so the outputs are
bit-identical (checked by vfusetest61.py on the real shapes); they only remove launches and memory round trips:
  * RMSNorm: kernel A squares x into float32 (= x.float().pow(2)), vLLM's own batch-invariant mean kernel reduces it
    (the same kernel the reference runs, so the sum order is identical), kernel B applies + eps, rsqrt, x * r, * w and
    the bf16 cast (optionally + a bf16 residual, = `residual + norm(x)`).
  * RoPE (apply_multidimensional_rope, 2-D axial): (x*cos) + (rotate_half(x)*sin) per 36-channel part, bf16 roundings
    after each product and after the sum, in one kernel.
  * MLP: gelu_tanh(gate) * up with ATen's tanh-GELU formula (float32 opmath, bf16 result) and the bf16 product.
  * VIS_ENC_GRAPH=1: the 27-layer encoder loop is captured as a CUDA graph per input shape (batch of images x 2520
    padded patches) and replayed: the same kernels with the same arguments, without the per-kernel Python launch cost
    (a lone image's encoder call 60 -> 31 ms); mask and rotary tables are computed eagerly and copied in.
  * VIS_ATTN=1 (vattn.py, gemm/vattn.so): the tower attention runs our exact replica of PyTorch's mem-efficient SDPA
    kernel. The encoder wrapper builds the stock 4-D mask (create_bidirectional_mask, as HF) and registers it with
    the 2-D key mask in a vattn context around the layer loop (eager or captured); the attention interface takes the
    kernel only for that mask object and the stock q/k/v layout, else stock SDPA. No padding (no mask): stock path.
  * VIS_UNPAD=1 (with VIS_ATTN=1, A2): when the padding is a suffix of the same length for every image, the layer
    loop runs on the real rows only (rotary tables computed at the padded shape, then sliced; attention without a
    mask = the padded reference's rows), and the padded rows of the output are zeros (the pooler zeroes them anyway).
Patched on import of transformers.models.gemma4.modeling_gemma4 (only modules of the vision/audio towers use these
classes; vLLM's text model has its own layers). Any unsupported input falls back to the original code.
"""
from __future__ import annotations

import importlib.abc
import os
import sys

ON = os.environ.get("VIS_FUSE", "0") == "1"
ARM_FILE = os.environ.get("VIS_ARM_FILE", "")      # tests only: {"fuse": 0|1, "bi": 0|1} switches per encoder call
_ACTIVE = [True]
GRAPH = os.environ.get("VIS_ENC_GRAPH", "0") == "1"   # CUDA-graph the 27-layer encoder loop per input shape
GRAPH_BMAX = int(os.environ.get("VIS_ENC_GRAPH_BMAX", "12"))
GRAPH_PRE = os.environ.get("VIS_ENC_GRAPH_PRE", "1") == "1"   # capture every batch size 1..BMAX at the first call
_GRAPH_ON = [GRAPH]
_G = {}
FUSE_RES = os.environ.get("VIS_FUSE_RES", "1") == "1"
_K = {}
ATTN = os.environ.get("VIS_ATTN", "0") == "1"          # exact tower attention kernel (vattn.py)
UNPAD = os.environ.get("VIS_UNPAD", "0") == "1"        # A2: real rows only (needs VIS_ATTN)
_ATTN_ON = [ATTN]
_UNPAD_ON = [UNPAD]
_VA = {}
_GA = {}
# V3 lever E5 (VIS_E5=1, default off): the whole image encoder of a board-grid picture batch (patch embedding -> 27
# layers -> pooling -> standardize -> embed_vision norm + projection) as ONE CUDA-graph replay per batch size 1..BMAX,
# with no device->host sync. Exact-match hook: only pictures whose CPU-side position ids equal the canonical padded grid
# of a configured size (VIS_E5_GRIDS, W x H patches; the board's 320x240 -> 912x672 is 57x42) and the known config;
# anything else runs vLLM's stock _process_image_input (with v5's layer-loop graphs when VIS_ENC_GRAPH=1).
E5 = os.environ.get("VIS_E5", "0") == "1"
E5_BMAX = int(os.environ.get("VIS_E5_BMAX", "12"))
E5_GRIDS = tuple(tuple(int(v) for v in g.split("x")) for g in os.environ.get("VIS_E5_GRIDS", "57x42").split(",") if g)
_E5ON = [E5]
_E5 = {"cpu": None, "grids": {}, "pool": None, "canon": {}, "chunk": {},
       "capture_ok": None}       # an engine may set capture_ok = callable -> bool: E5 captures only when it is True
E5_STATS = {"calls": 0, "images": 0, "replays": 0, "graphs": 0, "fallback": 0, "nocpu": 0, "capture_s": 0.0}
# V3 levers E3 (VIS_E3=1: q/k/v as one N=3456 GEMM, gate/up + GELU-tanh x up as one CUTLASS dual GEMM with an exact
# epilogue) and E4 (VIS_E4=1: one-pass RMSNorm chains + fused q/k/v head norms and 2-D RoPE), both default off; native
# kernels in gemm/vis_ops.so (vis_ops.cu), per-M launch configs in vis_e3.json (speed only: every config is exact).
E3 = os.environ.get("VIS_E3", "0") == "1"
E4 = os.environ.get("VIS_E4", "0") == "1"
_E3ON = [E3]
_E4ON = [E4]
_VO = {}
E34_STATS = {"e3_qkv": 0, "e3_mlp": 0, "e4_loop": 0, "e4_qkv": 0, "e3_nobuf": 0, "fallback": 0}
E4_DEBUG = [None]          # tests: a list -> the E4 layer loop appends each layer's output (eager only)


def _lever_sig():
    """E3 / E4 state baked into a captured graph: part of every graph key, so an in-process A/B never replays a graph
    captured with the other setting"""
    return (bool(_E3ON[0]), bool(_E4ON[0]))


def _kernels():
    if _K:
        return _K
    import triton
    import triton.language as tl

    @triton.jit
    def _sq_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + offs, x * x, mask=m)

    @triton.jit
    def _norm_epi(x_ptr, ms_ptr, w_ptr, res_ptr, out_ptr, N, eps, HAS_W: tl.constexpr, HAS_RES: tl.constexpr,
                  RSQRT: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        ms = tl.load(ms_ptr + row) + eps
        if RSQRT == 0:
            r = tl.math.rsqrt(ms)
        elif RSQRT == 1:
            r = 1.0 / tl.sqrt_rn(ms)
        else:
            r = tl.extra.cuda.libdevice.rsqrt(ms)
        for n0 in range(0, N, BLOCK):
            offs = n0 + tl.arange(0, BLOCK)
            m = offs < N
            x = tl.load(x_ptr + row * N + offs, mask=m, other=0.0).to(tl.float32)
            y = x * r
            if HAS_W:
                w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
                y = y * w
            yb = y.to(tl.bfloat16)
            if HAS_RES:
                res = tl.load(res_ptr + row * N + offs, mask=m, other=0.0).to(tl.float32)
                yb = (res + yb.to(tl.float32)).to(tl.bfloat16)
            tl.store(out_ptr + row * N + offs, yb, mask=m)

    @triton.jit
    def _rope2(x_ptr, cos_ptr, sin_ptr, out_ptr, H, D: tl.constexpr, HALF: tl.constexpr, QUART: tl.constexpr,
               BLOCK: tl.constexpr):
        # x: [T, H, D] (T = batch*patches), cos/sin: [T, D]; D = 2 parts of HALF channels, rotate_half per part
        t = tl.program_id(0)
        h = tl.program_id(1)
        j = tl.arange(0, BLOCK)
        m = j < D
        part = j // HALF
        jj = j - part * HALF
        src = tl.where(jj < QUART, part * HALF + jj + QUART, part * HALF + jj - QUART)
        base = (t * H + h) * D
        x = tl.load(x_ptr + base + j, mask=m, other=0.0).to(tl.float32)
        xs = tl.load(x_ptr + base + src, mask=m, other=0.0).to(tl.float32)
        xr = tl.where(jj < QUART, -xs, xs)
        c = tl.load(cos_ptr + t * D + j, mask=m, other=0.0).to(tl.float32)
        s = tl.load(sin_ptr + t * D + j, mask=m, other=0.0).to(tl.float32)
        a = (x * c).to(tl.bfloat16).to(tl.float32)
        b = (xr * s).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + base + j, (a + b).to(tl.bfloat16), mask=m)

    @triton.jit
    def _gelu_mul(g_ptr, u_ptr, out_ptr, n, USE_FMA: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + offs, mask=m, other=0.0).to(tl.float32)
        kBeta = 0.7978845608028654
        kKappa = 0.044715
        x3 = x * x * x
        if USE_FMA:
            inner = kBeta * tl.fma(kKappa, x3, x)
        else:
            inner = kBeta * (x + kKappa * x3)
        g = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
        gb = g.to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + offs, (gb * u).to(tl.bfloat16), mask=m)

    _K.update(sq=_sq_kernel, epi=_norm_epi, rope=_rope2, gelu=_gelu_mul)
    return _K


NOFMA = {"enable_fp_fusion": False}
RSQRT_MODE = int(os.environ.get("VIS_FUSE_RSQRT", "0"))


def rms_norm(x, weight, eps, residual=None):
    """bf16 x [..., N] -> bf16, = HF Gemma4RMSNorm(x) (+ residual) under vLLM batch-invariant mean."""
    import torch
    from vllm.model_executor.determinism import batch_invariant as bi
    k = _kernels()
    N = x.shape[-1]
    xc = x.contiguous()
    sq = torch.empty(xc.shape, device=x.device, dtype=torch.float32)
    n = xc.numel()
    k["sq"][(triton_cdiv(n, 4096),)](xc, sq, n, BLOCK=4096, **NOFMA)
    ms = bi.mean_dim(sq, dim=sq.ndim - 1, keepdim=True)      # vLLM's batch-invariant mean (the reference's kernel)
    out = torch.empty(xc.shape, device=x.device, dtype=torch.bfloat16)
    rows = n // N
    w = weight if weight is not None else xc
    res = residual.contiguous() if residual is not None else xc
    k["epi"][(rows,)](xc, ms, w, res, out, N, float(eps), HAS_W=weight is not None, HAS_RES=residual is not None,
                      RSQRT=RSQRT_MODE, BLOCK=triton_next_pow2(min(N, 2048)), **NOFMA)
    return out


def triton_cdiv(a, b):
    return (a + b - 1) // b


def triton_next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _install(m):
    import torch
    orig_norm = m.Gemma4RMSNorm.forward
    orig_rope = m.apply_multidimensional_rope
    orig_mlp = m.Gemma4VisionMLP.forward
    orig_layer = m.Gemma4VisionEncoderLayer.forward
    stats = {"norm": 0, "rope": 0, "mlp": 0, "fallback": 0}

    def ok(t):
        return _ACTIVE[0] and t.is_cuda and t.dtype == torch.bfloat16

    def norm_forward(self, hidden_states):
        if not ok(hidden_states) or (self.with_scale and self.weight.dtype != torch.bfloat16):
            stats["fallback"] += 1
            return orig_norm(self, hidden_states)
        stats["norm"] += 1
        return rms_norm(hidden_states, self.weight if self.with_scale else None, self.eps)

    def rope(x, cos, sin, position_ids, unsqueeze_dim=2):
        D = x.shape[-1]
        if (not ok(x) or x.ndim != 4 or unsqueeze_dim != 2 or position_ids.shape[-1] != 2 or cos.dtype != x.dtype
                or D % 4 or cos.shape[-1] != D or 2 * (D // 4) * 2 != D):
            stats["fallback"] += 1
            return orig_rope(x, cos, sin, position_ids, unsqueeze_dim)
        stats["rope"] += 1
        B, P, H, _ = x.shape
        xc = x.contiguous()
        c = cos.contiguous().reshape(B * P, D)
        s = sin.contiguous().reshape(B * P, D)
        out = torch.empty_like(xc)
        _kernels()["rope"][(B * P, H)](xc, c, s, out, H, D=D, HALF=D // 2, QUART=D // 4, BLOCK=triton_next_pow2(D),
                                       **NOFMA)
        return out

    gelu_fma = os.environ.get("VIS_FUSE_GELU_FMA", "1") == "1"

    def mlp_forward(self, x):
        if (not ok(x) or self.config.hidden_activation != "gelu_pytorch_tanh"
                or getattr(self.gate_proj, "use_clipped_linears", False)):
            stats["fallback"] += 1
            return orig_mlp(self, x)
        if _E3ON[0]:
            out = _e3_mlp(self, x)
            if out is not None:
                return out
        stats["mlp"] += 1
        g = self.gate_proj(x).contiguous()
        u = self.up_proj(x).contiguous()
        h = torch.empty_like(g)
        n = g.numel()
        _kernels()["gelu"][(triton_cdiv(n, 2048),)](g, u, h, n, USE_FMA=gelu_fma, BLOCK=2048, **NOFMA)
        return self.down_proj(h)

    def layer_forward(self, hidden_states, position_embeddings=None, attention_mask=None, position_ids=None, **kwargs):
        if not ok(hidden_states) or self.post_attention_layernorm.weight.dtype != torch.bfloat16:
            stats["fallback"] += 1
            return orig_layer(self, hidden_states, position_embeddings, attention_mask, position_ids, **kwargs)
        residual = hidden_states
        hs = self.input_layernorm(hidden_states)
        hs, _ = self.self_attn(hidden_states=hs, position_embeddings=position_embeddings,
                               attention_mask=attention_mask, position_ids=position_ids, **kwargs)
        n1 = self.post_attention_layernorm
        hidden_states = rms_norm(hs, n1.weight, n1.eps, residual=residual)       # = residual + norm(hs)
        residual = hidden_states
        hs = self.pre_feedforward_layernorm(hidden_states)
        hs = self.mlp(hs)
        n2 = self.post_feedforward_layernorm
        return rms_norm(hs, n2.weight, n2.eps, residual=residual)

    orig_enc = m.Gemma4VisionEncoder.forward

    def layers_loop(self, hs, mask, cos, sin, pos):
        if _E4ON[0] and _ACTIVE[0]:
            out = _e4_loop(self, hs, mask, cos, sin, pos)
            if out is not None:
                return out
        for layer in self.layers[: self.config.num_hidden_layers]:
            hs = layer(hs, attention_mask=mask, position_embeddings=(cos, sin), position_ids=pos)
        return hs

    def capture(self, shape, mask_meta, pos_shape, dev):
        """CUDA graph of the layer loop for one input shape (same kernels as eager, replayed without Python)."""
        _e3_prepare(self)
        if "pool" not in _G:
            _G["pool"] = torch.cuda.graph_pool_handle()
        st = {"hs": torch.zeros(shape, device=dev, dtype=torch.bfloat16),
              "cos": torch.zeros((*shape[:2], self.config.head_dim), device=dev, dtype=torch.bfloat16),
              "sin": torch.zeros((*shape[:2], self.config.head_dim), device=dev, dtype=torch.bfloat16),
              "pos": torch.zeros(pos_shape, device=dev, dtype=torch.long),
              "mask": None if mask_meta is None else torch.zeros(mask_meta[0], device=dev, dtype=mask_meta[1])}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                layers_loop(self, st["hs"], st["mask"], st["cos"], st["sin"], st["pos"])
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=_G["pool"]):
            st["out"] = layers_loop(self, st["hs"], st["mask"], st["cos"], st["sin"], st["pos"])
        st["g"] = g
        return st

    def enc_forward(self, inputs_embeds=None, attention_mask=None, pixel_position_ids=None, **kw):
        if ARM_FILE:      # test arms: read once per encoder call (the frontend writes the file between runs)
            try:
                import json
                with open(ARM_FILE) as fh:
                    arm = json.load(fh)
                _ACTIVE[0] = bool(arm.get("fuse", 1))
                _GRAPH_ON[0] = GRAPH and bool(arm.get("graph", 1))
                bb = sys.modules.get("blink_bi")
                if bb is not None:
                    bb.ON = bool(arm.get("bi", 1))
                _ATTN_ON[0] = ATTN and bool(arm.get("attn", 1))
                _UNPAD_ON[0] = UNPAD and bool(arm.get("unpad", 1))
                _E5ON[0] = E5 and bool(arm.get("e5", 1))
                _E3ON[0] = E3 and bool(arm.get("e3", 1))
                _E4ON[0] = E4 and bool(arm.get("e4", 1))
            except (OSError, ValueError):
                pass
        if (_ATTN_ON[0] and "m" in _VA and _ACTIVE[0] and not kw and inputs_embeds is not None
                and pixel_position_ids is not None and isinstance(attention_mask, torch.Tensor)
                and attention_mask.dim() == 2 and attention_mask.dtype == torch.bool and inputs_embeds.is_cuda
                and inputs_embeds.dtype == torch.bfloat16 and not torch.cuda.is_current_stream_capturing()):
            out = enc_forward_attn(self, inputs_embeds, attention_mask, pixel_position_ids)
            if out is not None:
                from transformers.modeling_outputs import BaseModelOutputWithPast
                return BaseModelOutputWithPast(last_hidden_state=out)
        if (not _GRAPH_ON[0] or not _ACTIVE[0] or kw or inputs_embeds is None or pixel_position_ids is None
                or not inputs_embeds.is_cuda or inputs_embeds.dtype != torch.bfloat16
                or torch.cuda.is_current_stream_capturing()):
            return orig_enc(self, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                            pixel_position_ids=pixel_position_ids, **kw)
        mask = m.create_bidirectional_mask(config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        if mask is not None and not isinstance(mask, torch.Tensor):
            stats["fallback"] += 1
            return orig_enc(self, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                            pixel_position_ids=pixel_position_ids)
        cos, sin = self.rotary_emb(inputs_embeds, pixel_position_ids)
        B = inputs_embeds.shape[0]
        mmeta = None if mask is None else (tuple(mask.shape), mask.dtype)
        key = (id(self), tuple(inputs_embeds.shape), mmeta, tuple(pixel_position_ids.shape), _lever_sig())
        st = _G.get(key)
        if st is None:
            if B > GRAPH_BMAX or len(_G) > 64:
                stats["fallback"] += 1
                return orig_enc(self, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                pixel_position_ids=pixel_position_ids)
            st = capture(self, tuple(inputs_embeds.shape), mmeta, tuple(pixel_position_ids.shape), inputs_embeds.device)
            _G[key] = st
            stats["graphs"] = stats.get("graphs", 0) + 1
            if GRAPH_PRE and mmeta is not None:
                # the encoder input is always B x 2520 padded patches, so the other batch sizes of this shape can be
                # captured now (start-up warm-up) instead of during the first timed requests
                import time as _time
                t0 = _time.time()
                for b in range(1, GRAPH_BMAX + 1):
                    shp = (b, *inputs_embeds.shape[1:])
                    mm = ((b, *mmeta[0][1:]), mmeta[1])
                    ps = (b, *pixel_position_ids.shape[1:])
                    k2 = (id(self), shp, mm, ps, _lever_sig())
                    if k2 not in _G:
                        _G[k2] = capture(self, shp, mm, ps, inputs_embeds.device)
                        stats["graphs"] += 1
                print(f"[vfuse] encoder graphs for batch 1..{GRAPH_BMAX} x {tuple(inputs_embeds.shape[1:])} captured "
                      f"in {_time.time() - t0:.1f} s", file=sys.stderr, flush=True)
        st["hs"].copy_(inputs_embeds)
        st["cos"].copy_(cos)
        st["sin"].copy_(sin)
        st["pos"].copy_(pixel_position_ids)
        if mask is not None:
            st["mask"].copy_(mask)
        st["g"].replay()
        stats["replay"] = stats.get("replay", 0) + 1
        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(last_hidden_state=st["out"].clone())

    def capture_attn(self, shape, mmeta, kvshape, unpad, dev):
        """CUDA graph of the layer loop with the vattn context bound to persistent mask buffers."""
        _e3_prepare(self)
        va = _VA["m"]
        if "pool" not in _G:
            _G["pool"] = torch.cuda.graph_pool_handle()
        st = {"hs": torch.zeros(shape, device=dev, dtype=torch.bfloat16),
              "cos": torch.zeros((*shape[:2], self.config.head_dim), device=dev, dtype=torch.bfloat16),
              "sin": torch.zeros((*shape[:2], self.config.head_dim), device=dev, dtype=torch.bfloat16),
              "pos": torch.zeros((*shape[:2], 2), device=dev, dtype=torch.long),
              "mask": None if mmeta is None else torch.zeros(mmeta[0], device=dev, dtype=mmeta[1]),
              "kvalid": None if kvshape is None else torch.zeros(kvshape, device=dev, dtype=torch.bool)}
        ctx = va.Ctx(st["mask"], st["kvalid"], unpad)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), va.active(ctx):
            for _ in range(2):
                layers_loop(self, st["hs"], st["mask"], st["cos"], st["sin"], st["pos"])
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        ctx.used = 0
        with torch.cuda.graph(g, pool=_G["pool"]), va.active(ctx):
            st["out"] = layers_loop(self, st["hs"], st["mask"], st["cos"], st["sin"], st["pos"])
        st["g"] = g
        st["used"] = ctx.used
        return st

    def run_attn(self, hs, mask, kvalid, cos, sin, pos, unpad):
        va = _VA["m"]
        if not _GRAPH_ON[0] or hs.shape[0] > GRAPH_BMAX:
            with va.active(va.Ctx(mask, kvalid, unpad)):
                return layers_loop(self, hs, mask, cos, sin, pos)
        mmeta = None if mask is None else (tuple(mask.shape), mask.dtype)
        kvs = None if kvalid is None else tuple(kvalid.shape)
        key = (id(self), tuple(hs.shape), mmeta, kvs, unpad, _lever_sig())
        st = _GA.get(key)
        if st is None:
            if len(_GA) > 64:
                with va.active(va.Ctx(mask, kvalid, unpad)):
                    return layers_loop(self, hs, mask, cos, sin, pos)
            import time as _time
            t0 = _time.time()
            pre = range(1, GRAPH_BMAX + 1) if GRAPH_PRE else (hs.shape[0],)
            for b in pre:
                shp = (b, *hs.shape[1:])
                mm = None if mmeta is None else ((b, *mmeta[0][1:]), mmeta[1])
                kk = None if kvs is None else (b, *kvs[1:])
                k2 = (id(self), shp, mm, kk, unpad, _lever_sig())
                if k2 not in _GA:
                    _GA[k2] = capture_attn(self, shp, mm, kk, unpad, hs.device)
                    stats["graphs"] = stats.get("graphs", 0) + 1
            st = _GA[key]
            print(f"[vfuse] vattn encoder graphs ({'unpadded' if unpad else 'padded'}) for batch {pre[0]}..{pre[-1]} "
                  f"x {tuple(hs.shape[1:])} captured in {_time.time() - t0:.1f} s; vattn layers per graph "
                  f"{st['used']}", file=sys.stderr, flush=True)
        st["hs"].copy_(hs)
        st["cos"].copy_(cos)
        st["sin"].copy_(sin)
        st["pos"].copy_(pos)
        if mask is not None:
            st["mask"].copy_(mask)
            st["kvalid"].copy_(kvalid)
        st["g"].replay()
        stats["replay_attn"] = stats.get("replay_attn", 0) + 1
        return st["out"].clone()

    def enc_forward_attn(self, inputs_embeds, attention_mask, pixel_position_ids):
        """VIS_ATTN path: the stock 4-D mask (None = no padding: stock path), then the layer loop under a vattn
        context, padded or (VIS_UNPAD) on the real rows only. Returns the last hidden state or None (stock)."""
        va = _VA["m"]
        mask = m.create_bidirectional_mask(config=self.config, inputs_embeds=inputs_embeds,
                                           attention_mask=attention_mask)
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.dim() != 4:
            stats["attn_stock"] = stats.get("attn_stock", 0) + 1
            return None
        cos, sin = self.rotary_emb(inputs_embeds, pixel_position_ids)
        S = inputs_embeds.shape[1]
        n = va.suffix_len(attention_mask) if _UNPAD_ON[0] else None
        if n is not None and n < S:
            out_n = run_attn(self, inputs_embeds[:, :n], None, None, cos[:, :n], sin[:, :n],
                             pixel_position_ids[:, :n], True)
            out = torch.zeros_like(inputs_embeds)
            out[:, :n] = out_n
            stats["attn_unpad"] = stats.get("attn_unpad", 0) + 1
            return out
        kv = attention_mask if attention_mask.is_contiguous() else attention_mask.contiguous()
        stats["attn_pad"] = stats.get("attn_pad", 0) + 1
        return run_attn(self, inputs_embeds, mask, kv, cos, sin, pixel_position_ids, False)

    m.Gemma4VisionEncoder.forward = enc_forward
    if ATTN:
        try:
            if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import vattn as _vattn_mod
            _vattn_mod.install(m)
            _VA["m"] = _vattn_mod
        except Exception as exc:  # noqa: BLE001
            print(f"[vfuse] vattn not installed: {exc!r}", file=sys.stderr, flush=True)
    m.Gemma4RMSNorm.forward = norm_forward
    m.apply_multidimensional_rope = rope
    m.Gemma4VisionMLP.forward = mlp_forward
    if E3 or E4:
        orig_attn = m.Gemma4VisionAttention.forward

        def attn_forward(self, hidden_states, position_embeddings=None, attention_mask=None, position_ids=None,
                         **kwargs):
            out = None
            if (_E3ON[0] or _E4ON[0]) and _ACTIVE[0] and ok(hidden_states):
                out = _e34_attn(m, self, hidden_states, position_embeddings, attention_mask, position_ids, kwargs)
            if out is None:
                return orig_attn(self, hidden_states, position_embeddings=position_embeddings,
                                 attention_mask=attention_mask, position_ids=position_ids, **kwargs)
            return out

        m.Gemma4VisionAttention.forward = attn_forward
        m._vfuse_e34_stats = E34_STATS
    if FUSE_RES:
        m.Gemma4VisionEncoderLayer.forward = layer_forward
    m._vfuse_stats = stats
    print(f"[vfuse] fused Gemma4 vision element-wise kernels installed (pid {os.getpid()}, residual fusion {FUSE_RES}, "
          f"attention kernel {'m' in _VA}, unpadded rows {UNPAD and 'm' in _VA})", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------------------------
# E3 / E4 (V3): vis_ops.so. Exact-match hooks: bf16 CUDA tensors of the known shapes (hidden 1152, 16 x 72 heads,
# MLP 4304), plain unquantized bias-free linears, Gemma4RMSNorm with bf16 weights and one eps; anything else -> the
# vfuse / stock code above.


def _vo():
    """gemm/vis_ops.so (vis_ops.cu, built by build_vis_ops.py) + the per-M config table vis_e3.json"""
    if "m" not in _VO:
        import glob
        import importlib.util
        import json
        d = os.environ.get("VIS_OPS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemm"))
        so = sorted(glob.glob(os.path.join(d, "vis_ops*.so")))
        if not so:
            raise RuntimeError(f"vis_ops extension not found in {d}")
        spec = importlib.util.spec_from_file_location("vis_ops", so[0])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _VO["m"] = mod
        tab = {}
        tp = os.environ.get("VIS_E3_TABLE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "vis_e3.json"))
        try:
            with open(tp) as fh:
                raw = json.load(fh)
            tab = {k: sorted((int(a), int(c)) for a, c in v) for k, v in raw.get("tables", {}).items()}
        except (OSError, ValueError):
            pass
        _VO["tab"] = tab
        print(f"[vfuse-e34] vis_ops loaded ({so[0]}; E3 {E3}, E4 {E4}; config tables {sorted(tab)})",
              file=sys.stderr, flush=True)
    return _VO["m"]


def _e3_cfg(name, M, default):
    for mm, c in _VO.get("tab", {}).get(name, ()):
        if M <= mm:
            return c
    return default


def _bgemm():
    """the CUTLASS sm80 bf16 GEMM blink_bi uses (gemm/blink_gemm.so): the fused QKV GEMM is the same kernel family"""
    if "bg" not in _VO:
        import glob
        import importlib.util
        d = os.environ.get("BLINK_GEMM_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemm"))
        so = sorted(glob.glob(os.path.join(d, "blink_gemm*.so")))
        spec = importlib.util.spec_from_file_location("blink_gemm", so[0])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _VO["bg"] = mod
    return _VO["bg"]


def _lin_w(mod):
    """weight of an unquantized, bias-free linear (HF Gemma4ClippableLinear -> vLLM ReplicatedLinear), else None"""
    import torch
    if getattr(mod, "use_clipped_linears", False):
        return None
    lin = getattr(mod, "linear", mod)
    w = getattr(lin, "weight", None)
    if (w is None or getattr(lin, "bias", None) is not None or not isinstance(w, torch.Tensor) or w.ndim != 2
            or w.dtype != torch.bfloat16 or not w.is_cuda or not w.is_contiguous()):
        return None
    qm = getattr(lin, "quant_method", None)
    if qm is not None and type(qm).__name__ != "UnquantizedLinearMethod":
        return None
    return w


def _e3_prepare(enc):
    """build every layer's [Wq; Wk; Wv] (N = 3456) once, outside any graph capture (a copy: the original weights stay,
    graphs captured earlier keep their pointers)"""
    import torch
    if not E3 or torch.cuda.is_current_stream_capturing():
        return
    for layer in getattr(enc, "layers", ()):
        at = getattr(layer, "self_attn", None)
        if at is None or hasattr(at, "_v3_wqkv"):
            continue
        ws = [_lin_w(getattr(at, n, None)) for n in ("q_proj", "k_proj", "v_proj")]
        ok = all(w is not None and tuple(w.shape) == (1152, 1152) for w in ws)
        at._v3_wqkv = torch.cat(ws, 0).contiguous() if ok else None


def _e3_qkv(at, x2):
    """[M, 1152] -> [M, 3456] = x2 @ [Wq; Wk; Wv].T (our CUTLASS GEMM: each column's k chain is the reference's)"""
    import torch
    W = getattr(at, "_v3_wqkv", "missing")
    if isinstance(W, str):
        if torch.cuda.is_current_stream_capturing():
            E34_STATS["e3_nobuf"] += 1
            return None
        _e3_prepare_one(at)
        W = at._v3_wqkv
    if W is None or x2.stride(1) != 1 or x2.stride(0) % 8 or x2.data_ptr() % 16:
        return None
    _vo()
    cfg = _e3_cfg("qkv", x2.shape[0], 0)
    if cfg < 0:                       # table: three separate GEMMs are faster at this row count
        return None
    E34_STATS["e3_qkv"] += 1
    return _bgemm().mm(x2, W, cfg)


def _e3_prepare_one(at):
    import torch
    ws = [_lin_w(getattr(at, n, None)) for n in ("q_proj", "k_proj", "v_proj")]
    ok = all(w is not None and tuple(w.shape) == (1152, 1152) for w in ws)
    at._v3_wqkv = torch.cat(ws, 0).contiguous() if ok else None


def _e3_mlp(mlp, x):
    """E3 MLP: one CUTLASS dual GEMM (gate, up, exact GELU-tanh x up epilogue), then the stock down projection"""
    wg, wu = _lin_w(mlp.gate_proj), _lin_w(mlp.up_proj)
    H = x.shape[-1]
    if (wg is None or wu is None or tuple(wg.shape) != (4304, 1152) or tuple(wu.shape) != (4304, 1152) or H != 1152
            or not x.is_contiguous()):
        E34_STATS["fallback"] += 1
        return None
    x2 = x.reshape(-1, H)
    if x2.stride(0) % 8 or x2.data_ptr() % 16 or wg.data_ptr() % 16 or wu.data_ptr() % 16:
        E34_STATS["fallback"] += 1
        return None
    vo = _vo()
    cfg = _e3_cfg("gelu_up", x2.shape[0], 0)
    if cfg < 0:                       # table: v5's two GEMMs + GELU kernel are faster at this row count
        return None
    E34_STATS["e3_mlp"] += 1
    h = vo.gelu_up(x2, wg, wu, cfg)
    return mlp.down_proj(h.view(*x.shape[:-1], wg.shape[0]))


def _e34_attn(m, at, hs, pe, mask, pos, kwargs):
    """HF Gemma4VisionAttention.forward with E3's fused q/k/v GEMM and/or E4's fused head norms + RoPE; the attention
    call (transformers' interface, V2's vattn wrapper) and o_proj are the stock ones. None = not eligible."""
    import torch
    if pe is None or hs.dim() != 3 or hs.shape[-1] != 1152 or not hs.is_contiguous() or at.head_dim != 72:
        return None
    cos, sin = pe
    B, S, H = hs.shape
    M = B * S
    x2 = hs.view(M, H)
    qkv = _e3_qkv(at, x2) if _E3ON[0] else None
    if qkv is not None:
        q2, k2, v2 = qkv[:, :1152], qkv[:, 1152:2304], qkv[:, 2304:]
    else:
        q2, k2, v2 = (getattr(at, n)(hs).view(M, 1152) for n in ("q_proj", "k_proj", "v_proj"))
    hidden_shape = (B, S, -1, at.head_dim)
    e4 = (_E4ON[0] and at.q_norm.with_scale and at.k_norm.with_scale and not at.v_norm.with_scale
          and at.q_norm.eps == at.k_norm.eps == at.v_norm.eps and at.q_norm.weight.dtype == torch.bfloat16
          and at.k_norm.weight.dtype == torch.bfloat16 and cos.dtype == torch.bfloat16 and sin.dtype == torch.bfloat16
          and tuple(cos.shape) == (B, S, 72) and tuple(sin.shape) == (B, S, 72) and pos is not None
          and pos.shape[-1] == 2)
    if e4:
        c2 = cos.reshape(M, 72) if cos.is_contiguous() else cos.contiguous().view(M, 72)
        s2 = sin.reshape(M, 72) if sin.is_contiguous() else sin.contiguous().view(M, 72)
        qo, ko, vo = _vo().qkv_norm_rope(q2, k2, v2, at.q_norm.weight, at.k_norm.weight, c2, s2, float(at.q_norm.eps))
        E34_STATS["e4_qkv"] += 1
        query_states = qo.view(hidden_shape).transpose(1, 2)
        key_states = ko.view(hidden_shape).transpose(1, 2)
        value_states = vo.view(hidden_shape).transpose(1, 2)
    else:
        query_states = at.q_norm(q2.view(hidden_shape))
        query_states = m.apply_multidimensional_rope(query_states, cos, sin, pos).transpose(1, 2)
        key_states = at.k_norm(k2.view(hidden_shape))
        key_states = m.apply_multidimensional_rope(key_states, cos, sin, pos).transpose(1, 2)
        value_states = at.v_norm(v2.view(hidden_shape)).transpose(1, 2)
    attention_interface = m.ALL_ATTENTION_FUNCTIONS.get_interface(at.config._attn_implementation,
                                                                  m.eager_attention_forward)
    attn_output, attn_weights = attention_interface(at, query_states, key_states, value_states, mask,
                                                    dropout=at.attention_dropout if at.training else 0.0,
                                                    scaling=at.scaling, **kwargs)
    attn_output = attn_output.reshape(B, S, -1).contiguous()
    return at.o_proj(attn_output), attn_weights


def _e4_ok(enc):
    import torch
    ok = enc.__dict__.get("_v3_e4ok")
    if ok is None:
        ok = True
        eps = None
        for layer in enc.layers[: enc.config.num_hidden_layers]:
            for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                      "post_feedforward_layernorm"):
                nm = getattr(layer, n, None)
                w = getattr(nm, "weight", None)
                if (nm is None or not getattr(nm, "with_scale", False) or w is None or w.dtype != torch.bfloat16
                        or w.numel() != 1152 or not w.is_contiguous() or (eps is not None and nm.eps != eps)):
                    ok = False
                eps = getattr(nm, "eps", eps)
        enc.__dict__["_v3_e4ok"] = ok
        enc.__dict__["_v3_eps"] = float(eps) if eps is not None else 1e-6
    return ok


def _e4_loop(enc, hs, mask, cos, sin, pos):
    """the 27-layer loop with one-pass norm chains: (A) layer 0's input norm; per layer (B) post-attention norm +
    residual + pre-feed-forward norm, (C) post-feed-forward norm + residual + the next layer's input norm. The
    attention and MLP modules are called as in the stock layer. None = not eligible."""
    import torch
    if (hs.dim() != 3 or hs.shape[-1] != 1152 or hs.dtype != torch.bfloat16 or not hs.is_cuda
            or not _e4_ok(enc)):
        return None
    vo = _vo()
    B, S, H = hs.shape
    M = B * S
    eps = enc.__dict__["_v3_eps"]
    layers = enc.layers[: enc.config.num_hidden_layers]
    hh = hs.reshape(M, H)
    if hh.stride(0) != H:
        hh = hh.contiguous()
    x_in = vo.rms_norm(hh, layers[0].input_layernorm.weight, eps)
    dbg = E4_DEBUG[0]
    for i, layer in enumerate(layers):
        a, _ = layer.self_attn(hidden_states=x_in.view(B, S, H), position_embeddings=(cos, sin), attention_mask=mask,
                               position_ids=pos)
        hh, mm = vo.norm_res_norm(a.reshape(M, H), hh, layer.post_attention_layernorm.weight,
                                  layer.pre_feedforward_layernorm.weight, eps, eps)
        f = layer.mlp(mm.view(B, S, H))
        nxt = layers[i + 1].input_layernorm.weight if i + 1 < len(layers) else None
        hh, x_in = vo.norm_res_norm(f.reshape(M, H), hh, layer.post_feedforward_layernorm.weight, nxt, eps, eps)
        if dbg is not None and not torch.cuda.is_current_stream_capturing():
            dbg.append(hh.view(B, S, H))
    E34_STATS["e4_loop"] += 1
    return hh.view(B, S, H)


# ------------------------------------------------------------------------------------------------------------------
# E5: whole-encoder graph (V3). Stock reference = vLLM 0.30.0 gemma4_mm.Gemma4ForConditionalGeneration.
# _process_image_input + HF transformers 5.17.0 Gemma4 vision modules. Every op below is the stock op on the stock
# shapes (or a batch of the stock per-row / per-image op, all of which are batch-invariant: elementwise ops, vLLM's
# batch-invariant bmm with one program per (batch, tile), the BI RMSNorm and linears); only host control flow changes:
#  * the CPU-side position ids (taken from the encoder runner's MultiModalKwargsItems before the H2D copy) decide
#    eligibility, the padding mask, the pooled-token count and which pooled rows are valid, so no `.all()` probe,
#    no boolean index and no other device->host read happens;
#  * position-dependent tensors are computed ONCE per grid at the first eligible call with the stock code: the patch
#    position embedding (F.embedding x/y + add + padding where, B=1, broadcast in the add), the bidirectional mask
#    (create_bidirectional_mask, B=BMAX, contiguous copy as v5's graphs use), the axial RoPE cos/sin (B=BMAX) and the
#    pooling weights (one_hot / k^2, B=1); they live outside the graph pool;
#  * the graph output is cloned before it is handed out (the next replay overwrites it).


def _e5_canon(W, H, P):
    """the processor's padded position ids for a W x H patch grid (Gemma4ImageProcessor._preprocess)"""
    import torch
    g = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    real = torch.stack(g, dim=-1).reshape(W * H, 2)
    return torch.nn.functional.pad(real, (0, 0, 0, P - W * H), mode="constant", value=-1)


def _e5_grid_of(c):
    """grid key (W, H, P) if the CPU position ids c [P, 2] are exactly a configured canonical grid, else None"""
    import torch
    if (not isinstance(c, torch.Tensor) or c.device.type != "cpu" or c.dtype != torch.int64 or c.ndim != 2
            or c.shape[1] != 2):
        return None
    P = int(c.shape[0])
    for W, H in E5_GRIDS:
        if W * H >= P:
            continue            # no padding: the stock mask builder would skip the mask (another SDPA variant)
        key = (W, H, P)
        ref = _E5["canon"].get(key)
        if ref is None:
            ref = _E5["canon"][key] = _e5_canon(W, H, P)
        if torch.equal(c, ref):
            return key
    return None


def _e5_cfg_ok(self):
    vt = self.vision_tower
    c = vt.config
    return (not getattr(self, "_enable_mm_lora", False) and getattr(c, "patch_size", 0) == 16
            and getattr(c, "pooling_kernel_size", 0) == 3 and getattr(c, "_attn_implementation", "") == "sdpa"
            and hasattr(vt, "patch_embedder") and hasattr(vt, "encoder") and hasattr(vt, "pooler")
            and hasattr(self, "embed_vision") and not getattr(c, "use_clipped_linears", False))


def _e5_grid_state(self, key, dev, pv_dtype, feat):
    """per-grid constants, all computed with the stock code, plus the per-batch-size graphs (captured here)"""
    import time
    import torch
    import torch.nn.functional as F
    import transformers.models.gemma4.modeling_gemma4 as m
    W, H, P = key
    vt = self.vision_tower
    vcfg = vt.config
    hid = vcfg.hidden_size
    md = self.model_dtype
    k = vcfg.pooling_kernel_size
    k2 = k * k
    length = P // k2                                  # stock: output_length = chunk_hidden.shape[0] // pooling_k2
    if k2 * length != P or int((P // length) ** 0.5) != k:
        return None
    t0 = time.time()
    try:
        from vllm.model_executor.custom_op import CustomOp
        cops = sorted({type(mod).__name__ for mod in vt.modules() if isinstance(mod, CustomOp)})
    except Exception as exc:  # noqa: BLE001
        cops = [f"check failed: {exc!r}"]
    if cops:
        print(f"[vfuse-e5] vLLM CustomOps inside the vision tower ({cops}): their binding depends on the engine mode, "
              f"so E5 stays off for grid {W}x{H}", file=sys.stderr, flush=True)
        return None
    _e3_prepare(vt.encoder)
    S = {"key": key, "P": P}
    canon = _E5["canon"][key]
    ppB = canon.unsqueeze(0).expand(E5_BMAX, P, 2).contiguous().to(dev)        # (BMAX, P, 2) int64
    padB = (ppB == -1).all(dim=-1)                                              # stock pad_tensor
    S["pp"] = ppB
    # patch position embedding, stock function, B=1 (the rows are per patch; the add broadcasts over the batch)
    S["pe"] = vt.patch_embedder._position_embeddings(ppB[:1], padB[:1])        # (1, P, hid)
    # bidirectional mask (stock builder; it probes padding_mask.all() once, here at set-up)
    dummy = torch.zeros((E5_BMAX, P, hid), dtype=md, device=dev)
    mask = m.create_bidirectional_mask(config=vt.encoder.config, inputs_embeds=dummy, attention_mask=~padB)
    if not isinstance(mask, torch.Tensor):
        return None
    S["mask"] = torch.empty(tuple(mask.shape), dtype=mask.dtype, device=dev)
    S["mask"].copy_(mask)                                                       # contiguous, as v5's graph buffers
    del mask
    cos, sin = vt.encoder.rotary_emb(dummy, ppB)
    S["cos"], S["sin"] = cos.contiguous(), sin.contiguous()
    del dummy
    # pooling weights and the valid pooled rows (stock _avg_pool_by_positions lines, one image)
    clamped = ppB[:1].clamp(min=0)
    max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
    kidx = torch.div(clamped, k, rounding_mode="floor")
    kidx = kidx[..., 0] + (max_x // k) * kidx[..., 1]
    wts = F.one_hot(kidx.long(), length).float() / k2                           # (1, P, length) float32
    S["wt"] = wts.transpose(1, 2)                                               # the stock matmul's left operand
    S["pad1"] = padB[:1]
    # host-side: which pooled rows are valid (= the stock boolean index), from the CPU position ids
    cl = canon.unsqueeze(0).clamp(min=0)
    mx = cl[..., 0].max(dim=-1, keepdim=True)[0] + 1
    ki = torch.div(cl, k, rounding_mode="floor")
    ki = ki[..., 0] + (mx // k) * ki[..., 1]
    validc = torch.zeros(length, dtype=torch.bool)
    validc[ki[0]] = True
    nv = int(validc.sum())
    S["nv"] = nv
    S["prefix"] = bool(validc[:nv].all())
    S["vidx"] = None if S["prefix"] else torch.nonzero(validc).flatten().to(dev)
    # device check of the same mask (set-up only): the stock valid_mask must equal the host one
    validd = torch.logical_not((wts == 0).all(dim=1))[0].cpu()
    if not torch.equal(validd, validc):
        print(f"[vfuse-e5] pooled-row mask mismatch for grid {key}; E5 off for it", file=sys.stderr, flush=True)
        return None
    S["std"] = bool(getattr(vcfg, "standardize", False))
    # VIS_ATTN (V2's exact attention kernel): the 2-D key mask the vattn context carries (padded mode), and the real
    # row count n for VIS_UNPAD (the canonical grid's padding is a suffix of n = W * H rows in every picture; the
    # rotary tables are the padded-shape ones, sliced, as vfuse's enc_forward_attn does)
    S["kv"] = (~padB).contiguous()
    S["n"] = W * H
    S["cos_n"], S["sin_n"] = S["cos"][:, : W * H].contiguous(), S["sin"][:, : W * H].contiguous()
    S["pv"] = torch.zeros((E5_BMAX, P, feat), dtype=pv_dtype, device=dev)
    S["g"] = {}
    S["t_setup"] = time.time() - t0
    return S


def _e5_mode():
    """the attention path vfuse's encoder wrapper would take for a padded board grid right now"""
    if _ATTN_ON[0] and "m" in _VA and _ACTIVE[0]:
        return "unpad" if _UNPAD_ON[0] else "pad"
    return "sdpa"


def _e5_graphs(self, S, mode, sig):
    """capture the whole-encoder graphs B=1..BMAX of one grid for one attention mode (first use of that mode)"""
    import time
    import torch
    t0 = time.time()
    if _E5["pool"] is None:
        _E5["pool"] = torch.cuda.graph_pool_handle()
    used = None
    for b in range(1, E5_BMAX + 1):
        S["g"][(b, mode, sig)] = st = _e5_capture(self, S, b, mode)
        E5_STATS["graphs"] += 1
        used = st["used"] if used is None else used
    E5_STATS["capture_s"] += time.time() - t0
    W, H, P = S["key"]
    msg = f"; vattn layers per graph {used}" if mode != "sdpa" else ""
    print(f"[vfuse-e5] grid {W}x{H} (P={P}, {S['nv']} soft tokens{'' if S['prefix'] else ', scattered'}), attention "
          f"{mode}{' (' + str(S['n']) + ' real rows)' if mode == 'unpad' else ''}, E3/E4 {sig}: whole-encoder graphs "
          f"B=1..{E5_BMAX} "
          f"captured in {time.time() - t0:.1f} s (set-up {S['t_setup']:.1f} s, pid {os.getpid()}){msg}",
          file=sys.stderr, flush=True)


def _e5_layers(self, S, b, h, mode):
    """the 27-layer loop for b pictures of grid S, as vfuse's encoder wrapper runs it in this mode: stock SDPA
    ("sdpa"), V2's vattn kernel on the padded rows with the stock 4-D mask registered in a vattn context ("pad"), or
    V2's A2 on the real rows only, no mask, padded output rows = zeros ("unpad"). Returns (hidden, vattn context)."""
    import torch
    enc = self.vision_tower.encoder
    layers = enc.layers[: enc.config.num_hidden_layers]
    def loop(x, msk, c, s_, p):
        if _E4ON[0] and _ACTIVE[0]:
            y = _e4_loop(enc, x, msk, c, s_, p)
            if y is not None:
                return y
        for layer in layers:
            x = layer(x, attention_mask=msk, position_embeddings=(c, s_), position_ids=p)
        return x

    if mode == "unpad":
        va = _VA["m"]
        n = S["n"]
        hn = h[:, :n].contiguous()
        cos, sin, pos = S["cos_n"][:b], S["sin_n"][:b], S["pp"][:b, :n]
        ctx = va.Ctx(None, None, True)
        with va.active(ctx):
            hn = loop(hn, None, cos, sin, pos)
        out = torch.zeros_like(h)
        out[:, :n] = hn
        return out, ctx
    cos, sin, pos = S["cos"][:b], S["sin"][:b], S["pp"][:b]
    mask = S["mask"][:b]                 # one view object: the vattn context matches it by identity
    if mode == "pad":
        va = _VA["m"]
        ctx = va.Ctx(mask, S["kv"][:b], False)
        with va.active(ctx):
            h = loop(h, mask, cos, sin, pos)
        return h, ctx
    return loop(h, mask, cos, sin, pos), None


def _e5_body(self, S, b, pv, mode):
    """patch embedding -> encoder -> pooling -> standardize for b pictures of one grid (stock ops). embed_vision is NOT
    in the graph: its RMSNorm is a vLLM CustomOp whose binding depends on the engine mode (a compiled engine binds
    forward_native, P-mode / eager forward_cuda), so it runs eagerly after the replay with whatever binding is live."""
    vt = self.vision_tower
    pe_mod = vt.patch_embedder
    # Gemma4VisionPatchEmbedder.forward (+ the caller's .to(model_dtype))
    x = 2 * (pv - 0.5)
    if (td := pe_mod.input_proj.weight.dtype).is_floating_point:
        x = x.to(td)
    h = pe_mod.input_proj(x)
    h = (h + S["pe"]).to(self.model_dtype)
    # Gemma4VisionEncoder.forward (mask and rotary tables precomputed for this grid)
    h, ctx = _e5_layers(self, S, b, h, mode)
    # Gemma4VisionPooler.forward + _avg_pool_by_positions, batched over the b pictures (per-picture stock calls)
    hs = h.masked_fill(S["pad1"].unsqueeze(-1), 0.0)
    out = S["wt"].expand(b, *S["wt"].shape[1:]) @ hs.float()
    out = out.to(hs.dtype)
    out = out.float() * vt.pooler.root_hidden_size
    # pooled_states[valid_mask] per picture, then torch.cat over the pictures
    if S["prefix"]:
        valid = out[:, : S["nv"]]
    else:
        valid = out.index_select(1, S["vidx"])
    if S["std"]:
        valid = (valid - vt.std_bias) * vt.std_scale
    # torch.cat over the pictures + .to(model_dtype); embed_vision runs after the replay, eagerly (see _e5_run)
    return valid.reshape(b * S["nv"], valid.shape[-1]).to(self.model_dtype), ctx


def _e5_capture(self, S, b, mode):
    import torch
    st = {"pv": S["pv"][:b]}
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            _e5_body(self, S, b, st["pv"], mode)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=_E5["pool"]):
        st["out"], ctx = _e5_body(self, S, b, st["pv"], mode)
    st["g"] = g
    st["used"] = 0 if ctx is None else ctx.used
    return st


def _e5_run(self, image_input, cpu):
    """E5 path for one _process_image_input call; None = not eligible (the caller runs the stock path)"""
    import torch
    if torch.cuda.is_current_stream_capturing():
        return None
    if not _e5_cfg_ok(self):
        if not _E5.get("cfg_logged"):
            _E5["cfg_logged"] = True
            c = self.vision_tower.config
            print(f"[vfuse-e5] config not the expected one (attn {getattr(c, '_attn_implementation', None)!r}, patch "
                  f"{getattr(c, 'patch_size', None)}, pool {getattr(c, 'pooling_kernel_size', None)}, lora "
                  f"{getattr(self, '_enable_mm_lora', None)}): stock path", file=sys.stderr, flush=True)
        return None
    pv = image_input["pixel_values"]
    pp = image_input["pixel_position_ids"]
    if isinstance(pv, torch.Tensor):
        n = pv.shape[0] if pv.ndim == 3 else -1
        pvl = None
    elif isinstance(pv, (list, tuple)) and pv and all(isinstance(t, torch.Tensor) and t.ndim == 2 for t in pv):
        n = len(pv)
        pvl = pv
    else:
        return None
    if n <= 0 or len(cpu) != n:
        return None
    key = None
    for c in cpu:
        kk = _e5_grid_of(c)
        if kk is None or (key is not None and kk != key):
            return None
        key = kk
    P = key[2]
    first = pv[0] if pvl is not None else pv[0]
    if (not first.is_cuda or first.dtype not in (torch.float32, torch.bfloat16) or first.ndim != 2 or first.shape[0] != P
            or (pvl is not None and any(t.shape != first.shape or t.dtype != first.dtype for t in pvl))):
        return None
    if isinstance(pp, torch.Tensor):
        if tuple(pp.shape) != (n, P, 2):
            return None
    elif not (isinstance(pp, (list, tuple)) and len(pp) == n and all(tuple(t.shape) == (P, 2) for t in pp)):
        return None
    dev = first.device
    S = _E5["grids"].get(key)
    if S is None:
        if key in _E5["grids"]:
            return None                       # set-up failed before: stock path for this grid
        _E5["grids"][key] = None
        S = _e5_grid_state(self, key, dev, first.dtype, int(first.shape[1]))
        if S is None:
            return None
        _E5["grids"][key] = S
    if int(first.shape[1]) != S["pv"].shape[-1] or first.dtype != S["pv"].dtype:
        return None             # (vLLM hands the tower bf16 pixel values: the model dtype; the stock ops run on them)
    # the stock chunk size (gemma4_mm _encoder_chunk; with total10 it does not depend on free memory)
    vcfg = self.config.vision_config
    if os.environ.get("REAX_VISION_ENCODER_BUDGET") == "total10":
        mb = _E5["chunk"].get(P)
        if mb is None:
            free, total = torch.accelerator.get_memory_info()
            mb = _E5["chunk"][P] = self._encoder_chunk(P, free, total, vcfg.position_embedding_size)
    else:
        free, total = torch.accelerator.get_memory_info()
        mb = self._encoder_chunk(P, free, total, vcfg.position_embedding_size)
    mb = max(1, min(n, mb))
    nv = S["nv"]
    mode = _e5_mode()
    sig = _lever_sig()
    if (1, mode, sig) not in S["g"]:
        ok = _E5.get("capture_ok")         # optional engine predicate (e.g. /opt/blink: "inside the P-mode context")
        if ok is not None and not ok():
            E5_STATS["capture_refused"] = E5_STATS.get("capture_refused", 0) + 1
            return None
        _e5_graphs(self, S, mode, sig)
    flats = []
    reps = 0
    for c0 in range(0, n, mb):
        c1 = min(n, c0 + mb)
        for s0 in range(c0, c1, E5_BMAX):
            s1 = min(c1, s0 + E5_BMAX)
            b = s1 - s0
            st = S["g"][(b, mode, sig)]
            if pvl is None:
                st["pv"].copy_(pv[s0:s1])
            else:
                for i in range(b):
                    st["pv"][i].copy_(pvl[s0 + i])
            st["g"].replay()
            flats.append(st["out"].clone())
            E5_STATS["replays"] += 1
            reps += 1
    # stock: one embed_vision call (pre-projection RMSNorm + projection) over every picture of the call, then views
    flat = flats[0] if len(flats) == 1 else torch.cat(flats, dim=0)
    proj = self.embed_vision(inputs_embeds=flat.unsqueeze(0)).squeeze(0)
    outs = [proj[i * nv:(i + 1) * nv] for i in range(n)]
    E5_STATS["calls"] += 1
    E5_STATS["images"] += n
    E5_STATS[mode] = E5_STATS.get(mode, 0) + 1
    vs = getattr(sys.modules.get("transformers.models.gemma4.modeling_gemma4"), "_vfuse_stats", None)
    if vs is not None and mode != "sdpa":      # V2's census: one attn_pad / attn_unpad per encoder call
        k = "attn_unpad" if mode == "unpad" else "attn_pad"
        vs[k] = vs.get(k, 0) + 1
        vs["replay_attn"] = vs.get("replay_attn", 0) + reps
    return outs


def _e5_install_model(mod):
    """vllm.model_executor.models.gemma4_mm: wrap Gemma4ForConditionalGeneration._process_image_input"""
    cls = getattr(mod, "Gemma4ForConditionalGeneration", None)
    if cls is None or getattr(cls, "_vfuse_e5", False):
        return
    orig = cls._process_image_input

    def _process_image_input(self, image_input):
        cpu = _E5["cpu"]
        if _E5ON[0]:
            if cpu is None:
                E5_STATS["nocpu"] += 1
            else:
                r = _e5_run(self, image_input, cpu)
                if r is not None:
                    return r
            E5_STATS["fallback"] += 1
        return orig(self, image_input)

    cls._process_image_input = _process_image_input
    cls._vfuse_e5 = True
    mod._vfuse_e5_stats = E5_STATS
    print(f"[vfuse-e5] whole-encoder graphs installed (grids {E5_GRIDS}, B<={E5_BMAX}, pid {os.getpid()})",
          file=sys.stderr, flush=True)


def _e5_install_runner(mod):
    """the encoder runners: hand the CPU-side position ids of each image batch to _process_image_input"""
    orig = getattr(mod, "group_and_batch_mm_kwargs", None)
    if orig is None or getattr(orig, "_vfuse_e5", False):
        return

    def group_and_batch_mm_kwargs(mm_kwargs, *a, **kw):
        lst = mm_kwargs if isinstance(mm_kwargs, list) else None
        cur = 0
        for modality, num_items, batch in orig(mm_kwargs, *a, **kw):
            cpu = None
            if lst is not None and modality == "image":
                try:
                    cpu = [it["pixel_position_ids"].data for _, it in lst[cur:cur + num_items]]
                except Exception:  # noqa: BLE001
                    cpu = None
            cur += num_items
            _E5["cpu"] = cpu
            try:
                yield modality, num_items, batch
            finally:
                _E5["cpu"] = None

    group_and_batch_mm_kwargs._vfuse_e5 = True
    mod.group_and_batch_mm_kwargs = group_and_batch_mm_kwargs


_HOOKS = {}
if ON:
    _HOOKS["transformers.models.gemma4.modeling_gemma4"] = _install
if E5:
    _HOOKS["vllm.model_executor.models.gemma4_mm"] = _e5_install_model
    _HOOKS["vllm.v1.worker.gpu.mm.encoder_runner"] = _e5_install_runner
    _HOOKS["vllm.v1.worker.gpu_model_runner"] = _e5_install_runner


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        hook = _HOOKS.get(fullname)
        if hook is None:
            return None
        spec = None
        for f in sys.meta_path:
            if f is self:
                continue
            try:
                spec = f.find_spec(fullname, path, target)
            except Exception:  # noqa: BLE001
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig, _hook=hook):
            _orig(module)
            _hook(module)

        spec.loader.exec_module = exec_module
        return spec


if _HOOKS:
    _todo = [n for n in _HOOKS if n not in sys.modules]
    for _n in [n for n in _HOOKS if n in sys.modules]:
        _HOOKS[_n](sys.modules[_n])
    if _todo:
        sys.meta_path.insert(0, _Finder())
