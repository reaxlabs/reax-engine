# SPDX-License-Identifier: Apache-2.0
"""G1 (development, BLINK_PM_GRAPHS=1, default off): FULL CUDA graphs for picture steps (P-mode).

A second family of FULL graphs, captured INSIDE the P-mode context (the eager forward with forward_cuda CustomOps,
eager IR priorities and the P lever set), replays the language-model forward of a picture step instead of launching
it kernel by kernel (~1,200 launches, ~100 ms of host time per step). The vision encoder still runs before it, as in
any P step; only the LM forward is a replay.

The one thing a graph could freeze wrongly is the attention variant: vLLM's Triton attention picks the bidirectional
image variant (tl.constexpr USE_MM_PREFIX / MAX_MM_RANGES) from the presence and shape of the per-step image-range
tensor. Here every P graph step and every P capture gets that tensor from ONE persistent (max_num_reqs, 1, 2) int32
buffer (rows without a picture (0, 0), which the kernel skips), so the sliding-window layers are captured with
USE_MM_PREFIX=1, MAX_MM_RANGES=1 - the variant eager P steps use for one picture per request - and read the step's
ranges from the buffer at replay. Gemma4's full-attention layers clear the ranges on their own group's metadata before
the forward (gemma4_mm.py), so they capture the causal variant, as eager. A step qualifies only if every request
carries at most one picture and the step fits a captured size; anything else runs eager P.
Padding rows of a graph size cannot reach real rows: every kernel of the P forward is batch-invariant (row-wise
norms / rotary / activation / quantisation, the BI and K1 GEMMs, the MoE family, the 2-D attention kernel), and the
padding requests' ranges are (0, 0). The proof is the oracle: every answer bit-identical, as for M0.
"""
import os
import sys
import time

ON = os.environ.get("BLINK_PM_GRAPHS", "0") == "1"
_DEF = list(range(448, 3072, 64)) + list(range(3072, 4096 + 1, 128))
SIZES = sorted({int(x) for x in os.environ.get("BLINK_PM_GRAPH_SIZES", "").split(",") if x.strip()} or _DEF)
_G = {"mgr": None, "buf": None, "active": False, "capture_s": 0.0, "replays": 0, "eager": 0, "why": "off"}


def _log(msg):
    print(f"[blink-pmg] {msg}", file=sys.stderr, flush=True)


def patch_builder(mod):
    """TritonAttentionMetadataBuilder.build: while a P graph step or capture runs, the image ranges live in the
    persistent buffer (same address at capture and replay)"""
    cls = mod.TritonAttentionMetadataBuilder
    if getattr(cls.build, "_pmg", False):
        return
    orig = cls.build

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = orig(self, common_prefix_len, common_attn_metadata, fast_build)
        buf = _G["buf"]
        if _G["active"] and buf is not None:
            import torch
            n = common_attn_metadata.num_reqs
            rows = [[[0, 0]] for _ in range(n)]
            rng = common_attn_metadata.mm_req_doc_ranges or {}
            for i, rs in rng.items():
                if i < n and rs:
                    rows[i][0] = [int(rs[0][0]), int(rs[0][1])]
            src = torch.tensor(rows, dtype=torch.int32, pin_memory=True)
            buf[:n].copy_(src, non_blocking=True)
            md.mm_prefix_range = rng
            md.mm_prefix_range_tensor = buf[:n]
        return md

    build._pmg = True
    cls.build = build


def patch_cg_context(mod):
    """cudagraph_utils' own set_forward_context reference: skip the compiled model while P graphs are captured"""
    orig = mod.set_forward_context
    if getattr(orig, "_pmg", False):
        return
    pm = sys.modules.get("blink_pmode")

    def set_forward_context(*a, **k):
        if pm is not None and pm._PM.on:
            k["skip_compiled"] = True
        return orig(*a, **k)

    set_forward_context._pmg = True
    mod.set_forward_context = set_forward_context


def capture(runner, pctx):
    """after the T capture: the P graph family (inside the P-mode context)"""
    if not ON or _G["mgr"] is not None:
        return
    import copy
    import torch
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
    t0 = time.time()
    vc = copy.copy(runner.vllm_config)
    cc = copy.copy(vc.compilation_config)
    sizes = [s for s in SIZES if s <= runner.max_num_tokens]
    cc.cudagraph_capture_sizes = sizes
    cc.max_cudagraph_capture_size = max(sizes)
    object.__setattr__(vc, "compilation_config", cc)
    mgr = ModelCudaGraphManager(vc, runner.device, CUDAGraphMode.FULL, runner.decode_query_len)
    _G["buf"] = torch.zeros((runner.max_num_reqs, 1, 2), dtype=torch.int32, device=runner.device)
    _G["active"] = True
    try:
        with pctx():
            mgr.capture(runner.model, runner.model_state, runner.input_buffers, runner.intermediate_tensors,
                        runner.block_tables, runner.attn_groups, runner.kv_cache_config,
                        use_aux_hidden_state_outputs=runner.use_aux_hidden_state_outputs,
                        progress_bar_desc="Capturing P-mode CUDA graphs")
    finally:
        _G["active"] = False
    _G["mgr"] = mgr
    _G["capture_s"] = time.time() - t0
    _G["why"] = "on"
    _log(f"P graphs captured: {len(mgr.graphs)} sizes {sizes[0]}..{sizes[-1]} in {_G['capture_s']:.1f} s")


def qualifies(so):
    """a picture step that may replay a P graph: one picture at most per request, total tokens within the sizes"""
    if _G["mgr"] is None:
        return False
    if so.total_num_scheduled_tokens > SIZES[-1]:
        return False
    for r in so.scheduled_new_reqs:
        if len(r.mm_features or ()) > 1:
            return False
    return True
