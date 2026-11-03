# SPDX-License-Identifier: Apache-2.0
"""E64 B7 (BLINK_LAST=1): the last decoder layer computes only the rows that reach the logits.

Every Blink request is one prefill that samples one token (max_tokens=1, no prompt logprobs, no speculative decoding),
and vLLM's model runner reads the model output only at each request's last scheduled row
(logits_indices = query_start_loc[1:] - 1). In the last decoder layer every other row feeds only the final norm, so
this module restricts the expensive opaque custom ops of that layer to the rows that matter:
  - qkv_proj (blink::bi_mm): the Q columns on the selected rows only (zeros elsewhere); the K/V columns on every row
    (the KV cache and the prefix cache need them);
  - o_proj, the dense MLP's gate_up_proj and down_proj (blink::bi_mm): the selected rows only (zeros elsewhere);
  - the routed MoE (blink_par's fork): the selected rows only (zeros elsewhere).
The selected rows are read on the device from the runner's persistent query_start_loc buffer (max_num_reqs + 1
entries; vLLM pads it with num_tokens, so a padded request slot repeats the last real row), clamped to the batch, so
the same captured CUDA graph serves every batch.

Exactness: each output row of every kernel involved depends only on its own input row (the dense GEMMs keep the same
k-ordered MMA chain for any M or tile config, the NVFP4 MoE routes and computes each row on its own, attention reads
the other rows only through K/V), so the selected rows are bit-identical to the full computation. The other rows hold
zeros or finite values derived from zeros and never reach the logits. Repeated slots compute bit-identical rows, so
the scatter is deterministic. The traced graph is unchanged (the ops are opaque), so Inductor's kernels and the
autotune pins are unchanged.

When the K and V slices of the layer's qkv weight are bitwise equal (Gemma-4 full-attention layers with
attention_k_eq_v load the K weights into both slots), the K columns are computed once and copied into the V slot:
equal weights and the same input rows give bit-identical columns.

Installed through blink_bi.py; patches GPUModelRunner.load_model (V2 model runner) to find the last layer.
"""
import importlib.abc
import os
import sys

ON = os.environ.get("BLINK_LAST", "0") == "1"
MIN_M = int(os.environ.get("BLINK_LAST_MIN_M", "96"))     # smaller batches run the stock path
_TARGET = "vllm.v1.worker.gpu.model_runner"
_S = {"qsl": None, "w": {}, "moe": None, "cur": None, "moe_cur": None, "why": "not registered", "n": {}}


def _log(msg):
    print(f"[blink-last] {msg}", file=sys.stderr, flush=True)


def _register(runner):
    import torch
    _S.update(qsl=None, w={}, moe=None, cur=None, moe_cur=None)
    if getattr(runner, "speculative_config", None) is not None:
        _S["why"] = "speculative decoding configured"
        return
    buf = getattr(getattr(runner, "input_buffers", None), "query_start_loc", None)
    if buf is None or buf.dtype != torch.int32 or buf.dim() != 1 or buf.numel() < 2:
        _S["why"] = "no persistent query_start_loc buffer"
        return
    layers = [mod for mod in runner.model.modules() if type(mod).__name__ == "Gemma4DecoderLayer"]
    if not layers:
        _S["why"] = "no Gemma4DecoderLayer"
        return
    last = layers[-1]
    attn, mlp = last.self_attn, getattr(last, "mlp", None)
    try:
        qkv = attn.qkv_proj.weight
        ws = {"o": attn.o_proj.weight, "gu": mlp.gate_up_proj.weight, "dn": mlp.down_proj.weight}
        q, kv = int(attn.q_size), int(attn.kv_size)
    except AttributeError as exc:
        _S["why"] = f"unexpected layer structure ({exc})"
        return
    for w in [qkv] + list(ws.values()):
        if w.dtype != torch.bfloat16 or w.dim() != 2:
            _S["why"] = f"weight {tuple(w.shape)} {w.dtype} is not a 2-D bf16 matrix"
            return
    if qkv.shape[0] != q + 2 * kv or getattr(last, "per_layer_input_gate", None) is not None:
        _S["why"] = "unexpected qkv layout or per-layer inputs"
        return
    keqv = bool(torch.equal(qkv[q:q + kv], qkv[q + kv:]))
    _S["w"] = {qkv.data_ptr(): ("qkv", q, kv, keqv)}
    for name, w in ws.items():
        _S["w"][w.data_ptr()] = ("rows", name)
    if getattr(last, "enable_moe_block", False):
        _S["moe"] = str(last.moe.experts.layer_name)
    _S["qsl"] = buf
    _S["why"] = "on"
    _log(f"on: last layer of {len(layers)}, {buf.numel() - 1} request slots, q {q} kv {kv} K=V {keqv}, "
         f"MoE {'restricted' if _S['moe'] is not None else 'absent'}, batches of >= {MIN_M} rows")


def _rows(m, device):
    """the selected rows of an m-row batch: last row of every request slot, clamped (device tensor), or None"""
    import torch
    qsl = _S["qsl"]
    if qsl is None or m < MIN_M or qsl.device != device:
        return None
    r = min(qsl.numel() - 1, m)
    return qsl[1:r + 1].to(torch.int64).sub_(1).clamp_(0, m - 1)


def mm(a, weight, launch):
    """blink::bi_mm for a registered last-layer weight, or None (caller runs the stock path).
    launch(x, w) runs the batch-invariant GEMM x @ w.T"""
    import torch
    kind = _S["w"].get(weight.data_ptr()) if _S["w"] else None
    if kind is None or a.dim() != 2:
        return None
    m = a.shape[0]
    if kind[0] == "qkv":            # first op of the last layer: select the rows for this forward
        idx = _rows(m, a.device)
        _S["cur"] = _S["moe_cur"] = (m, idx) if idx is not None else None
        if idx is None:
            return None
        _, q, kv, keqv = kind
        out = torch.empty((m, weight.shape[0]), device=a.device, dtype=a.dtype)
        if keqv:
            k = launch(a, weight[q:q + kv])
            out[:, q:q + kv].copy_(k)
            out[:, q + kv:].copy_(k)
        else:
            out[:, q:].copy_(launch(a, weight[q:]))
        oq = out[:, :q]
        oq.zero_()
        oq.index_copy_(0, idx, launch(a.index_select(0, idx), weight[:q]))
        _S["n"]["qkv"] = _S["n"].get("qkv", 0) + 1
        if _S["n"]["qkv"] == 1:
            _log(f"first restricted step: {m} rows -> {idx.numel()} selected (qkv Q, o_proj, MLP)")
        return out
    cur = _S["cur"]
    if kind[1] == "dn":         # the last restricted op of the layer
        _S["cur"] = None
    if cur is None or cur[0] != m:
        return None
    idx = cur[1]
    out = torch.zeros((m, weight.shape[0]), device=a.device, dtype=a.dtype)
    out.index_copy_(0, idx, launch(a.index_select(0, idx), weight))
    return out


def moe_rows(layer_name, m):
    """the selected rows for the last layer's MoE fork, or None. One-shot: set by the last layer's qkv op of the same
    forward and consumed by the next fork, which must resolve to the last layer's MoE (vLLM may pass the sentinel
    "from_forward_context" and resolve the layer by its call counter)"""
    pend, _S["moe_cur"] = _S["moe_cur"], None
    if pend is None or pend[0] != m or _S["moe"] is None:
        return None
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr
    name = mr._resolve_layer_name(layer_name)
    if name == "from_forward_context":
        from vllm.forward_context import get_forward_context
        fc = get_forward_context()
        name = fc.all_moe_layers[fc.moe_layer_index]
    if str(name) != _S["moe"]:
        return None
    _S["n"]["moe"] = _S["n"].get("moe", 0) + 1
    if _S["n"]["moe"] == 1:
        _log(f"first restricted MoE: {m} rows -> {pend[1].numel()} selected ({name})")
    return pend[1]


def _patch(mod):
    cls = mod.GPUModelRunner
    if getattr(cls.load_model, "_blink_last", False):     # already patched (idempotent)
        return
    orig = cls.load_model

    def load_model(self, *args, **kwargs):
        r = orig(self, *args, **kwargs)
        try:
            _register(self)
        except Exception as exc:  # noqa: BLE001
            _S.update(qsl=None, w={}, moe=None, cur=None, moe_cur=None, why=f"registration failed ({exc!r})")
        if _S["qsl"] is None:
            _log(f"off: {_S['why']}")
        return r

    load_model._blink_last = True
    cls.load_model = load_model


class _Finder(importlib.abc.MetaPathFinder):
    # E64: re-entrancy guard. Another import hook may target the same module and delegate back to this finder (each
    # asks every other finder for the spec); without the guard the two recurse into each other and wrap exec_module
    # hundreds of times (RecursionError at import). With it, each hook wraps the module's exec_module exactly once.
    _busy = set()

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET or fullname in _Finder._busy:
            return None
        spec = None
        _Finder._busy.add(fullname)
        try:
            for f in sys.meta_path:
                if f is self:
                    continue
                try:
                    spec = f.find_spec(fullname, path, target)
                except Exception:  # noqa: BLE001
                    spec = None
                if spec is not None:
                    break
        finally:
            _Finder._busy.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


if ON:
    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
    else:
        sys.meta_path.insert(0, _Finder())
