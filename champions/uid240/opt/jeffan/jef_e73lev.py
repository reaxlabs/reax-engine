# SPDX-License-Identifier: Apache-2.0
# E73 lever LLP for the Jeffan engine (on pjs's engine with team ruj's levers); off unless JEF_LLP=1.
"""Engine-core lever, called from jef_levers.py (no import hooks of its own).

JEF_LLP=1  last-layer row pruning. The answer of a read is the next-token distribution at its LAST prompt row; the last
           decoder layer (layer 59) matters only at those rows, except for its K/V, which later rows of the same
           prompt attend to (and which the prefix cache keeps). So in layer 59 the qkv GEMM computes its K/V columns for
           every row and its Q columns only for the read rows, and o_proj / gate_up / down compute only the read rows;
           the other rows are zero (rows never mix outside attention, whose K/V are complete). The read rows are each
           request's last scheduled row, read on the device from the V2 runner's persistent query_start_loc buffer
           (padded with num_tokens: a padding entry selects the last real row again), so the CUDA graphs replay it.
           The selected rows are gathered (e4m3 rows + per-token scales), multiplied by team ruj's jef::fp8_mm (same
           kernel family, same per-element arithmetic: fp32 accumulation over K in increasing order, the same epilogue;
           ruj: every tile configuration gives the same bits) and scattered back: the read rows get the same bits.
           Implemented as a Python kernel for _C::cutlass_scaled_mm (CUDA) in front of jef::fp8_mm; the layer is found
           by its weights' data_ptr (every full-attention layer has the same GEMM shapes), set at the first
           compute_logits call (the profile run, after the weights are final).
JEF_LLP_ATTN=1 (with JEF_LLP=1) the last layer's attention only for the read rows: each request's last scheduled row
           becomes a one-token query over the same paged K/V (same 32-key tiles, same online softmax, same [BLOCK_M, TILE]
           layout per row; its q-block's tile range is the same because the row is its request's last); padded
           request slots repeat the last real request (query row, sequence length, block-table row). The layer's K/V
           cache write is a separate op and stays complete; the other rows' outputs are zero.
"""
from __future__ import annotations

import os
import sys

LLP = os.environ.get("JEF_LLP", "0").strip() == "1"
LLPA = LLP and os.environ.get("JEF_LLP_ATTN", "0").strip() == "1"   # needs LLP (o_proj reads only the read rows)
_S = {"w": None, "lib": None}


def _log(msg):
    print(f"[jef-e73] {msg}", file=sys.stderr, flush=True)


def _qsl():
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if not is_forward_context_available():
        return None
    md = get_forward_context().attn_metadata
    if isinstance(md, list):
        md = md[0] if md else None
    if not md:
        return None
    for m in md.values():
        q = getattr(m, "query_start_loc", None)
        if q is not None:
            return q
    return None


def _mm(out, a, b, sa, sb, bias):
    import torch
    w = _S["w"]
    info = w.get(b.data_ptr()) if w else None
    if info is None or bias is not None:
        torch.ops.jef.fp8_mm(out, a, b, sa, sb)
        return
    qsl = _qsl()
    if qsl is None or qsl.shape[0] - 1 >= a.shape[0]:
        torch.ops.jef.fp8_mm(out, a, b, sa, sb)
        return
    idx = (qsl[1:] - 1).clamp_(min=0).to(torch.long)
    a_sel = a.index_select(0, idx)
    sa_sel = sa.index_select(0, idx)
    kind, qc = info
    if kind == "qkv":
        torch.ops.jef.fp8_mm(out[:, qc:], a, b[:, qc:], sa, sb)      # K / V columns: every row
        outq = out[:, :qc]
        outq.zero_()
        tmp = torch.empty((idx.shape[0], qc), dtype=out.dtype, device=out.device)
        torch.ops.jef.fp8_mm(tmp, a_sel, b[:, :qc], sa_sel, sb)      # Q columns: the read rows
        outq.index_copy_(0, idx, tmp)
    else:
        out.zero_()
        tmp = torch.empty((idx.shape[0], out.shape[1]), dtype=out.dtype, device=out.device)
        torch.ops.jef.fp8_mm(tmp, a_sel, b, sa_sel, sb)
        out.index_copy_(0, idx, tmp)


def install_gemm_override():
    """Called by jef_levers._install_gemm after team ruj's jef_fp8override.so: a Python CUDA kernel for
    _C::cutlass_scaled_mm in front of jef::fp8_mm (only with JEF_LLP=1)."""
    if not LLP or _S["lib"] is not None:
        return
    import warnings

    import torch
    lib = torch.library.Library("_C", "IMPL")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lib.impl("cutlass_scaled_mm", _mm, "CUDA")
    _S["lib"] = lib
    _log("JEF_LLP=1: _C::cutlass_scaled_mm -> last-layer row pruning in front of jef::fp8_mm")


def on_model(model):
    """First compute_logits call (profile run, weights final): the last decoder layer's four FP8 weights."""
    if _S["w"] is not None:
        return
    _S["w"] = {}
    if not LLP:
        return
    try:
        last = model.model.layers[-1]
        at, mlp = last.self_attn, last.mlp
        w = {at.qkv_proj.weight.data_ptr(): ("qkv", int(at.q_size)), at.o_proj.weight.data_ptr(): ("o", 0),
             mlp.gate_up_proj.weight.data_ptr(): ("gu", 0), mlp.down_proj.weight.data_ptr(): ("d", 0)}
        assert len(w) == 4 and at.q_size == at.num_heads * at.head_dim
        assert at.qkv_proj.weight.shape[1] == at.q_size + 2 * at.kv_size, at.qkv_proj.weight.shape
        _S["w"] = w
        if LLPA:
            _S["last_attn"] = last.self_attn.attn.layer_name
            _patch_llpa()
        _log(f"JEF_LLP: last layer {len(model.model.layers) - 1}: qkv {tuple(at.qkv_proj.weight.shape)} (Q columns "
             f"{at.q_size}), o {tuple(at.o_proj.weight.shape)}, gate_up {tuple(mlp.gate_up_proj.weight.shape)}, "
             f"down {tuple(mlp.down_proj.weight.shape)}")
    except Exception as exc:  # noqa: BLE001
        _S["w"] = {}
        _log(f"JEF_LLP: last layer not found ({exc!r}); pruning off")


def _patch_llpa():
    import dataclasses
    m = sys.modules.get("vllm.v1.attention.backends.triton_attn")
    if m is None:
        _log("JEF_LLP_ATTN: triton_attn backend not imported; off")
        return
    cls = m.TritonAttentionImpl
    ofwd = cls.forward
    if getattr(ofwd, "_e73_llpa", False):
        return

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
                output_block_scale=None):
        if (attn_metadata is None or output_scale is not None or output_block_scale is not None
                or getattr(layer, "layer_name", None) != _S.get("last_attn")):
            return ofwd(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                        output_block_scale)
        import torch
        qsl, sl = attn_metadata.query_start_loc, attn_metadata.seq_lens
        r = sl.shape[0]
        if qsl.shape[0] != r + 1 or r >= attn_metadata.num_actual_tokens:
            return ofwd(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                        output_block_scale)
        ar = torch.arange(r, device=qsl.device)
        valid = qsl[1:] > qsl[:-1]
        last = torch.where(valid, ar, torch.zeros_like(ar)).max()
        src = torch.where(valid, ar, last)
        idx = (qsl[1:] - 1).clamp(min=0).to(torch.long).index_select(0, src)
        md = dataclasses.replace(
            attn_metadata, num_actual_tokens=r, max_query_len=1,
            query_start_loc=torch.arange(r + 1, dtype=qsl.dtype, device=qsl.device), seq_lens=sl.index_select(0, src),
            block_table=attn_metadata.block_table.index_select(0, src))
        q_sel = query.index_select(0, idx)
        o_sel = torch.empty((r,) + tuple(output.shape[1:]), dtype=output.dtype, device=output.device)
        ofwd(self, layer, q_sel, key, value, kv_cache, md, o_sel, None, None)
        output.zero_()
        output.index_copy_(0, idx, o_sel)
        return output

    forward._e73_llpa = True
    cls.forward = forward
    _log(f"JEF_LLP_ATTN: attention of {_S.get('last_attn')} only for the read rows")
