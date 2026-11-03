# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 pjs team (vision engine)
"""Exact attention kernel for the Gemma4 VISION tower (VIS_ATTN=1; A2 unpadded rows with VIS_UNPAD=1).

HF Gemma4VisionAttention calls transformers' "sdpa" attention interface, i.e. torch SDPA with the encoder's 4-D bool
padding mask, which PyTorch runs as its memory-efficient kernel (bias variant, fmha_cutlassF_bf16_aligned_64x128_rf).
gemm/vattn.so (source gemm/vattn.cu, built by build_vattn.py) reproduces that kernel's arithmetic bit for bit; it was
byte-tested against SDPA on synthetic/adversarial inputs and on the real tower (every B = 1..12, padded and unpadded).

This module wraps ALL_ATTENTION_FUNCTIONS["sdpa"]. A call takes the kernel only when ALL of these hold:
  * an encoder context is active (set by vfuse's encoder wrapper around the 27-layer loop, see vfuse.py);
  * the module is a Gemma4VisionAttention, dropout 0, not causal, no extra kwargs, a scaling value is given;
  * the attention mask is the very tensor object the context registered (identity), i.e. the 4-D mask that HF's
    create_bidirectional_mask built from the 2-D padding mask the context carries as the key mask (keys only);
  * q, k and v have exactly the stock layout: bf16 [B, H, S, 72] views of contiguous [B, S, H, 72] tensors.
Anything else runs the stock SDPA path. In the unpadded mode (A2: the encoder runs on the real rows only, no mask) the
stock fallback is SDPA with an all-true bool mask, which keeps the same mem-efficient bias kernel (byte-tested equal),
because SDPA without a mask would pick another kernel.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys

ON = os.environ.get("VIS_ATTN", "0") == "1"
UNPAD = os.environ.get("VIS_UNPAD", "0") == "1"
SO_DIR = os.environ.get("VIS_ATTN_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemm"))
VARIANT = int(os.environ.get("VIS_ATTN_VARIANT", "0"))
HEAD_DIM = 72
_EXT = {}
_CTX = [None]
STATS = {"vattn": 0, "fallback": 0, "lite": 0}


def ext():
    """the prebuilt vattn extension (gemm/vattn.so)"""
    if "m" not in _EXT:
        so = sorted(glob.glob(os.path.join(SO_DIR, "vattn*.so")))
        if not so:
            raise RuntimeError(f"vattn extension not found in {SO_DIR}")
        spec = importlib.util.spec_from_file_location("vattn", so[0])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXT["m"] = mod
    return _EXT["m"]


class Ctx:
    """Encoder context: mask4d = the 4-D mask object the layers receive (None in the unpadded mode), kvalid = the
    [B, S] bool key mask it was built from (None = every key visible), unpad = rows are the real rows only."""
    __slots__ = ("mask4d", "kvalid", "unpad", "used")

    def __init__(self, mask4d, kvalid, unpad):
        self.mask4d = mask4d
        self.kvalid = kvalid
        self.unpad = unpad
        self.used = 0


class active:
    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        self.prev = _CTX[0]
        _CTX[0] = self.ctx
        return self.ctx

    def __exit__(self, *exc):
        _CTX[0] = self.prev
        return False


def _stock_layout(x, B, H, S):
    import torch
    return (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 4 and tuple(x.shape) == (B, H, S, HEAD_DIM)
            and x.stride() == (S * H * HEAD_DIM, HEAD_DIM, H * HEAD_DIM, 1) and x.data_ptr() % 16 == 0)


def install(m):
    """m = transformers.models.gemma4.modeling_gemma4 (after import)."""
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if getattr(m, "_vattn_installed", False):
        return
    if getattr(m, "ALL_ATTENTION_FUNCTIONS", ALL_ATTENTION_FUNCTIONS) is not ALL_ATTENTION_FUNCTIONS:
        print("[vattn] modeling_gemma4 uses another attention registry; not installed", file=sys.stderr, flush=True)
        return
    orig = ALL_ATTENTION_FUNCTIONS["sdpa"]
    cls = m.Gemma4VisionAttention
    ext()

    def sdpa_or_vattn(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None,
                      **kwargs):
        ctx = _CTX[0]
        if ctx is not None and type(module) is cls:
            if (attention_mask is ctx.mask4d and not kwargs and dropout == 0.0 and not is_causal
                    and scaling is not None and query.dim() == 4):
                B, H, S = query.shape[0], query.shape[1], query.shape[2]
                if (_stock_layout(query, B, H, S) and _stock_layout(key, B, H, S) and _stock_layout(value, B, H, S)
                        and (ctx.kvalid is None or (tuple(ctx.kvalid.shape) == (B, S) and ctx.kvalid.dtype == torch.bool
                                                    and ctx.kvalid.is_contiguous()))):
                    out = _EXT["m"].fwd(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), ctx.kvalid,
                                        float(scaling), VARIANT)
                    STATS["vattn"] += 1
                    ctx.used += 1
                    return out, None          # [B, S, H, 72] contiguous, as sdpa_attention_forward returns
            if ctx.unpad:
                # real rows only: keep the reference's mem-efficient bias kernel with an all-true key mask
                STATS["lite"] += 1
                full = torch.ones((query.shape[0], 1, query.shape[2], key.shape[2]), dtype=torch.bool,
                                  device=query.device)
                return orig(module, query, key, value, full, dropout=dropout, scaling=scaling, is_causal=is_causal,
                            **kwargs)
            STATS["fallback"] += 1
        return orig(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal,
                    **kwargs)

    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_or_vattn
    m._vattn_installed = True
    m._vattn_stats = STATS
    print(f"[vattn] exact vision attention kernel installed (pid {os.getpid()}, {SO_DIR}, variant {VARIANT}, "
          f"unpad {UNPAD})", file=sys.stderr, flush=True)


def suffix_len(attention_mask):
    """[B, S] bool padding mask -> n if every row is n True values then only False (the same n for all rows), else
    None. One small device-to-host copy (the caller runs where the stock mask builder syncs anyway)."""
    import torch
    m = attention_mask
    if m.dim() != 2 or m.dtype != torch.bool or m.shape[1] < 2:
        return None
    nv = m.sum(-1)
    rise = (m[:, 1:] & ~m[:, :-1]).any()          # a True after a False: not a suffix padding
    lo, hi, bad = torch.stack([nv.min(), nv.max(), rise.to(nv.dtype)]).tolist()
    if bad or lo != hi or lo < 1:
        return None
    return int(lo)
