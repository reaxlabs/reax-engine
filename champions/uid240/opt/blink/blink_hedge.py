# SPDX-License-Identifier: Apache-2.0
"""RMS-flip hedge for the Blink engine (BLINK_HEDGE=1; off by default).

Why. The Blink reference compiles vLLM with Inductor and times two launch configs of ONE kernel at its start: the
2816-wide RMSNorm triton_red_fused_bi_mm_rms_norm_0 (v8 key cqltnak7..; reference graph cugpcm4v..), R0_BLOCK 4096 /
16 warps ("RMS") or R0_BLOCK 1024 / 8 warps ("F1"). The two sum x^2 in different orders, and the winner differs between
reference starts, so a pinned engine matches the reference in some sessions and moves ~15 % of the items in the others.

What makes an exact hedge cheap. That kernel runs once per forward: layer 0's input_layernorm (compiled piece 0:
rms_norm -> blink::bi_mm qkv). Every later input norm is a fused add+mul+rms kernel with a single config. Its input is
inputs_embeds = embed_tokens(id) * normalizer, so for text its output row is a pure function of the token id. At start
(when Inductor autotunes the kernel; blink_pin.py calls on_rms_autotune) this module runs BOTH of Inductor's own
compiled launchers of that kernel on every vocabulary row (exact by construction: the same cubins) and keeps:
  * the tokens whose two bf16 output rows differ in any bit (Blink: 1,410 of 262,144), and
  * the OTHER config's output row for each of them (TAB).
An item whose prompt holds none of these tokens is bit-identical under both configs (nothing else differs). For an item
that holds one, the frontend also runs an "alt" copy whose flagged tokens are replaced by shadow ids V + k:
  * the embedding hook maps shadow ids back to the real token for the lookup (same x, same residual stream) and writes
    k for that row into a static per-row buffer (ALT);
  * a Triton kernel launched right after the RMSNorm kernel (inside the same CUDA graphs) overwrites the rows with
    ALT >= 0 by TAB[k] = exactly what the other config writes for that token. Rows of other requests are untouched;
  * shadow ids also give the alt copy its own prefix-cache blocks from the first flagged token on, and share the blocks
    before it with the main copy (identical there by construction).
The answer is the average of the two answers in the validator's centred-log space (s1scoring.logit_distance is linear
there), so it lies at exactly half the two variants' distance from either reference; unflagged items stay exact.

Process roles. Engine core: blink_pin.py calls on_rms_autotune() at the kernel's first launch (the model is loaded):
it finds the Gemma4SelfDecoderLayers instance, builds the tables, warms the patch kernel, wraps the autotuner's run(),
replaces that instance's embed_input_ids (shadow ids) and writes TABLE_FILE. No import hook, no sitecustomize entry.
Frontends: Frontend (blink_engine.py) loads TABLE_FILE, flags prompts, builds alt ids, lets vLLM's input check accept
shadow ids, and averages. blink_core.py counts alt requests (ALT_PREFIX) outside the convoy group.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

ON = os.environ.get("BLINK_HEDGE", "0") == "1"
DEBUG = os.environ.get("BLINK_HEDGE_DEBUG", "0") == "1"      # development: both raw answers in the HTTP reply
TABLE_FILE = os.environ.get("BLINK_HEDGE_FILE", "/tmp/blink_hedge/table.npz")
ALT_PREFIX = "h66-"                 # request-id prefix of alt copies (blink_core.py keeps them out of the group size)
MAIN_PREFIX = "m66-"                # request-id prefix of a main request that has an alt copy (sent right after it)
KERNEL = "triton_red_fused_bi_mm_rms_norm_0"
D = 2816
CHUNK = 8192
MAX_ROWS = 16384                    # >= the largest CUDA graph / batch (8192 tokens)
LOG_FLOOR = 1e-6                    # reax_subnet s1scoring.LOG_FLOOR
_R = {}                             # engine-core state


def _log(msg):
    print(f"[blink-hedge] {msg}", file=sys.stderr, flush=True)


def is_alt(request_id) -> bool:
    return str(request_id).startswith(ALT_PREFIX)


# ------------------------------------------------------------------------------------------------ engine core
def _patch_kernel():
    if "kernel" in _R:
        return _R["kernel"]
    import triton
    import triton.language as tl

    @triton.jit
    def _blink_hedge_patch(out_ptr, alt_ptr, tab_ptr, D: tl.constexpr, BLOCK: tl.constexpr):
        """row r: if ALT[r] = k >= 0, out[r, :] = TAB[k, :] (bf16 bits copied unchanged)"""
        r = tl.program_id(0)
        k = tl.load(alt_ptr + r)
        if k >= 0:
            offs = tl.arange(0, BLOCK)
            m = offs < D
            v = tl.load(tab_ptr + k.to(tl.int64) * D + offs, mask=m)
            tl.store(out_ptr + r.to(tl.int64) * D + offs, v, mask=m)

    _R["kernel"] = _blink_hedge_patch
    return _blink_hedge_patch


def _cfg(launcher):
    c = launcher.config
    d = dict(getattr(c, "kwargs", {}) or {})
    d["num_warps"] = getattr(c, "num_warps", None)
    d["num_stages"] = getattr(c, "num_stages", None)
    return d


def on_rms_autotune(at, pick, args):
    """blink_pin.py, inside CachingAutotuner.benchmark_all_configs of the layer-0 RMSNorm kernel: `at.launchers` holds
    every candidate, `pick` is the pinned one, `args` the kernel's arguments of this first launch."""
    if not ON or getattr(at, "_blink_hedge", False):
        return
    at._blink_hedge = True
    try:
        _build(at, pick, args)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        _log(f"DISABLED: table build failed: {exc!r}")
        _R.pop("ready", None)
        _write_table(error=repr(exc))      # the frontends then run without the hedge at once


def _write_table(**kw):
    import numpy as np
    os.makedirs(os.path.dirname(TABLE_FILE), exist_ok=True)
    tmp = TABLE_FILE + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **kw)
    os.replace(tmp, TABLE_FILE)


def _build(at, pick, args):
    import numpy as np
    import torch
    import types
    t0 = time.time()
    dec = _find_decoder()
    _R["orig_embed"] = type(dec).embed_input_ids
    launchers = list(at.launchers)
    if len(launchers) != 2 or int(args[4]) != D:
        raise RuntimeError(f"unexpected kernel: {len(launchers)} candidates, r0_numel {args[4]}")
    main, alt = launchers[pick], launchers[1 - pick]
    # Inductor autotunes this kernel at COMPILE time on generated example values (the wrapper's "compile-time
    # auto-tuning block"), so args are not the model's tensors: x comes from the model's own embedding and w is
    # layer 0's input_layernorm weight (what the compiled graph passes as the kernel's in_ptr1)
    w = dec.decoder_layers[0].input_layernorm.weight
    if tuple(w.shape) != (D,) or w.dtype != args[1].dtype:
        raise RuntimeError(f"unexpected layer-0 input_layernorm weight {tuple(w.shape)} {w.dtype}")
    same = bool(torch.equal(w, args[1]))
    emb = dec.embed_tokens
    V = int(getattr(emb, "org_vocab_size", 0) or dec.config.vocab_size)
    dev = w.device
    stream = torch.cuda.current_stream(dev).cuda_stream
    embed = _R["orig_embed"]
    ids_f, rows = [], []
    with torch.no_grad():
        for s in range(0, V, CHUNK):
            ids = torch.arange(s, min(V, s + CHUNK), device=dev, dtype=torch.int32)
            x = embed(dec, ids)                     # the model's own embedding path (embed_tokens * normalizer)
            ya = torch.empty_like(x)
            yb = torch.empty_like(x)
            main(x, w, ya, x.shape[0], D, stream=stream)
            alt(x, w, yb, x.shape[0], D, stream=stream)
            if s == 0:                              # run-to-run determinism of both compiled configs
                y2 = torch.empty_like(x)
                main(x, w, y2, x.shape[0], D, stream=stream)
                y3 = torch.empty_like(x)
                alt(x, w, y3, x.shape[0], D, stream=stream)
                if not (torch.equal(ya.view(torch.int16), y2.view(torch.int16))
                        and torch.equal(yb.view(torch.int16), y3.view(torch.int16))):
                    raise RuntimeError("RMSNorm launchers are not deterministic")
            f = (ya.view(torch.int16) != yb.view(torch.int16)).any(dim=1).nonzero().squeeze(1)
            if f.numel():
                ids_f.append(f.to(torch.int64) + s)
                rows.append(yb.index_select(0, f))
    ids_f = torch.cat(ids_f) if ids_f else torch.zeros(0, dtype=torch.int64, device=dev)
    n = int(ids_f.numel())
    tab = torch.cat(rows).contiguous() if rows else torch.zeros((1, D), dtype=w.dtype, device=dev)
    base = torch.arange(V + n, dtype=torch.int32, device=dev)     # int32 like the runner's input_ids
    if n:
        base[V:] = ids_f.to(torch.int32)
    alt_buf = torch.full((MAX_ROWS,), -1, dtype=torch.int32, device=dev)
    kern = _patch_kernel()
    # warm-up (Triton compiles here, outside any CUDA-graph capture), and a self-check of the patch kernel
    if n:
        probe = torch.zeros((4, D), dtype=w.dtype, device=dev)
        alt_buf[:4] = torch.tensor([-1, 0, n - 1, -1], dtype=torch.int32, device=dev)
        kern[(4,)](probe, alt_buf, tab, D=D, BLOCK=4096)
        ok = (torch.equal(probe[1].view(torch.int16), tab[0].view(torch.int16))
              and torch.equal(probe[2].view(torch.int16), tab[n - 1].view(torch.int16))
              and not probe[0].view(torch.int16).any() and not probe[3].view(torch.int16).any())
        alt_buf.fill_(-1)
        if not ok:
            raise RuntimeError("patch kernel self-check failed")
    torch.cuda.synchronize(dev)
    _R.update(V=V, n=n, tab=tab, base=base, alt=alt_buf, hw=0, at=at,
              cfg_main=_cfg(main), cfg_alt=_cfg(alt))
    # every later launch of this kernel (CUDA-graph captures and eager steps): the pinned config, then the patch.
    # Patched on the class and recognised by kernel name: it covers whichever CachingAutotuner object the compiled
    # wrapper launches, and keeps the instances picklable (Inductor pickles the compiled graph for its FX cache)
    _patch_run_class()
    dec.embed_input_ids = types.MethodType(_embed_input_ids, dec)
    _R["ready"] = True
    ids_np = ids_f.cpu().numpy().astype(np.int64)
    _write_table(V=np.int64(V), ids=ids_np, cfg_main=json.dumps(_cfg(main)), cfg_alt=json.dumps(_cfg(alt)))
    _log(f"table: V={V}, {n} tokens whose layer-0 RMSNorm rows differ between {_cfg(main)} (pinned) and "
         f"{_cfg(alt)} (alt); {tab.numel() * 2 / 2**20:.1f} MiB; {time.time() - t0:.1f} s; autotune args were the "
         f"model's weight: {same}; -> {TABLE_FILE}")


def _patch_run_class():
    from torch._inductor.runtime import triton_heuristics as th
    cls = th.CachingAutotuner
    if getattr(cls, "_blink_hedge", False):
        return
    orig = cls.run
    kern = _patch_kernel()

    def run(self, *a, stream, **kw):
        r = orig(self, *a, stream=stream, **kw)
        if (_R.get("ready") and len(a) == 5 and a[4] == D
                and getattr(getattr(self, "fn", None), "__name__", "") == KERNEL):
            rows_n = int(a[3])
            if rows_n > 0:
                kern[(rows_n,)](a[2], _R["alt"], _R["tab"], D=D, BLOCK=4096)
                _R["launches"] = _R.get("launches", 0) + 1
        return r

    cls.run = run
    cls._blink_hedge = True


def _find_decoder():
    """the model's Gemma4SelfDecoderLayers instance (it owns embed_tokens and normalizer). Found once, at the first
    forward (the model is loaded by then), without any import hook on vllm's gemma4 module (blink_par.py has one)."""
    import gc
    mod = sys.modules.get("vllm.model_executor.models.gemma4")
    cls = getattr(mod, "Gemma4SelfDecoderLayers", None)
    if cls is None:
        raise RuntimeError("vllm.model_executor.models.gemma4.Gemma4SelfDecoderLayers is not loaded")
    found = [o for o in gc.get_objects() if issubclass(type(o), cls)]
    if len(found) != 1:
        raise RuntimeError(f"{len(found)} Gemma4SelfDecoderLayers instances")
    return found[0]


def _embed_input_ids(self, input_ids):
    """instance-level replacement of Gemma4SelfDecoderLayers.embed_input_ids (the model runner calls it eagerly before
    every forward): shadow ids V + k -> ALT[row] = k and the real token for the lookup; other ids -> ALT = id - V < 0"""
    import torch
    m = input_ids.shape[0]
    alt = _R["alt"]
    torch.sub(input_ids, _R["V"], out=alt[:m])
    hw = _R["hw"]
    if hw > m:
        alt[m:hw].fill_(-1)
    _R["hw"] = m
    return _R["orig_embed"](self, torch.index_select(_R["base"], 0, input_ids))


# ------------------------------------------------------------------------------------------------ frontend
def centred_logs(p):
    """reax_subnet s1scoring.centred_logs (validator), verbatim"""
    logs = [math.log(max(float(x), LOG_FLOOR)) for x in p]
    mean = math.fsum(logs) / len(logs)
    return [x - mean for x in logs]


def average(qtype, keys, ans_a, ans_b):
    """the answer whose validator-view centred logs are the mean of those of ans_a and ans_b (two readout() answers of
    the same question). Validator view (normalize_answer): noul -> (p, 1 - p) from 'noul'; else 'probabilities' in
    key order."""
    if qtype == "noul":
        pa = [float(ans_a["noul"]), 1.0 - float(ans_a["noul"])]
        pb = [float(ans_b["noul"]), 1.0 - float(ans_b["noul"])]
    else:
        pa = [float(ans_a["probabilities"][k]) for k in keys]
        pb = [float(ans_b["probabilities"][k]) for k in keys]
    c = [(x + y) / 2 for x, y in zip(centred_logs(pa), centred_logs(pb))]
    top = max(c)
    e = [math.exp(x - top) for x in c]
    s = math.fsum(e)
    p = [x / s for x in e]
    if qtype == "noul":
        pt = p[0]
        dist = {"false": 1.0 - pt, "true": pt}
        out = {"type": qtype, "probabilities": {k: dist[k] for k in keys}, "confidence": max(pt, 1.0 - pt),
               "noul": pt}
        return out
    dist = dict(zip(keys, p))
    out = {"type": qtype, "probabilities": dist, "confidence": max(p)}
    if qtype == "choice":
        out["choice"] = keys[max(range(len(p)), key=p.__getitem__)]
    else:
        out["score"] = sum(i * x for i, x in enumerate(p))
    return out


# BLINK_HEDGE_SELECT (frontend): which flagged items get the alt copy, from prompt-side facts only (so the alt still runs in
# the same step as its main). E66 curve on 6,144 RMS / F1 oracle items (distance of an unhedged flagged item to a
# mismatched reference, T 1.3): it grows with the number of options (codes); 2-option items stay <= 0.44.
#   all   every flagged item (17.3 % of items, +10.6 % prompt tokens); max unhedged 0
#   n3    >= 3 options (12.1 %, +7.6 %); max unhedged 0.44
#   n4c3  >= 4 options, or 3-option choice (10.2 %, +6.2 %); max unhedged 0.44
#   n4    >= 4 options (8.4 %, +5.1 %); max unhedged 0.79
SELECT = os.environ.get("BLINK_HEDGE_SELECT", "all")


def selected(sel, kind, n):
    if sel == "n3":
        return n >= 3
    if sel == "n4c3":
        return n >= 4 or (n == 3 and kind == "choice")
    if sel == "n4":
        return n >= 4
    return True


class Frontend:
    """per frontend process: the flip-token table (from the engine core) and the alt prompt of an item"""

    def __init__(self, path=TABLE_FILE, wait_s=120.0):
        import numpy as np
        t0 = time.time()
        while not os.path.exists(path):
            if time.time() - t0 > wait_s:
                raise RuntimeError(f"hedge table {path} not written by the engine core")
            time.sleep(0.2)
        z = np.load(path)
        if "error" in z.files:
            raise RuntimeError(f"engine core could not build the hedge table: {z['error']}")
        self.V = int(z["V"])
        ids = z["ids"].astype(np.int64)
        self.n = len(ids)
        self.shadow = np.full(self.V, -1, dtype=np.int64)
        self.shadow[ids] = self.V + np.arange(self.n, dtype=np.int64)
        self.base = ids
        self.cfg = (str(z["cfg_main"]), str(z["cfg_alt"]))
        self.np = np
        _patch_input_check(self)

    def alt_ids(self, ids):
        """prompt ids of the alt copy, or None when the prompt holds no flip token"""
        np = self.np
        a = np.asarray(ids, dtype=np.int64)
        s = self.shadow[a]
        m = s >= 0
        if not m.any():
            return None
        return np.where(m, s, a).tolist()


def _patch_input_check(fe):
    """vLLM's input check rejects ids >= the vocabulary: check an alt prompt with its shadow ids mapped back"""
    from vllm.v1.engine import input_processor as ip
    cls = ip.InputProcessor
    if getattr(cls, "_blink_hedge", False):
        return
    orig = cls._validate_model_input
    V, base = fe.V, fe.base

    def _validate_model_input(self, prompt_input, prompt_type, *a, **kw):
        try:
            ids = prompt_input.get("prompt_token_ids") if isinstance(prompt_input, dict) else None
        except Exception:  # noqa: BLE001
            ids = None
        if ids and max(ids) >= V:
            ids = [int(base[i - V]) if V <= i < V + len(base) else i for i in ids]
            prompt_input = dict(prompt_input)
            prompt_input["prompt_token_ids"] = ids
        return orig(self, prompt_input, prompt_type, *a, **kw)

    cls._validate_model_input = _validate_model_input
    cls._blink_hedge = True
