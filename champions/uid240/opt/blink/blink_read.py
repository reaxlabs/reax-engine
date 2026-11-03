# SPDX-License-Identifier: Apache-2.0
"""Label-only readout through vLLM's sampler (BLINK_READ=1): the decision's option-letter logprobs without the
full-vocabulary sampler passes.

Every Blink request asks for one greedy token restricted to the first n option letters (allowed_token_ids) with n
logprobs. vLLM's V2 sampler (vllm/v1/worker/gpu/sample) handles it over the whole 262,144-wide vocabulary: an fp32 copy
of the logits, the allowed-token mask (every other logit set to -inf), a greedy argmax, top-k and a log-softmax kernel
that streams the row in 1,024-wide blocks, and a rank count. All letters of the tokenizer lie in a few 1,024-aligned
vocabulary blocks (Blink's 26 letters: block 231). This module runs the same steps on those blocks only:

  logits (vLLM's own LM head + soft-cap, unchanged)  ->  letter columns in fp32  ->  [S, 1024 x B] blocks: letters at
  their in-block offsets, -inf elsewhere and for letters >= the request's n  ->  argmax, top-k, vLLM's own
  _topk_log_softmax_kernel (BLOCK_SIZE 1024) and _ranks_kernel on the blocks  ->  ids + block base.

Why the letter logprobs are bit-identical: the max is order-free; in the full-vocabulary kernel every block without an
allowed letter sums exp(-inf) = +0.0 exactly, so its running sum equals the sum over the live blocks, which this module
feeds to the same compiled kernel in the same block order with every letter at the same lane. The answers read only
the letter logprobs (blink_engine.py readout); token ids of -inf filler entries and tie order are irrelevant to them.

A batch takes this path only when every request in it qualifies (checked from its SamplingParams when it is added:
greedy, allowed_token_ids = the first n letters, logprobs = n, no penalties / bias / bad words / min_tokens / custom
logprob ids / thinking budget) and the runner uses the plain Sampler (no watermark, trace replay, sampling mask, NaN
count, batch sharding, grammar or spec decode); otherwise vLLM's sampler runs unchanged.
Development: BLINK_TUNE_FILE key "label_readout" switches it per batch (A/B in one engine start).
Installed by sitecustomize.py (BLINK_READ=1); patches vllm.v1.worker.gpu.sample.sampler and
vllm.v1.worker.gpu.model_runner when they are imported (in the engine core process).
"""
import importlib.abc
import json
import os
import sys
import time

ON = os.environ.get("BLINK_READ", "0") == "1"
# BLINK_READ_HEAD=1 (default): with blink_bi's label-only LM head (BLINK_LABEL_HEAD=1) the letter logits come from the
# same op on the same rows (blink::bi_mm(h, W[letters])) and the same bf16 soft-cap, without the zero-filled
# [S, 262144] tensor that the stock compute_logits builds around them
HEAD = os.environ.get("BLINK_READ_HEAD", "1") == "1" and os.environ.get("BLINK_LABEL_HEAD", "0") == "1"
TUNE_FILE = os.environ.get("BLINK_TUNE_FILE", "")
MODEL_DIR = os.environ.get("REAX_MODEL_DIR", "/model")
BLOCK = 1024                      # _topk_log_softmax_kernel's BLOCK_SIZE in vllm/v1/worker/gpu/sample/logprob.py
_S = {"on": ON, "t": 0.0, "letters": None, "dev": {}, "n_fast": 0, "n_stock": 0}


def _on():
    if TUNE_FILE:
        now = time.monotonic()
        if now - _S["t"] > 0.5:
            _S["t"] = now
            try:
                with open(TUNE_FILE) as fh:
                    _S["on"] = bool(json.load(fh).get("label_readout", _S["on"]))
            except Exception:  # noqa: BLE001
                pass
    return _S["on"]


def _letters():
    """the option-letter token ids, exactly as blink_engine.Encoder computes them"""
    if _S["letters"] is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        ids = []
        for i in range(26):
            t = tok.encode(chr(65 + i), add_special_tokens=False)
            if len(t) != 1 or tok.decode(t) != chr(65 + i):
                raise RuntimeError(f"letter {chr(65 + i)} is not one token")
            ids.append(t[0])
        _S["letters"] = ids
        blocks = sorted({i // BLOCK for i in ids})
        print(f"[blink-read] label readout: {len(ids)} letters in vocabulary block(s) {blocks}", file=sys.stderr,
              flush=True)
    return _S["letters"]


def _qualifies(sp, letters):
    """n (number of allowed letters) if this request's sampling reduces to the letter block, else 0"""
    try:
        ids = sp.allowed_token_ids
        if not ids:
            return 0
        n = len(ids)
        if n > len(letters) or sorted(ids) != sorted(letters[:n]) or len(set(ids)) != n:
            return 0
        if sp.temperature >= 1e-5 or sp.logprobs != n or sp.prompt_logprobs:
            return 0
        if sp.top_p != 1.0 or sp.top_k not in (0, -1) or sp.min_p != 0.0:
            return 0
        if (sp.logit_bias or sp.bad_words or sp.min_tokens or sp.repetition_penalty != 1.0 or sp.presence_penalty
                or sp.frequency_penalty or getattr(sp, "logprob_token_ids", None)
                or getattr(sp, "thinking_token_budget", None) is not None
                or getattr(sp, "trace_decode_token_ids", None) or getattr(sp, "structured_outputs", None)):
            return 0
        return n
    except Exception:  # noqa: BLE001
        return 0


def _patch_sampler(m):
    import numpy as np
    cls = m.Sampler
    if getattr(cls.add_request, "_blink_read", False):     # already patched (idempotent)
        return
    orig_add = cls.add_request

    def add_request(self, req_idx, prompt_len, sampling_params):
        orig_add(self, req_idx, prompt_len, sampling_params)
        n = getattr(self, "_e64_n", None)
        if n is None:
            n = self._e64_n = np.zeros(len(self.needs_logits_processing), dtype=np.int64)
        n[req_idx] = _qualifies(sampling_params, _letters()) if ON else 0

    add_request._blink_read = True
    cls.add_request = add_request
    _S["Sampler"] = cls


def _dev(device):
    """device constants: letter ids, live blocks, each letter's column in the concatenated live blocks"""
    d = _S["dev"].get(device)
    if d is None:
        import torch
        letters = _letters()
        blocks = sorted({i // BLOCK for i in letters})
        col = [blocks.index(i // BLOCK) * BLOCK + i % BLOCK for i in letters]
        base = torch.tensor([b * BLOCK for b in blocks], dtype=torch.int64, device=device)
        d = {"ids": torch.tensor(letters, dtype=torch.int64, device=device),
             "col": torch.tensor(col, dtype=torch.int64, device=device),
             "ar": torch.arange(len(letters), dtype=torch.int64, device=device),
             "width": BLOCK * len(blocks), "blocks": blocks, "base": base}
        _S["dev"][device] = d
    return d


def _head_model(model):
    """the module whose compute_logits is logits_processor(lm_head, h), if the narrow head reproduces the model's
    compute_logits on the letter columns (Gemma4ForConditionalGeneration: language_model + suppress_tokens), else None"""
    if not HEAD:
        return None
    c = _S.setdefault("head", {})
    if id(model) not in c:
        lm = model if hasattr(model, "logits_processor") else getattr(model, "language_model", None)
        why = ""
        if lm is None or not hasattr(lm, "logits_processor") or not hasattr(lm, "lm_head"):
            why = f"no logits_processor / lm_head on {type(model).__name__}"
        else:
            lp = lm.logits_processor
            supp = set(getattr(model, "_suppress_token_ids", None) or ()) | set(getattr(lm, "_suppress_token_ids", None) or ())
            qm = type(lm.lm_head.quant_method).__name__
            hd = getattr(lp, "head_dtype", None)
            if lp.logits_as_input:
                why = "logits_as_input"
            elif hd not in (None, getattr(lm.lm_head.weight, "dtype", None)):
                why = f"head_dtype {hd}"
            elif supp & set(_letters()):
                why = "suppressed letters"
            elif qm not in ("UnquantizedEmbeddingMethod", "UnquantizedLinearMethod"):
                why = f"lm_head quant method {qm}"
        c[id(model)] = None if why else lm
        print(f"[blink-read] narrow letter head: {'on (' + type(lm).__name__ + ')' if not why else 'off: ' + why}",
              file=sys.stderr, flush=True)
    return c[id(model)]


def _letter_weight(weight):
    """the LM head's letter rows [26, K], contiguous, as blink_bi's label head builds them (cached per weight)"""
    key = (weight.data_ptr(), tuple(weight.shape))
    w = _S.setdefault("w", {}).get(key)
    if w is None:
        w = weight.index_select(0, _dev(weight.device)["ids"]).contiguous()
        _S["w"][key] = w
    return w


def letter_scores(logits, n, letter_cols=False):
    """(token ids [S, 1 + max n], logprobs [S, 1 + max n], rank of the sampled token [S]) as vLLM's Sampler computes
    them (greedy, allowed = the first n[i] letters of row i, processed logprobs) from bf16 logits [S, V] (any V that
    holds the letters; letter_cols=True: [S, 26], column j = letter j), working on the live 1,024-blocks only. Column 0
    is the sampled (argmax) letter."""
    import torch
    from vllm.v1.worker.gpu.sample.logprob import _ranks_kernel, compute_token_logprobs
    dev = logits.device
    d = _dev(dev)
    S, W = logits.shape[0], d["width"]
    # the sampler's fp32 copy of the allowed logits; letters >= n and every other column -inf (allowed-token mask)
    vals = (logits if letter_cols else logits.index_select(1, d["ids"])).to(torch.float32)
    n_dev = torch.from_numpy(n).to(dev, non_blocking=True)
    vals = torch.where(d["ar"].unsqueeze(0) < n_dev.unsqueeze(1), vals, float("-inf"))
    blk = torch.full((S, W), float("-inf"), dtype=torch.float32, device=dev)
    blk.index_copy_(1, d["col"], vals)
    # greedy sample, top-k ids and their logprobs, rank of the sampled token: compute_topk_scores on the blocks
    sampled_c = blk.argmax(dim=-1)
    tok_c = torch.cat((sampled_c.unsqueeze(-1), torch.topk(blk, int(n.max()), dim=-1).indices), dim=1)
    scores = compute_token_logprobs(blk, tok_c)
    ranks = torch.empty(S, dtype=torch.int64, device=dev)
    _ranks_kernel[(S,)](ranks, blk, blk.stride(0), sampled_c, W, BLOCK_SIZE=8192)
    # column in the concatenated blocks -> vocabulary id
    tok = d["base"][tok_c // BLOCK] + tok_c % BLOCK
    return tok, scores, ranks


def _fast(runner, hidden_states, input_batch, grammar_output):
    """the stock (sampler_output, num_sampled, num_rejected) for this batch via the letter blocks, or None"""
    sampler = runner.sampler
    if (grammar_output is not None or runner.batch_sharder is not None or type(sampler) is not _S.get("Sampler")
            or input_batch.num_reqs == 0 or input_batch.num_draft_tokens != 0 or sampler.trace_replay_state is not None
            or sampler.return_sampling_mask or sampler.compute_nans or sampler.logprobs_mode != "processed_logprobs"):
        return None
    n_all = getattr(sampler, "_e64_n", None)
    idx = input_batch.idx_mapping_np
    if n_all is None or idx.shape[0] != input_batch.num_reqs or input_batch.logits_indices.shape[0] != idx.shape[0]:
        return None
    n = n_all[idx]
    if not n.all():
        return None
    import torch
    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    h = hidden_states[input_batch.logits_indices]
    lm = _head_model(runner.model)
    if lm is not None and h.dtype == torch.bfloat16:
        lp = lm.logits_processor
        # the label-only head's letter logits (same op, same rows) and vLLM's bf16 soft-cap / scale on [S, 26]
        lg = torch.ops.blink.bi_mm(h, _letter_weight(lm.lm_head.weight))
        if lp.soft_cap is not None:
            lg = lg / lp.soft_cap
            lg = torch.tanh(lg)
            lg = lg * lp.soft_cap
        if lp.scale != 1.0:
            lg *= lp.scale
        if not _S.get("head_ok"):
            # first use: the narrow head must equal the stock LM head + soft-cap on the letter columns, bit for bit
            ref = runner.model.compute_logits(h).index_select(1, _dev(h.device)["ids"])
            same = bool(torch.equal(ref.view(torch.int16), lg.view(torch.int16)))
            print(f"[blink-read] narrow letter head self-check ({h.shape[0]} rows): "
                  f"{'bit-identical, on' if same else 'DIFFERENT, off'}", file=sys.stderr, flush=True)
            if not same:
                _S["head"][id(runner.model)] = None
                tok, scores, ranks = letter_scores(ref, n, letter_cols=True)
                sampled = tok[:, 0].contiguous()
                lg = None
            _S["head_ok"] = True
        if lg is not None:
            tok, scores, ranks = letter_scores(lg, n, letter_cols=True)
    else:
        # vLLM's own LM head and soft-cap (label-only head: the letter columns are the real logits)
        logits = runner.model.compute_logits(h)
        tok, scores, ranks = letter_scores(logits, n)
    sampled = tok[:, 0].contiguous()
    num_sampled, num_rejected = get_num_sampled_and_rejected(
        input_batch.seq_lens.new_ones(input_batch.num_reqs), input_batch.seq_lens, input_batch.cu_num_logits,
        input_batch.idx_mapping, sampler.req_states.prefill_len.gpu)
    out = SamplerOutput(sampled_token_ids=sampled.view(-1, 1),
                        logprobs_tensors=LogprobsTensors(logprob_token_ids=tok, logprobs=scores,
                                                         selected_token_ranks=ranks, cu_num_generated_tokens=None),
                        num_nans=None, num_sampled=num_sampled, num_rejected=num_rejected)
    return out, out.num_sampled, out.num_rejected


def _patch_runner(m):
    cls = m.GPUModelRunner
    if getattr(cls.sample, "_blink_read", False):     # already patched (idempotent)
        return
    orig = cls.sample

    def sample(self, hidden_states, input_batch, grammar_output):
        if _on():
            r = _fast(self, hidden_states, input_batch, grammar_output)
            if r is not None:
                _S["n_fast"] += 1
                return r
        _S["n_stock"] += 1
        return orig(self, hidden_states, input_batch, grammar_output)

    sample._blink_read = True
    cls.sample = sample
    print(f"[blink-read] label readout installed (GPUModelRunner.sample; narrow head {HEAD})", file=sys.stderr,
          flush=True)


_TARGETS = {"vllm.v1.worker.gpu.sample.sampler": _patch_sampler, "vllm.v1.worker.gpu.model_runner": _patch_runner}


class _Finder(importlib.abc.MetaPathFinder):
    # E64: re-entrancy guard. Another import hook may target the same module and delegate back to this finder (each
    # asks every other finder for the spec); without the guard the two recurse into each other and wrap exec_module
    # hundreds of times (RecursionError at import). With it, each hook wraps the module's exec_module exactly once.
    _busy = set()

    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS or fullname in _Finder._busy:
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

        def exec_module(module, _orig=orig, _fn=_TARGETS[fullname]):
            _orig(module)
            _fn(module)

        spec.loader.exec_module = exec_module
        return spec


if ON:
    for _name, _fn in _TARGETS.items():
        if _name in sys.modules:
            _fn(sys.modules[_name])
    sys.meta_path.insert(0, _Finder())
