"""Backend for the Qwen3.5-hybrid decision models served by the native PplxEngine: pplx-decider-v1.1-27b and d3.

Both checkpoints share the architecture (48 Gated-DeltaNet + 16 full-attention layers, hidden 5120, 255-way
last-token readout) and the engine; they differ in prompt/answer contract:

  family "pplx"  perplexity-ai/pplx-decider-v1.1-27b   prompt in pplx_engine.py (adapted from the author's code)
  family "d3"    vllm-sr/d3                            prompt/answer contract in d3_contract.py (vendored, upstream/d3)

Both have NONCAUSAL full attention: their decision_config.json says "attention_mode": "noncausal_full_attention"
(d3 as of the 15 Oct 2026 revision; PplxEngine asserts it at load).

The torch/transformers imports are lazy so the module (and the registry) import on a CPU-only machine.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

from .. import d3_contract
from .base import Encoded

_ENGINE_DIR = str(Path(__file__).resolve().parent.parent)
FAMILIES = ("pplx", "d3")


def _pplx_engine_module():
    """pplx_engine.py and its kernels import each other as top-level modules (legacy flat layout)."""
    if _ENGINE_DIR not in sys.path:
        sys.path.insert(0, _ENGINE_DIR)
    import pplx_engine
    return pplx_engine


class Qwen35Backend:
    kind = "native-gpu"
    serve_warmup = False  # serve.py leaves warming to the `warm_steps` option (runs in the constructor)

    def __init__(self, model_dir, *, family="pplx", name=None, linear="fp8", compile="1", gva="1", fp8_skip="", nvfp4="",
                 nvfp4_scales="", budget=16384, max_input_tokens=None, warm_steps=0, device="cuda"):
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        self.family = family
        self.name = name or {"pplx": "pplx-decider-v1.1-27b", "d3": "d3"}[family]
        # pplx: the author's 8192-token limit; d3: the author's server has none (long prompts were validated to 64k+).
        self.max_input_tokens = max_input_tokens if max_input_tokens is not None else (8192 if family == "pplx" else 262144)
        self.max_batch_tokens = budget
        self._pp = _pplx_engine_module()
        model_dir = str(model_dir)
        if family == "d3":
            from transformers import AutoTokenizer
            self._d3 = d3_contract
            cfg = json.loads((Path(model_dir) / "decision_config.json").read_text())
            self._tok, self._codes = AutoTokenizer.from_pretrained(model_dir), cfg["codes"]
            if cfg.get("readout_dtype", "float32") == "float32":
                os.environ.setdefault("PPLX_READOUT_FP32", "1")  # author runtime: hidden.float() @ readout.T in fp32
        else:
            self._prompter = self._pp.Prompter(model_dir)
        self.engine = self._pp.PplxEngine(model_dir, device=device, linear_mode=linear, compile=compile,
                                          fp8_skip=tuple(x for x in fp8_skip.split(",") if x), gva_native=gva,
                                          nvfp4_groups=tuple(g for g in nvfp4.split(",") if g), nvfp4_scales=nvfp4_scales or None)
        self._device = device
        self.steps = 0
        if warm_steps:
            self.warmup(warm_steps)

    # ---------------------------------------------------------------- Backend API
    def options(self, question):
        if self.family == "d3":
            return list(self._d3.normalize_question(question).keys)
        return list(self._pp.options(question)[0])

    def encode(self, state, question):
        if self.family == "d3":
            n = self._d3.normalize_question(question)  # ValueError with the reason
            ids = self._tok(self._d3.render(self._tok, state, n.rendered, self._codes), add_special_tokens=False)["input_ids"]
            n_opt = len(n.keys)
        else:
            validate_pplx_question(question)
            ids, n_opt = self._prompter.ids(state, question)
        if len(ids) > self.max_input_tokens:
            raise ValueError(f"maximum context length exceeded: {len(ids)} > {self.max_input_tokens} tokens")
        return Encoded(ids, len(ids), n_opt)

    def run(self, batch):
        import torch
        ids = torch.tensor([t for e in batch for t in e.payload], device=self._device)
        logits = self.engine.forward(ids, [len(e.payload) for e in batch])
        self.steps += 1
        return self.engine.probs(logits, [e.n_options for e in batch])

    # ---------------------------------------------------------------- family answer shapes (/v1/systemone)
    def answer(self, question, probs):
        if self.family == "d3":
            return self._d3.product_answer(self._d3.normalize_question(question), probs)
        a = self._pp.answer(question, probs)
        if question["type"] == "noul":
            a["probabilities"] = {"false": 1.0 - a["noul"], "true": a["noul"]}
        return a

    # ---------------------------------------------------------------- warm-up
    def warmup(self, steps=200):
        """Trigger every torch.compile specialisation / Triton autotune before serving (first requests of each new batch
        shape otherwise stall the GPU loop for seconds). Synthetic token ids; results discarded."""
        cap = min(self.max_input_tokens, 8192)

        def go(lens, n_opt=2):
            lens = [min(n, cap) for n in lens]
            while sum(lens) > self.max_batch_tokens and len(lens) > 1:
                lens.pop()
            self.run([Encoded([1] * n, n, n_opt) for n in lens])

        shapes = [[n] for n in (17, 64, 300, 700, 1500, 3000, 6000, 8192)]
        shapes += [[300] * k for k in (2, 3, 4, 5, 7, 8, 12, 16, 24, 32)]
        shapes += [[700] * 11 + [64], [4000, 4000, 4000, 4000], [2000] * 8, [8192, 8000], [500] * 32, [16000]]
        for _ in range(2):
            for lens in shapes:
                go(lens)
        rng = random.Random(7)
        for _ in range(steps):
            lens, n = [], rng.choice([1, 1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 40])
            for _ in range(n):
                length = int(min(cap, max(16, rng.lognormvariate(6.3, 0.8))))
                if sum(lens) + length > self.max_batch_tokens:
                    break
                lens.append(length)
            go(lens or [64], rng.randint(2, 12))


def validate_pplx_question(question):
    """Same checks as the legacy server (engine/server.py validate) for one question; raises ValueError."""
    if not isinstance(question, dict) or question.get("type") not in ("choice", "score", "noul"):
        raise ValueError("each question needs type choice|score|noul")
    crit = question.get("criteria")
    if question["type"] == "choice" and (not isinstance(crit, dict) or not 1 <= len(crit) <= 255):
        raise ValueError("choice questions take 1 to 255 options per choice")
    if question["type"] == "score" and (not isinstance(crit, list) or not 2 <= len(crit) <= 10):
        raise ValueError("a score takes 2 to 10 levels")
