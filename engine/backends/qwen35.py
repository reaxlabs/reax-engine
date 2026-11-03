"""Backend for the Qwen3.5-hybrid decision models served by the native PplxEngine: pplx-decider-v1.1-27b, d3, d3-turbo, reax-1.

All checkpoints share the architecture (48 Gated-DeltaNet + 16 full-attention layers, hidden 5120, 255-way
readout) and the engine; they differ in prompt/answer contract and attention mode:

  family "pplx"  perplexity-ai/pplx-decider-v1.1-27b   prompt in pplx_engine.py (adapted from the author's code)
  family "d3"    vllm-sr/d3, vllm-sr/d3-turbo          prompt/answer contract in d3_contract.py (vendored, upstream/d3)
  family "reax1" decisionmodels/reax-1                 pplx-decider v1.2 joint format (protocol v3), tokenised by the author's
                                                       own Encoder from the checkpoint's source/src/pplx_decider (foreign code,
                                                       read from the read-only weights mount), readout at the question marker

pplx and d3 have NONCAUSAL full attention (decision_config.json: "attention_mode": "noncausal_full_attention"). d3-turbo
says "prefix_bidir": the state block attends only to itself, the question block reads the whole row. With `shared_state`
every single decision is then run state-once through PplxEngine.forward_groups (prefix = state block, one suffix), and
queued calls with a token-identical state block are merged into one group (`shared_group`): the production recipe of
the s1-pro d3-turbo node (private engine branch d3-turbo-shared2-20261022 @75d0659, `--shared-state 1 --shared-until 64
--shared-mixq 0 --shared-group 1`), so an answer does not depend on which sibling calls arrive with it.

The torch/transformers imports are lazy so the module (and the registry) import on a CPU-only machine.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

from .. import d3_contract, qev_contract
from .base import Encoded

_ENGINE_DIR = str(Path(__file__).resolve().parent.parent)
FAMILIES = ("pplx", "d3", "reax1", "mammoth")
_DEFAULT_NAMES = {"pplx": "pplx-decider-v1.1-27b", "d3": "d3", "reax1": "reax-1", "mammoth": "mammoth-1"}


def _pplx_engine_module():
    """pplx_engine.py and its kernels import each other as top-level modules (legacy flat layout)."""
    if _ENGINE_DIR not in sys.path:
        sys.path.insert(0, _ENGINE_DIR)
    import pplx_engine
    return pplx_engine


class Qwen35Backend:
    kind = "native-gpu"
    serve_warmup = False  # serve.py leaves warming to the `warm_steps` option (runs in the constructor)
    # d3-turbo (prefix_bidir) state-once routing, set in __init__; class defaults keep stubbed instances valid
    prefix_bidir = False
    shared = False
    shared_group = False
    shared_min = 64

    def __init__(self, model_dir, *, family="pplx", name=None, linear="fp8", compile="1", gva="1", fp8_skip="", nvfp4="",
                 nvfp4_scales="", budget=16384, max_input_tokens=None, warm_steps=0, device="cuda", shared_state=False,
                 shared_group=False, shared_min=64, shared_until=None, shared_mixq=None):
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        self.family = family
        self.name = name or _DEFAULT_NAMES[family]
        # pplx / reax-1: the 8192-token limit; d3: the author's server has none (long prompts were validated to 64k+).
        self.max_input_tokens = max_input_tokens if max_input_tokens is not None else (262144 if family == "d3" else 8192)
        self.max_batch_tokens = budget
        self._pp = _pplx_engine_module()
        model_dir = str(model_dir)
        if family == "mammoth":
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(model_dir)
        elif family == "d3":
            from transformers import AutoTokenizer
            self._d3 = d3_contract
            cfg = json.loads((Path(model_dir) / "decision_config.json").read_text())
            self._tok, self._codes = AutoTokenizer.from_pretrained(model_dir), cfg["codes"]
            if cfg.get("readout_dtype", "float32") == "float32":
                os.environ.setdefault("PPLX_READOUT_FP32", "1")  # author runtime: hidden.float() @ readout.T in fp32
        elif family == "reax1":
            self._init_reax1(model_dir)
        else:
            self._prompter = self._pp.Prompter(model_dir)
        self.engine = self._pp.PplxEngine(model_dir, device=device, linear_mode=linear, compile=compile,
                                          fp8_skip=tuple(x for x in fp8_skip.split(",") if x), gva_native=gva,
                                          nvfp4_groups=tuple(g for g in nvfp4.split(",") if g), nvfp4_scales=nvfp4_scales or None)
        self._device = device
        self.steps = 0
        self.prefix_bidir = bool(getattr(self.engine, "prefix_bidir", False))
        if shared_until is not None:
            self.engine.shared_until = shared_until
        if shared_mixq is not None:
            self.engine.shared_mixq = bool(shared_mixq)
        if shared_state or shared_group:
            if not self.prefix_bidir:
                raise ValueError("shared_state / shared_group need a prefix_bidir checkpoint (vllm-sr/d3-turbo): the sharing "
                                 "is exact only when the state block attends to itself")
            self.shared, self.shared_group, self.shared_min = bool(shared_state), bool(shared_group), shared_min
            if self.shared:
                print(f"state-once mode: until layer {self.engine.shared_until} of {self.engine.n_layers}, "
                      f"state rows see {'all questions' if self.engine.shared_mixq else 'the state only'}", flush=True)
        if warm_steps:
            self.warmup(warm_steps)

    def _init_reax1(self, model_dir):
        """decisionmodels/reax-1: pplx-decider v1.2 joint protocol v3. Tokenisation is the author's own Encoder (identical
        prompts and validation); the production node runs it with `--v12-split single` (one sequence per question, the
        gateway fans multi-question requests out), which is what the one-question-per-call contract does as well."""
        mdir = Path(model_dir)
        sys.path.insert(0, str(mdir / "source" / "src"))
        from pplx_decider.protocol import PROTOCOL, Encoder, LengthError, RequestError
        from transformers import AutoProcessor
        joint = json.loads((mdir / "joint_config.json").read_text())
        assert joint == PROTOCOL, f"joint_config.json does not match the author's PROTOCOL: {joint} != {PROTOCOL}"
        codes = json.loads((mdir / "decision_config.json").read_text())["codes"]
        self._encoder = Encoder(AutoProcessor.from_pretrained(model_dir), codes)
        self._v12_errors = (LengthError, RequestError)
        assert self.max_input_tokens <= self.max_batch_tokens, "max_input_tokens must fit one step (budget)"

    # ---------------------------------------------------------------- Backend API
    def options(self, question):
        if self.family == "mammoth":
            return list(qev_contract.normalize_question(question).keys)
        if self.family == "d3":
            return list(self._d3.normalize_question(question).keys)
        return list(self._pp.options(question)[0])

    def encode(self, state, question):
        aux = None
        if self.family == "mammoth":
            n = qev_contract.normalize_question(question)
            full, aux = qev_contract.encode_request(self._tok, state, [n])
            ids, n_opt = full[0], len(n.keys)
        elif self.family == "d3":
            n = self._d3.normalize_question(question)  # ValueError with the reason
            text = self._d3.render(self._tok, state, n.rendered, self._codes)
            if self.prefix_bidir:
                ids, aux = self._d3.encode_split(self._tok, text, state)
            else:
                ids = self._tok(text, add_special_tokens=False)["input_ids"]
            n_opt = len(n.keys)
        elif self.family == "reax1":
            return self._encode_reax1(state, question)
        else:
            validate_pplx_question(question)
            ids, n_opt = self._prompter.ids(state, question)
        if len(ids) > self.max_input_tokens:
            raise ValueError(f"maximum context length exceeded: {len(ids)} > {self.max_input_tokens} tokens")
        return Encoded(ids, len(ids), n_opt, aux)

    def _encode_reax1(self, state, question):
        if not isinstance(question, dict) or question.get("type") not in ("choice", "score", "noul"):
            raise ValueError("each question needs type choice|score|noul")
        try:
            chunks = self._encoder.plan({"state": state, "questions": {"q": question}}, self.max_input_tokens)
        except self._v12_errors[0] as exc:  # over-long single question: the engine's 422 context_length
            raise ValueError(f"maximum context length exceeded: {exc}") from None
        except self._v12_errors[1] as exc:
            raise ValueError(str(exc)) from None
        if len(chunks) != 1:  # one question cannot be split
            raise ValueError("maximum context length exceeded: the question needs more than one sequence")
        c = chunks[0]
        return Encoded(c.input_ids.tolist(), int(c.length), int(c.counts[0]), int(c.markers[0]))

    def run(self, batch):
        import torch
        if self.family == "reax1":
            ids = torch.tensor([t for e in batch for t in e.payload], device=self._device)
            logits = self.engine.forward_marked(ids, [len(e.payload) for e in batch], [[e.aux] for e in batch])
            self.steps += 1
            return self.engine.probs(logits, [e.n_options for e in batch])
        if self.shared:
            return self._run_shared(batch)
        ids = torch.tensor([t for e in batch for t in e.payload], device=self._device)
        states = [e.aux for e in batch] if self.prefix_bidir else None
        logits = self.engine.forward(ids, [len(e.payload) for e in batch], states)
        self.steps += 1
        return self.engine.probs(logits, [e.n_options for e in batch])

    # ---------------------------------------------------------------- d3-turbo: state-once routing
    def plan_shared(self, batch):
        """Split a step into (plain, groups) the way the production d3-turbo node does (engine/server.py of 75d0659).

        groups: [(prefix_ids, [(suffix_ids, batch_index), ...]), ...] for PplxEngine.forward_groups; plain: batch indices that
        cannot use the shared path (state block shorter than the 3-token conv window) and run through forward().
        With shared_group, queued calls whose first `shared_min` tokens AND state-block length agree are merged into one
        group when their common prefix covers exactly the state block (>= shared_min tokens); every other single decision
        becomes a group with one suffix, so the answer does not depend on sibling co-arrival."""
        items = [(e.payload, e.aux, i) for i, e in enumerate(batch)]
        singles, merged = [], []
        if self.shared_group:
            buckets = {}
            for it in items:
                if len(it[0]) <= self.shared_min:
                    singles.append(it)
                else:
                    buckets.setdefault((tuple(it[0][:self.shared_min]), it[1]), []).append(it)
            for its in buckets.values():
                if len(its) < 2:
                    singles += its
                    continue
                seqs = [x[0] for x in its]
                m = min(map(len, seqs)) - 1
                k = 0
                while k < m and all(x[k] == seqs[0][k] for x in seqs):
                    k += 1
                sb = its[0][1]
                k = sb if k >= sb else 0  # the shared prefix is exactly the (token-identical) state block
                if k < self.shared_min:
                    singles += its
                else:
                    merged.append((seqs[0][:k], [(x[0][k:], x[2]) for x in its]))
        else:
            singles = items
        conv, plain = [], []
        for ids, sb, i in singles:
            if sb is not None and 3 <= sb <= len(ids) - 1:  # forward_groups needs a prefix >= the conv window (3)
                conv.append((ids[:sb], [(ids[sb:], i)]))
            else:
                plain.append(i)
        return plain, conv + merged

    def _run_shared(self, batch):
        import torch
        out = [None] * len(batch)
        plain, groups = self.plan_shared(batch)
        if plain:
            ids = torch.tensor([t for i in plain for t in batch[i].payload], device=self._device)
            logits = self.engine.forward(ids, [len(batch[i].payload) for i in plain], [batch[i].aux for i in plain])
            for i, p in zip(plain, self.engine.probs(logits, [batch[i].n_options for i in plain])):
                out[i] = p
        if groups:
            order = [i for _, ss in groups for _, i in ss]
            logits = self.engine.forward_groups([(p, [s for s, _ in ss]) for p, ss in groups])
            for i, p in zip(order, self.engine.probs(logits, [batch[i].n_options for i in order])):
                out[i] = p
        self.steps += 1
        return out

    # ---------------------------------------------------------------- family answer shapes (/v1/systemone)
    def answer(self, question, probs):
        if self.family == "mammoth":
            return qev_contract.product_answer(qev_contract.normalize_question(question), probs)
        if self.family == "d3":
            return self._d3.product_answer(self._d3.normalize_question(question), probs)
        a = self._pp.answer(question, probs)
        if question["type"] == "noul":
            a["probabilities"] = {"false": 1.0 - a["noul"], "true": a["noul"]}
        return a

    # ---------------------------------------------------------------- warm-up
    def _enc(self, n, n_opt=2):
        """Synthetic step item of n tokens (ids discarded); d3-turbo needs a state block, the marker is the last token."""
        if self.family == "reax1":
            return Encoded([1] * n, n, n_opt, n - 1)
        return Encoded([1] * n, n, n_opt, max(1, n // 2) if self.prefix_bidir else None)

    def warmup(self, steps=200):
        """Trigger every torch.compile specialisation / Triton autotune before serving (first requests of each new batch
        shape otherwise stall the GPU loop for seconds). Synthetic token ids; results discarded."""
        cap = min(self.max_input_tokens, 8192)

        def go(lens, n_opt=2):
            lens = [min(n, cap) for n in lens]
            while sum(lens) > self.max_batch_tokens and len(lens) > 1:
                lens.pop()
            self.run([self._enc(n, n_opt) for n in lens])

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
        if self.shared:
            self._warmup_shared(steps, rng, cap)

    def _warmup_shared(self, steps, rng, cap):
        """The shared-state kernels as the production node warms them: a deterministic sweep over every step-size bucket
        (fla's conv kernel specialises and re-autotunes per 1024-token bucket of the step; a size first seen while serving
        stalls the GPU loop for tens of seconds), then random steps of shared groups (long prefix + short suffixes)."""
        eng, budget = self.engine, self.max_batch_tokens
        full = eng.shared_until < eng.n_layers
        for T in range(512, budget + 1, 512):
            for n_seq in (1, 3, 9):
                lens = [max(16, T // n_seq)] * n_seq
                if max(lens) > cap:
                    continue
                self.run([self._enc(n) for n in lens])
                if n_seq > 1:  # one shared group of this full size: prefix + n_seq suffixes
                    P = T // n_seq - 40 if full else T - 40 * n_seq
                    if P < 64:
                        continue
                    self._warm_groups([([1] * P, [([1] * 40, 2)] * n_seq)])
        for _ in range(max(40, steps // 2)):
            groups, tot = [], 0
            for _ in range(rng.choice([1, 2, 3, 4, 6, 8, 12])):
                P = int(min(cap - 600, max(64, rng.lognormvariate(6.8, 0.8))))
                subs = [([1] * int(max(8, rng.lognormvariate(4.5, 0.7))), rng.randint(2, 12))
                        for _ in range(rng.choice([1, 2, 3, 5, 8, 12, 32, 64]))]
                n = (P * len(subs) if full else P) + sum(len(s) for s, _ in subs)
                if tot + n > budget:
                    break
                groups.append(([1] * P, subs))
                tot += n
            if groups:
                self._warm_groups(groups)

    def _warm_groups(self, groups):
        logits = self.engine.forward_groups([(p, [s for s, _ in ss]) for p, ss in groups])
        self.engine.probs(logits, [n for _, ss in groups for _, n in ss])
        self.steps += 1


def validate_pplx_question(question):
    """Same checks as the legacy server (engine/server.py validate) for one question; raises ValueError."""
    if not isinstance(question, dict) or question.get("type") not in ("choice", "score", "noul"):
        raise ValueError("each question needs type choice|score|noul")
    crit = question.get("criteria")
    if question["type"] == "choice" and (not isinstance(crit, dict) or not 1 <= len(crit) <= 255):
        raise ValueError("choice questions take 1 to 255 options per choice")
    if question["type"] == "score" and (not isinstance(crit, list) or not 2 <= len(crit) <= 10):
        raise ValueError("a score takes 2 to 10 levels")
