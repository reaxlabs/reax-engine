"""Native engine for Plumb-4B (crh225/plumb-4b, the "s1-fast" model): Qwen3.5-4B hybrid (24 Gated-DeltaNet + 8 full
attention layers, CAUSAL), JevK5 decision readout.

The forward pass is the shared REAX Engine machinery of `pplx_engine.PplxEngine` (packed varlen batches, fla varlen
DeltaNet, varlen flash attention, fused projections, last-layer pruning, readout on the last token of every decision
only) with `causal=True`; this file only adds what is specific to Plumb:

  * weight names of `Qwen3_5ForCausalLM` (`model.language_model.*`, tied embeddings, no readout file): the readout matrix is
    the embedding rows of the letter tokens A..P, exactly the `lm_head.weight[slots]` the reference multiplies with,
  * the prompt, token for token the one of the author's `jevk5.runtime.JevK5.encode` (Apache-2.0, see NOTICE): the JSON
    payload (evidence, criterion, lettered options), the Qwen3.5 chat template with thinking off, rendered here as a plain
    string (no jinja at request time) and tokenised with the Rust tokenizer directly,
  * `PlumbBackend`, the `Backend` implementation used by `serve.py`.

Readout: letter logits = last hidden state @ embedding rows of the letters (bf16), / T (T from jevk5_config.json, 2.07),
float32 softmax over the options of the question.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from pplx_engine import PplxEngine

LETTERS = "ABCDEFGHIJKLMNOP"
MAX_OPTIONS = len(LETTERS)
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
# Qwen3.5 chat template (chat_template.jinja) for [system, user] with add_generation_prompt and enable_thinking=False.
PREFIX = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def decision_options(question: dict) -> list[tuple[str, str]]:
    """(option id, option text) of a typed question, as jevk5.runtime.decision_options: noul -> true/false, choice -> its
    criteria, score -> level indices. Texts are "id: description"."""
    crit = question.get("criteria")
    if question["type"] == "noul":
        pairs = [(k, (crit or {}).get(k) or f"The proposition is {k}.") for k in ("true", "false")]
    elif question["type"] == "choice":
        if isinstance(crit, list):
            crit = dict.fromkeys(crit)
        pairs = [(k, v or k) for k, v in crit.items()]
    else:
        pairs = [(str(i), level) for i, level in enumerate(crit)]
    return [(k, f"{k}: {d}") for k, d in pairs]


def prompt_text(state, criterion, options: list[str]) -> str:
    payload = {
        "evidence": state,
        "criterion": criterion,
        "options": [{"letter": LETTERS[i], "description": d} for i, d in enumerate(options)],
    }
    return PREFIX + json.dumps(payload, ensure_ascii=False).strip() + SUFFIX


class PlumbEngine(PplxEngine):
    def __init__(self, model_dir, **kw):
        kw.setdefault("causal", True)
        self.letter_ids = None
        super().__init__(model_dir, **kw)

    def _read_weights(self, device):
        from safetensors import safe_open
        from tokenizers import Tokenizer
        self.temperature = float(json.loads((self.dir / "jevk5_config.json").read_text())["temperature"])
        tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
        slots = [tok.encode(c, add_special_tokens=False).ids for c in LETTERS]
        assert all(len(s) == 1 for s in slots), "every answer letter must be one token"
        self.letter_ids = [s[0] for s in slots]
        pre = "model.language_model."
        W = {}
        with safe_open(str(self.dir / "model.safetensors"), framework="pt", device=device) as sf:
            for k in sf.keys():
                if k.startswith(pre):
                    W[k[len(pre):]] = sf.get_tensor(k)
        self.embed = W.pop("embed_tokens.weight")
        self.norm_w = W.pop("norm.weight")
        # tied embeddings: lm_head.weight[letters] == embed_tokens.weight[letters]
        self.readout = self.embed[torch.tensor(self.letter_ids, device=device)].to(self.dtype).contiguous()
        return W

    @torch.inference_mode()
    def letter_logits(self, ids: torch.Tensor, lens: list[int]) -> torch.Tensor:
        """ids: packed [T] int64 on device -> [N, 16] float32 letter logits (before / T)."""
        return self.forward(ids, lens)


class PlumbBackend:
    """`Backend` (engine/backends/base.py) for Plumb-4B."""

    name = "plumb-4b"
    kind = "native-gpu"

    def __init__(self, model_dir=None, max_input_tokens=4096, max_batch_tokens=None, device="cuda", **engine_kw):
        from backends.base import Encoded
        from tokenizers import Tokenizer
        self._Encoded = Encoded
        self.dir = Path(model_dir or os.environ.get("PLUMB_MODEL_DIR", "/model"))
        self.max_input_tokens = max_input_tokens
        self.max_batch_tokens = int(max_batch_tokens or os.environ.get("PLUMB_BATCH_TOKENS", 16384))
        self.tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
        self.engine = PlumbEngine(self.dir, device=device, **engine_kw)
        self.temperature = self.engine.temperature
        if not 2.06 < self.temperature < 2.08:
            raise SystemExit("Plumb calibration temperature does not match the pinned release")
        self.device = device

    # --------------------------------------------------------------- request threads
    def options(self, question: dict) -> list[str]:
        return [k for k, _ in decision_options(question)]

    def encode(self, state, question: dict):
        if question.get("type") not in ("choice", "noul", "score"):
            raise ValueError("bad question type")
        try:
            opts = decision_options(question)
        except (AttributeError, TypeError, KeyError) as exc:
            raise ValueError(f"bad criteria: {type(exc).__name__}") from None
        if not 1 <= len(opts) <= MAX_OPTIONS:
            raise ValueError(f"a question takes 1 to {MAX_OPTIONS} options")
        text = prompt_text(state, question.get("instructions", ""), [t for _, t in opts])
        ids = self.tok.encode(text, add_special_tokens=False).ids
        if len(ids) > self.max_input_tokens:
            raise ValueError(f"input exceeds {self.max_input_tokens} model tokens")
        return self._Encoded(payload=np.asarray(ids, dtype=np.int64), n_tokens=len(ids), n_options=len(opts))

    # --------------------------------------------------------------- scheduler thread
    def run(self, batch):
        lens = [e.n_tokens for e in batch]
        ids = torch.from_numpy(np.concatenate([e.payload for e in batch])).to(self.device, non_blocking=True)
        logits = self.engine.letter_logits(ids, lens).cpu().numpy()
        out = []
        for row, e in zip(logits, batch):
            r = row[: e.n_options] / self.temperature
            x = np.exp(r - r.max())
            out.append((x / x.sum()).tolist())
        return out

    def warmup(self, log=print):
        """Compile / autotune every kernel shape class before serving."""
        import random
        rng = random.Random(7)
        E = self._Encoded

        def item(n):
            return E(payload=np.ones(n, dtype=np.int64), n_tokens=n, n_options=rng.randint(2, 8))
        shapes = [[n] for n in (17, 64, 192, 300, 700, 1500, 3000, 4096)]
        shapes += [[200] * k for k in (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)]
        for lens in shapes:
            for _ in range(2):
                self.run([item(n) for n in lens])
        for _ in range(int(os.environ.get("PLUMB_WARM_STEPS", "60"))):
            n = rng.choice([1, 1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 40])
            lens = [int(min(4096, max(16, rng.lognormvariate(5.2, 0.6)))) for _ in range(n)]
            while sum(lens) > self.max_batch_tokens:
                lens.pop()
            self.run([item(m) for m in lens or [64]])
        torch.cuda.synchronize()
