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
# (sequences, padded tokens per sequence) of the CUDA graphs captured at start-up
GRAPH_SHAPES = ([(1, n) for n in (128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)]
                + [(2, n) for n in (128, 192, 256, 384, 512)] + [(4, n) for n in (128, 192, 256, 384)]
                + [(8, n) for n in (128, 192, 256)])
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
        self.graphs = {}
        self.graph_waste = float(os.environ.get("PLUMB_GRAPH_WASTE", "1.5"))  # use a graph if padded tokens <= waste * real
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

    # ------------------------------------------------------------ batches
    @torch.inference_mode()
    def letter_logits(self, ids: torch.Tensor, lens: list[int]) -> torch.Tensor:
        """ids: packed [T] int64 on device -> [N, 16] float32 letter logits (before / T)."""
        return self.forward(ids, lens)

    def logits(self, arrays: list[np.ndarray]) -> np.ndarray:
        """Token-id arrays of N decisions -> [N, 16] float32 letter logits on the host. Small batches replay a CUDA graph
        of the padded shape [B sequences x n tokens] (right padding: every layer is causal, so the read last tokens are
        unaffected); everything else runs the packed varlen path."""
        lens = [len(a) for a in arrays]
        g = self._pick_graph(lens)
        if g is not None:
            return g.run(arrays, lens)
        ids = torch.from_numpy(np.concatenate(arrays)).to(self.device, non_blocking=True)
        return self.letter_logits(ids, lens).cpu().numpy()

    def _pick_graph(self, lens):
        if not self.graphs:
            return None
        B, n, tot = len(lens), max(lens), sum(lens)
        best = None
        for (gb, gn), g in self.graphs.items():
            if gb >= B and gn >= n and gb * gn <= self.graph_waste * tot + 64 and (best is None or gb * gn < best[0]):
                best = (gb * gn, g)
        return best[1] if best else None

    @torch.inference_mode()
    def capture_graphs(self, shapes):
        pool = torch.cuda.graph_pool_handle()
        for B, n in shapes:
            if (B, n) not in self.graphs:
                self.graphs[(B, n)] = _Graph(self, B, n, pool)


class _Graph:
    """CUDA graph of the whole forward for B sequences padded to n tokens (static ids / last-token index buffers)."""

    def __init__(self, eng: PlumbEngine, B: int, n: int, pool):
        self.eng, self.B, self.n = eng, B, n
        dev = eng.device
        self.ids = torch.zeros(B * n, dtype=torch.int64, device=dev)
        self.last = torch.arange(B, device=dev, dtype=torch.int64) * n + (n - 1)
        cu = torch.arange(B + 1, device=dev, dtype=torch.int32) * n
        offs = list(range(0, (B + 1) * n, n))
        cos, sin = eng._rope_tables([n] * B, dev)
        self.host_ids = torch.zeros(B * n, dtype=torch.int64).pin_memory()
        self.host_last = torch.zeros(B, dtype=torch.int64).pin_memory()

        def fwd():
            return eng._drive([eng._core_gen(self.ids, cu, cu.long(), offs, n, self.last, cos, sin)], [None])[0]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fwd()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self.out = fwd()

    def run(self, arrays, lens):
        n, hi, hl = self.n, self.host_ids.numpy(), self.host_last.numpy()
        hi[:] = 0
        for b, a in enumerate(arrays):
            hi[b * n: b * n + len(a)] = a
            hl[b] = b * n + len(a) - 1
        # unused sequences: read their own last position (garbage rows, dropped below)
        for b in range(len(arrays), self.B):
            hl[b] = b * n + n - 1
        self.ids.copy_(self.host_ids, non_blocking=True)
        self.last.copy_(self.host_last, non_blocking=True)
        self.graph.replay()
        return self.out[: len(arrays)].cpu().numpy()


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
        logits = self.engine.logits([e.payload for e in batch])
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
        if os.environ.get("PLUMB_GRAPHS", "1") == "1":
            self.engine.capture_graphs(GRAPH_SHAPES)
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
