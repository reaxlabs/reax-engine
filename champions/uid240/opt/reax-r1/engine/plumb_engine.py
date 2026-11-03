# SPDX-License-Identifier: LicenseRef-REAX-Engine-1.1 AND Apache-2.0
# REAX Engine code (LICENSE) with parts adapted from Apache-2.0 work named below and in NOTICE; modified by the REAX Engine licensor.
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

os.environ.setdefault("FLA_TENSOR_CACHE_SIZE", "16")  # read when fla is first imported (see _fla_cache_entries)
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


def pin_fla_autotune():
    """fla autotunes its Triton kernels by timing, at random per process. For l2norm_fwd_kernel (q/k normalisation of the
    DeltaNet layers, a row reduction over 128 values) some choices (e.g. BT=8, 8 warps) sum in a different order, which
    changes bf16 results at the last bit and, through 24 layers, the answers (mean logit distance ~0.06 vs the other
    processes; seen in ~1 of 10 process starts, in the reference runtime as well as here). Fix one of the configs the
    majority of processes end up with, so every start of this engine computes the same bits."""
    import triton
    from fla.modules import l2norm
    k = l2norm.l2norm_fwd_kernel
    k.configs = [triton.Config({"BT": 16}, num_warps=4)]
    k.cache.clear()


class PlumbEngine(PplxEngine):
    HOST_CU_DEFAULT = "1"

    def __init__(self, model_dir, exact=None, **kw):
        # exact (default): op-for-op the HF module sequence the reference graphs capture (separate projections, bf16 torch
        # conv, SDPA, full last layer), so a decision padded to its reference bucket is BIT-identical to the reference;
        # fast: fused projections / fla conv / flash varlen / last-layer pruning (packed batches, not bit-identical).
        self.exact = (os.environ.get("PLUMB_MODE", "exact") == "exact") if exact is None else exact
        if self.exact:
            kw.setdefault("fuse", False)
            kw.setdefault("conv_impl", "torch")
            kw.setdefault("attn_impl", "sdpa_cmask")
            kw.setdefault("prune_last", False)
            kw["compile"] = False  # fused/compiled elementwise kernels would change rounding
        pin_fla_autotune()
        if not self.exact:
            kw.setdefault("keep_full_last", True)  # padded CUDA-graph runs need the unpruned last layer
        kw.setdefault("causal", True)
        kw.setdefault("gva_native", os.environ.get("PLUMB_GVA", "1"))
        self.letter_ids = None
        self.graphs = {}
        self.runner = None
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

    @torch.inference_mode()
    def logits(self, arrays: list[np.ndarray]) -> np.ndarray:
        """Token-id arrays of N decisions -> [N, 16] float32 letter logits on the host. Small batches replay a CUDA graph
        of the padded shape [B sequences x n tokens] (right padding: every layer is causal, so the read last tokens are
        unaffected); everything else runs the packed varlen path."""
        if self.runner is not None:
            return self.runner.logits(arrays)
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

    def start_runner(self, n_streams=None, buckets=None):
        """Exact mode: capture the per-bucket graphs on every stream and route logits() through them."""
        r = ExactRunner(self, n_streams or int(os.environ.get("PLUMB_STREAMS", "4")))
        r.capture(buckets)
        self.runner = r

    @torch.inference_mode()
    def capture_graphs(self, shapes):
        pool = torch.cuda.graph_pool_handle()
        for B, n in shapes:
            if (B, n) not in self.graphs:
                self.graphs[(B, n)] = _Graph(self, B, n, pool)


def _fla_cache_entries(*keys):
    """All (args, kwargs, result) triples in fla's tensor_cache deques whose arguments include one of `keys`."""
    import collections
    import sys
    found = []
    for name, mod in list(sys.modules.items()):
        if not name.startswith("fla") or mod is None:
            continue
        for obj in list(vars(mod).values()):
            for cell in getattr(obj, "__closure__", None) or ():
                try:
                    c = cell.cell_contents
                except ValueError:
                    continue
                if isinstance(c, collections.deque):
                    found += [e for e in c if isinstance(e, tuple) and any(a is k for a in (*e[0], *e[1].values()) for k in keys)]
    return found


class _Graph:
    """CUDA graph of the whole forward for B sequences padded to n tokens (static ids / last-token index buffers)."""

    def __init__(self, eng: PlumbEngine, B: int, n: int, pool, stream=None):
        self.eng, self.B, self.n = eng, B, n
        dev = eng.device
        self.ids = torch.zeros(B * n, dtype=torch.int64, device=dev)
        self.last = torch.arange(B, device=dev, dtype=torch.int64) * n + (n - 1)
        cu = torch.arange(B + 1, device=dev, dtype=torch.int32) * n
        offs = list(range(0, (B + 1) * n, n))
        cos, sin = eng._rope_tables([n] * B, dev)
        self.host_ids = torch.zeros(B * n, dtype=torch.int64).pin_memory()
        self.host_last = torch.zeros(B, dtype=torch.int64).pin_memory()

        cu64 = cu.long()
        cu_cpu = torch.tensor(offs, dtype=torch.int64)

        def fwd():
            return eng._drive([eng._core_gen(self.ids, cu, cu64, offs, n, self.last, cos, sin, cu_cpu, prune=False)], [None])[0]
        stream = stream or torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fwd()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        # fla memoises index tensors derived from cu_seqlens by object identity in small bounded caches; the graph reads
        # them, so keep them alive (and the cache entries hit during capture) for as long as the graph exists
        self._keep = (cu, cu64, cu_cpu, cos, sin, offs, _fla_cache_entries(cu64, cu_cpu))
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool, stream=stream):
            self.out = fwd()
        torch.cuda.synchronize()

    @torch.inference_mode()
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


# The reference's padded lengths (jevk5.runtime.GRAPH_LENGTHS): a decision of L tokens is right-padded to the smallest of
# these >= L, which fixes the GEMM shapes (hence cuBLAS kernels and rounding) of the reference run.
REF_BUCKETS = (128, 192, 256, 320, 384, 512, 640, 768, 1024, 1536, 2048, 3072, 4096)


class ExactRunner:
    """Bit-faithful multi-stream executor. One CUDA graph per (stream, reference bucket) holds the whole forward of ONE
    decision padded to its bucket; decisions of a step are spread over `n_streams` streams and replayed concurrently, so
    every decision gets exactly the numbers the reference computes, whatever the concurrency, while the GPU overlaps the
    latency-bound kernels of several decisions."""

    def __init__(self, eng: PlumbEngine, n_streams: int, buckets=REF_BUCKETS, max_items=256):
        self.eng, self.buckets = eng, tuple(buckets)
        self.streams = [torch.cuda.Stream() for _ in range(n_streams)]
        self.pools = [torch.cuda.graph_pool_handle() for _ in range(n_streams)]
        self.graphs: dict = {}
        self.max_items = max_items
        self.stage_ids = torch.zeros(max_items, self.buckets[-1], dtype=torch.int64).pin_memory()
        self.stage_last = torch.zeros(max_items, dtype=torch.int64).pin_memory()
        self.stage_out = torch.zeros(max_items, MAX_OPTIONS, dtype=torch.float32).pin_memory()

    @torch.inference_mode()
    def capture(self, buckets=None):
        for si, st in enumerate(self.streams):
            for n in buckets or self.buckets:
                if (si, n) not in self.graphs:
                    self.graphs[(si, n)] = _Graph(self.eng, 1, n, self.pools[si], stream=st)

    def bucket(self, n: int) -> int:
        for b in self.buckets:
            if b >= n:
                return b
        raise ValueError("input too long")

    @torch.inference_mode()
    def logits(self, arrays):
        """-> [N, 16] float32 numpy (letter logits before / T). All decisions are enqueued first, then one sync."""
        N = len(arrays)
        out = np.empty((N, MAX_OPTIONS), dtype=np.float32)
        for lo in range(0, N, self.max_items):
            self._chunk(arrays[lo: lo + self.max_items], out[lo: lo + self.max_items])
        return out

    def _chunk(self, arrays, out):
        S = len(self.streams)
        load = [0] * S
        used = [False] * S
        ids_np, last_np = self.stage_ids.numpy(), self.stage_last.numpy()
        order = sorted(range(len(arrays)), key=lambda i: -len(arrays[i]))  # longest first: balance the streams
        for k, i in enumerate(order):
            a = arrays[i]
            n = self.bucket(len(a))
            si = min(range(S), key=load.__getitem__)
            load[si] += n
            used[si] = True
            g = self.graphs[(si, n)]
            ids_np[k, : len(a)] = a
            ids_np[k, len(a): n] = 0
            last_np[k] = len(a) - 1
            with torch.cuda.stream(self.streams[si]):
                g.ids.copy_(self.stage_ids[k, :n], non_blocking=True)
                g.last.copy_(self.stage_last[k: k + 1], non_blocking=True)
                g.graph.replay()
                self.stage_out[k].copy_(g.out[0], non_blocking=True)
        for si, st in enumerate(self.streams):
            if used[si]:
                st.synchronize()
        for k, i in enumerate(order):
            out[i] = self.stage_out[k].numpy()


class PlumbBackend:
    """`Backend` (engine/backends/base.py) for Plumb-4B."""

    name = "plumb-4b"
    kind = "native-gpu"
    encode_inline = True  # Rust tokenizer: serve.py encodes on the event loop (no thread hop)

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
        if self.engine.exact:
            self.engine.start_runner(buckets=[b for b in REF_BUCKETS if b <= self.max_input_tokens])
        elif os.environ.get("PLUMB_GRAPHS", "1") == "1":
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
