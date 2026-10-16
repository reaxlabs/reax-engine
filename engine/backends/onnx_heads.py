"""`classifier_heads` backend: one encoder pass, N fixed classification heads, executed with onnxruntime (CPU).

This is a different model kind from the readout decision models. A readout model sees the question in its prompt and
answers it from the answer-option logits at the last prompt position. A classifier-heads model has no prompt for the
question: its heads are trained for a fixed set of questions, the request text goes in, all heads come out of one pass.

The first model is s1-llm-auto-router (encoder + 7 heads). The execution is deliberately identical to the reference
path of that model (same ONNX file, same fast tokenizer and pair encoding, same truncation, float64 softmax / sigmoid
with the per-head temperatures from meta.json), so results are bit-identical to it, see docs/models/s1-llm-auto-router.md.

Classifier API (used by engine/serve.py; the decision `Backend` protocol of base.py is not implemented on purpose):
    encode(state) -> Encoded         tokenise (thread-safe), raises ValueError for bad input
    run(batch)    -> list[Result]    one onnxruntime call per item, called from the single scheduler thread
    classify(request, context)       encode + run for one request
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .base import Encoded
except ImportError:  # imported as a top-level module (engine/ on sys.path, as engine/serve.py does)
    from backends.base import Encoded

import numpy as np

CONTEXT_CHARS = 600  # only the first 600 characters of the conversation summary are read (reference behaviour)
MAX_STATE_BYTES = 16 * 1024  # size cap of the state (UTF-8 JSON), as in the reference server


@dataclass(frozen=True)
class Head:
    name: str
    kind: str                 # "choice" | "score" | "noul"
    output: int               # index of the ONNX output
    column: int | None        # column inside a multi-binary output (flags), else None
    labels: tuple             # class labels in model order (noul: ("false", "true") semantic order is handled in answer())
    temperature: str          # key in meta.json["temperatures"]


@dataclass(frozen=True)
class HeadsSpec:
    model_id: str
    onnx_file: str
    heads: tuple
    outputs: tuple            # output kind per ONNX output: "softmax" | "sigmoid"
    # pinned sha256 of the files at the registry revision (empty = not checked)
    sha256: dict = field(default_factory=dict)


ROUTER_CATEGORIES = ("coding", "agentic", "math", "knowledge", "long_context", "tool_use", "design", "summarisation", "general")
ROUTER_DIFFICULTY = ("trivial", "easy", "moderate", "hard", "frontier")
ROUTER_STAKES = ("negligible", "low", "medium", "high")
ROUTER_FLAGS = ("needs_tools", "needs_vision", "needs_long_context", "follow_up")

S1_LLM_AUTO_ROUTER = HeadsSpec(
    model_id="s1-llm-auto-router",
    onnx_file="model.gq.onnx",
    outputs=("softmax", "softmax", "softmax", "sigmoid"),   # ONNX outputs: category, difficulty, stakes, flags[4]
    heads=(
        Head("category", "choice", 0, None, ROUTER_CATEGORIES, "category"),
        Head("difficulty", "score", 1, None, ROUTER_DIFFICULTY, "difficulty"),
        Head("stakes", "score", 2, None, ROUTER_STAKES, "stakes"),
        *(Head(f, "noul", 3, i, ("false", "true"), "flags") for i, f in enumerate(ROUTER_FLAGS)),
    ),
    sha256={  # revision 572c5b6cdb53b77fbb26ed33568ebbc5b71c69c0
        "model.gq.onnx": "53a02107e46be7913f148037b63b4f4735a400f91221789b0aaefcde0ad8da80",
        "meta.json": "b6689ba21a44e0910cc2ae831c6ecf2bcc6f542b3fc5cd29d570e012865649c2",
        "tokenizer/tokenizer.json": "c01f8492531b2e569bd8d28ef1de1c66779275753c78cb497f7118762ae3230d",
    },
)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class OnnxHeadsBackend:
    kind = "onnx-cpu"
    model_type = "classifier_heads"
    max_batch_tokens = 0        # default: one item per run(). REAX_ROUTER_BATCH_TOKENS>0 packs queued items into one padded run

    def __init__(self, model_dir, spec: HeadsSpec = S1_LLM_AUTO_ROUTER, threads: int | None = None,
                 providers: list | None = None, verify: bool = True):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        root = Path(model_dir)
        if verify:
            for rel, want in spec.sha256.items():
                if sha256_file(root / rel) != want:
                    raise ValueError(f"{spec.model_id}: {rel} does not match the pinned sha256")
        meta = json.loads((root / "meta.json").read_text())
        self.spec, self.name = spec, spec.model_id
        self.temps = meta.get("temperatures") or {}
        self.max_input_tokens = min(int(meta.get("max_len") or 512), 512)
        self.tok = Tokenizer.from_file(os.path.realpath(root / "tokenizer" / "tokenizer.json"))
        self.tok.enable_truncation(self.max_input_tokens, strategy="only_second")
        self.tok.no_padding()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or int(os.environ.get("REAX_THREADS", "4"))
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(os.path.realpath(root / spec.onnx_file), so,
                                         providers=providers or ["CPUExecutionProvider"])
        self.providers = self.sess.get_providers()
        # Padded batches were bit-identical to single runs on the measured CPU (docs/models/s1-llm-auto-router.md), but
        # that is a property of the kernels on that CPU, so batching is opt-in.
        self.max_batch_tokens = int(os.environ.get("REAX_ROUTER_BATCH_TOKENS", "0"))
        self._lock = threading.Lock()   # direct classify() callers; the server uses the single scheduler thread

    # ---------------------------------------------------------------- request -> tokens
    @staticmethod
    def split_state(state) -> tuple[str, str]:
        """state is text or {"request": str, "context": str|obj}; same rules as the reference server."""
        if isinstance(state, dict) and isinstance(state.get("request"), str):
            request, context = state["request"], state.get("context")
            context = context if isinstance(context, str) else json.dumps(context, ensure_ascii=False) if context else ""
        elif isinstance(state, (str, dict, list)):
            request, context = (state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)), ""
        else:
            raise ValueError("state must be text or an object with a request")
        return request, context

    def encode(self, state, question=None) -> Encoded:
        request, context = self.split_state(state)
        if not request.strip():
            raise ValueError("request must be nonempty text")
        if len(json.dumps(state, ensure_ascii=False).encode()) > MAX_STATE_BYTES:
            raise ValueError("state exceeds 16 KiB")
        return self.tokenize(request, context)

    def tokenize(self, request: str, context: str = "") -> Encoded:
        try:
            enc = self.tok.encode(context[:CONTEXT_CHARS], request)
        except Exception as exc:  # the pair does not fit max_len even with the request truncated to nothing
            raise ValueError("context too long to fit the model's input length") from exc
        return Encoded(payload=enc.ids, n_tokens=len(enc.ids), n_options=0)

    # ---------------------------------------------------------------- forward + heads
    def logits(self, ids: list[int]) -> list:
        arr = np.asarray([ids], dtype=np.int64)
        return [o[0] for o in self.sess.run(None, {"input_ids": arr, "attention_mask": np.ones_like(arr)})]

    def probabilities(self, outs: list) -> dict:
        """Per-head probability arrays (float64): softmax(logits / T) or sigmoid(logit / T), as the reference."""
        res = {}
        for h in self.spec.heads:
            x = outs[h.output].astype(np.float64) / self.temps.get(h.temperature, 1.0)
            if self.spec.outputs[h.output] == "softmax":
                res[h.name] = _softmax(x)
            else:
                p = float(1 / (1 + np.exp(-x[h.column])))
                res[h.name] = np.array([1.0 - p, p])    # (false, true)
        return res

    def run(self, batch: list[Encoded]) -> list[dict]:
        if len(batch) == 1:
            return [self.probabilities(self.logits(batch[0].payload))]
        width = max(len(e.payload) for e in batch)       # right-pad with id 0, attention mask 0 on the padding
        ids = np.zeros((len(batch), width), dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, e in enumerate(batch):
            ids[i, :len(e.payload)] = e.payload
            mask[i, :len(e.payload)] = 1
        outs = self.sess.run(None, {"input_ids": ids, "attention_mask": mask})
        return [self.probabilities([o[i] for o in outs]) for i in range(len(batch))]

    def classify(self, request: str, context: str = "") -> tuple[dict, int]:
        enc = self.tokenize(request, context)
        with self._lock:
            return self.run([enc])[0], enc.n_tokens

    def warmup(self):
        self.classify("warm up", "")

    # ---------------------------------------------------------------- answer shaping
    def heads_answer(self, probs: dict) -> dict:
        """/v1/classify shape: {head: {"label", "probabilities": {class: p}, ...}}"""
        out = {}
        for h in self.spec.heads:
            p = probs[h.name]
            k = int(p.argmax())
            entry = {"label": h.labels[k], "confidence": float(p.max()),
                     "probabilities": {lab: float(v) for lab, v in zip(h.labels, p)}}
            if h.kind == "score":
                entry["score"] = float(sum(i * v for i, v in enumerate(p)))
            out[h.name] = entry
        return out

    def sidecar_answers(self, questions: dict, probs: dict) -> dict:
        """Production-compatible shape (/internal/v1/decide, /v1/systemone) for the fixed questions."""
        by_name = {h.name: h for h in self.spec.heads}
        out = {}
        for name, q in questions.items():
            h, p = by_name[name], probs[name]
            if h.kind == "choice":
                byk = {k: float(v) for k, v in zip(h.labels, p)}
                d = {k: byk[k] for k in q["criteria"]}   # option order of the request, as the reference
                out[name] = {"type": "choice", "choice": max(d, key=d.get), "confidence": max(d.values()),
                             "probabilities": d}
            elif h.kind == "score":
                out[name] = {"type": "score", "score": float(sum(i * v for i, v in enumerate(p))),
                             "confidence": float(max(p)),
                             "legend": {str(i): lvl for i, lvl in enumerate(q["criteria"])},
                             "probabilities": {str(i): float(v) for i, v in enumerate(p)}}
            else:
                v = float(p[1])
                out[name] = {"type": "noul", "noul": v, "probabilities": {"true": v, "false": 1.0 - v}}
        return out

    def check_questions(self, questions: dict) -> None:
        """The fixed-question schema of the production path: unknown names / types / option sets are rejected."""
        by_name = {h.name: h for h in self.spec.heads}
        for name, q in questions.items():
            h = by_name.get(name)
            if h is None or not isinstance(q, dict) or q.get("type") != h.kind:
                raise ValueError(f"{self.name} answers only its fixed questions: {', '.join(by_name)}")
            crit = q.get("criteria")
            if h.kind == "choice" and (not isinstance(crit, dict) or set(crit) != set(h.labels)):
                raise ValueError("category options must be exactly: " + ", ".join(h.labels))
            if h.kind == "score" and (not isinstance(crit, list) or len(crit) != len(h.labels)):
                raise ValueError(f"{name} must have {len(h.labels)} score levels")


def load(model_dir, **opts) -> OnnxHeadsBackend:
    return OnnxHeadsBackend(model_dir, **opts)
