"""Backend API of the REAX multi-model decision server (engine/serve.py).

A *decision model* answers a structured question about a state with ONE forward pass and a readout over the answer
options at the last prompt position (no generation): probabilities = softmax(option logits / T).

The server tokenises in request threads (`encode`), packs queued items into batches and calls `run` from a single
scheduler thread, so a backend never needs its own locking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Encoded:
    payload: Any  # backend-specific, e.g. token ids (list[int]) or a prompt string
    n_tokens: int  # prompt length in model tokens (0 if unknown)
    n_options: int  # number of answer options
    aux: Any = None  # backend-specific extra, e.g. the state-block token count of d3-turbo or the marker row of reax-1


class Backend(Protocol):
    name: str  # registry id, e.g. "plumb-4b"
    kind: str  # "native-gpu" | "vllm" | "gguf-cpu" | "onnx-cpu"
    max_input_tokens: int
    max_batch_tokens: int  # the scheduler packs encoded items up to this many tokens per run(); 0 = one item per run()

    def encode(self, state, question: dict) -> Encoded:
        """CPU work, thread-safe, called in request threads. Raises ValueError for bad or too long input."""

    def run(self, batch: list[Encoded]) -> list[list[float]]:
        """Option probabilities (sum 1, in the option order of `options(question)`) for each item.
        Called from ONE scheduler thread."""

    def options(self, question: dict) -> list[str]:
        """Option keys in model order."""


def sidecar_answer(question: dict, keys: list[str], probs: list[float]) -> dict:
    """Answer shape of the s1 runtime sidecar (`POST /internal/v1/decide`)."""
    p = {k: float(v) for k, v in zip(keys, probs)}
    answer = {"type": question["type"], "probabilities": p, "confidence": max(p.values())}
    if question["type"] == "noul":
        answer["noul"] = p["true"]
    elif question["type"] == "choice":
        answer["choice"] = max(p, key=p.get)
    else:
        answer["score"] = sum(int(k) * v for k, v in p.items())
    return answer
