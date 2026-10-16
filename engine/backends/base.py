"""Backend API of the REAX multi-model engine (frozen)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Encoded:
    payload: Any          # e.g. token ids (list[int]) or a prompt string; backend-specific
    n_tokens: int         # prompt length in model tokens (0 if unknown)
    n_options: int        # number of answer options


class Backend(Protocol):
    name: str             # registry id, e.g. "plumb-4b"
    kind: str             # "native-gpu" | "vllm" | "gguf-cpu" | "onnx-cpu"
    max_input_tokens: int
    max_batch_tokens: int  # scheduler packs encoded items up to this many tokens per run(); 0 = one item per run()

    def encode(self, state, question: dict) -> Encoded:
        """CPU, thread-safe, called in request threads; raises ValueError for bad/too-long input."""

    def run(self, batch: list[Encoded]) -> list[list[float]]:
        """Option probabilities (sum 1, option order of options(question)); called from ONE scheduler thread."""

    def options(self, question: dict) -> list[str]:
        """Option keys in model order."""
