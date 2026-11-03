"""Helpers shared by the registry-slice backends: the s1 sidecar answer shape and small numeric utilities."""
from __future__ import annotations

import math
from collections.abc import Sequence


def softmax_t(logits: Sequence[float], temperature: float) -> list[float]:
    """softmax(logits / T) in float64 (max-subtracted)."""
    z = [float(x) / temperature for x in logits]
    top = max(z)
    e = [math.exp(x - top) for x in z]
    s = sum(e)
    return [v / s for v in e]


def s1_answer(qtype: str, keys: Sequence[str], probs: Sequence[float]) -> dict:
    """Answer in the s1 sidecar shape (POST /internal/v1/decide): probabilities per option key, confidence = max p,
    plus `choice` (argmax key), `noul` (P(true)) or `score` (expected level index)."""
    if len(keys) != len(probs):
        raise ValueError("need one probability per option key")
    values = [float(p) for p in probs]
    if any(not math.isfinite(p) or p < 0 for p in values) or sum(values) <= 0:
        raise ValueError("probabilities must be finite, non-negative and not all zero")
    total = sum(values)
    values = [p / total for p in values]
    dist = dict(zip(keys, values))
    out = {"type": qtype, "probabilities": dist, "confidence": max(values)}
    if qtype == "noul":
        out["noul"] = dist["true"]
    elif qtype == "choice":
        out["choice"] = keys[max(range(len(values)), key=values.__getitem__)]
    elif qtype == "score":
        out["score"] = sum(i * p for i, p in enumerate(values))
    else:
        raise ValueError(f"unknown question type {qtype!r}")
    return out
