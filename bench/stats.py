"""Metrics shared by the bench scripts (pure Python, no numpy)."""
from __future__ import annotations

import math
import random

FLOOR = 1e-12


def probs_dict(answer: dict, keys: list[str]) -> dict[str, float] | None:
    """Option probabilities from either answer shape: `probabilities` {key: p}, or `noul` = P(true) only."""
    p = answer.get("probabilities")
    if isinstance(p, dict) and p:
        out = {str(k): float(v) for k, v in p.items()}
        if set(keys) <= set(out):
            return {k: out[k] for k in keys}
    if answer.get("type") == "noul" and "noul" in answer:
        t = float(answer["noul"])
        return {"false": 1.0 - t, "true": t}
    return None


def argmax(p: dict[str, float]) -> str:
    return max(p, key=p.get)


def compare_pair(a: dict[str, float], b: dict[str, float]) -> dict:
    keys = list(a)
    d = [abs(a[k] - b[k]) for k in keys]
    dl = [abs(math.log(max(a[k], FLOOR)) - math.log(max(b[k], FLOOR))) for k in keys]
    return {"agree": argmax(a) == argmax(b), "mean_dp": sum(d) / len(d), "max_dp": max(d),
            "mean_dlogp": sum(dl) / len(dl), "max_dlogp": max(dl)}


def summarize(pairs: list[dict]) -> dict:
    if not pairs:
        return {"n": 0}
    n = len(pairs)
    return {"n": n, "top_choice_agreement": sum(p["agree"] for p in pairs) / n,
            "mean_abs_dp": sum(p["mean_dp"] for p in pairs) / n, "max_abs_dp": max(p["max_dp"] for p in pairs),
            "mean_abs_dlogp": sum(p["mean_dlogp"] for p in pairs) / n, "max_abs_dlogp": max(p["max_dlogp"] for p in pairs),
            "frac_max_dp_gt_0.005": sum(p["max_dp"] > 0.005 for p in pairs) / n}


def gold_scores(p: dict[str, float], gold: str) -> tuple[float, float]:
    """(argmax correct, probability assigned to the gold option)."""
    return float(argmax(p) == gold), p.get(gold, 0.0)


def paired_bootstrap(x: list[float], y: list[float], groups: list[str] | None = None, reps: int = 1000, seed: int = 16) -> dict:
    """Mean of (y - x) in percent points plus 95% CI. Resamples groups (requests) when given, else single items."""
    assert len(x) == len(y) and x
    d = [b - a for a, b in zip(x, y)]
    idx: dict[str, list[int]] = {}
    for i, g in enumerate(groups or range(len(d))):
        idx.setdefault(str(g), []).append(i)
    gs = list(idx.values())
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        tot, cnt = 0.0, 0
        for _ in gs:
            for i in gs[rng.randrange(len(gs))]:
                tot += d[i]
                cnt += 1
        means.append(100.0 * tot / cnt)
    means.sort()
    return {"delta_pp": 100.0 * sum(d) / len(d), "ci95": [means[int(0.025 * (reps - 1))], means[int(0.975 * (reps - 1))]],
            "reps": reps, "n": len(d)}


def quantile(sorted_vals: list[float], q: float) -> float:
    return sorted_vals[min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1) + 0.5))]
