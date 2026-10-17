#!/usr/bin/env python3
"""Collect run speed/latency/quality for the v1.2 job into runs/summary.json and print a table.
usage: summary12.py [run ...]   (runs/ must hold fetched run dirs; compare-vs-*.json from tools/score.sh)"""
import json, sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "runs"


def get(p, *keys, default=None):
    try:
        d = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return default
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d if d is not None else default


def main():
    runs = sys.argv[1:] or sorted(p.name for p in R.iterdir() if (p / "di-native" / "run.json").exists())
    out = {}
    for t in runs:
        d = R / t
        row = {"args": (d / "args.txt").read_text().strip() if (d / "args.txt").exists() else "",
               "di_dec_s": get(d / "di-native" / "run.json", "decisions_per_s"),
               "di_status": get(d / "di-native" / "run.json", "status"),
               "c1_p50_s": get(d / "latency.json", "p50_s"), "c1_p95_s": get(d / "latency.json", "p95_s"),
               "p300_items_s": get(d / "p300.jsonl.run.json", "items_per_s")}
        tok = 0
        for line in open(d / "di-native" / "results.jsonl"):
            tok += ((json.loads(line).get("response") or {}).get("usage") or {}).get("input_tokens", 0) or 0
        row["di_input_tokens"] = tok
        for c in d.glob("compare-vs-*.json"):
            row[c.stem] = c.name
        out[t] = row
        print(f"{t:24s} {row['di_dec_s'] or 0:7.2f} dec/s  c1 p50 {1000 * (row['c1_p50_s'] or 0):5.0f} ms  "
              f"p95 {1000 * (row['c1_p95_s'] or 0):5.0f} ms  tokens {tok}")
    (R / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
