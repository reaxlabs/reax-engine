#!/usr/bin/env python3
"""Merge the raw outputs of one pod_validate.sh run into results/<model>/<date>.json (stdlib only).

  python -m bench.summarize --model ID --spec SPEC.json --env ENV.json --agreement P300.json [--agreement-di DI.json] \
      --ref-load REF.json --cand-load CAND.json [--ref-load-di ...] [--cand-load-di ...] --out results/ID/DATE.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _load(p):
    return json.loads(Path(p).read_text()) if p and Path(p).exists() else None


def speedups(ref, cand):
    """Per concurrency level present in both: candidate / reference decisions per second."""
    if not ref or not cand:
        return {}
    r = {x["concurrency"]: x for x in ref["levels"]}
    return {str(x["concurrency"]): round(x["decisions_per_s"] / r[x["concurrency"]]["decisions_per_s"], 3)
            for x in cand["levels"] if x["concurrency"] in r and r[x["concurrency"]]["decisions_per_s"]}


def merge(a) -> dict:
    out = {"model": a.model, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "spec": _load(a.spec), "env": _load(a.env),
           "agreement": {"p300": _load(a.agreement), "di": _load(a.agreement_di)},
           "throughput": {"p300": {"reference": _load(a.ref_load), "reax": _load(a.cand_load),
                                   "reax_over_reference_decisions_per_s": speedups(_load(a.ref_load), _load(a.cand_load))},
                          "di_grouped": {"reference": _load(a.ref_load_di), "reax": _load(a.cand_load_di),
                                         "reax_over_reference_decisions_per_s": speedups(_load(a.ref_load_di), _load(a.cand_load_di))}}}
    p = (out["agreement"]["p300"] or {}).get("paired") or {}
    out["headline"] = {"p300_top_choice_agreement": p.get("top_choice_agreement"), "p300_mean_abs_dp": p.get("mean_abs_dp"),
                       "p300_max_abs_dp": p.get("max_abs_dp"), "speedup_by_concurrency_p300": out["throughput"]["p300"]["reax_over_reference_decisions_per_s"]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ("model", "out"):
        ap.add_argument(f"--{n}", required=True)
    for n in ("spec", "env", "agreement", "agreement-di", "ref-load", "cand-load", "ref-load-di", "cand-load-di"):
        ap.add_argument(f"--{n}")
    a = ap.parse_args(argv)
    res = merge(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["headline"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
