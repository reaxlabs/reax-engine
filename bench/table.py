#!/usr/bin/env python3
"""Markdown tables from results/<model>/<run>.json files written by bench/summarize.py (stdlib only).

  python -m bench.table results/*/*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _f(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def _agree(a):
    p = (a or {}).get("paired") or {}
    if not p:
        return "-"
    return (f"{100 * p['top_choice_agreement']:.2f}% / {p['mean_abs_dp']:.4f} / {p['max_abs_dp']:.3f} / "
            f"{p['mean_abs_dlogp']:.3f} / {p['max_abs_dlogp']:.2f} (n={p['n']})")


def _acc(a):
    d = ((a or {}).get("gold") or {}).get("accuracy_delta")
    if not d:
        return "-"
    return f"{d['delta_pp']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}] pp (n={d['n']})"


def _levels(block):
    return {x["concurrency"]: x for x in ((block or {}).get("levels") or [])}


def _tp(res, key):
    t = res["throughput"][key]
    ref, rx = _levels(t.get("reference")), _levels(t.get("reax"))
    out = []
    for c in sorted(set(ref) & set(rx)):
        out.append(f"c{c}: {ref[c]['decisions_per_s']:g} -> {rx[c]['decisions_per_s']:g} ({rx[c]['decisions_per_s'] / ref[c]['decisions_per_s']:.2f}x)"
                   if ref[c]["decisions_per_s"] else f"c{c}: -")
    return "; ".join(out) or "-"


def _p50(res):
    t = res["throughput"]["p300"]
    ref, rx = _levels(t.get("reference")), _levels(t.get("reax"))
    c = 1
    if c in ref and c in rx:
        return f"{ref[c].get('req_p50_ms', '-')} -> {rx[c].get('req_p50_ms', '-')} ms"
    return "-"


def rows(paths):
    for p in sorted(paths):
        r = json.loads(Path(p).read_text())
        env = r.get("env") or {}
        tag = Path(p).stem
        yield r, env, tag


def main(argv=None):
    paths = argv if argv is not None else sys.argv[1:]
    print("| run | GPU / date | agreement P300: top-1 / mean abs dp / max abs dp / mean abs dlogp / max abs dlogp | agreement DI-1000 (same) | accuracy delta P300 (vs reference) | accuracy delta DI | P300 dec/s ref -> REAX | DI dec/s ref -> REAX | c1 p50 | peak GPU MiB ref / REAX |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r, env, tag in rows(paths):
        a = r.get("agreement") or {}
        mem = env.get("peak_gpu_mem_mib") or {}
        print(f"| {r['model']} `{tag}` | {env.get('gpu', '-').split(',')[0]}, {r['utc'][:10]} | {_agree(a.get('p300'))} | {_agree(a.get('di'))} | "
              f"{_acc(a.get('p300'))} | {_acc(a.get('di'))} | {_tp(r, 'p300')} | {_tp(r, 'di_grouped')} | {_p50(r)} | "
              f"{mem.get('reference', '-')} / {mem.get('reax', '-')} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
