#!/usr/bin/env python3
"""Closed-loop throughput / latency test: C concurrent clients replay evaluation items for D seconds.

  python -m bench.loadgen --url http://127.0.0.1:8002 --set p300 --data-dir DATA --concurrency 1,16 --seconds 60 \
      [--contract systemone|s1] [--grouped] [--out loadgen.json]

Prints decisions/s, requests/s and latency p50/p95 per concurrency level (and for single-question requests).
Default is one question per request (P300-style). --grouped sends all questions of a state together (DI-style
/v1/systemone requests that share a state). Decisions that error are counted separately and excluded from latency.
Ported from the legacy engine/loadgen.py (closed loop, one pass at c1 is just --seconds long).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import sets, stats  # noqa: E402


def build_requests(items, grouped, model):
    if grouped:
        groups: dict[str, dict] = {}
        for it in items:
            g = groups.setdefault(it.req or it.key, {"state": it.state, "questions": {}})
            g["questions"][it.qid] = it.question
        reqs = list(groups.values())
    else:
        reqs = [{"state": it.state, "questions": {it.qid: it.question}} for it in items]
    return [(json.dumps({"model": model, **r}).encode(), len(r["questions"])) for r in reqs]


def run_level(url, path, headers, bodies, conc, seconds):
    lock, it = threading.Lock(), itertools.count()
    lat, codes, decisions = [], {}, [0]
    stop = time.perf_counter() + seconds

    def post(i):
        req = urllib.request.Request(url + path, data=bodies[i][0], headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                r.read()
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
        except Exception:  # noqa: BLE001
            return 0

    def worker():
        while time.perf_counter() < stop:
            i = next(it) % len(bodies)
            t = time.perf_counter()
            code = post(i)
            dt = time.perf_counter() - t
            with lock:
                codes[code] = codes.get(code, 0) + 1
                if code == 200:
                    lat.append((dt, bodies[i][1]))
                    decisions[0] += bodies[i][1]

    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker) for _ in range(conc)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    wall = time.perf_counter() - t0
    allv, one = sorted(x for x, _ in lat), sorted(x for x, n in lat if n == 1)
    res = {"concurrency": conc, "wall_s": round(wall, 1), "requests": len(lat), "decisions": decisions[0],
           "decisions_per_s": round(decisions[0] / wall, 2), "requests_per_s": round(len(lat) / wall, 2), "http_codes": codes}
    if allv:
        res.update(req_p50_ms=round(1000 * statistics.median(allv), 1), req_p95_ms=round(1000 * stats.quantile(allv, 0.95), 1),
                   req_max_ms=round(1000 * allv[-1], 1))
    if one:
        res.update(single_q_p50_ms=round(1000 * statistics.median(one), 1), single_q_p95_ms=round(1000 * stats.quantile(one, 0.95), 1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--set", dest="set_name", default="p300", choices=sets.SETS)
    ap.add_argument("--data-dir")
    ap.add_argument("--data-file")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--contract", default="systemone", choices=["systemone", "s1"])
    ap.add_argument("--model", default="model")
    ap.add_argument("--concurrency", default="1,16")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--grouped", action="store_true")
    ap.add_argument("--max-chars", type=int, default=0,
                    help="drop items whose state is longer than this many JSON characters (0 = keep all); models without an input cap "
                         "(d3) otherwise spend the run on a handful of 100k-character P300 items")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    items = sets.load(a.set_name, a.data_dir, a.limit, a.data_file)
    if a.max_chars:
        items = [it for it in items if len(json.dumps(it.state)) <= a.max_chars]
    if a.contract == "s1" and a.grouped:
        raise SystemExit("--grouped needs --contract systemone (the s1 sidecar contract takes one question per call)")
    bodies = build_requests(items, a.grouped, a.model)
    headers = {"Content-Type": "application/json"}
    if a.contract == "s1" and os.environ.get("S1M_INTERNAL_TOKEN"):
        headers["X-S1M-Internal-Token"] = os.environ["S1M_INTERNAL_TOKEN"]
    path = "/v1/systemone" if a.contract == "systemone" else "/internal/v1/decide"
    results = []
    for c in (int(x) for x in a.concurrency.split(",")):
        r = run_level(a.url.rstrip("/"), path, headers, bodies, c, a.seconds)
        print(json.dumps(r), flush=True)
        results.append(r)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"set": a.set_name, "contract": a.contract, "grouped": a.grouped, "max_chars": a.max_chars, "n_requests": len(bodies),
                                           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "levels": results}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
