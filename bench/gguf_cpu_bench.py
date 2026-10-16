"""Load generator and agreement check for decision servers (``POST /internal/v1/decide``).

  run:      python bench/gguf_cpu_bench.py run --url http://127.0.0.1:8103 --items items.jsonl --out out.json \
                --concurrency 4 [--limit N] [--token T] [--warmup 4]
  compare:  python bench/gguf_cpu_bench.py compare ref.json test.json

``items.jsonl``: one ``{"id", "state", "question"}`` per line (s1 contract, one question per request).
``run`` writes per-item probabilities, latency (ms) and input tokens plus the throughput summary; ``compare`` reports
top-choice agreement and the mean / max |log p| and |p| differences of two result files. Standard library only.
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse


def _post(conn, token, item):
    body = json.dumps({"state": item["state"], "questions": {"q": item["question"]}}).encode()
    hdr = {"content-type": "application/json", "x-s1m-internal-token": token}
    t0 = time.perf_counter()
    conn.request("POST", "/internal/v1/decide", body, hdr)
    resp = conn.getresponse()
    raw = resp.read()
    ms = (time.perf_counter() - t0) * 1000
    if resp.status == 400:  # e.g. input over the token limit: recorded per item, same on every server
        return {"error": json.loads(raw)["error"]["message"]}, 0, ms
    if resp.status != 200:
        raise RuntimeError(f"{resp.status} {raw[:200]!r}")
    out = json.loads(raw)
    return out["answers"]["q"], out.get("usage", {}).get("input_tokens", 0), ms


def run(a):
    items = [json.loads(line) for line in open(a.items)]
    if a.limit:
        items = items[: a.limit]
    u = urlparse(a.url)
    local = threading.local()

    def conn():
        if not hasattr(local, "c"):
            local.c = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
        return local.c

    for it in items[: a.warmup]:
        _post(conn(), a.token, it)
    res: dict = {}

    def one(it):
        ans, ntok, ms = _post(conn(), a.token, it)
        if "error" in ans:
            res[it["id"]] = {"error": ans["error"]}
            return
        res[it["id"]] = {"probs": ans["probabilities"], "ms": ms, "tokens": ntok, "type": ans["type"]}

    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(one, items))
    wall = time.perf_counter() - t0
    errors = sum("error" in r for r in res.values())
    res_ok = {k: v for k, v in res.items() if "error" not in v}
    lat = sorted(r["ms"] for r in res_ok.values())
    pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]  # noqa: E731
    summ = {"n": len(items), "errors_400": errors, "concurrency": a.concurrency, "wall_s": round(wall, 2),
            "decisions_per_s": round(len(res_ok) / wall, 3), "p50_ms": round(pct(0.50), 1),
            "p95_ms": round(pct(0.95), 1), "mean_ms": round(statistics.mean(lat), 1),
            "mean_tokens": round(statistics.mean(r["tokens"] for r in res_ok.values()), 1), "label": a.label}
    json.dump({"summary": summ, "items": res}, open(a.out, "w"))
    print(json.dumps(summ))


def compare(a):
    ref, test = (json.load(open(p))["items"] for p in (a.ref, a.test))
    err_mismatch = sum(("error" in ref[i]) != ("error" in test.get(i, {"error": 1})) for i in ref)
    ids = [i for i in ref if i in test and "error" not in ref[i] and "error" not in test[i]]
    agree = 0
    dl, dp, exact = [], [], 0
    for i in ids:
        pr, pt = ref[i]["probs"], test[i]["probs"]
        agree += max(pr, key=pr.get) == max(pt, key=pt.get)
        d = [abs(math.log(max(pr[k], 1e-12)) - math.log(max(pt[k], 1e-12))) for k in pr]
        dl.extend(d)
        dp.extend(abs(pr[k] - pt[k]) for k in pr)
        exact += all(pr[k] == pt[k] for k in pr)
    out = {"n": len(ids), "error_mismatch": err_mismatch, "top_choice_agreement": round(agree / len(ids), 6), "bit_identical_items": exact,
           "mean_abs_logp_diff": sum(dl) / len(dl), "max_abs_logp_diff": max(dl),
           "mean_abs_p_diff": sum(dp) / len(dp), "max_abs_p_diff": max(dp)}
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--items", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--concurrency", type=int, default=1)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--warmup", type=int, default=4)
    r.add_argument("--token", default="bench")
    r.add_argument("--label", default="")
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("test")
    a = ap.parse_args()
    {"run": run, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
