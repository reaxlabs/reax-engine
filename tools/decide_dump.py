#!/usr/bin/env python3
"""Dump and compare the answers of a decision server on one item set (stdlib only).

  decide_dump.py dump  --url http://127.0.0.1:8741 --items p300-input.jsonl --conc 16 --out a.json [--contract systemone|s1]
  decide_dump.py compare A.json B.json

Items: JSONL rows {"task_id"|"id", "state", "question"}. `dump` sends one question per request (closed loop, --conc requests in
flight, one pass) and stores the raw answer of every item; `compare` reports, over the items answered in both dumps, how many
have bit-identical answers, the maximum absolute probability difference, the top-choice agreement and the answers that
differ in shape (keys, type). Exit status 1 if any answered item differs.
"""
import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def dump(a):
    rows = [json.loads(line) for line in open(a.items) if line.strip()]
    path = "/v1/systemone" if a.contract == "systemone" else "/internal/v1/decide"
    headers = {"Content-Type": "application/json"}
    if os.environ.get("S1M_INTERNAL_TOKEN"):
        headers["X-S1M-Internal-Token"] = os.environ["S1M_INTERNAL_TOKEN"]
    out, lock = {}, threading.Lock()

    def one(r):
        key = str(r.get("task_id", r.get("id")))
        body = {"state": r["state"], "questions": {"q": r["question"]}}
        if a.model:
            body["model"] = a.model
        req = urllib.request.Request(a.url.rstrip("/") + path, data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                rec = {"http": resp.status, "answer": json.load(resp)["answers"]["q"]}
        except urllib.error.HTTPError as e:
            rec = {"http": e.code}
        with lock:
            out[key] = rec
    with ThreadPoolExecutor(a.conc) as ex:
        list(ex.map(one, rows))
    json.dump(out, open(a.out, "w"))
    print(json.dumps({"items": len(rows), "answered": sum(v["http"] == 200 for v in out.values()), "out": a.out}))


def vec(ans):
    """Option probabilities of an answer ({key: p}); noul answers of some models carry only the `noul` value."""
    return ans["probabilities"] if "probabilities" in ans else {"noul": ans["noul"]}


def compare(a):
    x, y = json.load(open(a.a)), json.load(open(a.b))
    both = [k for k in x if k in y and x[k]["http"] == 200 and y[k]["http"] == 200]
    ident = shape = top = 0
    worst = 0.0
    for k in both:
        p, q = vec(x[k]["answer"]), vec(y[k]["answer"])
        if list(p) != list(q) or x[k]["answer"]["type"] != y[k]["answer"]["type"]:
            shape += 1
            continue
        worst = max(worst, max(abs(p[i] - q[i]) for i in p))
        ident += x[k]["answer"] == y[k]["answer"]
        top += max(p, key=p.get) == max(q, key=q.get)
    n = len(both) - shape
    http_diff = sum(1 for k in x if k in y and x[k]["http"] != y[k]["http"])
    res = {"n": n, "identical": ident, "max_abs_diff": worst, "top_agreement": round(top / n, 5) if n else None,
           "shape_mismatch": shape, "status_mismatch": http_diff}
    print(json.dumps(res))
    return 0 if ident == n and not shape and not http_diff else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--url", required=True)
    d.add_argument("--items", required=True)
    d.add_argument("--conc", type=int, default=1)
    d.add_argument("--out", required=True)
    d.add_argument("--contract", default="systemone", choices=["systemone", "s1"])
    d.add_argument("--model", default="")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    sys.exit(compare(a) if a.cmd == "compare" else dump(a))


if __name__ == "__main__":
    main()
