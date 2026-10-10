#!/usr/bin/env python3
"""Closed-loop load test against /v1/systemone: C concurrent clients replay val.jsonl requests (grouped by `req`,
P300 = 1 question, DI = up to 8 questions sharing a state) for D seconds (or one pass at c1).
Prints decisions/s, requests/s, latency p50/p95 per request and per single-question request.
Usage: loadgen.py VAL URL --conc 1,8,32,64 [--seconds 60] [--src all|p300|di] [--out FILE]"""
import argparse, itertools, json, statistics, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("val")
ap.add_argument("url")
ap.add_argument("--conc", default="1,8,32,64")
ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--src", default="all")
ap.add_argument("--out")
a = ap.parse_args()
reqs = {}
for l in open(a.val):
    if l.strip():
        r = json.loads(l)
        if a.src in ("all", r["src"]):
            reqs.setdefault(r["req"], {"state": r["state"], "questions": {}})["questions"][r["id"].split("::")[-1]] = r["question"]
bodies = [json.dumps({"model": "pplx", **v}).encode() for v in reqs.values()]
nq = [len(v["questions"]) for v in reqs.values()]


def post(i):
    req = urllib.request.Request(a.url + "/v1/systemone", data=bodies[i], headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


results = []
for c in [int(x) for x in a.conc.split(",")]:
    lock = threading.Lock()
    it = itertools.count()
    lat, done_q, codes = [], [0], {}
    stop = time.perf_counter() + a.seconds

    def worker():
        while time.perf_counter() < stop:
            i = next(it) % len(bodies)
            t = time.perf_counter()
            code = post(i)
            dt = time.perf_counter() - t
            with lock:
                codes[code] = codes.get(code, 0) + 1
                if code == 200:
                    lat.append((dt, nq[i]))
                    done_q[0] += nq[i]

    t0 = time.perf_counter()
    th = [threading.Thread(target=worker) for _ in range(c)]
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.perf_counter() - t0
    all_l = sorted(x for x, _ in lat)
    one_l = sorted(x for x, n in lat if n == 1)
    res = {"conc": c, "wall_s": round(wall, 1), "requests": len(lat), "decisions": done_q[0], "dec_per_s": round(done_q[0] / wall, 2),
           "req_p50_s": round(statistics.median(all_l), 3), "req_p95_s": round(all_l[int(0.95 * (len(all_l) - 1))], 3),
           "single_q_p50_s": round(statistics.median(one_l), 3) if one_l else None,
           "single_q_p95_s": round(one_l[int(0.95 * (len(one_l) - 1))], 3) if one_l else None, "codes": codes}
    print(json.dumps(res), flush=True)
    results.append(res)
if a.out:
    json.dump(results, open(a.out, "w"), indent=1)
