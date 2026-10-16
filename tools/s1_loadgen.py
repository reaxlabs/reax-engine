#!/usr/bin/env python3
"""Closed-loop load generator for the s1 runtime sidecar contract (POST /internal/v1/decide), TCP or unix socket.

Each of C workers sends one request at a time (as the engine competition's timed runs do); items cycle through a JSONL
of {"state","question"} rows. Prints decisions/s and latency percentiles per concurrency; --dump writes the answers of
the first pass (probabilities per item) for s1_compare.py.

  s1_loadgen.py ITEMS.jsonl --uds /run/reax/engine.sock --token-file /run/reax/token --conc 1,16,64 --seconds 20
Needs aiohttp.
"""
import argparse
import asyncio
import json
import statistics
import time

import aiohttp


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("--uds", default="")
    ap.add_argument("--url", default="http://127.0.0.1:8101")
    ap.add_argument("--token", default="")
    ap.add_argument("--token-file", default="")
    ap.add_argument("--conc", default="1,16,64")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--warmup", type=int, default=64, help="untimed requests before the first run")
    ap.add_argument("--dump", default="", help="write the probabilities of one pass over the items (concurrency 16)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    token = a.token or (open(a.token_file).read().strip() if a.token_file else "")
    rows = [json.loads(line) for line in open(a.items) if line.strip()]
    bodies = [json.dumps({"state": r["state"], "questions": {"q": r["question"]}}, ensure_ascii=False).encode() for r in rows]
    headers = {"X-S1M-Internal-Token": token, "Content-Type": "application/json"}
    conn = aiohttp.UnixConnector(path=a.uds, limit=0) if a.uds else aiohttp.TCPConnector(limit=0)
    base = "http://engine" if a.uds else a.url
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=120), trust_env=False) as s:
        async def post(i):
            async with s.post(base + "/internal/v1/decide", data=bodies[i % len(bodies)], headers=headers) as r:
                raw = await r.read()
                return r.status, raw

        async def run_n(c, total=None, seconds=None, start=0):
            idx = iter(range(start, start + (total or 10**9)))
            lat, errs, out = [], 0, {}
            stop = time.perf_counter() + seconds if seconds else None

            async def worker():
                nonlocal errs
                while stop is None or time.perf_counter() < stop:
                    try:
                        i = next(idx)
                    except StopIteration:
                        return
                    t = time.perf_counter()
                    code, raw = await post(i)
                    lat.append(time.perf_counter() - t)
                    if code != 200:
                        errs += 1
                    else:
                        out[i] = json.loads(raw)["answers"]["q"]["probabilities"]
            t0 = time.perf_counter()
            await asyncio.gather(*(worker() for _ in range(c)))
            return time.perf_counter() - t0, lat, errs, out

        if a.warmup:
            await run_n(16, total=a.warmup)
        results = []
        if a.dump:
            _, _, _, out = await run_n(16, total=len(bodies))
            json.dump({str(i): p for i, p in out.items()}, open(a.dump, "w"))
        offset = 10_000
        for c in [int(x) for x in a.conc.split(",")]:
            wall, lat, errs, _ = await run_n(c, seconds=a.seconds, start=offset)
            offset += 1_000_000
            q = sorted(lat)
            res = {"conc": c, "decisions": len(lat), "wall_s": round(wall, 2), "dec_per_s": round(len(lat) / wall, 1),
                   "p50_ms": round(statistics.median(q) * 1000, 2), "p95_ms": round(q[int(0.95 * (len(q) - 1))] * 1000, 2),
                   "p99_ms": round(q[int(0.99 * (len(q) - 1))] * 1000, 2), "errors": errs}
            print(json.dumps(res), flush=True)
            results.append(res)
        if a.out:
            json.dump(results, open(a.out, "w"), indent=1)


asyncio.run(main())
