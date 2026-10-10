#!/usr/bin/env python3
"""Serving memory at concurrency 16 with ~8k-token single-question requests. Run with a server already ready on :8741.
Samples nvidia-smi before and during the load; prints ready MiB, serving peak MiB, tokens/request, latency."""
import json, subprocess, threading, time, urllib.request

def mem():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])

ready = mem()
peak, stop = [ready], [False]
def sampler():
    while not stop[0]:
        peak.append(mem()); time.sleep(0.3)
state = "Log line: " + " ".join(f"event {i} user{i%97} status ok latency {i%13}ms" for i in range(int(__import__("os").environ.get("EVENTS", "600"))))
body = json.dumps({"model": "p", "state": state, "questions": {"q": {"type": "choice", "instructions": "Is anything broken?",
        "criteria": {"yes": "something failed", "no": "all fine", "unclear": "not enough info"}}}}).encode()
lat, tok = [], [0]
def w():
    for _ in range(6):
        t = time.perf_counter()
        req = urllib.request.Request("http://127.0.0.1:8741/v1/systemone", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            tok[0] = json.load(r)["usage"]["input_tokens"]
        lat.append(time.perf_counter() - t)
s = threading.Thread(target=sampler); s.start()
th = [threading.Thread(target=w) for _ in range(16)]; [t.start() for t in th]; [t.join() for t in th]
stop[0] = True; s.join()
lat.sort()
print(json.dumps({"ready_mib": ready, "serving_peak_mib": max(peak), "tokens_per_request": tok[0], "requests": len(lat),
                  "p50_s": round(lat[len(lat) // 2], 2), "p95_s": round(lat[int(.95 * len(lat))], 2)}))
