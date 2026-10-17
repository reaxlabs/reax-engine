"""Offline timing of forward() vs forward_groups() on real DI-1000 requests (same packing budget).
usage: python tools/bench_shared.py MODEL DI_ROWS DI_IDS [n_rows] [until] [engine opts JSON]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
import torch
from pplx_engine import PplxEngine, Prompter

model, rows_p, ids_p = sys.argv[1:4]
n_rows = int(sys.argv[4]) if len(sys.argv) > 4 else 200
until = int(sys.argv[5]) if len(sys.argv) > 5 else 16
opts = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
ids = json.load(open(ids_p))["ids"]
step = len(ids) / 1000
sel = [ids[int(k * step)] for k in range(1000)][:n_rows]
rows = {}
for l in open(rows_p):
    r = json.loads(l)
    if r["_evaluation"]["run_id"] in set(sel):
        rows[r["_evaluation"]["run_id"]] = r
pr = Prompter(model)
reqs = []
for rid in sel:
    r = rows[rid]
    enc = [pr.ids(r["state"], q)[0] for q in r["questions"].values()]
    enc = [e for e in enc if len(e) <= 8192]
    if enc:
        reqs.append(enc)
def lcp(seqs):
    n = min(map(len, seqs)) - 1; k = 0
    while k < n and all(s[k] == seqs[0][k] for s in seqs): k += 1
    return k
groups = []
for enc in reqs:
    k = lcp(enc) if len(enc) > 1 else 0
    groups.append((enc[0][:k], [e[k:] for e in enc]) if k >= 64 else None)
ndec = sum(map(len, reqs))
print("requests", len(reqs), "decisions", ndec, "tokens full", sum(len(e) for r in reqs for e in r),
      "shared", sum(len(g[0]) + sum(map(len, g[1])) if g else sum(map(len, r)) for g, r in zip(groups, reqs)), flush=True)
eng = PplxEngine(model, **opts)
B = 16384
def full_steps():
    cur, n = [], 0
    for r in reqs:
        t = sum(map(len, r))
        if cur and n + t > B:
            yield cur; cur, n = [], 0
        cur.append(r); n += t
    if cur: yield cur
def run_full():
    for st in full_steps():
        flat = [e for r in st for e in r]
        eng.forward(torch.tensor([t for e in flat for t in e], device="cuda"), [len(e) for e in flat])
    torch.cuda.synchronize()
def run_shared():
    cur, n = [], 0
    def go(cur):
        gs = []
        for g, r in cur:
            gs += [g] if g else [(e, []) for e in r]
        eng.forward_groups(gs, until=until)
    for g, r in zip(groups, reqs):
        t = (len(g[0]) + sum(map(len, g[1]))) if g and until >= 64 else sum(map(len, r))
        if cur and n + t > B:
            go(cur); cur, n = [], 0
        cur.append((g, r)); n += t
    if cur: go(cur)
    torch.cuda.synchronize()
for name, fn in (("full", run_full), ("shared", run_shared), ("full", run_full), ("shared", run_shared)):
    t0 = time.time(); fn(); dt = time.time() - t0
    print(f"{name}: {dt:.2f}s  {ndec/dt:.2f} dec/s", flush=True)
if "--prof" in sys.argv:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
        run_shared()
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25))
