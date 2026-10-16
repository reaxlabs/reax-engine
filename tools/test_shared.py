"""Correctness test of PplxEngine.forward_groups (shared-state path).
1. causal=True engine: shared state is EXACT under causal attention, so forward_groups(prefix,[suffixes]) must match
   forward() on the full sequences (checks conv tail, DeltaNet state hand-off, rope offsets, key gathering).
2. noncausal engine, groups without suffixes == forward() (ordinary decisions).
3. noncausal, real multi-question requests: report agreement of shared vs full (the approximation itself).
usage: python tools/test_shared.py MODEL VAL [engine opts as JSON]"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
import torch
from pplx_engine import PplxEngine, Prompter

model, val = sys.argv[1], sys.argv[2]
opts = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
mode = sys.argv[4] if len(sys.argv) > 4 else "all"
pr = Prompter(model)
rows = [json.loads(l) for l in open(val)]
by = {}
for r in rows:
    by.setdefault(r.get("req") or r["id"], []).append(r)
multi = [v for v in by.values() if len(v) >= 2][:40]
singles = [v[0] for v in by.values()][:24]
print("multi-question requests", len([v for v in by.values() if len(v) >= 2]), "used", len(multi))

def lcp(seqs):
    n = min(map(len, seqs)) - 1
    k = 0
    while k < n and all(s[k] == seqs[0][k] for s in seqs):
        k += 1
    return k

def groups_of(reqs):
    gs, flat = [], []
    for v in reqs:
        enc = [pr.ids(r["state"], r["question"])[0] for r in v]
        k = lcp(enc)
        gs.append((enc[0][:k], [e[k:] for e in enc]))
        flat += enc
    return gs, flat

def full(eng, flat, bs=8):
    out = []
    for a in range(0, len(flat), bs):
        b = flat[a:a + bs]
        out.append(eng.forward(torch.tensor([t for x in b for t in x], device="cuda"), [len(x) for x in b]))
    return torch.cat(out)

def shared(eng, gs, bs=4, until=None):
    return torch.cat([eng.forward_groups(gs[a:a + bs], until=until) for a in range(0, len(gs), bs)])

def cmp(name, a, b):
    pa, pb = a.float().softmax(-1), b.float().softmax(-1)
    d = (pa - pb).abs().max(-1).values
    print(f"{name}: n={a.shape[0]} top1 agree {(a.argmax(-1) == b.argmax(-1)).float().mean():.4f} max|dp| {d.max():.5f} mean|dp| {d.mean():.6f} max|dlogit| {(a-b).abs().max():.4f}", flush=True)

gs, flat = groups_of(multi)
print("tokens full", sum(map(len, flat)), "shared", sum(len(p) + sum(map(len, ss)) for p, ss in gs))
if mode in ("all", "causal"):
    eng = PplxEngine(model, causal=True, **opts)
    f8 = full(eng, flat)
    cmp("causal noise floor: full bs8 vs full bs3", f8, full(eng, flat, 3))
    cmp("causal shared vs full (must be at the noise floor)", f8, shared(eng, gs))
    cmp("causal shared bs1 vs full", f8, shared(eng, gs, 1))
    del eng; torch.cuda.empty_cache()
if mode in ("all", "noncausal"):
    eng = PplxEngine(model, **opts)
    sg = [(pr.ids(r["state"], r["question"])[0], []) for r in singles]
    cmp("noncausal singles via groups vs forward (must be ~0)", full(eng, [p for p, _ in sg]), shared(eng, sg))
    cmp("noncausal until=0 (unshare at once) vs full: must be noise", full(eng, flat), shared(eng, gs, until=1))
    ff = full(eng, flat)
    cmp("noncausal noise floor: full bs8 vs bs3", ff, full(eng, flat, 3))
    for u in [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "64").split(",")]:
        cmp(f"noncausal shared until layer {u} vs full", ff, shared(eng, gs, until=u))
