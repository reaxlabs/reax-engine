#!/usr/bin/env python3
"""Reference = the author's own code path (autojev.model.DecisionModel, transformers + SDPA, bf16, noncausal hook).

Runs every decision of val.jsonl exactly like the author's server: one request at a time, its questions in chunks of
`--batch` (server default 8, left-padded). Writes per-decision fp32 probabilities over the question's options plus
per-request wall time, so it is both the quality anchor and the "author server" throughput baseline.
Usage: ref_run.py MODEL_DIR VAL_JSONL OUT_JSONL [--batch 8] [--limit N]
"""
import argparse, json, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("val")
ap.add_argument("out")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--offline", action="store_true", help="ignore request boundaries: length-sorted batches of --batch")
a = ap.parse_args()
sys.path.insert(0, str(Path(a.model) / "source" / "src"))
import torch  # noqa: E402
from autojev.model import DecisionModel  # noqa: E402

rows = [json.loads(l) for l in open(a.val) if l.strip()]
if a.limit:
    rows = rows[: a.limit]
reqs = {}
for r in rows:
    reqs.setdefault(r["req"], []).append(r)
t0 = time.time()
model = DecisionModel(a.model, device="cuda")
load_s = time.time() - t0
print(f"loaded in {load_s:.0f}s; peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
# warm-up (kernels, autotune) on the first request, not timed
model.predict([{"state": rows[0]["state"], "question": rows[0]["question"]}], batch_size=1)
torch.cuda.synchronize()
n_dec, t_all = 0, time.time()
if a.offline:
    dec = [{"state": r["state"], "question": r["question"]} for r in rows]
    lens = [len(model.prepare([d], max_length=10**9).inputs["input_ids"][0]) for d in dec]
    order = sorted((i for i in range(len(dec)) if lens[i] <= 8192), key=lens.__getitem__)
    torch.cuda.synchronize()
    t_all = time.time()
    out = [None] * len(dec)
    for s0 in range(0, len(order), a.batch):
        idx = order[s0:s0 + a.batch]
        for i, p in zip(idx, model.predict([dec[i] for i in idx], batch_size=a.batch)):
            out[i] = p
    torch.cuda.synchronize()
    wall = time.time() - t_all
    with open(a.out, "w") as f:
        for r, p, nt in zip(rows, out, lens):
            if p is None:
                continue
            f.write(json.dumps({"id": r["id"], "src": r["src"], "req": r["req"], "probs": p, "tokens": nt}) + "\n")
    info = {"mode": "offline", "decisions": len(dec), "wall_s": wall, "dec_per_s": len(dec) / wall, "batch": a.batch,
            "tokens": sum(lens), "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30}
    json.dump(info, open(a.out + ".run.json", "w"), indent=1)
    print(json.dumps(info))
    sys.exit(0)
with open(a.out, "w") as f:
    for k, (rid, items) in enumerate(reqs.items()):
        dec = [{"state": it["state"], "question": it["question"]} for it in items]
        torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            probs = model.predict(dec, batch_size=a.batch)
        except ValueError:  # author refuses > 8192 tokens per question branch (server: 422)
            probs = []
            for d in dec:
                try:
                    probs.append(model.predict([d], batch_size=1)[0])
                except ValueError:
                    probs.append(None)
        torch.cuda.synchronize()
        dt_s = time.perf_counter() - t
        toks = [len(model.prepare([d], max_length=10**9).inputs["input_ids"][0]) for d in dec]
        for it, p, nt in zip(items, probs, toks):
            if p is None:
                continue
            f.write(json.dumps({"id": it["id"], "src": it["src"], "req": rid, "probs": p, "tokens": nt,
                                "req_s": dt_s, "req_n": len(items)}) + "\n")
        n_dec += len(items)
        if k % 50 == 0:
            print(f"{k}/{len(reqs)} requests, {n_dec} decisions, {n_dec/(time.time()-t_all):.2f} dec/s", flush=True)
wall = time.time() - t_all
info = {"decisions": n_dec, "requests": len(reqs), "wall_s": wall, "dec_per_s": n_dec / wall, "batch": a.batch,
        "load_s": load_s, "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
json.dump(info, open(a.out + ".run.json", "w"), indent=1)
print(json.dumps(info))
