#!/usr/bin/env python3
"""Run the custom engine on val.jsonl: quality output (probs per decision) + offline throughput + c1 latency.

Usage: fast_run.py MODEL_DIR VAL OUT_JSONL [--linear bf16|fp8] [--attn auto|varlen|sdpa] [--budget 16384]
                   [--gva] [--fp8-skip 0.,63.] [--tp-reps 3] [--lat-n 50]
Throughput = all val decisions (672, mixed P300/DI lengths), sorted into packed steps of <= budget tokens,
repeated --tp-reps times; c1 latency = first --lat-n P300 decisions one at a time (tokenisation included).
"""
import argparse, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from pplx_engine import PplxEngine, Prompter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("val")
ap.add_argument("out")
ap.add_argument("--linear", default="bf16")
ap.add_argument("--attn", default="auto")
ap.add_argument("--budget", type=int, default=16384)
ap.add_argument("--gva", action="store_true")
ap.add_argument("--fp8-skip", default="")
ap.add_argument("--tp-reps", type=int, default=3)
ap.add_argument("--lat-n", type=int, default=50)
ap.add_argument("--fuse", default="1")
ap.add_argument("--compile", default="0")
ap.add_argument("--fused-quant", default="1")
ap.add_argument("--nvfp4", default="", help="comma list of layer-group suffixes in NVFP4: gu,down,in_qkvz,out,qkv,o")
ap.add_argument("--conv", default="fla")
ap.add_argument("--sort", action="store_true", help="sort by length before packing (throughput only)")
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.val) if l.strip()]
pr = Prompter(a.model)
t0 = time.time()
enc = [pr.ids(r["state"], r["question"]) for r in rows]
keep = [i for i, e in enumerate(enc) if len(e[0]) <= 8192]  # author refuses longer branches
rows, enc = [rows[i] for i in keep], [enc[i] for i in keep]
tok_s = time.time() - t0
t0 = time.time()
eng = PplxEngine(a.model, linear_mode=a.linear, attn_impl=a.attn, gva_native=a.gva,
                 fp8_skip=tuple(s for s in a.fp8_skip.split(",") if s), fuse=a.fuse, conv_impl=a.conv, compile=a.compile, fused_quant=a.fused_quant,
                 nvfp4_groups=tuple("." + g for g in a.nvfp4.split(",") if g))
load_s = time.time() - t0
print(f"tokenised {len(rows)} in {tok_s:.1f}s; engine loaded in {load_s:.0f}s; mem {torch.cuda.memory_allocated()/2**30:.1f} GiB",
      flush=True)


def steps(order, budget):
    cur, n = [], 0
    for i in order:
        L = len(enc[i][0])
        if cur and n + L > budget:
            yield cur
            cur, n = [], 0
        cur.append(i)
        n += L
    if cur:
        yield cur


def run_step(idx):
    ids = torch.tensor([t for i in idx for t in enc[i][0]], device="cuda")
    logits = eng.forward(ids, [len(enc[i][0]) for i in idx])
    return eng.probs(logits, [enc[i][1] for i in idx])


# warm-up (also triggers torch.compile for the dynamic-shape kernels)
run_step([0])
run_step(list(range(8)))
for st in list(steps(range(len(rows)), a.budget))[:2]:
    run_step(st)
torch.cuda.synchronize()
# ---- quality pass (original order)
probs = [None] * len(rows)
for st in steps(range(len(rows)), a.budget):
    for i, p in zip(st, run_step(st)):
        probs[i] = p
with open(a.out, "w") as f:
    for r, p, (ids, _) in zip(rows, probs, enc):
        f.write(json.dumps({"id": r["id"], "src": r["src"], "req": r["req"], "probs": p, "tokens": len(ids)}) + "\n")
# ---- throughput
order = sorted(range(len(rows)), key=lambda i: len(enc[i][0])) if a.sort else list(range(len(rows)))
plan = list(steps(order, a.budget))
torch.cuda.synchronize()
t0 = time.time()
for _ in range(a.tp_reps):
    for st in plan:
        run_step(st)
torch.cuda.synchronize()
tp_wall = time.time() - t0
n_dec = len(rows) * a.tp_reps
n_tok = sum(len(e[0]) for e in enc) * a.tp_reps
# ---- c1 latency (tokenise + forward + softmax), P300 rows
lat = []
p300 = [i for i, r in enumerate(rows) if r["src"] == "p300"][: a.lat_n]
for i in p300:
    torch.cuda.synchronize()
    t = time.perf_counter()
    ids, n = pr.ids(rows[i]["state"], rows[i]["question"])
    x = torch.tensor(ids, device="cuda")
    eng.probs(eng.forward(x, [len(ids)]), [n])
    lat.append(time.perf_counter() - t)
lat.sort()
info = {"nvfp4": a.nvfp4, "fused_quant": a.fused_quant, "compile": a.compile, "fuse": a.fuse, "conv": a.conv, "linear": a.linear, "attn": a.attn, "budget": a.budget, "gva": a.gva, "fp8_skip": a.fp8_skip,
        "decisions": n_dec, "tokens": n_tok, "wall_s": tp_wall, "dec_per_s": n_dec / tp_wall, "tok_per_s": n_tok / tp_wall,
        "steps_per_rep": len(plan), "c1_p50_s": statistics.median(lat), "c1_p95_s": lat[int(0.95 * (len(lat) - 1))],
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30, "load_s": load_s,
        "attn_backend": "varlen" if eng._varlen is not None else "sdpa-loop", "torch": torch.__version__}
json.dump(info, open(a.out + ".run.json", "w"), indent=1)
print(json.dumps(info))
