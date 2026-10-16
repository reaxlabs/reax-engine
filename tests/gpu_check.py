"""GPU parity check: native Plumb engine vs the author's jevk5 (HF forward) on a JSONL of {"state","question"} items.
Reports top-choice agreement and the competition's logit distance for: eager single, graph single, packed batches of 16.
usage (reference image, GPU): gpu_check.py MODEL_DIR ITEMS.jsonl [N]"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jevk5.runtime import JevK5, decision_options  # noqa: E402
from plumb_engine import GRAPH_SHAPES, PlumbBackend  # noqa: E402

T = 2.07
model_dir, items_path = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
rows = [json.loads(line) for line in open(items_path)][:N]
for r in rows:
    r["question"].setdefault("instructions", "")


def dist(a, b):
    ca, cb = [math.log(x) for x in a], [math.log(x) for x in b]
    ma, mb = sum(ca) / len(ca), sum(cb) / len(cb)
    return T * max(abs((x - ma) - (y - mb)) for x, y in zip(ca, cb))


ref = JevK5(model_dir, graphs=os.environ.get("REF_GRAPHS", "0") == "1")
refp = [list(ref.probabilities(r["state"], r["question"])[0].values()) for r in rows]
kw = {k: (os.environ[v] == "1" if k == "exact" else os.environ[v]) for k, v in (("attn_impl", "ATTN"), ("conv_impl", "CONV"), ("gva_native", "PLUMB_GVA"), ("fuse", "FUSE"), ("exact", "EXACT")) if v in os.environ}
be = PlumbBackend(model_dir, compile=os.environ.get("REAX_COMPILE", "1"), **kw)
print("engine kw", kw, "compile", os.environ.get("REAX_COMPILE", "1"), flush=True)
keep, encs = [], []
for i, r in enumerate(rows):
    try:
        encs.append(be.encode(r["state"], r["question"]))
        keep.append(i)
    except ValueError:
        pass
rows, refp = [rows[i] for i in keep], [refp[i] for i in keep]
ref_ids = [ref.encode(r["state"], r["question"]["instructions"], [t for _, t in decision_options(r["question"])]) for r in rows]
assert all(list(e.payload) == i for e, i in zip(encs, ref_ids)), "token ids differ"


def report(name, got):
    ds = [dist(a, b) for a, b in zip(refp, got)]
    top = sum(max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__) for a, b in zip(refp, got)) / len(ds)
    print(json.dumps({"path": name, "n": len(ds), "top_agreement": round(top, 4), "mean_dist": round(sum(ds) / len(ds), 5),
                      "max_dist": round(max(ds), 4)}), flush=True)


t = time.time()
if be.engine.exact and not os.environ.get('NO_RUNNER'):
    be.engine.start_runner(n_streams=int(os.environ.get("PLUMB_STREAMS", "4")))
    print("runner capture s", round(time.time() - t, 1), flush=True)
else:
    single = [be.run([e])[0] for e in encs] if not be.engine.exact else None
    if single:
        report("eager_single", single)
    if not be.engine.exact:
        be.engine.capture_graphs(GRAPH_SHAPES)
        print("graph capture s", round(time.time() - t, 1), flush=True)
report("single", [be.run([e])[0] for e in encs])
rng = random.Random(1)
order = list(range(len(encs)))
rng.shuffle(order)
for B in (1, 16, 64) if os.environ.get("NO_RUNNER") else (3, 16, 64):
    got = [None] * len(encs)
    for k in range(0, len(order), B):
        idx = order[k:k + B]
        for i, p in zip(idx, be.run([encs[i] for i in idx])):
            got[i] = p
    report(f"batch{B}", got)
