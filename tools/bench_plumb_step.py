"""In-process step benchmark of the native Plumb engine (no HTTP): time per forward step and decisions/s for B decisions of
about N tokens, on the packed varlen path and (small B) the CUDA-graph path.  usage: bench_plumb_step.py MODEL_DIR
[--lens 180] [--batches 1,2,4,8,16,32,64] [--profile B]"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plumb_engine import GRAPH_SHAPES, PlumbBackend  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("--lens", type=int, default=180)
ap.add_argument("--batches", default="1,2,4,8,16,32,64")
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--profile", type=int, default=0, help="print the kernel table of one step of this batch size")
ap.add_argument("--compile", default=os.environ.get("REAX_COMPILE", "1"))
a = ap.parse_args()
be = PlumbBackend(a.model_dir, compile=a.compile)
eng = be.engine
rng = np.random.default_rng(0)


def mk(B):
    return [rng.integers(1000, 200000, size=int(a.lens * (0.8 + 0.4 * rng.random()))).astype(np.int64) for _ in range(B)]


def bench(B, graph):
    saved = eng.graphs
    if not graph:
        eng.graphs = {}
    try:
        for _ in range(3):
            eng.logits(mk(B))
        torch.cuda.synchronize()
        ts = []
        for _ in range(a.iters):
            arr = mk(B)
            t = time.perf_counter()
            eng.logits(arr)
            ts.append(time.perf_counter() - t)
        ts.sort()
        return ts[len(ts) // 2] * 1000
    finally:
        eng.graphs = saved


t0 = time.time()
if eng.exact:
    eng.start_runner()
else:
    eng.capture_graphs(GRAPH_SHAPES)
print(f"graphs captured in {time.time() - t0:.0f}s (exact={eng.exact})")
for B in [int(x) for x in a.batches.split(",")]:
    ms_e = bench(B, False) if not eng.exact else None
    ms_g = bench(B, True) if (eng.exact or eng._pick_graph([a.lens] * B) is not None) else None
    print(f"B={B:3d}" + (f" eager {ms_e:7.2f} ms ({B / ms_e * 1000:7.1f} dec/s)" if ms_e else "")
          + (f"   graph {ms_g:7.2f} ms ({B / ms_g * 1000:7.1f} dec/s)" if ms_g else ""), flush=True)
if a.profile:
    from torch.profiler import ProfilerActivity, profile
    arr = [np.ones(192, dtype=np.int64) * 5 for _ in range(a.profile)]
    eng.graphs, eng.runner = {}, None
    eng.logits(arr)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        eng.logits(arr)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40, max_name_column_width=70))
