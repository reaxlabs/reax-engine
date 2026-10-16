"""Find where the reference's CUDA-graph replay stops matching its own eager forward: truncate the HF model to k layers,
capture it as jevk5 does, compare with eager on the same ids. usage: ref_graph_probe.py MODEL_DIR ITEMS.jsonl"""
import json
import sys

import torch
from jevk5.runtime import JevK5

ref = JevK5(sys.argv[1], graphs=False)
rows = [json.loads(line) for line in open(sys.argv[2])]
ids_list = []
m = ref.model.model
full = list(m.layers)
n = 256
x = torch.randint(1000, 100000, (1, n), device="cuda")
for k in (1, 2, 3, 4, 5, 8, 16, 32):
    m.layers = torch.nn.ModuleList(full[:k])
    m.config.num_hidden_layers = k
    with torch.inference_mode():
        eager = m(input_ids=x, use_cache=False).last_hidden_state.clone()
        sid = torch.zeros((1, n), dtype=torch.long, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                m(input_ids=sid, use_cache=False)
        torch.cuda.current_stream().wait_stream(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = m(input_ids=sid, use_cache=False).last_hidden_state
        sid.copy_(x)
        g.replay()
        torch.cuda.synchronize()
        eager2 = m(input_ids=x, use_cache=False).last_hidden_state
        d = (out.float() - eager.float()).abs()
        print(f"k={k:2d}: graph vs eager max|d| {d.max().item():.5f}  equal frac {(d == 0).float().mean().item():.4f}   eager vs eager2 equal {(eager2 == eager).all().item()}", flush=True)
