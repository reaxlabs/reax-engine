"""Per-layer divergence of the native engine from the HF forward (last token residual stream). Debug tool; GPU.
usage: layer_diff.py MODEL_DIR ITEMS.jsonl [idx]"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jevk5.runtime import JevK5, decision_options  # noqa: E402
from plumb_engine import PlumbBackend  # noqa: E402

model_dir, items = sys.argv[1], sys.argv[2]
idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
row = [json.loads(line) for line in open(items)][idx]
q = row["question"]
q.setdefault("instructions", "")
ref = JevK5(model_dir, graphs=False)
kw = {k: os.environ[v] for k, v in (("attn_impl", "ATTN"), ("conv_impl", "CONV"), ("gva_native", "PLUMB_GVA"), ("fuse", "FUSE")) if v in os.environ}
be = PlumbBackend(model_dir, compile=os.environ.get("REAX_COMPILE", "0"), **kw)
enc = be.encode(row["state"], q)
ids = torch.tensor([enc.payload.tolist()], device="cuda")
with torch.inference_mode():
    out = ref.model.model(input_ids=ids, use_cache=False, output_hidden_states=True)
    hs = out.hidden_states  # embeddings + each layer output (last one is post final-norm)
    be.engine._trace = []
    lg = be.engine.logits([enc.payload])
tr = be.engine._trace
print("n tokens", ids.shape[1], "layers traced", len(tr), "hidden", len(hs))
for i, t in enumerate(tr):
    r = hs[i + 1][0, -1].float() if i < len(tr) - 1 else None
    if r is None:
        continue
    d = (t[0] - r)
    print(f"layer {i:2d} {be.engine.layer_types[i][:6]} max|diff| {d.abs().max().item():8.4f}  rel {d.norm().item() / r.norm().item():.5f}")
