"""Where does the engine's exact mode differ from the reference graphs? Compares letter logits (bf16-quantised, so equal or
not) between: reference graph replay, reference eager padded / unpadded, engine eager padded, engine graph.
usage: exact_diff.py MODEL_DIR ITEMS.jsonl [N]"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jevk5.runtime import GRAPH_LENGTHS, JevK5  # noqa: E402
from plumb_engine import PlumbBackend  # noqa: E402

model_dir, items = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
rows = [json.loads(line) for line in open(items)][:N]
ref = JevK5(model_dir, graphs=True)
be = PlumbBackend(model_dir, compile="0")
eng = be.engine
eng.start_runner(n_streams=2)
dev = "cuda"
res = {k: [] for k in ("Rg", "Rpad", "Runpad", "Epad", "Eg")}
with torch.inference_mode():
    for r in rows:
        q = r["question"]
        q.setdefault("instructions", "")
        enc = be.encode(r["state"], q)
        ids = enc.payload.tolist()
        L = len(ids)
        nb = min(n for n in GRAPH_LENGTHS if n >= L)
        cnt = enc.n_options
        res["Rg"].append(ref.letter_logits(ids, 16))
        t = torch.tensor([ids + [0] * (nb - L)], device=dev)
        res["Rpad"].append(ref._slot_logits(t, torch.tensor([L - 1], device=dev))[0].float().cpu().numpy())
        t = torch.tensor([ids], device=dev)
        res["Runpad"].append(ref._slot_logits(t, torch.tensor([L - 1], device=dev))[0].float().cpu().numpy())
        # engine eager at padded length
        pid = torch.tensor(ids + [0] * (nb - L), device=dev)
        cu = torch.tensor([0, nb], device=dev, dtype=torch.int32)
        cos, sin = eng._rope_tables([nb], dev)
        last = torch.tensor([L - 1], device=dev)
        out = eng._drive([eng._core_gen(pid, cu, cu.long(), [0, nb], nb, last, cos, sin, torch.tensor([0, nb]))], [None])[0]
        res["Epad"].append(out[0].cpu().numpy())
        res["Eg"].append(eng.logits([enc.payload])[0])


def frac(a, b):
    return np.mean([np.array_equal(x, y) for x, y in zip(res[a], res[b])])


for a, b in (("Rg", "Rpad"), ("Rg", "Runpad"), ("Rpad", "Runpad"), ("Rpad", "Epad"), ("Runpad", "Epad"), ("Epad", "Eg"), ("Rg", "Eg"), ("Rg", "Epad")):
    print(f"{a:7s} == {b:7s}: {frac(a, b):.3f} of items identical")

d = np.array([np.abs(x - y).max() for x, y in zip(res["Rg"], res["Rpad"])])
print("Rg vs Rpad max|logit diff|: mean", d.mean(), "max", d.max(), "frac>0.1", (d > 0.1).mean())
print("example Rg  ", res["Rg"][0][:8])
print("example Rpad", res["Rpad"][0][:8])
again = [ref.letter_logits(enc_ids, 16) for enc_ids in []]
