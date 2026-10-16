"""Which autotuned Triton kernel choice changes the answers? Starts the exact engine, hashes its logits for a fixed item set and
prints every Autotuner cache entry (kernel, key, chosen config). Run many times and group by hash. GPU."""
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plumb_engine import PlumbBackend  # noqa: E402
from triton.runtime.autotuner import Autotuner  # noqa: E402

model_dir, items = sys.argv[1], sys.argv[2]
rows = [json.loads(line) for line in open(items)][:40]
be = PlumbBackend(model_dir, compile="0")
be.engine.start_runner(n_streams=2)
encs = [be.encode(r["state"], r["question"]) for r in rows]
out = np.concatenate([be.engine.logits([e.payload]) for e in encs])
h = hashlib.sha256(out.tobytes()).hexdigest()[:12]
cfgs = {}
for o in gc.get_objects():
    if isinstance(o, Autotuner):
        name = getattr(o.fn, "__name__", None) or getattr(getattr(o.fn, "fn", None), "__name__", "?")
        for k, v in o.cache.items():
            cfgs.setdefault(name, {})[str(k)] = str(v)
print(json.dumps({"hash": h, "cfgs": cfgs}))
