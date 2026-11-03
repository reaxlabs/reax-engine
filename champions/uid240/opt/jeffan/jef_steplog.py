# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 team ruj (screen instrumentation for pjs's Jeffan engine, E61/E62)
"""Step log for lever screens (JEF_STEPLOG=<seconds>; off unless set).

In the spawned vLLM engine core it wraps the V2 model runner's execute_model (the forward) and sample_tokens (the
readout) with CUDA timing events on the runner's stream. Nothing is synchronised: completed events are read with
Event.query() on later steps. Every <seconds> one line on stderr:
  [jef-step] {"steps", "tok", "tok_per_step", "reqs_per_step", "new_reqs", "cached_tok", "gpu_span_ms", "fwd_ms",
              "smp_ms", "busy", "hist": {bucket: [steps, mean fwd ms, mean tokens]}}
busy = (forward + sampler GPU time) / GPU span of the window (first forward start -> last sampler end).
"""
from __future__ import annotations

import importlib.abc
import json
import os
import sys
import time

PERIOD = float(os.environ.get("JEF_STEPLOG", "0") or 0)
BUCKETS = (64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192, 1 << 30)
_S = {"pending": [], "rows": [], "t_next": 0.0, "base": None, "cur": None}


def _ev():
    import torch
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def _capturing():
    import torch
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return True


def _drain():
    base = _S["base"]
    keep = []
    for rec in _S["pending"]:
        evs = rec["ev"]
        if evs[-1].query():
            rec["t"] = [base.elapsed_time(e) for e in evs]
            del rec["ev"]
            _S["rows"].append(rec)
        else:
            keep.append(rec)
    _S["pending"] = keep


def _report():
    rows, _S["rows"] = _S["rows"], []
    rows = [r for r in rows if len(r["t"]) == 4]
    if not rows:
        return
    fwd = sum(r["t"][1] - r["t"][0] for r in rows)
    smp = sum(r["t"][3] - r["t"][2] for r in rows)
    span = rows[-1]["t"][3] - rows[0]["t"][0]
    tok = sum(r["ntok"] for r in rows)
    hist = {}
    for r in rows:
        b = next(x for x in BUCKETS if r["ntok"] <= x)
        h = hist.setdefault(b, [0, 0.0, 0])
        h[0] += 1
        h[1] += r["t"][1] - r["t"][0]
        h[2] += r["ntok"]
    out = {
        "steps": len(rows), "tok": tok, "tok_per_step": round(tok / len(rows), 1),
        "reqs_per_step": round(sum(r["nreq"] for r in rows) / len(rows), 2),
        "new_reqs": sum(r["nnew"] for r in rows), "cached_tok": sum(r["ncached"] for r in rows),
        "gpu_span_ms": round(span, 1), "fwd_ms": round(fwd, 1), "smp_ms": round(smp, 1),
        "busy": round((fwd + smp) / span, 4) if span > 0 else None,
        "tok_per_s_gpu": round(1000.0 * tok / span, 1) if span > 0 else None,
        "hist": {str(b if b < (1 << 30) else "inf"): [h[0], round(h[1] / h[0], 2), round(h[2] / h[0], 1)]
                 for b, h in sorted(hist.items())},
    }
    print("[jef-step] " + json.dumps(out), file=sys.stderr, flush=True)


def _tick():
    _drain()
    now = time.monotonic()
    if now >= _S["t_next"]:
        _S["t_next"] = now + PERIOD
        _report()


def _patch(m):
    cls = m.GPUModelRunner
    orig_exec, orig_samp = cls.execute_model, cls.sample_tokens

    def execute_model(self, scheduler_output, *a, **kw):
        dummy = kw.get("dummy_run", a[1] if len(a) > 1 else False)
        ntok = getattr(scheduler_output, "total_num_scheduled_tokens", 0)
        if dummy or not ntok or _capturing():
            return orig_exec(self, scheduler_output, *a, **kw)
        if _S["base"] is None:
            _S["base"] = _ev()
        new = scheduler_output.scheduled_new_reqs
        rec = {"ntok": ntok, "nreq": len(scheduler_output.num_scheduled_tokens), "nnew": len(new),
               "ncached": sum(int(getattr(r, "num_computed_tokens", 0) or 0) for r in new), "ev": [_ev()]}
        out = orig_exec(self, scheduler_output, *a, **kw)
        rec["ev"].append(_ev())
        _S["cur"] = rec
        return out

    def sample_tokens(self, grammar_output, *a, **kw):
        rec, _S["cur"] = _S["cur"], None
        if rec is None or _capturing():
            return orig_samp(self, grammar_output, *a, **kw)
        rec["ev"].append(_ev())
        out = orig_samp(self, grammar_output, *a, **kw)
        rec["ev"].append(_ev())
        _S["pending"].append(rec)
        _tick()
        return out

    cls.execute_model, cls.sample_tokens = execute_model, sample_tokens
    print(f"[jef-step] step log on (every {PERIOD} s)", file=sys.stderr, flush=True)


_TARGET = "vllm.v1.worker.gpu.model_runner"


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        spec = None
        for f in sys.meta_path:
            if f is self:
                continue
            try:
                spec = f.find_spec(fullname, path, target)
            except Exception:  # noqa: BLE001
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


if PERIOD > 0:
    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
    else:
        sys.meta_path.insert(0, _Finder())
