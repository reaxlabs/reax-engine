# SPDX-License-Identifier: Apache-2.0
"""Numerics pin for the Blink engine: no start-time benchmarking may choose a kernel variant.

The reference's compiled vLLM picks a launch config for each Inductor-generated Triton kernel that has several
candidates by timing them at first launch (CachingAutotuner.benchmark_all_configs). Some candidates sum in a different
order, so which one wins - and with it the answers - depends on the start conditions (another engine resident on the
GPU, a profiler, ...). This module replaces the timing with a fixed table: for every kernel the config the reference
picks when it starts alone on an idle GPU (pins.json, recorded from such starts). A kernel not in the table falls back
to a fixed rule (first candidate) and is logged, never timed.

Install: imported by sitecustomize.py in every process; it patches torch._inductor.runtime.triton_heuristics when that
module is imported (in the engine core process). BLINK_PIN_LOG=<dir> writes one JSON line per decision.
"""
from __future__ import annotations

import importlib.abc
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TABLE_PATH = os.environ.get("BLINK_PIN_TABLE", os.path.join(_HERE, "pins.json"))
_LOG_DIR = os.environ.get("BLINK_PIN_LOG", "")
_HEDGE = os.environ.get("BLINK_HEDGE", "0") == "1"
_STATE = {"table": None, "n": 0, "miss": 0}


def _table():
    if _STATE["table"] is None:
        try:
            with open(_TABLE_PATH) as fh:
                _STATE["table"] = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            print(f"[blink-pin] no table at {_TABLE_PATH}: {exc!r}", file=sys.stderr, flush=True)
            _STATE["table"] = {"by_key": {}, "instances": []}
    return _STATE["table"]


def _cfg_dict(cfg):
    d = {k: v for k, v in (getattr(cfg, "kwargs", None) or {}).items()}
    d["num_warps"] = getattr(cfg, "num_warps", None)
    d["num_stages"] = getattr(cfg, "num_stages", None)
    return d


def _same(cfg, want: dict) -> bool:
    have = _cfg_dict(cfg)
    return all(have.get(k) == v for k, v in want.items())


def _log(rec):
    if not _LOG_DIR:
        return
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        with open(os.path.join(_LOG_DIR, f"pin_{os.getpid()}.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _cands_sig(cands):
    return json.dumps(sorted(json.dumps(c, sort_keys=True) for c in cands))


def _norm_name(name):
    """kernel name without the origin tags an opaque op in the compiled graph adds or removes (E59: the per-M matmul
    custom op turns 'triton_red_fused_rms_norm_t_0' into 'triton_red_fused_bi_mm_rms_norm_0')"""
    import re
    return re.sub(r"_t_(\d+)$", r"_\1", name.replace("_bi_mm", ""))


def _lookup(name, src, cands):
    """Pinned config for this autotune instance: by kernel source file + occurrence index (exact), else by kernel
    name + candidate list (same kernel generated on a GPU with other device properties), else by the normalised name +
    candidate list (_norm_name), else None."""
    tab = _table()
    occ = _STATE.setdefault("occ", {})
    k = occ.get(src, 0)
    occ[src] = k + 1
    want = tab.get("by_key", {}).get(f"{src}#{k}")
    if want is not None:
        return want, "table"
    sig = _cands_sig(cands)
    alts = [i["pick"] for i in tab.get("instances", []) if i.get("name") == name and _cands_sig(i.get("cands", [])) == sig]
    if alts:
        j = _STATE.setdefault("alt", {})
        n = j.get((name, sig), 0)
        j[(name, sig)] = n + 1
        return alts[min(n, len(alts) - 1)], "table-by-name"
    nn = _norm_name(name)
    alts = [i["pick"] for i in tab.get("instances", [])
            if _norm_name(i.get("name", "")) == nn and _cands_sig(i.get("cands", [])) == sig]
    if alts:
        j = _STATE.setdefault("alt", {})
        n = j.get(("norm", nn, sig), 0)
        j[("norm", nn, sig)] = n + 1
        return alts[min(n, len(alts) - 1)], "table-by-normalised-name"
    return None, "none"


def _add_config(at, want: dict):
    """Compile `want` (same kernel, other launch config) as an extra launcher of autotuner `at`; its index or None."""
    try:
        import copy
        cfg = copy.deepcopy(at.launchers[0].config)
        for k, v in want.items():
            if k in ("num_warps", "num_stages"):
                setattr(cfg, k, v)
            else:
                cfg.kwargs[k] = v
        if hasattr(at, "_ensure_kernel_loaded"):
            at._ensure_kernel_loaded()
        at.compile_results.append(at._precompile_config(cfg))
        at._make_launchers()
        for i, lc in enumerate(at.launchers):
            if _same(lc.config, want):
                return i
    except Exception as exc:  # noqa: BLE001
        print(f"[blink-pin] could not compile pinned config {want}: {exc!r}", file=sys.stderr, flush=True)
    return None


def _patch(m):
    cls = m.CachingAutotuner

    def benchmark_all_configs(self, *args, **kwargs):
        name = getattr(getattr(self, "fn", None), "__name__", "?")
        src = os.path.basename(getattr(self, "filename", None) or "?")
        cands = [_cfg_dict(lc.config) for lc in self.launchers]
        want, how = _lookup(name, src, cands)
        pick = None
        if want is not None:
            for i, lc in enumerate(self.launchers):
                if _same(lc.config, want):
                    pick = i
                    break
            if pick is None:
                # the candidate list differs from the reference's (it depends on how the kernel was compiled): compile
                # the pinned config itself, it is what the reference ran
                pick = _add_config(self, want)
                if pick is not None:
                    how = how + " (pinned config compiled)"
                    cands = [_cfg_dict(lc.config) for lc in self.launchers]
        if pick is None:
            how = "fallback-first" if want is None else "fallback-first (pinned config not among the candidates)"
            pick = 0
            _STATE["miss"] += 1
            print(f"[blink-pin] {name} ({src}): {how}; candidates {cands}", file=sys.stderr, flush=True)
        _STATE["n"] += 1
        _log({"name": name, "src": src, "how": how, "pick": cands[pick], "candidates": cands})
        if _HEDGE and name == "triton_red_fused_bi_mm_rms_norm_0":
            # E66 RMS-flip hedge (blink_hedge.py): both candidates of the layer-0 RMSNorm on every vocabulary row
            try:
                import blink_hedge
                blink_hedge.on_rms_autotune(self, pick, args)
            except Exception as exc:  # noqa: BLE001
                print(f"[blink-hedge] not installed: {exc!r}", file=sys.stderr, flush=True)
        return {lc: (0.0 if i == pick else 1.0) for i, lc in enumerate(self.launchers)}

    cls.benchmark_all_configs = benchmark_all_configs


def _patch_scheduler(m):
    """Inductor decides whether to fuse independent kernels into one combo kernel by timing both at compile time; for
    the Blink graph every candidate group is a 'small kernel' (< 0.3 ms) and gets fused. Keep the timing call (its
    side effects - first launches, pinned autotuning - stay in the reference's order) but always fuse."""
    cls = m.Scheduler
    orig = cls.speedup_by_combo_kernel

    def speedup_by_combo_kernel(self, nodes):
        r = orig(self, nodes)
        _log({"combo": len(nodes), "timed_decision": bool(r), "used": True})
        if not r:
            print(f"[blink-pin] combo group of {len(nodes)}: timing said no, fused anyway", file=sys.stderr, flush=True)
        return True

    cls.speedup_by_combo_kernel = speedup_by_combo_kernel


_TARGETS = {"torch._inductor.runtime.triton_heuristics": _patch, "torch._inductor.scheduler": _patch_scheduler}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS:
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

        def exec_module(module, _orig=orig, _fn=_TARGETS[fullname]):
            _orig(module)
            _fn(module)

        spec.loader.exec_module = exec_module
        return spec


for _name, _fn in _TARGETS.items():
    if _name in sys.modules:
        _fn(sys.modules[_name])
sys.meta_path.insert(0, _Finder())
