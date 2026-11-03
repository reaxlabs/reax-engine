# SPDX-License-Identifier: Apache-2.0
"""Numerics pin for the Jeffan engine: no start-time benchmarking may choose a kernel variant.

vLLM's compiled graph (torch Inductor) picks a launch config for each generated Triton kernel that has several
candidates by timing them at its first launch (CachingAutotuner.benchmark_all_configs), and decides whether to fuse
independent kernels into a combo kernel by timing both (Scheduler.speedup_by_combo_kernel). Candidates sum in different
orders, so which one wins - and with it the answers - depends on the start conditions. This module replaces the timing
with a fixed table (pins.json): the picks of the Jeffan reference started alone on an idle GPU. An autotune instance not
in the table falls back to the first candidate (logged), never to a timing.

Installed by sitecustomize.py in every process; it patches torch._inductor.runtime.triton_heuristics and
torch._inductor.scheduler when they are imported (in the spawned engine core). JEF_PIN_LOG=<dir>: one JSON line per
decision.
"""
from __future__ import annotations

import importlib.abc
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TABLE_PATH = os.environ.get("JEF_PIN_TABLE", os.path.join(_HERE, "pins.json"))
_LOG_DIR = os.environ.get("JEF_PIN_LOG", "")
_STATE = {"table": None, "n": 0, "miss": 0, "combo": 0}


def _table():
    if _STATE["table"] is None:
        try:
            with open(_TABLE_PATH) as fh:
                _STATE["table"] = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            print(f"[jef-pin] no table at {_TABLE_PATH}: {exc!r}", file=sys.stderr, flush=True)
            _STATE["table"] = {"by_key": {}, "instances": [], "combo": None}
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


def _lookup(name, src, cands):
    """Pinned config of this autotune instance: by kernel source file + occurrence index, else by kernel name +
    candidate list (the same kernel generated with other device properties), else None."""
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
    return None, "none"


def _add_config(at, want: dict):
    """Compile `want` (same kernel, another launch config) as an extra launcher of autotuner `at`; its index or None."""
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
        print(f"[jef-pin] could not compile pinned config {want}: {exc!r}", file=sys.stderr, flush=True)
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
                pick = _add_config(self, want)
                if pick is not None:
                    how = how + " (pinned config compiled)"
                    cands = [_cfg_dict(lc.config) for lc in self.launchers]
        if pick is None:
            how = "fallback-first" if want is None else "fallback-first (pinned config not among the candidates)"
            pick = 0
            _STATE["miss"] += 1
            print(f"[jef-pin] {name} ({src}): {how}; candidates {cands}", file=sys.stderr, flush=True)
        _STATE["n"] += 1
        _log({"name": name, "src": src, "how": how, "pick": cands[pick], "candidates": cands})
        return {lc: (0.0 if i == pick else 1.0) for i, lc in enumerate(self.launchers)}

    cls.benchmark_all_configs = benchmark_all_configs


def _patch_scheduler(m):
    """Combo-kernel decisions: the reference's decision for the i-th timed group (table "combo"), else fuse. The timing
    call itself is kept so that its side effects (first launches, pinned autotunes) happen in the reference's order."""
    cls = m.Scheduler
    orig = cls.speedup_by_combo_kernel

    def speedup_by_combo_kernel(self, nodes):
        r = orig(self, nodes)
        seq = _table().get("combo") or []
        i = _STATE["combo"]
        _STATE["combo"] = i + 1
        use = bool(seq[i]) if i < len(seq) else True
        _log({"combo": len(nodes), "index": i, "timed_decision": bool(r), "used": use})
        return use

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
