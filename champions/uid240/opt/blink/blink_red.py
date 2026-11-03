# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 pjs team (K2)
"""BLINK_RED=1: the two Inductor reductions at the tail of every Blink decoder layer run as exact one-pass CUDA
replicas (blink_red.cu -> gemm/blink_red.so) instead of their Triton builds:

  triton_red_fused_add_bi_mm_mul_rms_norm_0  (o_proj residual RMSNorms -> MoE / router / dense-MLP inputs)
  triton_red_fused_add_bi_mm_mul_rms_norm_2  (MoE + MLP join, new residual, next layer's input RMSNorm)

The replicas reproduce the Triton 3.7.1 sm_120 builds bit for bit (same per-element instructions, same tl.sum tree,
ptxas's FFMA(S, 1/2816, 1e-6) normalisation; see blink_red.cu), read each row once and skip red2's fp32 scratch output
when the calling Inductor wrapper provably never reads it.

Hook: CachingAutotuner.run is wrapped at class level (idempotent; chains with blink_pin, which patches
benchmark_all_configs / the scheduler). A kernel object is hooked only when all of these match exactly:
  kernel name, Inductor source file (content hash), triton signature, number of arguments, r0_numel 2816, and the
  launch config Triton runs (XBLOCK 1, R0_BLOCK 1024, num_warps 8, num_stages 1, a single launcher).
Anything else runs the Triton kernel unchanged. The first call of each kernel object always runs Triton (it compiles /
selects the launcher); the replica launches from the second call on (warm-up runs before CUDA-graph capture), on the
stream Triton would use (the wrapper's raw_stream0 = the current stream), with no allocation and no host sync.
red2's out_ptr2 (an fp32 scratch the wrapper allocates with empty_strided_cuda and deletes right after the call) is
skipped only when the caller's source shows exactly: one fresh allocation before the call, this call as the only read,
one `del` after it; otherwise it is written.

Logs (stderr): "[blink-red] armed ...", one "[blink-red] hooked k/2 ..." line per kernel the first time it engages
(+ "[blink-red] NOT hooked ..." with the reason for a name match that fails a check), and an exit summary.
BLINK_RED unset or != 1: this module is not imported (sitecustomize) and nothing changes.
"""
from __future__ import annotations

import ast
import atexit
import importlib.abc
import importlib.machinery
import importlib.util
import linecache
import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.environ.get("BLINK_RED_SO", os.path.join(_HERE, "gemm", "blink_red.so"))
_TARGET = "torch._inductor.runtime.triton_heuristics"
_CFG = {"XBLOCK": 1, "R0_BLOCK": 1024, "num_warps": 8, "num_stages": 1}
_SIG0 = {"in_ptr0": "*bf16", "in_ptr1": "*bf16", "in_ptr2": "*bf16", "in_ptr3": "*bf16", "in_ptr4": "*bf16",
         "in_ptr5": "*bf16", "in_ptr6": "*bf16", "out_ptr0": "*fp32", "out_ptr4": "*bf16", "out_ptr5": "*bf16",
         "out_ptr6": "*bf16", "xnumel": "i32", "r0_numel": "i32", "XBLOCK": "constexpr", "R0_BLOCK": "constexpr"}
_SIG2 = {"in_out_ptr0": "*bf16", "in_ptr0": "*bf16", "in_ptr1": "*bf16", "in_ptr2": "*bf16", "in_ptr3": "*bf16",
         "in_ptr4": "*bf16", "in_ptr5": "*fp32", "in_ptr6": "*bf16", "in_ptr7": "*bf16", "in_ptr8": "*bf16",
         "in_ptr9": "*bf16", "out_ptr2": "*fp32", "out_ptr5": "*bf16", "xnumel": "i32", "r0_numel": "i32",
         "XBLOCK": "constexpr", "R0_BLOCK": "constexpr"}
# kernel name -> (replica, Inductor source file, signature, number of positional run() arguments)
_SPEC = {
    "triton_red_fused_add_bi_mm_mul_rms_norm_0": ("red0", "c3wwo3b2e4upplnm2ju3ef4m4mi5mxufiqpgzw7nhgjptheg4xhi.py",
                                                  _SIG0, 13),
    "triton_red_fused_add_bi_mm_mul_rms_norm_2": ("red2", "cbpkosjpa5brpjsapkulaacetht4nwifxjojih4hildoqq5svlk6.py",
                                                  _SIG2, 15),
}
_ATTR = "_blink_red_state"
_STATE = {"ext": None, "hooked": {}, "calls": {"red0": 0, "red2": 0}, "fallback": 0, "notes": set(), "dead": {},
          "lock": threading.Lock()}


def _log(msg):
    print(f"[blink-red] {msg}", file=sys.stderr, flush=True)


def _note(key, msg):
    """log a message once per key"""
    with _STATE["lock"]:
        if key in _STATE["notes"]:
            return
        _STATE["notes"].add(key)
    _log(msg)


def _load_ext():
    spec = importlib.util.spec_from_loader("blink_red", importlib.machinery.ExtensionFileLoader("blink_red", _SO))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg_of(launcher):
    cfg = getattr(launcher, "config", None)
    d = dict(getattr(cfg, "kwargs", None) or {})
    d["num_warps"] = getattr(cfg, "num_warps", None)
    d["num_stages"] = getattr(cfg, "num_stages", None)
    return d


def _classify(at, name):
    """replica name if this CachingAutotuner object is exactly one of the two kernels, else None (+ log why)"""
    op, src_want, sig_want, _ = _SPEC[name]
    src = os.path.basename(getattr(at, "filename", None) or "?")
    why = None
    if src != src_want:
        why = f"source {src} != {src_want}"
    elif dict((getattr(at, "triton_meta", None) or {}).get("signature", {})) != sig_want:
        why = "triton signature differs"
    elif len(getattr(at, "launchers", ())) != 1:
        why = f"{len(getattr(at, 'launchers', ()))} launchers"
    else:
        cfg = _cfg_of(at.launchers[0])
        if any(cfg.get(k) != v for k, v in _CFG.items()) or set(cfg) - set(_CFG):
            why = f"launch config {cfg} != {_CFG}"
    if why is not None:
        _note(("not", name, src, why), f"NOT hooked {name}: {why} (Triton kernel runs)")
        return None
    return op


# ---- red2 out_ptr2: dead-store proof from the calling wrapper's source -----------------------------------------------
def _out2_dead(frame):
    """(dead, where): out_ptr2 of the red2 call at frame's current line is a fresh empty_strided_cuda buffer that the
    calling function never reads again (allocated once before the call, passed only here, deleted after it)."""
    co = frame.f_code
    key = (co.co_filename, co.co_firstlineno, frame.f_lineno)
    hit = _STATE["dead"].get(key)
    if hit is not None:
        return hit
    res = (False, f"{os.path.basename(co.co_filename)}:{frame.f_lineno} not analysable")
    try:
        lines = linecache.getlines(co.co_filename)
        tree = ast.parse("".join(lines)) if lines else None
        line = frame.f_lineno
        funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.lineno <= line <= (n.end_lineno or n.lineno)] if tree is not None else []
        func = min(funcs, key=lambda n: (n.end_lineno or n.lineno) - n.lineno) if funcs else None
        calls = [n for n in ast.walk(func) if isinstance(n, ast.Call) and n.lineno == line
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "run"] if func is not None else []
        if len(calls) == 1 and len(calls[0].args) == 15 and isinstance(calls[0].args[11], ast.Name):
            arg = calls[0].args[11]
            name = arg.id
            refs = [n for n in ast.walk(func) if isinstance(n, ast.Name) and n.id == name]
            stores = [n for n in refs if isinstance(n.ctx, ast.Store)]
            loads = [n for n in refs if isinstance(n.ctx, ast.Load)]
            dels = [n for n in refs if isinstance(n.ctx, ast.Del)]
            fresh = False
            if len(stores) == 1:
                for n in ast.walk(func):
                    if (isinstance(n, ast.Assign) and len(n.targets) == 1 and n.targets[0] is stores[0]
                            and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                            and n.value.func.id == "empty_strided_cuda"):
                        fresh = True
            ok = (fresh and stores[0].lineno < line and len(loads) == 1 and loads[0] is arg and len(dels) == 1
                  and dels[0].lineno > line)
            where = (f"{os.path.basename(co.co_filename)}:{line} {name}"
                     + (f" alloc L{stores[0].lineno} del L{dels[0].lineno}" if ok else
                        f" stores {len(stores)} loads {len(loads)} dels {len(dels)} fresh {fresh}"))
            res = (ok, where)
    except Exception as exc:  # noqa: BLE001
        res = (False, f"{os.path.basename(co.co_filename)}:{frame.f_lineno} analysis failed: {exc!r}")
    _STATE["dead"][key] = res
    return res


# ---- the run() wrapper --------------------------------------------------------------------------------------------------
_PASS = ("blink_hedge.py", "blink_red.py", "triton_heuristics.py")


def _caller(frame):
    """the Inductor wrapper frame that called run(): skip other class-level run() wrappers (blink_hedge)"""
    while frame is not None and os.path.basename(frame.f_code.co_filename) in _PASS:
        frame = frame.f_back
    return frame


def _hooked_run(self, args, stream, frame):
    """True when the replica was launched"""
    op = self.__dict__.get(_ATTR)
    name = self.fn.__name__
    nargs = _SPEC[name][3]
    if len(args) != nargs or args[-1] != 2816 or not isinstance(args[-2], int):
        _note(("args", name, len(args)), f"{name}: unexpected run() arguments ({len(args)}), Triton kernel runs")
        return False
    ext = _STATE["ext"]
    if op == "red0":
        ok = ext.red0(*args[:11], args[11], int(stream))
    else:
        dead, where = _out2_dead(frame) if frame is not None else (False, "no caller frame")
        _note(("out2", where), f"red2 out_ptr2 {'not written (dead: ' if dead else 'written ('}{where})")
        ok = ext.red2(*args[:13], args[13], not dead, int(stream))
    if not ok:
        with _STATE["lock"]:
            _STATE["fallback"] += 1
        _note(("operands", name), f"{name}: operand check failed (dtype / size / alignment / device), Triton kernel runs")
        return False
    _STATE["calls"][op] += 1
    return True


def _patch(m):
    cls = m.CachingAutotuner
    if getattr(cls.run, "_blink_red", False):
        return
    if _STATE["ext"] is None:
        try:  # torch is imported by now (triton_heuristics imports it); the extension links against it
            _STATE["ext"] = _load_ext()
        except Exception as exc:  # noqa: BLE001
            _log(f"inactive: cannot load {_SO}: {exc!r}")
            return
    orig = cls.run

    def run(self, *args, stream, benchmark_run=False, **kwargs):
        st = self.__dict__.get(_ATTR, 0)
        if st is None or benchmark_run or kwargs:
            return orig(self, *args, stream=stream, benchmark_run=benchmark_run, **kwargs)
        if st == 0:
            name = getattr(getattr(self, "fn", None), "__name__", None)
            if name not in _SPEC:
                self.__dict__[_ATTR] = None
                return orig(self, *args, stream=stream, benchmark_run=benchmark_run, **kwargs)
            if len(getattr(self, "launchers", ())) != 1:
                # first call: Triton compiles / picks its launcher and runs; classify on the next call
                return orig(self, *args, stream=stream, benchmark_run=benchmark_run, **kwargs)
            st = _classify(self, name)
            self.__dict__[_ATTR] = st
            if st is None:
                return orig(self, *args, stream=stream, benchmark_run=benchmark_run, **kwargs)
            with _STATE["lock"]:
                first = name not in _STATE["hooked"]
                _STATE["hooked"][name] = _STATE["hooked"].get(name, 0) + 1
                k = len(_STATE["hooked"])
            if first:
                _log(f"hooked {k}/2: {name} ({os.path.basename(self.filename)}, XBLOCK 1 R0_BLOCK 1024 num_warps 8) "
                     f"-> blink_red.{st}")
        if _hooked_run(self, args, stream, _caller(sys._getframe(1))):
            return None
        return orig(self, *args, stream=stream, benchmark_run=benchmark_run, **kwargs)

    run._blink_red = True
    run.__wrapped__ = orig
    cls.run = run


# ---- install: patch now if the module is loaded, else when it is imported -----------------------------------------
class _Finder(importlib.abc.MetaPathFinder):
    """Wraps exec_module of _TARGET's spec. Consults only the finders AFTER itself in sys.meta_path (those before it
    have been asked already and, if they chain like blink_pin, ask this one), with a re-entry guard: every patching
    finder runs exactly once whatever the import order of blink_pin / blink_red."""
    _busy: set = set()

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET or fullname in _Finder._busy:
            return None
        _Finder._busy.add(fullname)
        try:
            spec = None
            mp = list(sys.meta_path)
            after = mp[mp.index(self) + 1:] if self in mp else mp
            for f in after:
                fs = getattr(f, "find_spec", None)
                if fs is None:
                    continue
                try:
                    spec = fs(fullname, path, target)
                except Exception:  # noqa: BLE001
                    spec = None
                if spec is not None:
                    break
        finally:
            _Finder._busy.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


def _summary():
    h = _STATE["hooked"]
    if h:
        _log(f"summary: hooked {len(h)}/2 kernels ({', '.join(f'{k} x{v}' for k, v in sorted(h.items()))}); replica "
             f"launches red0 {_STATE['calls']['red0']} red2 {_STATE['calls']['red2']}; operand fallbacks "
             f"{_STATE['fallback']}")
    else:
        _log("summary: inactive - no matching kernel was ever launched")


def _install():
    if not os.path.isfile(_SO):
        _log(f"inactive: {_SO} not found")
        return
    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
    elif not any(isinstance(f, _Finder) or type(f).__module__ == __name__ for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    atexit.register(_summary)
    _log(f"armed: waiting for {', '.join(_SPEC)} ({os.path.basename(_SO)})")


if os.environ.get("BLINK_RED", "0") == "1":
    _install()
