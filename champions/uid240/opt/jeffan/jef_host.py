# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 team ruj. Line J4 round 2 (Jeffan): a port of team ruj's line-B round-4 Blink host levers
# (bj/blink4 blink_host.py, board #406) to pjs's Jeffan engine (E61 / E62, in-process vLLM 0.30.0, V2 model runner,
# one-group FULL-graph steps through team ruj's BF wait); the levers are unchanged, only the switch names and the log
# prefix differ. Credit pjs (the engine) and line B (the levers).
"""Exact host-side levers for the Jeffan engine core (vLLM 0.30.0, V2 model runner, FULL CUDA-graph steps).

Under the validator's closed loop at c8 the GPU idles ~10 ms between ~91 ms steps (line J4 screens): the outputs leave
the core, the frontend answers, the clients send the next questions and the re-reads come back, the core schedules them
(team ruj's BF wait) and execute_model prepares the step's inputs before cudaGraphLaunch. These levers shorten the host
work on that critical path. None of them touches a value the GPU computes: the same requests run in the same step with
the same token rows, KV blocks, request slots and graph, so every answer is bit-identical.

  JEF_HOST_NOMETA=1       in a FULL CUDA-graph replay, skip building the per-step attention-metadata objects
                          (model_state.prepare_attn): the captured graph reads only the persistent input buffers that
                          prepare_inputs / prepare_attn (block tables, slot mappings, positions, seq_lens,
                          query_start_loc) fill; the objects themselves feed only a speculator, a KV connector or a
                          non-FULL forward context, none of which this engine has (checked once at the first FULL step:
                          no speculator, no-op KV connector, no ubatching / context parallelism, every attention group a
                          TRITON_ATTN builder without R-SWA). Anything else builds them as before.
  JEF_HOST_LEAN=1         no per-request prefill statistics when the engine logs no statistics (JEF_LF=1 ->
                          disable_log_stats): the input thread sets Request.prefill_stats = None, so the scheduler does
                          not fill them, update_from_output does not estimate each finished request's cached tokens
                          (kv_cache_manager.estimate_cached_tokens over every KV-cache group) and the outputs carry no
                          PrefillStats to encode, send and decode; their only consumer is the frontend's metrics
                          (RequestOutput.num_cached_tokens), which this engine never reads. Answers untouched.
  JEF_HOST_DSAMP=1        deferred sampler adds: execute_model's add_requests stages each new request's sampling state
                          (sampler.add_request) and applies it (sampler.apply_staged_writes) before the graph launch, but
                          only the sampling after the graph reads it; the sampler calls of a step are queued and run right
                          after execute_model returns (the FULL graph is launched by then), in the same order on the same
                          stream, before sample_tokens. Same state, same kernels, later.
  JEF_HOST_DFREE=1        deferred block frees: update_from_output returns the KV blocks of each finished request to the
                          pool before the step's outputs leave the engine core; with DFREE the returns (Scheduler.
                          _free_blocks: kv_cache_manager.free + forgetting the request) are queued and run at the start of
                          the next schedule() call, i.e. after the outputs were sent and before any new allocation. The
                          finished-request bookkeeping the model runner needs (finished_req_ids) is unchanged. The blocks
                          and their prefix-cache entries are the same; only the time they are returned moves (the BF
                          wait's prefix-hit lookups in between are read-only and see the same cached blocks).
Installed by sitecustomize.py when any JEF_HOST_* switch is set; patches the V2 model runner, the scheduler and the
engine core when they are imported (the engine-core process; the frontend never imports them)."""
import importlib.abc
import os
import sys
import time

NOMETA = os.environ.get("JEF_HOST_NOMETA", "0") == "1"
LEAN = os.environ.get("JEF_HOST_LEAN", "0") == "1"
DSAMP = os.environ.get("JEF_HOST_DSAMP", "0") == "1"
DFREE = os.environ.get("JEF_HOST_DFREE", "0") == "1"
_S = {"nometa": None, "why": "", "patched": False, "dsamp": 0, "dfree": 0, "lean": 0, "patched_core": False,
      "patched_sched": False}


def _log(msg):
    print(f"[jef-host] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ NOMETA
def _nometa_ok(runner):
    """True when FULL replays of this runner never read the attention-metadata objects (see the module docstring)"""
    why = ""
    try:
        if getattr(runner, "speculator", None) is not None:
            why = "speculator"
        elif getattr(runner, "pcp_manager", None) is not None:
            why = "context parallelism"
        elif getattr(runner, "ubatch_runner", None) is not None:
            why = "ubatching"
        elif type(getattr(runner, "kv_connector", None)).__name__ != "KVConnector":
            # vLLM's no-op connector is an instance of the base class KVConnector (ActiveKVConnector otherwise)
            why = f"kv connector {type(getattr(runner, 'kv_connector', None)).__name__}"
        else:
            for groups in runner.attn_groups:
                for g in groups:
                    for i in range(4):
                        try:
                            b = g.get_metadata_builder(i)
                        except Exception:  # noqa: BLE001
                            break
                        if type(b).__name__ != "TritonAttentionMetadataBuilder":
                            why = f"attention builder {type(b).__name__}"
                        elif getattr(b, "rswa_window", None) is not None:
                            why = "R-SWA"
                        if why:
                            break
                    if why:
                        break
                if why:
                    break
    except Exception as exc:  # noqa: BLE001
        why = f"check failed: {exc!r}"
    _S["why"] = why
    return not why


def _patch_runner(m):
    if _S["patched"]:
        return
    _S["patched"] = True
    cls = m.GPUModelRunner
    orig_exec = cls.execute_model

    def execute_model(self, scheduler_output, *a, **k):
        dummy = k.get("dummy_run", False) or (len(a) >= 2 and bool(a[1]))
        if NOMETA and _S["nometa"] is None and not dummy:
            ms_cls = type(self.model_state)
            _S["nometa"] = _nometa_ok(self)
            if _S["nometa"]:
                _wrap_prepare_attn(ms_cls)
            _log(f"NOMETA {'on' if _S['nometa'] else 'off: ' + _S['why']} ({ms_cls.__name__})")
        smp = self.sampler if DSAMP and not dummy else None
        if smp is None or not scheduler_output.scheduled_new_reqs:
            return orig_exec(self, scheduler_output, *a, **k)
        # DSAMP: the step's sampler calls, queued while execute_model prepares and launches the graph
        pend = []
        smp.add_request = lambda req_idx, prompt_len, sp: pend.append((req_idx, prompt_len, sp))
        smp.apply_staged_writes = lambda: None
        try:
            return orig_exec(self, scheduler_output, *a, **k)
        finally:
            del smp.add_request, smp.apply_staged_writes        # the class methods again
            if pend:
                import torch
                with torch.inference_mode():                    # as inside execute_model
                    for args in pend:
                        smp.add_request(*args)
                    smp.apply_staged_writes()
                _S["dsamp"] += 1

    cls.execute_model = execute_model
    _log(f"installed in pid {os.getpid()}: nometa={NOMETA} lean={LEAN} dsamp={DSAMP} dfree={DFREE}")


def _wrap_prepare_attn(ms_cls):
    from vllm.config.compilation import CUDAGraphMode
    orig = ms_cls.prepare_attn
    if getattr(orig, "_b4_nometa", False):
        return

    def prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config,
                     for_capture=False, ubatch_idx=0, *a, **k):
        if cudagraph_mode == CUDAGraphMode.FULL and not for_capture and not a and not k:
            return {}       # FULL replay: the graph reads the persistent buffers, not these objects
        return orig(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config,
                    for_capture, ubatch_idx, *a, **k)

    prepare_attn._b4_nometa = True
    ms_cls.prepare_attn = prepare_attn


def _patch_core(m):
    """LEAN: Request.prefill_stats = None for every new request when the engine logs no statistics (JEF_LF=1)"""
    if _S["patched_core"]:
        return
    _S["patched_core"] = True
    base = m.EngineCore
    orig = base.preprocess_add_request

    def preprocess_add_request(self, request):
        res = orig(self, request)
        if not getattr(self, "log_stats", True):
            try:
                res[0].prefill_stats = None
                _S["lean"] += 1
            except Exception:  # noqa: BLE001
                pass
        return res

    base.preprocess_add_request = preprocess_add_request
    _log(f"LEAN installed in pid {os.getpid()}")


def _patch_sched(m):
    """DFREE: Scheduler._free_blocks queued inside update_from_output, run at the next schedule()"""
    if _S["patched_sched"]:
        return
    _S["patched_sched"] = True
    cls = m.Scheduler
    orig_fb = cls._free_blocks
    orig_upd = cls.update_from_output
    orig_sch = cls.schedule

    def _free_blocks(self, request):
        q = self.__dict__.get("_b4_dfree_in")
        if q is not None:
            q.append(request)
            return None
        return orig_fb(self, request)

    def flush(self):
        pend = self.__dict__.get("_b4_dfree_pend")
        if pend:
            self.__dict__["_b4_dfree_pend"] = []
            for r in pend:
                orig_fb(self, r)
            _S["dfree"] += 1

    def update_from_output(self, *a, **k):
        flush(self)                 # (never pending here in the engine's flow; kept for any other caller order)
        self.__dict__["_b4_dfree_in"] = []
        try:
            return orig_upd(self, *a, **k)
        finally:
            q = self.__dict__.pop("_b4_dfree_in", None) or []
            self.__dict__.setdefault("_b4_dfree_pend", []).extend(q)

    def schedule(self, *a, **k):
        flush(self)
        return orig_sch(self, *a, **k)

    cls._free_blocks = _free_blocks
    cls.update_from_output = update_from_output
    cls.schedule = schedule
    cls._b4_flush_dfree = flush
    _log(f"DFREE installed in pid {os.getpid()}")


_PATCHES = {"vllm.v1.worker.gpu.model_runner": lambda m: _patch_runner(m)}
if LEAN:
    _PATCHES["vllm.v1.engine.core"] = _patch_core
if DFREE:
    _PATCHES["vllm.v1.core.sched.scheduler"] = _patch_sched


class _Finder(importlib.abc.MetaPathFinder):
    _busy = set()      # re-entrancy guard (jef_levers.py / jef_steplog.py hook these modules too)

    def find_spec(self, fullname, path, target=None):
        if fullname not in _PATCHES or fullname in _Finder._busy:
            return None
        spec = None
        _Finder._busy.add(fullname)
        try:
            for f in sys.meta_path:
                if f is self:
                    continue
                try:
                    spec = f.find_spec(fullname, path, target)
                except Exception:  # noqa: BLE001
                    spec = None
                if spec is not None:
                    break
        finally:
            _Finder._busy.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig, _fn=_PATCHES[fullname]):
            _orig(module)
            _fn(module)

        spec.loader.exec_module = exec_module
        return spec


if NOMETA or LEAN or DSAMP or DFREE:
    _todo = False
    for _name, _fn in _PATCHES.items():
        if _name in sys.modules:
            _fn(sys.modules[_name])
        else:
            _todo = True
    if _todo:
        sys.meta_path.insert(0, _Finder())
