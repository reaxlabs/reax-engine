# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 team ruj (bj/blink4, line B, round 4 iterations 1-2; bj/blink5, line B, round 5). Builds on pjs's Blink engine
# (blink_core.py's one-group batch pipelining and pre-scheduling, which team ruj's #343 overlay added to his engine); credit pjs.
"""Exact host-side levers for the Blink engine core (vLLM 0.30.0, V2 model runner, one-group FULL-graph steps).

Under the validator's closed loop at c16 the GPU idles ~8.8 ms between one-group steps of ~30.7 ms (bj/blink4 p1b,
pjs's #396, runsc; ~7.5 ms between steps of ~30.3 ms on pjs's v1y): the outputs leave the core, the frontends answer,
the client sends the next 16 requests over ~3 ms, the core schedules them, and execute_model prepares the step's inputs
for ~1.4 ms before cudaGraphLaunch. These levers shorten the host work on that critical path. None of them changes a
value the GPU computes for a request's tokens: NOMETA, LEAN, DSAMP, DFREE and XDIET keep the same requests in the same
steps with the same token rows, KV blocks, request slots and graph; LATEFREE changes only which free KV blocks the next
requests receive (placement). The answers rely on the engine's batch invariance and on its answers not depending on KV
placement or prefix-cache hits; every screen checks them (1,024 timed, 24 c1 and 32 c1 re-run answers per config).

  BLINK_HOST_NOMETA=1     in a FULL CUDA-graph replay, skip building the per-step attention-metadata objects
                          (model_state.prepare_attn): the captured graph reads only the persistent input buffers that
                          prepare_inputs / prepare_attn (block tables, slot mappings, positions, seq_lens,
                          query_start_loc) fill; the objects themselves feed only a speculator, a KV connector or a
                          non-FULL forward context, none of which this engine has (checked once at the first FULL step:
                          no speculator, no-op KV connector, no ubatching / context parallelism, every attention group a
                          TRITON_ATTN builder without R-SWA). Anything else builds them as before.
  BLINK_HOST_LEAN=1       no per-request prefill statistics when the engine logs no statistics (pjs's BLINK_NOSTATS=1 ->
                          disable_log_stats): the input thread sets Request.prefill_stats = None, so the scheduler does
                          not fill them, update_from_output does not estimate each finished request's cached tokens
                          (kv_cache_manager.estimate_cached_tokens over every KV-cache group) and the outputs carry no
                          PrefillStats to encode, send and decode; their only consumer is the frontends' metrics
                          (RequestOutput.num_cached_tokens), which this engine never reads. Answers untouched.
  BLINK_HOST_DSAMP=1      deferred sampler adds: execute_model's add_requests stages each new request's sampling state
                          (sampler.add_request) and applies it (sampler.apply_staged_writes) before the graph launch, but
                          only the sampling after the graph reads it; the sampler calls of a step are queued and run right
                          after execute_model returns (the FULL graph is launched by then), in the same order on the same
                          stream, before sample_tokens. Same state, same kernels, later.
  BLINK_HOST_DFREE=1      deferred block frees: update_from_output returns the KV blocks of each finished request to the
                          pool before the step's outputs leave the engine core; with DFREE the returns (Scheduler.
                          _free_blocks: kv_cache_manager.free + forgetting the request) are queued and run at the start of
                          the next schedule() call, i.e. after the outputs were sent and before any new allocation. The
                          finished-request bookkeeping the model runner needs (finished_req_ids) is unchanged. The blocks
                          and their prefix-cache entries are the same; only the time they are returned moves.
  BLINK_HOST_XDIET=1      (round 2) a leaner execute_model before the graph launch, two exact parts:
                          NODRAFT - with no speculative steps the request state's draft_tokens tensor has shape
                          [max_num_reqs, 0], so RequestState.add_request's per-request draft_tokens[req_idx].zero_()
                          has no element to write; it is skipped (vLLM's add_request otherwise line for line);
                          FUSEW - RequestState.apply_staged_writes launches vLLM's staged-write kernel once per tensor
                          (total_len, all_token_ids, num_computed_tokens: 3 launches + 2 host-to-device copies), and the
                          block tables apply theirs with a fourth (fused) launch after update_requests; with FUSEW the
                          three request-state tensors and the six block tables are written by ONE launch of vLLM's own
                          multi-group write kernel (_apply_write_kernel, MULTI_GROUP) at the request state's point in
                          add_requests. Same values to the same addresses (the writes are disjoint: each tensor its own
                          buffer), on the same stream, before any reader (the sampler's staged writes and
                          prepare_inputs read them afterwards); the block tables' later call finds nothing staged for
                          the new requests (anything update_requests stages is applied there as before).
  BLINK_HOST_LATEFREE=1   (round 2, with pjs's BLINK_DEFER_FREE=1) pjs's E64 b2 queues each finished request's KV-block
                          frees and runs them in post_step right after the step's outputs are queued, where they hold
                          the GIL while the output thread wants to encode and send the outputs (v1y: 0.41 ms median from
                          update_from_output's end to the first output send). LATEFREE moves the queued frees (the
                          same Scheduler._free_blocks calls, in the same order) to the next post_step that runs while a
                          step is on the GPU, i.e. right after the next graph launch, into the GPU's busy time. Before
                          that, schedule() calls (pjs's pre-scheduling of the next group) see the finished requests'
                          blocks still allocated: the shared prompt prefix stays cached either way (those blocks are in
                          use, ref-counted, and hit as before), only which free blocks the new requests receive
                          changes. If the free pool ever drops below 25 % of the KV-cache blocks, the queued frees run
                          before the schedule() call. Answers rely on the engine's batch invariance and on its answers
                          not depending on KV block placement or prefix-cache hits (pjs's E72 L1 relies on the same;
                          the c1 re-run gate checks it).
  BLINK_HOST_SWITCH_US=N  (round 2, development) the engine-core process's GIL switch interval in microseconds
                          (sys.setswitchinterval; Python's default is 5000): bounds how long the input thread waits for
                          the GIL while the busy loop pre-schedules. Scheduling only.
  BLINK_HOST_UNIFY=1      (bj/blink5) one KV-cache group instead of six. Blink has 25 sliding-window layers (window 1,024
                          tokens, 8 KV heads x 256) and 5 full-attention layers (2 x 512); vLLM's hybrid KV-cache manager
                          keeps them in six groups (five sliding-window groups of 16-token blocks, one full-attention
                          group of 32-token blocks), so the scheduler, the prefix cache, the block pool and the model
                          runner's block tables do their per-request work six times (allocate_slots, cache lookups and
                          registrations, block-table staged writes). With UNIFY the engine core sets vLLM's own
                          scheduler_config.disable_hybrid_kv_cache_manager before it builds the KV cache: vLLM's
                          unify_hybrid_kv_cache_specs then allocates the sliding-window layers like full-attention layers
                          (32-token blocks for every token; the layers still compute sliding-window attention, as vLLM
                          documents for this mode) and all 30 layers share one block table. Same KV values in other
                          blocks: vLLM's Triton attention kernel walks each sequence in KV tiles whose size depends on
                          the head size, the layer's window and the dtype, not on the KV block size (its backend does not
                          enable tensor descriptors, the only path that clamps the tile to the block size), and masks by
                          the window; the board's s1-fast prompts (~170 tokens) lie far inside it. Prefix-cache hits stay on
                          32-token boundaries (the hybrid coordinator's granularity before). The KV cache holds ~159k
                          tokens either way (16.7 GiB). XDIET's FUSEW then also writes the one block table in its launch
                          (BlockTables.apply_staged_writes finds nothing staged afterwards). bj/blink5 s2: schedule()
                          cost per new request and the GIL wait of the last arrival drop; answers byte-identical.
                          (ours, #419 review: on vLLM 0.30.0 ced6857 the one group has 16-token blocks, not 32:
                          _promote_local_kv_cache_specs gives the sliding-window layers the full-attention layers' own
                          block size, cache_config.block_size = 16, and no page-size unification follows, so prefix-cache
                          hits land on 16-token boundaries, not 32; the start log line "UNIFY: KV cache 1 group(s), block
                          sizes [16]" shows it. Our copy flips the switch only on that vLLM file, see _KVU_SHA.)
Installed by sitecustomize.py when any BLINK_HOST_* switch is set; patches the V2 model runner, the scheduler and the
engine core when they are imported. The levers act in the engine-core process. The frontends also import
vllm.v1.engine.core, so LEAN's wrapper is installed there too ("LEAN installed" in their logs), but it is never called
there: only the engine core's input thread preprocesses requests (corrected after bj/blink4 VERIFY1).

Round 2 (bj/blink4 line B, on pjs's v1y): pjs's v1y has its own deferred KV frees (BLINK_DEFER_FREE, E64 b2: the frees
run in post_step after the step's outputs are queued) and VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=1 (E72 O2); on v1y this
module's BLINK_HOST_DFREE stays off and LEAN complements pjs's BLINK_LEAN_OUT (LEAN drops the PrefillStats objects,
LEAN_OUT only the cached-token estimate)."""
import importlib.abc
import os
import sys
import time

NOMETA = os.environ.get("BLINK_HOST_NOMETA", "0") == "1"
LEAN = os.environ.get("BLINK_HOST_LEAN", "0") == "1"
DSAMP = os.environ.get("BLINK_HOST_DSAMP", "0") == "1"
DFREE = os.environ.get("BLINK_HOST_DFREE", "0") == "1"
XDIET = os.environ.get("BLINK_HOST_XDIET", "0") == "1"
LATEFREE = os.environ.get("BLINK_HOST_LATEFREE", "0") == "1"
UNIFY = os.environ.get("BLINK_HOST_UNIFY", "0") == "1"
try:
    SWITCH_US = float(os.environ.get("BLINK_HOST_SWITCH_US", "0") or 0)
except ValueError:
    SWITCH_US = 0.0
_S = {"nometa": None, "why": "", "patched": False, "dsamp": 0, "dfree": 0, "lean": 0, "patched_core": False,
      "patched_states": False, "xdiet": None, "fusew": 0, "patched_late": False, "late": 0, "late_guard": 0,
      "patched_sched": False, "patched_unify": False}


# --- ours (not team ruj's): live A/B switches for one-start ABBA screens. Keys "host_nometa" / "host_dsamp" /
# "host_fusew" / "host_nodraft" in BLINK_TUNE_FILE, re-read at most every 0.5 s (as blink_read.py); a missing key keeps
# the current value. Without BLINK_TUNE_FILE nothing is read. Acts only where the env switch installed the patch.
_TUNE = os.environ.get("BLINK_TUNE_FILE", "")
_LIVE = {"t": 0.0, "nometa": True, "dsamp": True, "fusew": True, "nodraft": True}
# --- ours: XDIET's copy of RequestState.add_request (NODRAFT) and its replacement apply_staged_writes (FUSEW) are
# exact only for this file: vllm/v1/worker/gpu/states.py of vLLM 0.30.0 ced6857 (= the base image's); otherwise XDIET
# stays off (logged).
_STATES_SHA = "99418f5df43ca612ded72609fee011620b065b2cab2f249ccb387096bf4ae71a"
# --- ours: UNIFY flips vLLM's switch only on the file it was reviewed against, vllm/v1/core/kv_cache_utils.py of vLLM
# 0.30.0 ced6857 (= the base image's: unify_hybrid_kv_cache_specs gives Blink one group of 16-token blocks); otherwise
# UNIFY stays off (logged), so a vLLM bump can neither fail the start nor change the KV layout unreviewed.
_KVU_SHA = "2666c9f113584e52e7521058efd2c0d9598544559dc942d3a3da87001a3fedf2"


def _live():
    if _TUNE:
        now = time.monotonic()
        if now - _LIVE["t"] > 0.5:
            _LIVE["t"] = now
            try:
                import json
                with open(_TUNE) as fh:
                    j = json.load(fh)
                for k in ("nometa", "dsamp", "fusew", "nodraft"):
                    _LIVE[k] = bool(j.get("host_" + k, _LIVE[k]))
            except Exception:  # noqa: BLE001
                pass
    return _LIVE


def _log(msg):
    print(f"[blink-host] {msg}", file=sys.stderr, flush=True)


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
        lv = _live()
        dummy = k.get("dummy_run", False) or (len(a) >= 2 and bool(a[1]))
        if NOMETA and _S["nometa"] is None and not dummy:
            ms_cls = type(self.model_state)
            _S["nometa"] = _nometa_ok(self)
            if _S["nometa"]:
                _wrap_prepare_attn(ms_cls)
            _log(f"NOMETA {'on' if _S['nometa'] else 'off: ' + _S['why']} ({ms_cls.__name__})")
        if XDIET and _S["xdiet"] is None and not dummy:
            _xdiet_setup(self)
        smp = self.sampler if DSAMP and lv["dsamp"] and not dummy else None
        if smp is None or not scheduler_output.scheduled_new_reqs:
            return orig_exec(self, scheduler_output, *a, **k)
        # DSAMP: the step's sampler calls, queued while execute_model prepares and launches the graph
        pend = []
        smp.add_request = lambda req_idx, prompt_len, sp: pend.append((req_idx, prompt_len, sp))
        smp.apply_staged_writes = lambda: None
        try:
            return orig_exec(self, scheduler_output, *a, **k)
        finally:
            del smp.add_request, smp.apply_staged_writes        # the class methods again (pjs's blink_read wrapper)
            if pend:
                import torch
                with torch.inference_mode():                    # as inside execute_model
                    for args in pend:
                        smp.add_request(*args)
                    smp.apply_staged_writes()
                _S["dsamp"] += 1

    cls.execute_model = execute_model
    if SWITCH_US > 0:       # the model runner is imported in the engine-core process only
        sys.setswitchinterval(SWITCH_US / 1e6)
    _log(f"installed in pid {os.getpid()}: nometa={NOMETA} lean={LEAN} dsamp={DSAMP} dfree={DFREE} xdiet={XDIET} "
         f"latefree={LATEFREE} unify={UNIFY} switch={sys.getswitchinterval() * 1e6:.0f}us")


# ------------------------------------------------------------------------------------------------ XDIET
def _xdiet_setup(runner):
    """first real step: the combined writer for FUSEW (request-state tensors + block tables), or FUSEW off"""
    why = ""
    try:
        import torch
        from vllm.v1.worker.gpu.buffer_utils import FusedStagedWriter, StagedWriteTensor
        rs, bt = runner.req_states, runner.block_tables
        tensors = [rs.total_len, rs.all_token_ids, rs.num_computed_tokens] + list(bt.block_tables)
        if getattr(bt, "fused_writer", None) is None and not (UNIFY and len(bt.block_tables) == 1):
            # (bj/blink5) with UNIFY the one block table joins the same launch; BlockTables.apply_staged_writes then
            # finds nothing staged (StagedWriteTensor.apply_write returns at once) and only copies num_blocks
            why = "single KV-cache group"
        elif not _S["patched_states"]:
            why = "request state not patched"
        elif not all(isinstance(x, StagedWriteTensor) and x.dtype == torch.int32 for x in tensors):
            why = "unexpected tensor types"
        if not why:
            writer = FusedStagedWriter(runner.device, len(tensors) * rs.max_num_reqs)
            ptrs = torch.tensor([x.gpu.data_ptr() for x in tensors], dtype=torch.uint64, device=runner.device)
            strides = torch.tensor([x.gpu.stride(0) for x in tensors], dtype=torch.int64, device=runner.device)
            rs.__dict__["_b4_fuse"] = (writer, tensors, ptrs, strides)
    except Exception as exc:  # noqa: BLE001
        why = f"setup failed: {exc!r}"
    _S["xdiet"] = not why
    _log(f"XDIET: NODRAFT {'on' if _S['patched_states'] else 'off'}; FUSEW {'on' if not why else 'off: ' + why}")


def _patch_states(m):
    """XDIET: RequestState.add_request without the empty draft-token zeroing (NODRAFT) and apply_staged_writes through
    the combined writer once _xdiet_setup has attached it (FUSEW)"""
    if _S["patched_states"]:
        return
    try:        # ours: only on the states.py this copy was proven against
        import hashlib
        with open(m.__file__, "rb") as fh:
            same = hashlib.sha256(fh.read()).hexdigest() == _STATES_SHA
    except Exception:  # noqa: BLE001
        same = False
    if not same:
        _log("XDIET off: vllm/v1/worker/gpu/states.py is not vLLM 0.30.0 ced6857's")
        return
    _S["patched_states"] = True
    RS = m.RequestState
    orig_add = RS.add_request
    orig_apply = RS.apply_staged_writes

    def add_request(self, req_id, prompt_len, all_token_ids, num_computed_tokens, max_tokens):
        if self.num_speculative_steps or not _LIVE["nodraft"]:
            return orig_add(self, req_id, prompt_len, all_token_ids, num_computed_tokens, max_tokens)
        # vLLM 0.30.0 RequestState.add_request line for line, minus `self.draft_tokens[req_idx].zero_()`: with no
        # speculative steps draft_tokens is [max_num_reqs, 0] and the row has no element (Apache-2.0, vLLM project)
        assert len(self.free_indices) > 0, "No free indices"
        req_idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = req_idx
        self.index_to_req_id[req_idx] = req_id
        self.max_seq_len[req_idx] = prompt_len + max_tokens
        self.prompt_len.np[req_idx] = prompt_len
        prefill_len = len(all_token_ids)
        assert prefill_len >= prompt_len, f"prefill_len {prefill_len} < prompt_len {prompt_len}"
        self.prefill_len.np[req_idx] = prefill_len
        self.total_len.stage_write_elem(req_idx, prefill_len)
        self.all_token_ids.stage_write(req_idx, 0, all_token_ids)
        self.num_computed_prefill_tokens[req_idx] = num_computed_tokens
        self.num_computed_tokens_np[req_idx] = num_computed_tokens
        self.num_computed_tokens.stage_write_elem(req_idx, num_computed_tokens)

    def apply_staged_writes(self):
        ctx = self.__dict__.get("_b4_fuse")
        if ctx is None or not _LIVE["fusew"]:
            return orig_apply(self)
        writer, tensors, ptrs, strides = ctx
        self.prompt_len.copy_to_uva()          # vLLM's order: the two UVA-backed mirrors (host copies) first
        self.prefill_len.copy_to_uva()
        writer.apply(tensors, ptrs, strides)   # then every staged write of these tensors in one launch
        _S["fusew"] += 1

    RS.add_request = add_request
    RS.apply_staged_writes = apply_staged_writes
    _log(f"XDIET request-state patch installed in pid {os.getpid()}")


def _wrap_prepare_attn(ms_cls):
    from vllm.config.compilation import CUDAGraphMode
    orig = ms_cls.prepare_attn
    if getattr(orig, "_b4_nometa", False):
        return

    def prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config,
                     for_capture=False, ubatch_idx=0, *a, **k):
        if _LIVE["nometa"] and cudagraph_mode == CUDAGraphMode.FULL and not for_capture and not a and not k:
            return {}       # FULL replay: the graph reads the persistent buffers, not these objects
        return orig(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config,
                    for_capture, ubatch_idx, *a, **k)

    prepare_attn._b4_nometa = True
    ms_cls.prepare_attn = prepare_attn


def _patch_late(m):
    """LATEFREE: pjs's deferred KV-block frees (BLINK_DEFER_FREE, scheduler._e64_deferred) run after the next graph
    launch instead of right after the outputs (see the module docstring)"""
    if _S["patched_late"]:
        return
    _S["patched_late"] = True
    cls = m.EngineCoreProc
    orig_post = cls.post_step
    pjs_inside = bool(getattr(orig_post, "_e64_defer", False))

    def _flush(sched):
        late = sched.__dict__.get("_b4_late")
        while late:
            sched._free_blocks(late.pop(0))

    def post_step(self, model_executed):
        sched = self.scheduler
        d = sched.__dict__.get("_e64_deferred")
        late = sched.__dict__.get("_b4_late")
        if late is None:
            late = sched.__dict__["_b4_late"] = []
        if d:
            late.extend(d)      # pjs's post_step below then finds nothing to free (no sleep(0), no frees)
            d.clear()
        if late and self.batch_queue:       # a step is on the GPU: free the previous step's blocks in its shadow
            _flush(sched)
            _S["late"] += 1
        return orig_post(self, model_executed)

    post_step._b4_late = True
    cls.post_step = post_step
    from vllm.v1.core.sched import scheduler as _sm
    S = _sm.Scheduler
    orig_sched = S.schedule

    def schedule(self, *a, **k):
        late = self.__dict__.get("_b4_late")
        if late:
            bp = self.kv_cache_manager.block_pool
            if bp.get_num_free_blocks() < 0.25 * bp.num_gpu_blocks:
                _flush(self)
                _S["late_guard"] += 1
        return orig_sched(self, *a, **k)

    S.schedule = schedule
    _log(f"LATEFREE installed in pid {os.getpid()} (wraps {'pjs DEFER_FREE post_step' if pjs_inside else 'post_step WITHOUT pjs DEFER_FREE: inert'})")


def _patch_unify(m):
    """UNIFY: the engine core builds its KV cache with vLLM's hybrid KV-cache manager disabled (see the docstring)"""
    if _S["patched_unify"]:
        return
    _S["patched_unify"] = True
    EC = m.EngineCore
    orig_init = EC.__init__

    def __init__(self, vllm_config, *a, **k):
        try:
            import hashlib      # ours: the kv_cache_utils.py pin (_KVU_SHA)
            from vllm.v1.core import kv_cache_utils as _kvu
            with open(_kvu.__file__, "rb") as fh:
                pinned = hashlib.sha256(fh.read()).hexdigest() == _KVU_SHA
            sc = vllm_config.scheduler_config
            if not pinned:
                _log("UNIFY off: vllm/v1/core/kv_cache_utils.py is not vLLM 0.30.0 ced6857's")
            elif not sc.disable_hybrid_kv_cache_manager:
                sc.disable_hybrid_kv_cache_manager = True
                _log(f"UNIFY: hybrid KV-cache manager disabled in pid {os.getpid()} (one KV-cache group)")
        except Exception as exc:  # noqa: BLE001
            _log(f"UNIFY not applied: {exc!r}")
        r = orig_init(self, vllm_config, *a, **k)
        try:                    # ours: the KV-cache layout the engine core built (T1 reads this line)
            kc = self.scheduler.kv_cache_config
            _log(f"UNIFY: KV cache {len(kc.kv_cache_groups)} group(s), block sizes "
                 f"{[g.kv_cache_spec.block_size for g in kc.kv_cache_groups]}, {kc.num_blocks} blocks; hybrid manager "
                 f"{'off' if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager else 'on'}")
        except Exception as exc:  # noqa: BLE001
            _log(f"UNIFY: KV-cache layout not logged: {exc!r}")
        return r

    EC.__init__ = __init__


def _patch_core(m):
    """LEAN: Request.prefill_stats = None for every new request when the engine logs no statistics"""
    if UNIFY:
        _patch_unify(m)
    if LATEFREE:
        _patch_late(m)
    if not LEAN or _S["patched_core"]:
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
if XDIET:
    _PATCHES["vllm.v1.worker.gpu.states"] = _patch_states
if LEAN or LATEFREE or UNIFY:
    _PATCHES["vllm.v1.engine.core"] = _patch_core
if DFREE:
    _PATCHES["vllm.v1.core.sched.scheduler"] = _patch_sched


class _Finder(importlib.abc.MetaPathFinder):
    _busy = set()      # re-entrancy guard (pjs's blink_read.py / blink_last.py / blink_core.py hook these modules too)

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


if NOMETA or LEAN or DSAMP or DFREE or XDIET or LATEFREE or UNIFY or SWITCH_US > 0:
    _todo = False
    for _name, _fn in _PATCHES.items():
        if _name in sys.modules:
            _fn(sys.modules[_name])
        else:
            _todo = True
    if _todo:
        sys.meta_path.insert(0, _Finder())
