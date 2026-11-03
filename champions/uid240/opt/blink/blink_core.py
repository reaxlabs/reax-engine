# SPDX-License-Identifier: Apache-2.0
"""Convoy batching in the vLLM engine core (the spawned engine-core process).

Under a closed-loop client with C requests in flight, the requests of one engine step finish together and their
clients send their next requests together a few ms later (a convoy). vLLM's async scheduling starts a new step as
soon as any request waits, so a request that arrived while a big step was running runs alone in the next step (a
~20 ms step for one decision) while the convoy waits behind it, and the pattern repeats (big step, tiny step, ...).
This policy keeps the convoy together:
  defer: while a step of N >= MIN requests is in flight, a waiting set of fewer than DEFER_FRAC * N requests is not
         scheduled yet; it joins the convoy after that step;
  hold:  when the core has just finished a step of F >= MIN requests and fewer than (stragglers + F - SLACK) requests
         wait, it waits for the convoy: until that many wait, or FIRST_MS after the step without any arrival, or
         QUIET_MS without a further arrival, or HOLD_MS in total.
Numerics are untouched: the kernels are batch-invariant, only the grouping of requests into steps changes.

  BLINK_PIPE=1: eager pipelining (step_pipe below), BLINK_PIPE_POLL_MS (0.25)
  BLINK_CONVOY=1 (on), BLINK_CONVOY_MIN (4), BLINK_CONVOY_DEFER_FRAC (0.5), BLINK_CONVOY_SLACK (1),
  BLINK_CONVOY_FIRST_MS (10), BLINK_CONVOY_QUIET_MS (3), BLINK_CONVOY_HOLD_MS (25)
  development only: BLINK_TUNE_FILE (JSON, re-read every 0.5 s: convoy, c_min, c_defer_frac, c_slack, c_first_ms,
  c_quiet_ms, c_hold_ms), BLINK_CORE_STATS (JSON counters written every 2 s), BLINK_CORE_TRACE (one JSON line per step / hold)
Installed by sitecustomize.py; patches vllm.v1.engine.core.EngineCoreProc on import.

bj/blink overlay (team ruj, 2026-10-28; see /opt/blink/NOTICE.bj): defaults BLINK_PIPE_FULL=1 (one group per
step; pjs's two groups of 8 with BLINK_PIPE_FULL=0), BLINK_PIPE_ADAPT=1, BLINK_PIPE_QUIET_MS=1.0,
BLINK_PIPE_CWIN_S=60.0, BLINK_PIPE_PRESCHED=1; with FULL the launch cap (BLINK_PIPE_CAP) is ignored.
"""
from __future__ import annotations

import importlib.abc
import json
import os
import queue
import sys
import time


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


class _P:
    on = os.environ.get("BLINK_CONVOY", "0") == "1"
    min = _f("BLINK_CONVOY_MIN", 4)
    defer_frac = _f("BLINK_CONVOY_DEFER_FRAC", 0.5)
    slack = _f("BLINK_CONVOY_SLACK", 1)
    first = _f("BLINK_CONVOY_FIRST_MS", 10) / 1e3
    quiet = _f("BLINK_CONVOY_QUIET_MS", 3) / 1e3
    hold = _f("BLINK_CONVOY_HOLD_MS", 25) / 1e3
    pipe = os.environ.get("BLINK_PIPE", "0") == "1"
    poll = _f("BLINK_PIPE_POLL_MS", 0.25) / 1e3
    cwin = _f("BLINK_PIPE_CWIN_S", 60.0)          # window of the concurrency estimate (max requests in the engine)
    eager = os.environ.get("BLINK_PIPE_EAGER", "0") == "1"   # 1: launch for any waiting request (no group gather)
    half = os.environ.get("BLINK_PIPE_HALF", "0") == "1"     # 1: an idle GPU waits for half the clients (balance)
    full = os.environ.get("BLINK_PIPE_FULL", "1") == "1"     # 1: one group: an idle GPU waits for all the clients
    cap = _f("BLINK_PIPE_CAP", 0)       # > 0: a launch takes at most ceil(cap * concurrency) new requests (balance)
    idle_wait = _f("BLINK_PIPE_IDLE_MS", 4) / 1e3            # ... but at most this long
    # bj/blink (team ruj): BLINK_PIPE_ADAPT=1 with BLINK_PIPE_FULL: an idle GPU waits for as many requests as the
    # last step finished (+ the stragglers that were already waiting), not for the 1 s concurrency estimate, so a
    # drop to concurrency 1 costs one idle wait instead of a second of them; FULL ignores the launch cap
    adapt = os.environ.get("BLINK_PIPE_ADAPT", "1") == "1"
    # bj/blink: BLINK_PIPE_QUIET_MS > 0 with BLINK_PIPE_FULL: an idle GPU also launches once at least half the
    # expected group waits and no new request arrived for this long (the burst of a closed-loop round is over), so a
    # group does not wait the full idle time for a client that is not coming
    qwait = _f("BLINK_PIPE_QUIET_MS", 1.0) / 1e3
    # bj/blink: BLINK_PIPE_PRESCHED=1 with BLINK_PIPE_FULL: while an idle GPU waits for the rest of the group, the
    # requests that already arrived are scheduled (KV blocks, prefix-cache lookups: ~0.1 ms of host time each); at
    # the launch the partial scheduler outputs are merged into the one step, so only the last arrivals are scheduled
    # on the critical path. Requests other than ADD (abort, utility) that arrive meanwhile wait for that launch.
    presched = os.environ.get("BLINK_PIPE_PRESCHED", "1") == "1"
    pre_min = _f("BLINK_PIPE_PRESCHED_MIN", 4)     # schedule ahead once this many have arrived (fewer calls)
    # E62 EX-1c: a lone alt copy (its main launched without it) is launched once it has waited alone this long
    lone = _f("BLINK_HEDGE_LONE_MS", 20) / 1e3
    # E72 L1 (BLINK_E72_LEANCACHE=1): lean prefix-cache registration (see _lean_install)
    lean = os.environ.get("BLINK_E72_LEANCACHE", "0") == "1"
    # E72 L4 (BLINK_E72_ISL=1; live: tune "e72_isl"): skip the scheduler's full-sequence admission pre-check for a
    # request that is scheduled whole in this step (see _lean_install)
    isl = os.environ.get("BLINK_E72_ISL", "0") == "1"
    # E72 W1 (BLINK_E72_GPUWAIT=1; live: tune "e72_gpuwait"): near the expected end of a running step, wait on its
    # output copy event instead of re-polling every BLINK_PIPE_POLL_MS (see step_pipe)
    gpuwait = os.environ.get("BLINK_E72_GPUWAIT", "0") == "1"
    t_read = 0.0


TUNE_FILE = os.environ.get("BLINK_TUNE_FILE", "")
_cp = os.environ.get("BLINK_CPROF", "")       # development only: "N:M:dir"
CPROF = (int(_cp.split(":")[0]), int(_cp.split(":")[1]), _cp.split(":", 2)[2]) if _cp else None
_CP = {}
STATS_FILE = os.environ.get("BLINK_CORE_STATS", "")
TRACE_FILE = os.environ.get("BLINK_CORE_TRACE", "")
# E64 rank 2(b1): BLINK_LEAN_OUT=1 drops a telemetry-only walk from the output update (blink_out.py)
if os.environ.get("BLINK_LEAN_OUT", "0") == "1":
    try:
        import blink_out  # noqa: F401
    except Exception as _exc:  # noqa: BLE001
        print(f"[blink-core] blink_out unavailable ({_exc!r})", file=sys.stderr, flush=True)
_S = {"steps": 0, "step_reqs": 0, "step_tokens": 0, "holds": 0, "hold_ms": 0.0, "defers": 0, "why": {},
      "sizes": {}, "t_write": 0.0}
_T0 = time.monotonic()


def _params():
    if TUNE_FILE:
        now = time.monotonic()
        if now - _P.t_read > 0.5:
            _P.t_read = now
            try:
                with open(TUNE_FILE) as fh:
                    j = json.load(fh)
                _P.on = bool(j.get("convoy", _P.on))
                _P.pipe = bool(j.get("pipe", _P.pipe))
                _P.eager = bool(j.get("pipe_eager", _P.eager))
                _P.half = bool(j.get("pipe_half", _P.half))
                _P.full = bool(j.get("pipe_full", _P.full))
                if "pipe_cap" in j:
                    _P.cap = float(j["pipe_cap"])
                if "pipe_idle_ms" in j:
                    _P.idle_wait = float(j["pipe_idle_ms"]) / 1e3
                if "pipe_poll_ms" in j:
                    _P.poll = float(j["pipe_poll_ms"]) / 1e3
                if "pipe_cwin_s" in j:
                    _P.cwin = float(j["pipe_cwin_s"])
                _P.adapt = bool(j.get("pipe_adapt", _P.adapt))
                if "pipe_quiet_ms" in j:
                    _P.qwait = float(j["pipe_quiet_ms"]) / 1e3
                _P.presched = bool(j.get("pipe_presched", _P.presched))
                if "hedge_lone_ms" in j:
                    _P.lone = float(j["hedge_lone_ms"]) / 1e3
                if "pipe_presched_min" in j:
                    _P.pre_min = float(j["pipe_presched_min"])
                _P.lean = bool(j.get("e72_lean_cache", _P.lean))
                _P.isl = bool(j.get("e72_isl", _P.isl))
                _P.gpuwait = bool(j.get("e72_gpuwait", _P.gpuwait))
                for k in ("min", "defer_frac", "slack"):
                    if "c_" + k in j:
                        setattr(_P, k, float(j["c_" + k]))
                for k in ("first", "quiet", "hold"):
                    if "c_" + k + "_ms" in j:
                        setattr(_P, k, float(j["c_" + k + "_ms"]) / 1e3)
            except Exception:  # noqa: BLE001
                pass
    return _P


_TB = {"buf": [], "t": 0.0}


def _trace(rec):
    """buffered (development only): one write every 2 s instead of an open/write/close per event"""
    if not TRACE_FILE:
        return
    try:
        now = time.monotonic()
        rec["t"] = round((now - _T0) * 1e3, 3)
        rec["wt"] = round(time.time(), 6)
        _TB["buf"].append(json.dumps(rec))
        if now - _TB["t"] > 2.0:
            _TB["t"] = now
            with open(TRACE_FILE, "a") as fh:
                fh.write("\n".join(_TB["buf"]) + "\n")
            _TB["buf"].clear()
    except Exception:  # noqa: BLE001
        pass


def _stats():
    if not STATS_FILE:
        return
    now = time.monotonic()
    if now - _S["t_write"] < 2.0:
        return
    _S["t_write"] = now
    try:
        with open(STATS_FILE + ".tmp", "w") as fh:
            json.dump({k: v for k, v in _S.items() if k != "t_write"}, fh)
        os.replace(STATS_FILE + ".tmp", STATS_FILE)
    except Exception:  # noqa: BLE001
        pass


# E66 (BLINK_HEDGE=1, blink_hedge.py): the RMS-flip hedge's alt copies (request ids starting with "h66-") ride along
# with their main requests (the frontend sends each alt before its main); every count below that sizes the convoy
# group (waiting, in flight, finished, prescheduled) counts main requests only, so a group is still the C clients
_HEDGE = os.environ.get("BLINK_HEDGE", "0") == "1"
_VISION = os.environ.get("BLINK_VISION", "0") == "1"   # v2d: picture requests (blink_pmode.py) use their own step loop


class _PP:
    """v2d O1: the step loop for picture traffic. BLINK_PM_LOOP=stock: vLLM's own step_with_batch_queue (M0, v5's
    loop); BLINK_PM_LOOP=pipe: step_pipe with two pipelined groups (an idle GPU waits for half the clients at most
    BLINK_PM_IDLE_MS; a launch behind a running step takes at most ceil(BLINK_PM_CAP x concurrency) new requests and
    goes once the rest of the clients wait or the running step finishes). Scheduling only: the kernels are
    batch-invariant and every picture request is scheduled whole (blink_pmode's filter)."""
    loop = os.environ.get("BLINK_PM_LOOP", "stock")
    full = False
    eager = False
    half = os.environ.get("BLINK_PM_HALF", "1") == "1"
    cap = _f("BLINK_PM_CAP", 0.5)
    idle_wait = _f("BLINK_PM_IDLE_MS", 10) / 1e3
    poll = _f("BLINK_PIPE_POLL_MS", 0.25) / 1e3
    cwin = _f("BLINK_PM_CWIN_S", 10.0)     # concurrency estimate window (the start-up warm-up must not inflate it)
    adapt = False
    qwait = 0.0
    presched = False
    pre_min = 1 << 30
    lone = _f("BLINK_HEDGE_LONE_MS", 20) / 1e3
    gpuwait = False
    t_read = 0.0


_PM_TUNE = os.environ.get("BLINK_PM_TUNE_FILE", "")


def _pp_params():
    """development only: live P-loop keys pm_loop / pm_half / pm_cap / pm_idle_ms in BLINK_PM_TUNE_FILE"""
    if _PM_TUNE:
        now = time.monotonic()
        if now - _PP.t_read > 0.5:
            _PP.t_read = now
            try:
                with open(_PM_TUNE) as fh:
                    j = json.load(fh)
                _PP.loop = str(j.get("pm_loop", _PP.loop))
                _PP.half = bool(j.get("pm_half", _PP.half))
                _PP.cap = float(j.get("pm_cap", _PP.cap))
                _PP.idle_wait = float(j.get("pm_idle_ms", _PP.idle_wait * 1e3)) / 1e3
            except Exception:  # noqa: BLE001
                pass
    return _PP
_ALT = "h66-"


def _nmain(ids):
    """number of main (non-alt) request ids"""
    if not _HEDGE:
        return len(ids)
    return sum(1 for r in ids if not str(r).startswith(_ALT))


def _nmain_reqs(reqs):
    if not _HEDGE:
        return len(reqs)
    return sum(1 for r in reqs if not str(getattr(r, "request_id", "")).startswith(_ALT))


_MAIN = "m66-"


def _orphans(core, pre):
    """hedged mains (m66-) waiting or prescheduled whose alt copy (h66-, sent right after its main) has not arrived"""
    if not _HEDGE:
        return 0
    s = core.scheduler
    m = a = 0
    for q in (s.waiting, getattr(s, "skipped_waiting", ()) or ()):
        for r in q:
            rid = str(getattr(r, "request_id", ""))
            if rid.startswith(_MAIN):
                m += 1
            elif rid.startswith(_ALT):
                a += 1
    for o in pre or ():
        for rid in o.num_scheduled_tokens:
            rid = str(rid)
            if rid.startswith(_MAIN):
                m += 1
            elif rid.startswith(_ALT):
                a += 1
    return m - a


def _alts_waiting(core, pre):
    """E62 EX-1: True when an alt copy (h66-) is waiting or prescheduled"""
    s = core.scheduler
    for q in (s.waiting, getattr(s, "skipped_waiting", ()) or ()):
        for r in q:
            if str(getattr(r, "request_id", "")).startswith(_ALT):
                return True
    for o in pre or ():
        for rid in o.num_scheduled_tokens:
            if str(rid).startswith(_ALT):
                return True
    return False


def _waiting(core):
    s = core.scheduler
    return _nmain_reqs(s.waiting) + _nmain_reqs(getattr(s, "skipped_waiting", ()) or ())


def _finished(outs):
    n = 0
    for o in (outs or {}).values():
        for r in getattr(o, "outputs", None) or ():
            if getattr(r, "finish_reason", None) is not None and not (
                    _HEDGE and str(getattr(r, "request_id", "")).startswith(_ALT)):
                n += 1
    return n


def _patch(m):
    cls = m.EngineCoreProc
    orig_bq = cls.step_with_batch_queue

    def _drain(self):
        while not self.input_queue.empty():
            self._handle_client_request(*self.input_queue.get_nowait())

    def _pop(self):
        """Block on the oldest in-flight batch and process its output (what step_with_batch_queue does when its queue
        is full), without scheduling a new batch first."""
        future, scheduler_output, exec_model_fut = self.batch_queue.pop()
        with self.capture_iteration_details(scheduler_output) as details, self.log_error_detail(scheduler_output):
            model_output = future.result()
            if model_output is None:
                exec_model_fut.result()
                raise RuntimeError("unexpected error")
        st = getattr(self, "_e59", None)
        if st is not None and st.get("t_launch") is not None:     # E72 W1: EMA of launch -> output ready
            d = time.monotonic() - st.pop("t_launch")
            g = st.get("gpu_ema")
            st["gpu_ema"] = d if g is None else (0.8 * g + 0.2 * d)
        self._process_aborts_queue()
        t_u = time.monotonic()
        outs = self.scheduler.update_from_output(scheduler_output, model_output)
        self._attach_iteration_details(outs, details)
        if TRACE_FILE:      # development only: host time of the output processing (ms)
            _trace({"ev": "upd", "ms": round((time.monotonic() - t_u) * 1e3, 3)})
        return outs, True

    def _hold(self, P):
        st = self._e59
        t0 = time.monotonic()
        target = min(st["w_done"] + st["F"] - P.slack, self.scheduler.max_num_running_reqs)
        deadline = st["t_done"] + P.hold
        last = None
        why = "target"
        while _waiting(self) < target:
            now = time.monotonic()
            if now >= deadline:
                why = "max"
                break
            if last is None:
                wait = min(deadline, st["t_done"] + P.first) - now
                if wait <= 0:
                    why = "first"
                    break
            else:
                wait = min(deadline, last + P.quiet) - now
                if wait <= 0:
                    why = "quiet"
                    break
            try:
                req = self.input_queue.get(timeout=wait)
            except queue.Empty:
                continue
            self._handle_client_request(*req)
            _drain(self)
            last = time.monotonic()
        dt = (time.monotonic() - t0) * 1e3
        _S["holds"] += 1
        _S["hold_ms"] += dt
        _S["why"][why] = _S["why"].get(why, 0) + 1
        _trace({"ev": "hold", "ms": round(dt, 3), "why": why, "w": _waiting(self), "target": target})

    def _after(self, res, top_before):
        outs, _ = res
        bq = self.batch_queue
        if bq and (top_before is None or bq[0][1] is not top_before):
            so = bq[0][1]
            n = len(so.num_scheduled_tokens)
            if n:
                _S["steps"] += 1
                _S["step_reqs"] += n
                _S["step_tokens"] += so.total_num_scheduled_tokens
                k = str(n)
                _S["sizes"][k] = _S["sizes"].get(k, 0) + 1
                _trace({"ev": "sched", "n": n, "tok": so.total_num_scheduled_tokens, "w": _waiting(self),
                        "launch_ms": getattr(self, "_e59", {}).pop("launch_ms", None)})
        if outs:
            f = _finished(outs)
            if f:
                st = self._e59
                st["F"] = f
                st["t_done"] = time.monotonic()
                st["w_done"] = _waiting(self)
                _trace({"ev": "done", "F": f, "w": st["w_done"]})
        _stats()
        return res

    def _ready(fut):
        ev = getattr(getattr(fut, "async_output", None), "copy_event", None)
        return None if ev is None else ev.query()

    def _inflight(bq):
        return sum(_nmain(so.num_scheduled_tokens) for _, so, _ in bq)

    def _conc(self, P, n_now):
        """concurrency estimate: the most requests seen in the engine (waiting + in flight) within the last cwin s"""
        st = self._e59
        now = time.monotonic()
        if n_now >= st.get("c_max", 0) or now - st.get("t_max", 0.0) > P.cwin:
            st["c_max"], st["t_max"] = n_now, now
        return st["c_max"]

    def _schedule_capped(self, sched, P, c):
        """sched.schedule(), taking at most ceil(cap * c) new requests when BLINK_PIPE_CAP is set: the closed-loop
        clients then settle into two groups of ~c/2 instead of a lopsided split (e.g. 15 + 1), whose short step cannot
        hide the big group's turnaround. Requests not taken stay waiting and join the next launch."""
        g = int(-(-P.cap * c // 1)) if P.cap > 0 and not P.full else 0
        old = sched.max_num_running_reqs
        lim = len(sched.running) + g
        if g <= 0 or lim >= old:
            return sched.schedule(self._should_throttle_prefills())
        sched.max_num_running_reqs = lim
        try:
            return sched.schedule(self._should_throttle_prefills())
        finally:
            sched.max_num_running_reqs = old

    add_type = m.EngineCoreRequestType.ADD

    def _mergeable(so):
        """a scheduler output of new requests only, nothing that ties it to a step of its own"""
        try:
            return (not so.scheduled_cached_reqs.req_ids and not so.scheduled_spec_decode_tokens
                    and not so.scheduled_encoder_inputs and getattr(so, "kv_connector_metadata", None) is None
                    and getattr(so, "ec_connector_metadata", None) is None
                    and getattr(so, "ec_manager_metadata", None) is None
                    and not getattr(so, "has_structured_output_requests", False)
                    and not getattr(so, "num_spec_tokens_to_schedule", 0))
        except Exception:  # noqa: BLE001
            return False

    def _merge(outs):
        """one scheduler output for the requests of several schedule() calls of the same idle period"""
        import dataclasses
        if len(outs) == 1:
            return outs[0]
        mo = dataclasses.replace(outs[0])
        mo.scheduled_new_reqs = [r for o in outs for r in o.scheduled_new_reqs]
        nst = {}
        for o in outs:
            nst.update(o.num_scheduled_tokens)
        mo.num_scheduled_tokens = nst
        mo.total_num_scheduled_tokens = sum(nst.values())
        mo.finished_req_ids = set().union(*(o.finished_req_ids for o in outs))
        mo.free_encoder_mm_hashes = [h for o in outs for h in o.free_encoder_mm_hashes]
        ncp = [o.num_common_prefix_blocks for o in outs if o.num_common_prefix_blocks]
        mo.num_common_prefix_blocks = [min(x) for x in zip(*ncp)] if len(ncp) == len(outs) else (
            [0] * len(ncp[0]) if ncp else outs[0].num_common_prefix_blocks)
        pre = [o.preempted_req_ids for o in outs if o.preempted_req_ids]
        mo.preempted_req_ids = set().union(*pre) if pre else outs[0].preempted_req_ids
        for name in ("new_block_ids_to_zero", "kv_cache_block_copies"):
            vals = [getattr(o, name) for o in outs if getattr(o, name, None)]
            setattr(mo, name, [b for v in vals for b in v] if vals else getattr(outs[0], name))
        bt = [o.block_table_updates for o in outs if getattr(o, "block_table_updates", None)]
        if bt:
            d = {}
            for x in bt:
                d.update(x)
            mo.block_table_updates = d
        mo.pending_structured_output_tokens = any(o.pending_structured_output_tokens for o in outs)
        return mo

    def _enqueue(self, sched, so):
        """execute one scheduler output (as the launch below does) and put it in the batch queue"""
        with self.log_error_detail(so):
            exec_future = self.model_executor.execute_model(so, non_block=True)
        if so.total_num_scheduled_tokens > 0:
            future = self.model_executor.sample_tokens(sched.get_grammar_bitmask(so), non_block=True)
            self.batch_queue.appendleft((future, so, exec_future))
        else:
            self.batch_queue.appendleft((exec_future, so, exec_future))

    def _flush_deferred(self):
        for req in self._e59.pop("deferred", None) or ():
            self._handle_client_request(*req)

    def _take(self, req):
        """handle one input request; while partial scheduler outputs wait for their launch, defer everything but ADD"""
        st = self._e59
        if st.get("pre") and req[0] != add_type:
            st.setdefault("deferred", []).append(req)
            return
        self._handle_client_request(*req)

    def _take_all(self):
        while not self.input_queue.empty():
            _take(self, self.input_queue.get_nowait())

    def step_pipe(self, P, top):
        """Batch pipelining. vLLM's loop blocks on the oldest batch as soon as an empty batch has filled its 2-deep
        queue, so requests that arrive while a batch runs are launched only after it has finished (the GPU idles for
        the launch). Here, while a batch runs, the next one is launched (it queues behind it on the stream) as soon as
        the expected group has arrived - all requests of the estimated concurrency that are not in flight - or, at the
        latest, when the running batch finishes. With no batch in flight a waiting request is launched at once, or
        (BLINK_PIPE_HALF) once half the clients wait or BLINK_PIPE_IDLE_MS passed, which keeps two balanced groups."""
        bq = self.batch_queue
        sched = self.scheduler
        st = self._e59
        while True:
            w0 = _waiting(self)
            pre = st.get("pre")
            w = w0 + (sum(_nmain(o.num_scheduled_tokens) for o in pre) if pre else 0)
            fly = _inflight(bq)
            c = _conc(self, P, w + fly)
            if bq:
                st.pop("idle_t", None)
                if P.full:
                    want = 1 << 30      # one group: nothing is launched behind a running batch
                else:
                    want = 1 if P.eager else max(1, min(c - fly, sched.max_num_running_reqs))
            elif (P.half or P.full) and w > 0:
                # idle GPU: launch once half (or, BLINK_PIPE_FULL, all) the clients wait, or after idle_wait
                now = time.monotonic()
                t_idle = st.setdefault("idle_t", now)
                grp = c if P.full else c // 2
                if P.full and P.adapt:
                    grp = max(1, min(grp, st.get("F", 0) + st.get("w_done", 0)))
                if w != st.get("w_seen"):
                    st["w_seen"], st["t_arr"] = w, now
                quiet = P.full and P.qwait > 0 and 2 * w >= grp and now - st.get("t_arr", now) >= P.qwait
                want = 1 if (now - t_idle >= P.idle_wait or quiet) else max(1, min(grp, sched.max_num_running_reqs))
                if _HEDGE and now - t_idle < P.idle_wait and _orphans(self, pre) > 0:
                    want = max(want, w + 1)     # E66: a hedged main's alt copy is on its way; launch the pair together
            else:
                want = 1
            lone = False
            if not bq and w == 0:
                if sched.running:
                    # E62 C1: chunked prefill left a remainder in scheduler.running (a step above the token budget);
                    # w counts waiting / prescheduled mains only, so launch it now (vLLM schedules running first)
                    w = want = 1
                    _S["remainder"] = _S.get("remainder", 0) + 1
                    _trace({"ev": "remainder", "n": len(sched.running)})
                elif _HEDGE and _alts_waiting(self, pre):
                    # E62 EX-1 / EX-1c: an alt copy whose main was launched without it (the alt reached the core after
                    # the pair hold) waits alone; w counts mains only, so it was never launched when nothing else
                    # arrived (c1). It is launched once it has waited alone P.lone; before that this call returns as
                    # without the fix (at c16 the alt joins the next convoy step). Answers unchanged.
                    lone = True
                    now_l = time.monotonic()
                    if now_l - st.setdefault("lone_t", now_l) >= P.lone:
                        lone = False
                        w = want = 1
                        _S["lone_alt"] = _S.get("lone_alt", 0) + 1
                        _trace({"ev": "lone_alt", "n": _S["lone_alt"]})
            if not lone:
                st.pop("lone_t", None)
            if w > 0 and w < want and not bq:
                if P.presched and P.full and w0 >= P.pre_min:
                    # schedule the arrived requests now; they are launched with the rest of the group
                    so_p = sched.schedule(self._should_throttle_prefills())
                    if _mergeable(so_p):
                        st.setdefault("pre", []).append(so_p)
                        _S["presched"] = _S.get("presched", 0) + 1
                    else:       # unexpected: launch the earlier partial outputs, then this one, as two steps
                        pre = st.pop("pre", None)
                        if pre:
                            _enqueue(self, sched, _merge(pre))
                        _enqueue(self, sched, so_p)
                        _flush_deferred(self)
                        return _after(self, (None, True), top)
                try:      # idle GPU, group still forming: wait briefly for more requests
                    req = self.input_queue.get(timeout=P.poll)
                    _take(self, req)
                    _take_all(self)
                except queue.Empty:
                    pass
                continue
            if len(bq) < self.batch_queue_size and w >= want:
                st.pop("idle_t", None)
                st.pop("w_seen", None)
                t_a = time.monotonic()
                so = _schedule_capped(self, sched, P, c)
                pre = st.pop("pre", None)
                if pre:
                    if _mergeable(so):
                        so = _merge(pre + [so])
                    else:       # unexpected: the partial outputs first, as a step of their own
                        _enqueue(self, sched, _merge(pre))
                t_b = time.monotonic()
                # every scheduler output goes to the model runner (it carries the finished-request bookkeeping)
                with self.log_error_detail(so):
                    exec_future = self.model_executor.execute_model(so, non_block=True)
                t_c = time.monotonic()
                if so.total_num_scheduled_tokens > 0:
                    grammar_output = sched.get_grammar_bitmask(so)
                    future = self.model_executor.sample_tokens(grammar_output, non_block=True)
                    if TRACE_FILE:      # development only: host time of the launch (ms): schedule / execute / sample
                        st["launch_ms"] = (round((t_b - t_a) * 1e3, 3), round((t_c - t_b) * 1e3, 3),
                                           round((time.monotonic() - t_c) * 1e3, 3))
                    bq.appendleft((future, so, exec_future))
                    st["t_launch"] = time.monotonic()      # E72 W1: step duration estimate
                    _flush_deferred(self)
                    return _after(self, (None, True), top)
                bq.appendleft((exec_future, so, exec_future))   # empty batch: popped in order, as vLLM does
                _flush_deferred(self)
            if not bq:
                return None, False
            fut = bq[-1][0]
            if len(bq) >= self.batch_queue_size or _ready(fut) in (True, None):
                return _after(self, _pop(self), top)
            # E72 W1: the main thread notices a finished step only at its next poll (every P.poll = 0.25 ms, ~0.15 ms
            # late on average); within 1.5 ms of the step's expected end, with one batch in flight under PIPE_FULL
            # (nothing can launch behind it) and no input queued, block on the step's output copy event instead
            # (CUDA event sync, the GIL is released), so update_from_output starts as the copy lands. Timing only.
            if P.gpuwait and P.full and len(bq) == 1 and self.input_queue.empty():
                ev = getattr(getattr(fut, "async_output", None), "copy_event", None)
                t_l, g = st.get("t_launch"), st.get("gpu_ema")
                if ev is not None and t_l is not None and g and time.monotonic() >= t_l + g - 0.0015:
                    ev.synchronize()
                    _S["gpuwait"] = _S.get("gpuwait", 0) + 1
                    continue
            # the oldest batch is still on the GPU and the pipeline has room: wait for it or for a new request
            try:
                req = self.input_queue.get(timeout=P.poll)
            except queue.Empty:
                continue
            self._handle_client_request(*req)
            _drain(self)

    def _pm_active(self):
        """v2d (BLINK_VISION): a picture request waits, runs or is in flight: vLLM's own step loop schedules them
        (no PIPE group gather, no pre-scheduling; blink_pmode's scheduler filter keeps them out of text steps)"""
        s = self.scheduler
        for q in (s.waiting, getattr(s, "skipped_waiting", ()) or (), s.running):
            for r in q:
                if r.mm_features:
                    return True
        for _, so, _ in self.batch_queue:
            for r in so.scheduled_new_reqs:
                if r.mm_features:
                    return True
        return False

    def step_with_batch_queue(self):
        P = _params()
        if not hasattr(self, "_e59"):
            self._e59 = {"F": 0, "t_done": 0.0, "w_done": 0}
        bq = self.batch_queue
        top = bq[0][1] if bq else None
        if _VISION and not self._e59.get("pre") and _pm_active(self):
            _S["pm_steps"] = _S.get("pm_steps", 0) + 1
            pp = _pp_params()
            if pp.loop == "pipe":
                return step_pipe(self, pp, top)
            return _after(self, orig_bq(self), top)
        if P.pipe and not P.on:
            if CPROF:
                return _cprof_step(self, P, top)
            return step_pipe(self, P, top)
        if not P.on:
            if TRACE_FILE or STATS_FILE:
                return _after(self, orig_bq(self), top)
            return orig_bq(self)
        if bq:
            n_fly = sum(len(so.num_scheduled_tokens) for _, so, _ in bq)
            w = _waiting(self)
            if n_fly >= P.min and 0 < w < P.defer_frac * n_fly:
                _S["defers"] += 1
                _trace({"ev": "defer", "fly": n_fly, "w": w})
                return _after(self, _pop(self), top)
            return _after(self, orig_bq(self), top)
        st = self._e59
        if (st["F"] >= P.min and time.monotonic() - st["t_done"] < P.hold
                and _waiting(self) < st["w_done"] + st["F"] - P.slack):
            _hold(self, P)
            st["F"] = 0          # one hold per finished convoy
        return _after(self, orig_bq(self), top)

    def _cprof_step(self, P, top):
        """development only (BLINK_CPROF="N:M:dir"): cProfile of the engine-core loop for calls N..N+M of the step
        function, written to <dir>/core-<pid>.pstats (the wait in input_queue.get shows up as idle time)"""
        st = _CP
        st["n"] = st.get("n", 0) + 1
        n0, n1 = CPROF[0], CPROF[0] + CPROF[1]
        if st["n"] == n0:
            import cProfile
            st["p"] = cProfile.Profile()
        prof = st.get("p")
        if prof is None or st["n"] > n1:
            return step_pipe(self, P, top)
        prof.enable()
        try:
            return step_pipe(self, P, top)
        finally:
            prof.disable()
            if st["n"] == n1:
                try:
                    prof.dump_stats(os.path.join(CPROF[2], f"core-{os.getpid()}.pstats"))
                except Exception:  # noqa: BLE001
                    pass
                st["p"] = None

    cls.step_with_batch_queue = step_with_batch_queue

    _lean_install()


    # E64 rank 2(b2), BLINK_DEFER_FREE=1: a finished request's KV blocks are freed in post_step, after the step's outputs
    # went to the output queue and the output thread had the GIL, instead of inside update_from_output on the output
    # path. vLLM's own delay_free_blocks path keeps every other bookkeeping step in place (finished ids, encoder
    # cache); the deferred _free_blocks run in their original order before the next schedule(), so the block pool,
    # the prefix cache and every step are unchanged. Not with KV / EC connectors or deferred block frees.
    if os.environ.get("BLINK_DEFER_FREE", "0") == "1":
        from vllm.v1.core.sched import scheduler as _sm
        S = _sm.Scheduler
        if not getattr(S._free_request, "_e64_defer", False):
            orig_free = S._free_request

            def _free_request(self, request, delay_free_blocks=False):
                if (not delay_free_blocks and self.connector is None and self.ec_connector is None
                        and not self.defer_block_free):
                    r = orig_free(self, request, delay_free_blocks=True)
                    self.__dict__.setdefault("_e64_deferred", []).append(request)
                    return r
                return orig_free(self, request, delay_free_blocks)

            _free_request._e64_defer = True
            S._free_request = _free_request
        orig_post = cls.post_step
        if not getattr(orig_post, "_e64_defer", False):
            def post_step(self, model_executed):
                d = self.scheduler.__dict__.get("_e64_deferred")
                if d:
                    time.sleep(0)       # yield the GIL: the output thread sends this step's outputs first
                    sched = self.scheduler
                    while d:
                        sched._free_blocks(d.pop(0))
                return orig_post(self, model_executed)

            post_step._e64_defer = True
            cls.post_step = post_step
        print("[blink-core] E64: KV block frees after the step's outputs are queued (BLINK_DEFER_FREE)",
              file=sys.stderr, flush=True)

    if TRACE_FILE:
        orig_hcr = cls._handle_client_request
        add_type = m.EngineCoreRequestType.ADD

        def _handle_client_request(self, request_type, request):
            if request_type == add_type:
                try:
                    req = request[0]
                    _trace({"ev": "arr", "rid": str(req.request_id)[:12],
                            "lag": round((time.time() - req.arrival_time) * 1e3, 3), "w": _waiting(self)})
                except Exception:  # noqa: BLE001
                    pass
            return orig_hcr(self, request_type, request)

        cls._handle_client_request = _handle_client_request


def _lean_install():
    """E72 L1, lean prefix-cache registration (BLINK_E72_LEANCACHE=1; live: tune "e72_lean_cache").

    The scheduler registers every full block of a new request's prompt in the prefix cache, in every KV-cache group
    (Blink: 5 sliding-window groups of 16-token blocks and one full-attention group of 32-token blocks), and evicts the
    registrations again when the blocks are reused - about 66 hash-map insertions and later evictions per 173-token
    prompt, ~0.1 ms of engine-core Python per request on the arrival path. With this switch, a new request registers
    its prompt blocks only up to its prefix-cache hit plus one scheduler block (32 tokens, the hit granularity), so a
    prefix that the next request shares is still found and extended block by block, and the shared system prompt
    stays cached; the request-specific tail is not registered (the validator makes every timed item unique with a
    random record id at the start of the state, so a tail is only ever hit again by a repeated item).
    Exact by construction: the cache decides only which blocks a request's KV lives in and whether an identical
    token prefix is computed again or read back; the answer is the same bit for bit either way (batch-invariant
    kernels, E64 lesson 9; the 8192-item gate repeats 1,536 items within one start). Computed tokens of fresh items
    are unchanged (their only hit is the shared prefix)."""
    from vllm.v1.core import kv_cache_coordinator as _kvc
    from vllm.v1.core import kv_cache_manager as _kvm
    KM = _kvm.KVCacheManager
    if not getattr(KM.allocate_slots, "_e72_lean", False):
        orig_alloc = KM.allocate_slots

        def allocate_slots(self, request, num_new_tokens, *a, **k):
            if (_P.lean and request.num_computed_tokens == 0 and not request.num_output_tokens
                    and getattr(request, "_e72_cc", None) is None and not request.mm_features):
                hit = k["num_new_computed_tokens"] if "num_new_computed_tokens" in k else (a[0] if a else 0)
                hit += k.get("num_external_computed_tokens", 0)
                request._e72_cc = hit + self.coordinator.scheduler_block_size
                _S["lean"] = _S.get("lean", 0) + 1
            if _P.isl and k.get("full_sequence_must_fit") and not a:
                # E72 L4: the full-sequence admission pre-check counts the blocks for min(num_tokens, max_model_len)
                # tokens with the recycling-aware cap, which never asks for more than the per-step count that
                # allocate_slots checks right after; when this step schedules the whole remaining sequence (no
                # chunked prefill, no lookahead, no external / async load, nothing reserved) both counts cover the
                # same tokens, so the pre-check cannot fail where the step check passes: skip it (one
                # get_num_blocks_to_allocate over every KV-cache group per new request). Same decisions, same blocks.
                comp = request.num_computed_tokens + k.get("num_new_computed_tokens", 0)
                ok = self.__dict__.get("_e72_isl_ok")
                if ok is None:      # every KV-cache group counts blocks with the base rule (admission cap = a min())
                    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager as _ST
                    base_fn = _ST.get_num_blocks_to_allocate
                    ok = self.__dict__["_e72_isl_ok"] = all(
                        type(mg).get_num_blocks_to_allocate is base_fn
                        for mg in getattr(self.coordinator, "single_type_managers", ()) or ())
                    print(f"[blink-core] E72 L4: admission pre-check skip {'on' if ok else 'OFF (manager override)'}; "
                          f"managers {[type(mg).__name__ for mg in self.coordinator.single_type_managers]}",
                          file=sys.stderr, flush=True)
                if (ok and not k.get("num_lookahead_tokens") and not k.get("num_external_computed_tokens")
                        and not k.get("delay_cache_blocks") and not k.get("num_encoder_tokens")
                        and not k.get("reserved_blocks")
                        and comp + num_new_tokens >= min(request.num_tokens, self.max_model_len)):
                    k["full_sequence_must_fit"] = False
                    _S["isl_skip"] = _S.get("isl_skip", 0) + 1
            return orig_alloc(self, request, num_new_tokens, *a, **k)

        allocate_slots._e72_lean = True
        KM.allocate_slots = allocate_slots
    for C in (getattr(_kvc, "KVCacheCoordinator", None), getattr(_kvc, "HybridKVCacheCoordinator", None)):
        if C is None or "cache_blocks" not in C.__dict__ or getattr(C.__dict__["cache_blocks"], "_e72_lean", False):
            continue

        def cache_blocks(self, request, num_computed_tokens, _orig=C.__dict__["cache_blocks"]):
            cap = getattr(request, "_e72_cc", None)
            if cap is not None and num_computed_tokens > cap:
                num_computed_tokens = cap
            return _orig(self, request, num_computed_tokens)

        cache_blocks._e72_lean = True
        C.cache_blocks = cache_blocks
    if _P.lean:
        print("[blink-core] E72: lean prefix-cache registration (BLINK_E72_LEANCACHE)", file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    TARGET = "vllm.v1.engine.core"

    def find_spec(self, fullname, path, target=None):
        if fullname != self.TARGET:
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


if _P.on or _P.pipe or TUNE_FILE or TRACE_FILE or STATS_FILE or CPROF:
    if "vllm.v1.engine.core" in sys.modules:
        _patch(sys.modules["vllm.v1.engine.core"])
    else:
        sys.meta_path.insert(0, _Finder())
