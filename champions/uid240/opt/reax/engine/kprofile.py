"""kprofile: per-step profiling of a serving engine (dev switch KAI_PROFILE=1; tools/profile.sh drives it).

Unset or KAI_PROFILE=0: serve.profile_hook() returns before this module is even imported; nothing changes. With
KAI_PROFILE=1 the backend's run() (the one call per scheduler step, on the scheduler thread, the only thread that touches
the GPU) is wrapped. While idle the wrapper only polls for a trigger file (one stat per KAI_PROFILE_POLL_S, default 0.2 s)
and forwards the call; results are never touched.

Trigger: create ``$KAI_PROFILE_DIR/start`` (default /tmp/kai-prof; tools/profile.sh does it with `docker exec`), optionally
holding JSON: {"label": "c8", "timing_s": 15, "timing_steps": 0, "trace_s": 30, "trace_steps": 50, "shapes": true,
"annotate": true}. A window ends at its step count or its seconds, whichever comes first (0 = no limit of that kind;
both 0 = the window is skipped). Then:

1. timing window (no profiler, so no profiler overhead): per step wall time, gap since the previous step returned (the
   scheduler's collect + deliver + waiting for work), batch size and tokens, and around the model call inside run()
   (engine.forward / forward_marked / forward_groups, or E63's graph run): host time before it (input build, H2D),
   host time inside it (launch cost: the call is asynchronous), host time after it (waiting for the GPU, readout, D2H),
   plus CUDA events: GPU span of the step and of the forward. -> <label>/steps_timing.json
2. trace window: torch.profiler (CPU + CUDA activities, record_shapes) with annotations kai_step#<i> (each step),
   kai_fwd (the model call), kai_readout (engine.probs) and kai_lin <mode> K= N= M= (each pplx_engine Linear call, so
   GEMM kernels carry their shape). -> <label>/trace.json.gz (Chrome trace), key_averages.txt / .json,
   key_averages_shapes.txt, steps_trace.json
3. the export runs in a background thread (the scheduler keeps serving, unmeasured); then <label>/meta.json and
   ``$KAI_PROFILE_DIR/<label>.done`` ("ok" or "error: ..."); back to idle (re-triggerable).

Any failure inside the profiler writes "<label>.done" with the error and returns to idle; the step itself always runs
and its result or exception is passed through unchanged.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import shutil
import sys
import threading
import time
import traceback

DEFAULTS = {"label": "prof", "timing_s": 15.0, "timing_steps": 0, "trace_s": 30.0, "trace_steps": 50, "shapes": True,
            "annotate": True}
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
INNER = ("forward", "forward_marked", "forward_groups")  # model calls on backend.engine (PplxEngine)
ENV_KEYS = ("KAI_", "MAM_", "PPLX_", "E63_", "REAX_", "FLA_")
# lever I (gap attribution): encode functions the backends look up at call time, stamped per item during a window. The
# serving encode threads hold backend.encode as a bound method taken at start, so only these module functions can be
# wrapped; the stamp is keyed by the id of the token-id list that becomes Encoded.payload.
FRONT_FUNCS = (("backends.qwen35", "_encode_request_fast", "full"), ("qev_contract", "encode_request", "full"),
               ("qev_contract", "encode_once", "ids"), ("fastenc1", "encode1", "ids"))
FRONT_KEYS = ("enc_n", "enc_first_start_ms", "enc_last_start_ms", "enc_last_end_ms", "enc_sum_ms", "enc_max_ms", "close_ms")


def enabled(env=None) -> bool:
    v = (os.environ if env is None else env).get("KAI_PROFILE", "").strip()
    if v not in ("", "0", "1"):
        raise SystemExit(f"KAI_PROFILE must be 0 or 1 or unset, not {v!r}")
    return v == "1"


def install(backend, out_dir: str | None = None, poll_s: float | None = None):
    """Wrap backend.run once (idempotent: an already installed profiler is returned)."""
    have = getattr(backend, "_kai_profile", None)
    if have is not None:
        return have
    out_dir = out_dir or os.environ.get("KAI_PROFILE_DIR", "/tmp/kai-prof")
    poll_s = float(os.environ.get("KAI_PROFILE_POLL_S", "0.2")) if poll_s is None else poll_s
    p = StepProfiler(backend, out_dir, poll_s=poll_s)
    p.install()
    print(f"kprofile: installed on {type(backend).__name__}.run; trigger file {p.start_path}", file=sys.stderr, flush=True)
    return p


def _stats(xs):
    xs = sorted(x for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(x))
    if not xs:
        return None
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]  # noqa: E731
    return {"n": len(xs), "mean": sum(xs) / len(xs), "p10": q(0.1), "p50": q(0.5), "p90": q(0.9), "min": xs[0],
            "max": xs[-1], "sum": sum(xs)}


def summarize_steps(steps, window_s):
    """Aggregate per-step records (also used by tools/profile.py)."""
    keys = ("wall_ms", "gap_ms", "pre_ms", "fwd_host_ms", "post_ms", "readout_ms", "step_gpu_ms", "fwd_gpu_ms", "n", "tokens") + FRONT_KEYS
    out = {"steps": len(steps), "window_s": window_s}
    for k in keys:
        out[k] = _stats([s.get(k) for s in steps])
    wall = sum(s.get("wall_ms") or 0.0 for s in steps)
    gap = sum(s.get("gap_ms") or 0.0 for s in steps)
    out["decisions"] = sum(s.get("n") or 0 for s in steps)
    out["dps"] = out["decisions"] / window_s if window_s > 0 else None
    out["in_run_frac"] = wall / (wall + gap) if wall + gap > 0 else None
    return out


class StepProfiler:
    def __init__(self, backend, out_dir: str, poll_s: float = 0.2, clock=time.perf_counter):
        self.backend = backend
        self.out_dir = out_dir
        self.start_path = os.path.join(out_dir, "start")
        self.poll_s = float(poll_s)
        self.clock = clock
        self.state = "idle"
        self._orig = None
        self._next_poll = 0.0
        self._last_ret = None
        self._patches = []
        self._depth = 0
        self.cur = None
        self.prof = None
        self._torch = None
        self._stamps = {}  # lever I: id(token-id list) -> (encode start, encode end) during a window

    # ------------------------------------------------------------------------------------------------ install
    def install(self):
        if getattr(self.backend, "_kai_profile", None) is not None:
            return self.backend._kai_profile
        os.makedirs(self.out_dir, exist_ok=True)
        try:
            os.chmod(self.out_dir, 0o777)  # the trigger may come from `docker exec` as another uid
        except OSError:
            pass
        self._orig = self.backend.run
        self.backend.run = self._run
        self.backend._kai_profile = self
        return self

    # ------------------------------------------------------------------------------------------------ the wrapper
    def _run(self, batch):
        now = self.clock()
        if self.state == "idle" and now >= self._next_poll:
            self._next_poll = now + self.poll_s
            if os.path.exists(self.start_path):
                self._guard(self._begin)
        if self.state in ("idle", "exporting"):
            try:
                return self._orig(batch)
            finally:
                self._last_ret = self.clock()
        return self._measured(batch)

    def _guard(self, fn, *a):
        try:
            fn(*a)
        except Exception as exc:  # noqa: BLE001 - profiling must never take the server down
            self._fail(exc)

    def _fail(self, exc):
        label = getattr(self, "label", None) or "trigger"
        msg = f"error: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        print(f"kprofile: {label}: {msg}", file=sys.stderr, flush=True)
        try:
            if self.prof is not None:
                self.prof.stop()
        except Exception:  # noqa: BLE001
            pass
        self.prof = None
        self._unpatch()
        self.state = "idle"
        self.cur = None
        try:
            with open(os.path.join(self.out_dir, f"{label}.done"), "w") as fh:
                fh.write(msg)
        except OSError:
            pass
        self.label = None

    # ------------------------------------------------------------------------------------------------ windows
    def _begin(self):
        self.label = None
        raw = ""
        try:
            with open(self.start_path) as fh:
                raw = fh.read()
        finally:
            try:
                os.unlink(self.start_path)
            except OSError:
                pass
        cfg = dict(DEFAULTS)
        if raw.strip():
            got = json.loads(raw)
            if not isinstance(got, dict):
                raise ValueError("trigger must hold a JSON object")
            unknown = set(got) - set(DEFAULTS)
            if unknown:
                raise ValueError(f"unknown trigger keys {sorted(unknown)}")
            cfg.update(got)
        if not isinstance(cfg["label"], str) or not LABEL_RE.match(cfg["label"]):
            raise ValueError(f"bad label {cfg['label']!r}")
        for k in ("timing_s", "trace_s"):
            cfg[k] = float(cfg[k])
        for k in ("timing_steps", "trace_steps"):
            cfg[k] = int(cfg[k])
        self.cfg = cfg
        self.label = cfg["label"]
        self.dir = os.path.join(self.out_dir, self.label)
        shutil.rmtree(self.dir, ignore_errors=True)
        try:
            os.unlink(os.path.join(self.out_dir, f"{self.label}.done"))
        except OSError:
            pass
        os.makedirs(self.dir)
        import torch
        self._torch = torch
        self.cuda = bool(torch.cuda.is_available())
        self.meta = {"label": self.label, "cfg": cfg, "cuda": self.cuda, "torch": torch.__version__, "pid": os.getpid(),
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "backend": type(self.backend).__name__,
                     "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_KEYS)
                             and "TOKEN" not in k}, "windows": {}}
        if self.cuda:
            self.meta["device"] = torch.cuda.get_device_name()
        self._patch_inner()
        self._patch_front()
        self._enter("timing" if (cfg["timing_s"] > 0 or cfg["timing_steps"] > 0) else "trace")

    def _enter(self, state):
        if state == "trace" and not (self.cfg["trace_s"] > 0 or self.cfg["trace_steps"] > 0):
            return self._finish()
        self.state = state
        self.steps = []
        self.k = 0
        self.t_win0 = self.clock()
        if state == "trace" and self.cfg["annotate"]:
            self._patch_linear()

    def _limits(self):
        p = "timing" if self.state == "timing" else "trace"
        return self.cfg[f"{p}_steps"], self.cfg[f"{p}_s"]

    def _window_over(self, t):
        n, s = self._limits()
        return (n > 0 and len(self.steps) >= n) or (s > 0 and t - self.t_win0 >= s)

    def _end_window(self, t):
        window_s = t - self.t_win0
        name = self.state
        prof = self._stop_profiler() if name == "trace" else None
        self._resolve_events()
        rows = [{k: v for k, v in s.items() if not k.startswith("_")} for s in self.steps]
        summary = summarize_steps(rows, window_s)
        with open(os.path.join(self.dir, f"steps_{name}.json"), "w") as fh:
            json.dump({"label": self.label, "window": name, "summary": summary, "steps": rows}, fh)
        self.meta["windows"][name] = {"steps": len(rows), "window_s": window_s}
        if name == "timing":
            self._enter("trace")
            return
        # the trace export (seconds to a minute for a big trace) runs off the scheduler thread, which goes on serving
        self._unpatch()
        self.cur = None
        self.state = "exporting"

        def work():
            try:
                if prof is not None:
                    self._export(prof)
                self._finish()
            except Exception as exc:  # noqa: BLE001
                self._fail(exc)
        threading.Thread(target=work, name="kprofile-export", daemon=True).start()

    def _finish(self):
        self._unpatch()
        label, d = self.label, self.dir
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(self.meta, fh, indent=1, default=repr)
        self.label = None
        self.cur = None
        self.state = "idle"  # before the .done file: whoever reads "ok" may trigger again at once
        with open(os.path.join(self.out_dir, f"{label}.done"), "w") as fh:
            fh.write("ok\n")
        print(f"kprofile: {label} done -> {d}", file=sys.stderr, flush=True)

    # ------------------------------------------------------------------------------------------------ one step
    def _event(self):
        if not self.cuda:
            return None
        e = self._torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def _measured(self, batch):
        torch = self._torch
        t0 = self.clock()
        cur = {"i": self.k, "t_ms": (t0 - self.t_win0) * 1e3,
               "gap_ms": None if self._last_ret is None else (t0 - self._last_ret) * 1e3, "n": len(batch),
               "tokens": sum(int(getattr(e, "n_tokens", 0) or 0) for e in batch), "fwd_calls": 0, "fwd_host_ms": 0.0,
               "readout_ms": 0.0, "_fev": []}
        self._front(cur, batch, t0)
        self.cur = cur
        rf = None
        try:
            if self.state == "trace":
                if self.prof is None:
                    self._start_trace()
                rf = torch.profiler.record_function(f"kai_step#{self.k} n={cur['n']} tok={cur['tokens']}")
                rf.__enter__()
            cur["_ev0"] = self._event()
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)
            rf = None
        try:
            return self._orig(batch)
        finally:
            t1 = self.clock()
            self._last_ret = t1
            if self.state != "idle" and self.cur is cur:
                try:
                    cur["_ev1"] = self._event()
                    if rf is not None:
                        rf.__exit__(None, None, None)
                    cur["wall_ms"] = (t1 - t0) * 1e3
                    if cur["fwd_calls"]:
                        cur["pre_ms"] = (cur.pop("_tf0") - t0) * 1e3
                        cur["post_ms"] = (t1 - cur.pop("_tf1")) * 1e3
                    else:
                        cur["pre_ms"] = cur["post_ms"] = None
                    self.steps.append(cur)
                    self.k += 1
                    if self._window_over(t1):
                        self._end_window(t1)
                except Exception as exc:  # noqa: BLE001
                    self._fail(exc)

    def _front(self, cur, batch, t0):
        """lever I: where the gap before this step went, from the encode stamps of its items (ms after the previous
        return): request path = enc_first_start_ms, encode span up to enc_last_end_ms, close_ms = run() call - last end"""
        st = self._stamps
        if not st or self._last_ret is None:
            return
        got = [st.pop(id(getattr(e, "payload", None)), None) for e in batch]
        got = [g for g in got if g is not None]
        if len(st) > 50000:
            st.clear()
        if not got:
            return
        r = self._last_ret
        last_end = max(g[1] for g in got)
        cur.update(enc_n=len(got), enc_first_start_ms=(min(g[0] for g in got) - r) * 1e3,
                   enc_last_start_ms=(max(g[0] for g in got) - r) * 1e3, enc_last_end_ms=(last_end - r) * 1e3,
                   enc_sum_ms=sum(g[1] - g[0] for g in got) * 1e3, enc_max_ms=max(g[1] - g[0] for g in got) * 1e3,
                   close_ms=(t0 - last_end) * 1e3)

    def _patch_front(self):
        for mod in list(sys.modules.values()):
            d = getattr(mod, "__dict__", None)
            name = getattr(mod, "__name__", "") or ""
            if not isinstance(d, dict):
                continue
            for suffix, fn, kind in FRONT_FUNCS:
                if (name == suffix or name.endswith("." + suffix)) and callable(d.get(fn)):
                    if not any(o is mod and n == fn for o, n, *_ in self._patches):
                        self._patch(mod, fn, lambda old, kind=kind: self._wrap_encode(old, kind))

    def _wrap_encode(self, fn, kind):
        prof = self

        def kai_encode(*a, **k):
            t0 = prof.clock()
            out = fn(*a, **k)
            try:
                ids = (out[0][0] if kind == "full" else out[0]) if out is not None else None
                if ids is not None:
                    prof._stamps[id(ids)] = (t0, prof.clock())
            except (TypeError, IndexError, KeyError):
                pass
            return out
        kai_encode.__name__ = getattr(fn, "__name__", "kai_encode")
        return kai_encode

    def _resolve_events(self):
        if not self.cuda:
            for s in self.steps:
                s["step_gpu_ms"] = s["fwd_gpu_ms"] = None
            return
        self._torch.cuda.synchronize()
        for s in self.steps:
            e0, e1 = s.get("_ev0"), s.get("_ev1")
            s["step_gpu_ms"] = e0.elapsed_time(e1) if (e0 is not None and e1 is not None) else None
            fe = s.get("_fev") or []
            s["fwd_gpu_ms"] = sum(a.elapsed_time(b) for a, b in fe) if fe else None

    # ------------------------------------------------------------------------------------------------ inner wrappers
    def _patch(self, obj, name, make):
        old = getattr(obj, name, None)
        if old is None or not callable(old):
            return
        had = name in vars(obj) if hasattr(obj, "__dict__") else False
        prev = vars(obj).get(name) if had else None  # before setattr: restored as it was
        try:
            setattr(obj, name, make(old))
        except (AttributeError, TypeError):
            return
        self._patches.append((obj, name, had, prev, old))

    def _engines(self):
        be = self.backend
        out = []
        for e in (getattr(be, "engine", None), getattr(getattr(be, "be", None), "engine", None)):
            if e is not None and all(e is not x for x in out):
                out.append(e)
        return out

    def _patch_inner(self):
        for eng in self._engines():
            for name in INNER:
                self._patch(eng, name, self._wrap_fwd)
            self._patch(eng, "probs", self._wrap_readout)
        g = getattr(self.backend, "g", None)  # E63 GraphBackend: the whole step is one graph run
        if g is not None:
            self._patch(g, "run", self._wrap_fwd)

    def _patch_linear(self):
        for mod in list(sys.modules.values()):
            d = getattr(mod, "__dict__", None)  # the module dict only: getattr on lazy modules (transformers) imports
            if not isinstance(d, dict) or "PplxEngine" not in d:
                continue
            cls = d.get("Linear")
            if not isinstance(cls, type) or "__call__" not in vars(cls):
                continue
            if any(o is cls for o, *_ in self._patches):
                continue
            self._patch(cls, "__call__", self._wrap_linear)

    def _unpatch(self):
        while self._patches:
            obj, name, had, prev, _old = self._patches.pop()
            try:
                if had:
                    setattr(obj, name, prev)
                else:
                    delattr(obj, name)
            except (AttributeError, TypeError):
                pass

    def _wrap_fwd(self, fn):
        prof = self

        def kai_fwd(*a, **k):
            cur = prof.cur
            if cur is None or prof._depth or prof.state == "idle":
                return fn(*a, **k)
            prof._depth += 1
            rf = None
            t0 = prof.clock()
            cur.setdefault("_tf0", t0)
            e0 = prof._event()
            if prof.state == "trace":
                rf = prof._torch.profiler.record_function("kai_fwd")
                rf.__enter__()
            try:
                return fn(*a, **k)
            finally:
                if rf is not None:
                    rf.__exit__(None, None, None)
                e1 = prof._event()
                t1 = prof.clock()
                cur["_tf1"] = t1
                cur["fwd_host_ms"] += (t1 - t0) * 1e3
                cur["fwd_calls"] += 1
                if e0 is not None:
                    cur["_fev"].append((e0, e1))
                prof._depth -= 1
        return kai_fwd

    def _wrap_readout(self, fn):
        prof = self

        def kai_readout(*a, **k):
            cur = prof.cur
            if cur is None or prof.state == "idle":
                return fn(*a, **k)
            rf = None
            t0 = prof.clock()
            if prof.state == "trace":
                rf = prof._torch.profiler.record_function("kai_readout")
                rf.__enter__()
            try:
                return fn(*a, **k)
            finally:
                if rf is not None:
                    rf.__exit__(None, None, None)
                cur["readout_ms"] += (prof.clock() - t0) * 1e3
        return kai_readout

    def _wrap_linear(self, fn):
        prof = self

        def kai_linear(lin, x=None, q=None, *a, **k):
            if prof.state != "trace":
                return fn(lin, x, q, *a, **k)
            src = x if x is not None else (q[0] if isinstance(q, (tuple, list)) and q else getattr(q, "q", None))
            m = int(src.shape[0]) if getattr(src, "shape", None) is not None and len(src.shape) else -1
            name = (f"kai_lin {getattr(lin, 'mode', '?')} K={getattr(lin, 'in_features', '?')} "
                    f"N={getattr(lin, 'out_features', '?')} M={m}")
            with prof._torch.profiler.record_function(name):
                return fn(lin, x, q, *a, **k)
        return kai_linear

    # ------------------------------------------------------------------------------------------------ torch.profiler
    def _start_trace(self):
        torch = self._torch
        acts = [torch.profiler.ProfilerActivity.CPU]
        if self.cuda:
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        self.prof = torch.profiler.profile(activities=acts, record_shapes=bool(self.cfg["shapes"]), with_stack=False)
        self.prof.start()
        self.t_win0 = self.clock()  # the window starts once the profiler runs

    def _stop_profiler(self):
        prof, self.prof = self.prof, None
        if prof is None:
            return None
        if self.cuda:
            self._torch.cuda.synchronize()
        prof.stop()
        self._unpatch_linear_only()
        return prof

    def _export(self, prof):
        raw = os.path.join(self.dir, "trace.json")
        prof.export_chrome_trace(raw)
        with open(raw, "rb") as src, gzip.open(raw + ".gz", "wb", compresslevel=3) as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        os.unlink(raw)
        ka = prof.key_averages()
        sort = None
        for key in (("self_device_time_total", "self_cuda_time_total") if self.cuda else ()) + ("self_cpu_time_total",):
            try:
                text = ka.table(sort_by=key, row_limit=120, max_name_column_width=90)
                sort = key
                break
            except Exception:  # noqa: BLE001 - the key name differs between torch versions
                continue
        if sort is None:
            text = ka.table(row_limit=120)
        with open(os.path.join(self.dir, "key_averages.txt"), "w") as fh:
            fh.write(text)
        if self.cfg["shapes"]:
            try:
                with open(os.path.join(self.dir, "key_averages_shapes.txt"), "w") as fh:
                    fh.write(prof.key_averages(group_by_input_shape=True).table(sort_by=sort or "self_cpu_time_total",
                                                                                 row_limit=200, max_name_column_width=90))
            except Exception:  # noqa: BLE001
                pass
        rows = []
        for e in ka:
            g = lambda *names: next((getattr(e, n) for n in names if hasattr(e, n)), None)  # noqa: E731
            rows.append({"key": e.key, "count": e.count, "self_cpu_us": g("self_cpu_time_total"),
                         "cpu_total_us": g("cpu_time_total"),
                         "self_device_us": g("self_device_time_total", "self_cuda_time_total"),
                         "device_total_us": g("device_time_total", "cuda_time_total")})
        with open(os.path.join(self.dir, "key_averages.json"), "w") as fh:
            json.dump({"sort": sort, "rows": rows}, fh)

    def _unpatch_linear_only(self):
        keep = []
        for p in self._patches:
            obj, name = p[0], p[1]
            if isinstance(obj, type) and name == "__call__":
                try:
                    if p[2]:
                        setattr(obj, name, p[3])
                    else:
                        delattr(obj, name)
                except (AttributeError, TypeError):
                    pass
            else:
                keep.append(p)
        self._patches = keep
