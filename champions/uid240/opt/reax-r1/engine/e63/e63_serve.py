#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-REAX-Engine-1.1 AND Apache-2.0 AND MIT
# E63 mammoth server: the REAX Engine's decision server (engine/serve.py App: routes, token check, the qev answer shapes,
# health) over the E63 CUDA-graph engine (e63_engine.py) and an E63 step scheduler. Derived from reax-engine 546728669f91
# (MIT REAX Engine code, see NOTICE).
from __future__ import annotations

import asyncio
import os
import queue
import sys
import threading
import time

sys.path[:0] = ["/opt/e63", "/opt/reax/engine", "/opt/reax"]

import serve as S  # noqa: E402  (engine/serve.py of the reference image)

E63_LOG = os.environ.get("E63_STEPLOG", "0") == "1"


def _env_list(name, default):
    return [int(x) for x in os.environ.get(name, default).split(",") if x]


class GraphBackend:
    """Qwen35Backend (tokeniser, contract, answers) with the E63 graph engine for the steps."""

    def __init__(self, be):
        from e63_engine import E63Graph
        self.be = be
        self.kind, self.name, self.family = be.kind, be.name, be.family
        self.max_input_tokens = be.max_input_tokens
        t_b = _env_list("E63_T_BUCKETS", ",".join(str(x) for x in list(range(128, 2049, 128)) + list(range(2304, 4097, 256))))
        s_b = _env_list("E63_S_BUCKETS", "1,2,4,8,16")
        self.g = E63Graph(be.engine, t_b, s_b)
        self.max_batch_tokens = max(t_b)
        self.max_items = max(s_b)
        self.steps = 0

    def encode(self, state, question):
        return self.be.encode(state, question)

    def options(self, question):
        return self.be.options(question)

    def answer(self, question, probs):
        return self.be.answer(question, probs)

    def run(self, batch):
        lg = self.g.run([e.n_tokens for e in batch], [e.aux for e in batch], [e.payload for e in batch])
        self.steps += 1
        return self.be.engine.probs(lg, [e.n_options for e in batch])

    def warmup(self):
        import json
        import torch
        from e63_engine import dump_fla_choices, pin_fla
        pins = os.environ.get("E63_FLA_PINS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fla_pins.json"))  # kai: file-relative (pjs: /opt/e63)
        if pins and os.path.exists(pins):
            done = pin_fla(json.load(open(pins)))
            print(f"[e63] FLA pins: {len(done)} kernels {sorted(done)}", flush=True)
        t0 = time.time()
        s_b, t_b = self.g.s_buckets, self.g.t_buckets
        n = 0
        for s in sorted(s_b, reverse=True):
            for t in sorted(t_b, reverse=True):
                if t < 64 * s:
                    continue
                self.g.capture(t, s)
                n += 1
        torch.cuda.synchronize()
        print(f"[e63] captured {n} graphs in {time.time() - t0:.0f}s; GPU memory {torch.cuda.memory_reserved() / 2**30:.1f} GiB "
              f"reserved", flush=True)
        if os.environ.get("E63_DUMP_FLA"):
            json.dump(dump_fla_choices(), open(os.environ["E63_DUMP_FLA"], "w"), indent=1)


class Scheduler63:
    """One GPU thread. Requests are counted from their arrival (before tokenising); the client concurrency C is the peak
    number of requests not yet computed over the last E63_PEAK_S seconds. A step starts once ceil(C / E63_GROUPS) decisions
    are queued (or E63_MAX_WAIT_MS after the first one), up to the bucket caps. With E63_GROUPS=1 one step takes a whole
    wave of a closed-loop client; with 2 the waves split in two groups that alternate on the GPU. Idle = blocked on the
    queue (no polling, no GPU work)."""

    close = ""  # kai: our serve.App reads sched.close (lever B5) and sched.adapt (rgh MAM_ADAPT); neither applies here
    adapt = False  # kai

    def __init__(self, backend, app_ref):
        self.backend = backend
        self.app_ref = app_ref
        self.q: queue.SimpleQueue = queue.SimpleQueue()
        self.max_wait = float(os.environ.get("E63_MAX_WAIT_MS", "15")) / 1e3
        self.groups = max(1, int(os.environ.get("E63_GROUPS", "1")))
        self.peak_s = float(os.environ.get("E63_PEAK_S", "2"))
        self.max_items = backend.max_items
        self.limit = backend.max_batch_tokens
        self.steps = self.decisions = self.tokens = 0
        self.inflight = 0  # arrived, not yet computed
        self.c_peak, self.c_peak_t = 1, 0.0
        self._lock = threading.Lock()
        self._on_ready = None
        self.thread = threading.Thread(target=self._loop, name="scheduler", daemon=True)

    def start(self, on_ready=None):
        self._on_ready = on_ready
        S.profile_hook(self.backend)  # kai: KAI_PROFILE dev profiling (engine/kprofile.py); unset = nothing
        self.thread.start()

    def arrive(self, n=1):
        with self._lock:
            self.inflight += n
            now = time.perf_counter()
            if self.inflight >= self.c_peak or now - self.c_peak_t > self.peak_s:
                self.c_peak, self.c_peak_t = self.inflight, now

    def submit(self, enc, loop, fut):
        self.q.put((enc, loop, fut))

    def abandon(self, n=1):
        with self._lock:
            self.inflight -= n

    def _target(self):
        with self._lock:
            if time.perf_counter() - self.c_peak_t > self.peak_s:
                self.c_peak, self.c_peak_t = max(1, self.inflight), time.perf_counter()
            c = self.c_peak
        return max(1, min(self.max_items, -(-c // self.groups)))

    def _loop(self):
        be, q = self.backend, self.q
        if self._on_ready:
            self._on_ready()
        carry = []
        while True:
            batch = carry or [q.get()]
            carry = []
            n = sum(b[0].n_tokens for b in batch)
            t_first = time.perf_counter()
            target = self._target()
            while len(batch) < target:
                left = self.max_wait - (time.perf_counter() - t_first)
                if left <= 0:
                    break
                try:
                    item = q.get(timeout=left)
                except queue.Empty:
                    break
                if n + item[0].n_tokens > self.limit:
                    carry.append(item)
                    break
                batch.append(item)
                n += item[0].n_tokens
            while len(batch) < self.max_items and not carry:  # whatever else is already queued (up to the caps)
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break
                if n + item[0].n_tokens > self.limit or len(batch) >= target:
                    carry.append(item)
                    break
                batch.append(item)
                n += item[0].n_tokens
            t0 = time.perf_counter()
            try:
                res = be.run([b[0] for b in batch])
                err = None
            except Exception as e:  # noqa: BLE001
                res, err = None, e
                import traceback
                traceback.print_exc()
            t1 = time.perf_counter()
            self.abandon(len(batch))
            if E63_LOG:
                print(f"[e63-step] n={len(batch)} tok={n} target={target} gather_ms={1e3 * (t0 - t_first):.1f} "
                      f"run_ms={1e3 * (t1 - t0):.1f}", flush=True)
            self.steps += 1
            self.decisions += len(batch)
            self.tokens += n
            by_loop: dict = {}
            for i, b in enumerate(batch):
                by_loop.setdefault(b[1], []).append((b[2], None if err else res[i]))
            for loop, items in by_loop.items():
                loop.call_soon_threadsafe(S._deliver, items, err)


class App63(S.App):
    async def decide(self, state, questions: dict, path: str, long_body: bool = False, images=None, ticket=None):  # kai: ticket (B5) unused
        loop = asyncio.get_running_loop()
        n_q = len(questions)
        self.sched.arrive(n_q)
        submitted = 0
        try:
            encode = self.be.encode
            if long_body or n_q > 1:
                encs = dict(zip(questions, await asyncio.gather(
                    *(loop.run_in_executor(None, encode, state, q) for q in questions.values()))))
            else:
                (qid, q), = questions.items()
                encs = {qid: await loop.run_in_executor(None, encode, state, q)}
            futs = {}
            for qid, enc in encs.items():
                futs[qid] = loop.create_future()
                self.sched.submit(enc, loop, futs[qid])
                submitted += 1
        finally:
            if n_q - submitted:
                self.sched.abandon(n_q - submitted)
        probs = {qid: await f for qid, f in futs.items()}
        shape = self._shape[path]
        answers = {qid: shape(q, probs[qid]) for qid, q in questions.items()}
        return answers, sum(e.n_tokens for e in encs.values())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("REAX_MODEL") or "mammoth-1")
    ap.add_argument("--model-dir", default=os.environ.get("REAX_MODEL_DIR", "/model"))
    a = ap.parse_args()
    from engine import registry
    spec = registry.get(a.model)
    if spec.id != "mammoth-1":
        raise SystemExit(f"e63 serves mammoth-1 only, not {a.model!r}")
    token = S._token()
    import torch
    torch.set_num_threads(int(os.environ.get("E63_TORCH_THREADS", "1")))
    opts = dict(spec.production_opts)
    opts["warm_steps"] = 0
    t0 = time.time()
    be = registry.build(spec.id, a.model_dir, **opts)
    print(f"[e63] weights loaded in {time.time() - t0:.0f}s", flush=True)
    gb = GraphBackend(be)
    app_box = {}
    sched = Scheduler63(gb, app_box)
    app = App63(gb, sched, token, gb.name)
    app_box["app"] = app

    def warm():
        t1 = time.time()
        try:
            gb.warmup()
        except BaseException:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            os._exit(1)
        app.ready = True
        print(f"ready: {gb.name} (e63 graphs) warm-up {time.time() - t1:.0f}s, total {time.time() - t0:.0f}s", flush=True)

    import uvicorn
    kw = dict(log_level="warning", access_log=False, lifespan="on", timeout_keep_alive=75)
    if os.environ.get("RUNTIME_UDS"):
        kw["uds"] = os.environ["RUNTIME_UDS"]
    else:
        kw.update(host=os.environ.get("RUNTIME_HOST", "127.0.0.1"), port=int(os.environ.get("RUNTIME_PORT", "8101")))
    sched.start(on_ready=warm)
    uvicorn.run(app, **kw)


if __name__ == "__main__":
    main()
