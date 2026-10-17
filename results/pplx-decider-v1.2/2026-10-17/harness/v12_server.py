#!/usr/bin/env python3
"""/v1/systemone server around the AUTHOR's pplx_decider v1.2 code (reference or fast path), for side-by-side timing.

Same protocol as reax-engine engine/server.py (v1.1): POST /v1/systemone {model, state, questions} ->
{model, answers, usage}; GET /health. A GPU thread takes every queued request and calls Decider.decide(list) once,
so the author's fast path can pack concurrent requests (the author ships no server; this is the most favourable
thin wrapper). --split single sends every question as its own request (state repeated), like v1.1.
A single question longer than --max-length (default 8192, the v1.1 server rule) -> HTTP 422.
Usage: v12_server.py --model DIR --path fast|fastfp8|ref --split joint|single [--port 8741]
"""
from __future__ import annotations

import argparse, json, queue, sys, threading, time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--path", choices=["fast", "fastfp8", "ref"], default="fast")
    ap.add_argument("--split", choices=["joint", "single"], default="joint")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--buckets", default="", help="comma list, default author FastOptions buckets")
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--port", type=int, default=8741)
    ap.add_argument("--recompile-limit", type=int, default=0, help="re-apply the author's intended dynamo limit (256) in the GPU thread")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.model) / "source" / "src"))
    import torch
    from pplx_decider import Decider, FastOptions, LengthError, RequestError

    fast = False
    if a.path != "ref":
        kw = {"fp8": a.path == "fastfp8"}
        if a.buckets:
            kw["buckets"] = tuple(int(x) for x in a.buckets.split(","))
        fast = FastOptions(**kw)
    t0 = time.time()
    dec = Decider.from_pretrained(a.model, device="cuda", fast=fast, max_length=a.max_length)
    import torch._dynamo as dyn
    print(f"main thread dynamo limits after init: recompile={dyn.config.recompile_limit}", flush=True)
    dec.warmup()
    print(f"loaded+warm {time.time() - t0:.0f}s path={a.path} split={a.split} max_length={a.max_length}", flush=True)
    name = Path(a.model).name
    q: queue.Queue = queue.Queue()
    stats = {"steps": 0, "decisions": 0, "busy_s": 0.0}

    def expand(state, questions):
        if a.split == "joint":
            return [{"state": state, "questions": questions}]
        return [{"state": state, "questions": {k: v}} for k, v in questions.items()]

    def loop():
        import torch._dynamo as dyn
        print(f"gpu thread dynamo limits before: recompile={dyn.config.recompile_limit} cache={dyn.config.cache_size_limit}", flush=True)
        if a.recompile_limit:
            dyn.config.recompile_limit = max(dyn.config.recompile_limit, a.recompile_limit)
            dyn.config.accumulated_recompile_limit = max(dyn.config.accumulated_recompile_limit, 4096)
        print(f"gpu thread dynamo limits now: recompile={dyn.config.recompile_limit}", flush=True)
        while True:
            items = [q.get()]
            while len(items) < a.max_batch:
                try:
                    items.append(q.get_nowait())
                except queue.Empty:
                    break
            good, reqs = [], []
            for state, questions, fut in items:  # validate/plan per item so one bad request fails alone
                sub = expand(state, questions)
                try:
                    for r in sub:
                        dec.encoder.plan(r, a.max_length)
                except (LengthError, RequestError, ValueError) as e:
                    fut.set_exception(e)
                    continue
                good.append((questions, fut, len(sub)))
                reqs += sub
            if not reqs:
                continue
            ts = time.time()
            try:
                out = dec.decide(reqs)
            except Exception as e:  # noqa: BLE001
                for _, fut, _ in good:
                    fut.set_exception(e)
                continue
            stats["busy_s"] += time.time() - ts
            stats["steps"] += 1
            i = 0
            for questions, fut, n in good:
                answers, tokens = {}, 0
                for r in out[i:i + n]:
                    answers.update(r["answers"])
                    tokens += r["usage"]["input_tokens"]
                i += n
                for k, v in answers.items():
                    if v["type"] == "noul":
                        v["probabilities"] = {"false": 1.0 - v["noul"], "true": v["noul"]}
                stats["decisions"] += len(questions)
                fut.set_result(({k: answers[k] for k in questions}, tokens))

    threading.Thread(target=loop, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *x):
            pass

        def _send(self, code, obj):
            raw = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"status": "ready", "model": name, **{k: round(v, 3) for k, v in stats.items()}})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                qs = body.get("questions")
                if not isinstance(qs, dict) or not qs:
                    raise ValueError("questions must be a non-empty object")
                fut: Future = Future()
                q.put((body.get("state"), qs, fut))
                answers, tokens = fut.result()
                self._send(200, {"model": name, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}})
            except LengthError as e:
                self._send(422, {"error": f"maximum context length exceeded: {e}"})
            except (RequestError, ValueError, KeyError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": f"{type(e).__name__}: {e}"[:300]})

    class Srv(ThreadingHTTPServer):
        request_queue_size = 512
        daemon_threads = True

    print(f"ready on :{a.port}", flush=True)
    Srv(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
