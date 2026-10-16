#!/usr/bin/env python3
"""Minimal single-file server over any registry Backend (engine/backends/base.py). Stand-in until engine/serve.py
(continuous batcher, auth, UDS) is merged; same routes and answer shapes:

  POST /internal/v1/decide   s1 sidecar contract, one question per call: {"state", "questions": {qid: q}} ->
                             {"model", "answers": {qid: {type, probabilities, confidence, choice|noul|score}}, "usage"}
  POST /v1/systemone         multi-question; answers in the model's own shape (backend.answer) when it has one
  GET  /health               200 once the backend is loaded

Requests are encoded in their handler thread; one scheduler thread packs queued items up to backend.max_batch_tokens
tokens per run() (0 = one item per run), so concurrent requests batch without a lock around the GPU.
Usage: python -m engine.mini_serve --model quyet-1.0-large --model-dir /model --port 8741 [--opt preset=fp8-kv ...]
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import queue
import sys
import threading
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import registry  # noqa: E402
from engine.backends.common import s1_answer  # noqa: E402


class Scheduler:
    def __init__(self, backend):
        self.backend, self.q = backend, queue.Queue()
        self.steps = self.decisions = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, enc) -> Future:
        f = Future()
        self.q.put((enc, f))
        return f

    def _loop(self):
        carry = None
        cap = self.backend.max_batch_tokens
        while True:
            batch = [carry] if carry else [self.q.get()]
            carry, n = None, max(1, batch[0][0].n_tokens)
            while cap and n < cap:
                try:
                    item = self.q.get_nowait()
                except queue.Empty:
                    break
                if n + item[0].n_tokens > cap:
                    carry = item
                    break
                batch.append(item)
                n += item[0].n_tokens
            try:
                for (_, f), p in zip(batch, self.backend.run([e for e, _ in batch])):
                    f.set_result(p)
            except Exception as e:  # noqa: BLE001
                for _, f in batch:
                    if not f.done():
                        f.set_exception(e)
            self.steps += 1
            self.decisions += len(batch)


def make_handler(backend, sched, mode_token):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
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
                return self._send(200, {"ok": True, "model": backend.name, "steps": sched.steps, "decisions": sched.decisions})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            s1 = self.path == "/internal/v1/decide"
            if self.path not in ("/internal/v1/decide", "/v1/systemone"):
                return self._send(404, {"error": "not found"})
            if s1 and mode_token and not hmac.compare_digest(self.headers.get("X-S1M-Internal-Token", ""), mode_token):
                return self._send(401, {"error": {"message": "unauthorized"}})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                qs = body["questions"]
                if not isinstance(qs, dict) or not qs or (s1 and len(qs) != 1):
                    raise ValueError("questions must be a non-empty object" + (" with exactly one question" if s1 else ""))
                encs = {k: backend.encode(body.get("state"), q) for k, q in qs.items()}
                futs = {k: sched.submit(e) for k, e in encs.items()}
                answers = {}
                for k, q in qs.items():
                    probs = futs[k].result()
                    if not s1 and hasattr(backend, "answer"):
                        answers[k] = backend.answer(q, probs)
                    else:
                        answers[k] = s1_answer(q["type"], backend.options(q), probs)
                self._send(200, {"model": backend.name, "answers": answers,
                                 "usage": {"input_tokens": sum(e.n_tokens for e in encs.values()), "output_tokens": 0}})
            except ValueError as e:
                self._send(422 if "maximum context length" in str(e) else 400, {"error": str(e)})
            except (KeyError, TypeError, json.JSONDecodeError) as e:
                self._send(400, {"error": f"bad request: {type(e).__name__}: {e}"})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": f"{type(e).__name__}: {e}"[:300]})

    return H


def parse_opts(pairs):
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        try:
            out[k.replace("-", "_")] = json.loads(v)
        except json.JSONDecodeError:
            out[k.replace("-", "_")] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="registry id or alias")
    ap.add_argument("--model-dir", default=os.environ.get("REAX_MODEL_DIR", "/model"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8741)
    ap.add_argument("--opt", action="append", default=[], help="backend option key=value (JSON values), repeatable")
    a = ap.parse_args(argv)
    token = os.environ.get("S1M_INTERNAL_TOKEN", "")
    backend = registry.build(a.model, a.model_dir, **parse_opts(a.opt))
    sched = Scheduler(backend)

    class Srv(ThreadingHTTPServer):
        request_queue_size = 512
        daemon_threads = True

    srv = Srv((a.host, a.port), make_handler(backend, sched, token))
    print(f"ready: {backend.name} ({backend.kind}) on {a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
