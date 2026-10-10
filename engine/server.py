#!/usr/bin/env python3
"""/v1/systemone server for pplx-decider-v1.1-27b.

Backends
  engine  our prefill-only engine (pplx_engine.PplxEngine) behind a continuous batcher: every question of every
          in-flight request is tokenised in the request thread and queued; one GPU thread packs queued decisions up
          to --budget tokens per forward (no padding) and resolves them. Requests are never serialised behind a lock.
  ref     the author's DecisionModel, served like the author's server (one request at a time, questions in chunks of 8).
Protocol (as the author's server + harness needs): POST /v1/systemone {model, state, questions{id: question}} ->
{model, answers{id: answer}, usage{input_tokens, output_tokens:0}}. Noul answers additionally carry
probabilities {false, true}. A question branch > 8192 tokens -> HTTP 422 "maximum context length ..." (the author
refuses those too; the marker makes the Decision Index kit count it as unsupported). GET /health.
Usage: server.py --model ./model --backend engine [--linear fp8 --compile 1 --budget 16384] --port 8741
"""
from __future__ import annotations

import argparse, json, queue, sys, threading, time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from pplx_engine import MAX_OPTIONS, Prompter, answer, options  # noqa: E402

MAX_TOKENS = 8192
NAME = "pplx-decider-v1.1-27b"


class Capacity(Exception):
    pass


def validate(questions):
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object")
    for q in questions.values():
        if not isinstance(q, dict) or q.get("type") not in ("choice", "score", "noul"):
            raise ValueError("each question needs type choice|score|noul")
        n = len(options(q)[0])
        if q["type"] == "choice" and not 1 <= n <= MAX_OPTIONS:
            raise Capacity(f"choice questions take 1 to {MAX_OPTIONS} options per choice")
        if q["type"] == "score" and not 2 <= n <= 10:
            raise Capacity("a score takes 2 to 10 levels")


def shape(q, probs):
    a = answer(q, probs)
    if q["type"] == "noul":
        a["probabilities"] = {"false": 1.0 - a["noul"], "true": a["noul"]}
    return a


class EngineBackend:
    def __init__(self, a):
        from pplx_engine import PplxEngine
        self.pr = Prompter(a.model)
        self.eng = PplxEngine(a.model, linear_mode=a.linear, compile=a.compile, gva_native=a.gva,
                              fp8_skip=tuple(x for x in a.fp8_skip.split(",") if x))
        self.budget = a.budget
        self.max_wait = a.max_wait_ms / 1000
        a_warm_steps = a.warm_steps
        self.q: queue.Queue = queue.Queue()
        self.steps = self.decisions = 0
        threading.Thread(target=self._loop, daemon=True).start()
        # Warm-up: trigger every torch.compile specialization / Triton autotune BEFORE serving (otherwise the first
        # requests of each new batch shape stall the GPU loop for seconds). Covers 1..many sequences per step, short
        # and long sequences, and both DeltaNet conv paths (split path for steps >= 8192 tokens).
        t0 = time.time()
        shapes = [[n] for n in (17, 64, 300, 700, 1500, 3000, 6000, 8192)]
        shapes += [[300] * k for k in (2, 3, 4, 5, 7, 8, 12, 16, 24, 32)]
        shapes += [[700] * 11 + [64], [4000, 4000, 4000, 4000], [2000] * 8, [8192, 8000], [500] * 32, [16000]]
        for rep in range(2):
            for lens in shapes:
                lens = [min(n, MAX_TOKENS) for n in lens]
                while sum(lens) > self.budget:
                    lens.pop()
                fs = []
                for n in lens:
                    f = Future()
                    fs.append(f)
                    self.q.put(([1] * n, 2, f))
                for f in fs:
                    f.result()
        # Randomised realistic steps (seeded): many sequence counts x length mixes, so every size bucket of the generated
        # kernels is tuned before serving (first-pass cost measured 9.2 vs 20.7 items/s without this).
        import random
        rng = random.Random(7)
        for _ in range(a_warm_steps):
            lens, n = [], rng.choice([1, 1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 40])
            for _ in range(n):
                L = int(min(MAX_TOKENS, max(16, rng.lognormvariate(6.3, 0.8))))
                if sum(lens) + L > self.budget:
                    break
                lens.append(L)
            fs = []
            for L in lens or [64]:
                f = Future()
                fs.append(f)
                self.q.put(([1] * L, rng.randint(2, 12), f))
            for f in fs:
                f.result()
        print(f"warm-up done in {time.time() - t0:.0f}s ({self.steps} steps)", flush=True)

    def _loop(self):
        carry = None
        while True:
            batch = [carry] if carry else [self.q.get()]
            carry = None
            n = len(batch[0][0])
            deadline = time.perf_counter() + self.max_wait
            while n < self.budget:
                try:
                    item = self.q.get(timeout=max(0.0, deadline - time.perf_counter())) if self.max_wait else self.q.get_nowait()
                except queue.Empty:
                    break
                if n + len(item[0]) > self.budget:
                    carry = item
                    break
                batch.append(item)
                n += len(item[0])
            try:
                ids = torch.tensor([t for b in batch for t in b[0]], device="cuda")
                logits = self.eng.forward(ids, [len(b[0]) for b in batch])
                probs = self.eng.probs(logits, [b[1] for b in batch])
                for b, p in zip(batch, probs):
                    b[2].set_result(p)
            except Exception as e:  # noqa: BLE001
                for b in batch:
                    if not b[2].done():
                        b[2].set_exception(e)
            self.steps += 1
            self.decisions += len(batch)

    def decide(self, state, questions):
        enc = {k: self.pr.ids(state, q) for k, q in questions.items()}
        if any(len(ids) > MAX_TOKENS for ids, _ in enc.values()):
            raise Capacity(f"maximum context length exceeded: a question branch is longer than {MAX_TOKENS} tokens")
        futs = {}
        for k, (ids, n) in enc.items():
            futs[k] = Future()
            self.q.put((ids, n, futs[k]))
        answers = {k: shape(questions[k], futs[k].result()) for k in questions}
        return answers, sum(len(ids) for ids, _ in enc.values())


class RefBackend:
    def __init__(self, a):
        sys.path.insert(0, str(Path(a.model) / "source" / "src"))
        from autojev.model import DecisionModel
        self.m = DecisionModel(a.model, device="cuda")
        self.lock = threading.Lock()
        self.steps = self.decisions = 0

    def decide(self, state, questions):
        rows = [{"state": state, "question": q} for q in questions.values()]
        with self.lock, torch.inference_mode():
            probs, tokens = [], 0
            for s in range(0, len(rows), 8):
                try:
                    b = self.m.prepare(rows[s:s + 8])
                except ValueError as e:
                    raise Capacity(f"maximum context length exceeded: {e}") from None
                probs += [p[:c] for p, c in zip((self.m(b) / self.m.temperature).softmax(-1).cpu().tolist(), b.counts)]
                tokens += b.input_tokens
                self.steps += 1
            self.decisions += len(rows)
        return {k: shape(q, p) for (k, q), p in zip(questions.items(), probs)}, tokens


def handler(backend):
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
                return self._send(200, {"status": "ready", "model": NAME, "steps": backend.steps, "decisions": backend.decisions})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                validate(body.get("questions"))
                answers, tokens = backend.decide(body.get("state"), body["questions"])
                self._send(200, {"model": NAME, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}})
            except Capacity as e:
                self._send(422, {"error": str(e)})
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": f"{type(e).__name__}: {e}"[:300]})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["engine", "ref"], default="engine")
    ap.add_argument("--linear", default="bf16")
    ap.add_argument("--compile", default="1")
    ap.add_argument("--fp8-skip", default="")
    ap.add_argument("--gva", default="1", help="native grouped-value heads in fla (bit-identical, faster)")
    ap.add_argument("--budget", type=int, default=16384)
    ap.add_argument("--max-wait-ms", type=float, default=0.0, help="extra wait to fill a step (0 = take what is queued)")
    ap.add_argument("--port", type=int, default=8741)
    ap.add_argument("--warm-steps", type=int, default=200, help="randomised warm-up steps before /health is ready")
    a = ap.parse_args()
    backend = EngineBackend(a) if a.backend == "engine" else RefBackend(a)
    class Srv(ThreadingHTTPServer):
        request_queue_size = 512
    srv = Srv(("127.0.0.1", a.port), handler(backend))
    srv.daemon_threads = True
    print(f"ready on :{a.port} backend={a.backend}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
