#!/usr/bin/env python3
"""REAX decision server: one asyncio/uvicorn process, one continuous batcher over a `Backend` (backends/base.py).

Routes
  POST /internal/v1/decide   the s1 runtime sidecar contract: header X-S1M-Internal-Token, one question per call,
                             {"model","answers":{qid:{type,probabilities,confidence,choice|noul|score}},"usage"}
  POST /v1/systemone         the same with any number of questions per call
  GET  /health               200 only after warm-up (503 before)
Listens on $RUNTIME_UDS (unix socket, mode 0666) if set, else RUNTIME_HOST:RUNTIME_PORT (127.0.0.1:8101).
Token: S1M_INTERNAL_TOKEN or the file S1M_INTERNAL_TOKEN_FILE; RUNTIME_ALLOW_NO_AUTH=1 serves without one.

Scheduling: request handlers tokenise on the event loop (a worker thread for very long inputs) and put `Encoded` items
on a queue; ONE scheduler thread (the only one that touches the GPU) takes everything queued, packs it up to
`backend.max_batch_tokens` tokens (waiting at most --max-wait-ms for more when the queue was empty at the first item;
0 = never wait: a lone request runs immediately, a burst is packed because requests pile up while a step runs) and
hands the probabilities back to the event loop in one callback per step.

    python engine/serve.py --model plumb-4b            # weights at $PLUMB_MODEL_DIR (default /model)
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from backends.base import sidecar_answer  # noqa: E402

LONG_TEXT = 20000  # request bodies longer than this are tokenised in a worker thread, not on the event loop


# ------------------------------------------------------------------------------------------------ scheduler
class Scheduler:
    def __init__(self, backend, max_wait_ms: float = 0.0, max_items: int = 256):
        self.backend = backend
        self.q: queue.SimpleQueue = queue.SimpleQueue()
        self.max_wait = max_wait_ms / 1000.0
        self.max_items = max_items
        self.steps = self.decisions = self.tokens = 0
        self._on_ready = None
        self.thread = threading.Thread(target=self._loop, name="scheduler", daemon=True)

    def start(self, on_ready=None):
        self._on_ready = on_ready
        self.thread.start()

    def submit(self, enc, loop, fut):
        self.q.put((enc, loop, fut))

    def _loop(self):
        be, q = self.backend, self.q
        if self._on_ready:  # warm-up on the thread that serves afterwards (CUDA graphs, compile caches are per thread)
            self._on_ready()
        limit = be.max_batch_tokens
        carry = None
        while True:
            first = carry or q.get()
            carry = None
            batch, n = [first], first[0].n_tokens
            if limit == 0:
                pass  # one item per run()
            else:
                deadline = time.perf_counter() + self.max_wait if self.max_wait else 0.0
                while len(batch) < self.max_items:
                    try:
                        if deadline:
                            item = q.get(timeout=max(0.0, deadline - time.perf_counter()))
                        else:
                            item = q.get_nowait()
                    except queue.Empty:
                        break
                    if n + item[0].n_tokens > limit:
                        carry = item
                        break
                    batch.append(item)
                    n += item[0].n_tokens
            try:
                res = be.run([b[0] for b in batch])
                err = None
            except Exception as e:  # noqa: BLE001
                res, err = None, e
            self.steps += 1
            self.decisions += len(batch)
            self.tokens += n
            by_loop: dict = {}
            for i, b in enumerate(batch):
                by_loop.setdefault(b[1], []).append((b[2], None if err else res[i]))
            for loop, items in by_loop.items():
                loop.call_soon_threadsafe(_deliver, items, err)


def _deliver(items, err):
    for fut, val in items:
        if not fut.done():
            if err is not None:
                fut.set_exception(err)
            else:
                fut.set_result(val)


# ------------------------------------------------------------------------------------------------ app
def _token() -> str:
    tok = os.environ.get("S1M_INTERNAL_TOKEN", "")
    path = os.environ.get("S1M_INTERNAL_TOKEN_FILE", "")
    if not tok and path and os.path.exists(path):
        tok = open(path).read().strip()
    if not tok and os.environ.get("RUNTIME_ALLOW_NO_AUTH") != "1":
        sys.exit("serve: no token (S1M_INTERNAL_TOKEN or S1M_INTERNAL_TOKEN_FILE; RUNTIME_ALLOW_NO_AUTH=1 to disable)")
    return tok


class App:
    def __init__(self, backend, sched: Scheduler, token: str, model_name: str):
        self.be, self.sched, self.token, self.model = backend, sched, token.encode(), model_name
        self.ready = False
        self.loop_hdr = b"x-s1m-internal-token"
        self._shape = getattr(backend, "answer", None)

    # -------- one decision (all questions of a request are batched together)
    async def decide(self, state, questions: dict, long_body: bool = False):
        loop = asyncio.get_running_loop()
        encs = {}
        for qid, q in questions.items():
            encs[qid] = await loop.run_in_executor(None, self.be.encode, state, q) if long_body else self.be.encode(state, q)
        futs = {}
        for qid, enc in encs.items():
            futs[qid] = loop.create_future()
            self.sched.submit(enc, loop, futs[qid])
        probs = {qid: await f for qid, f in futs.items()}
        answers = {}
        for qid, q in questions.items():
            keys = self.be.options(q)
            answers[qid] = self._shape(q, probs[qid]) if self._shape else sidecar_answer(q, keys, probs[qid])
        return answers, sum(e.n_tokens for e in encs.values())

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        path, method = scope["path"], scope["method"]
        if path == "/health" and method == "GET":
            if not self.ready:
                return await self._json(send, 503, {"ok": False, "status": "warming up"})
            return await self._json(send, 200, {"ok": True, "model": self.model, "steps": self.sched.steps,
                                                "decisions": self.sched.decisions})
        if method != "POST" or path not in ("/internal/v1/decide", "/v1/systemone"):
            return await self._json(send, 404, {"error": {"message": "not found"}})
        if self.token:
            got = b""
            for k, v in scope["headers"]:
                if k == self.loop_hdr:
                    got = v
                    break
            if not hmac.compare_digest(got, self.token):
                return await self._json(send, 401, {"error": {"message": "unauthorized"}})
        if not self.ready:
            return await self._json(send, 503, {"error": {"message": "warming up"}})
        chunks = []
        while True:
            msg = await receive()
            chunks.append(msg.get("body", b""))
            if not msg.get("more_body"):
                break
        try:
            body = json.loads(b"".join(chunks))
            questions = body["questions"]
            state = body["state"]
            if not isinstance(questions, dict) or not questions:
                raise ValueError("questions must be a non-empty object")
            if path == "/internal/v1/decide" and len(questions) != 1:
                raise ValueError("exactly one question per call")
            for q in questions.values():
                if not isinstance(q, dict) or q.get("type") not in ("choice", "noul", "score"):
                    raise ValueError("bad question type")
                q.setdefault("instructions", "")
        except Exception as exc:  # noqa: BLE001
            return await self._json(send, 400, {"error": {"message": f"bad request: {type(exc).__name__}"}})
        try:
            answers, n = await self.decide(state, questions, sum(map(len, chunks)) > LONG_TEXT)
        except ValueError as exc:
            return await self._json(send, 400, {"error": {"message": str(exc)}})
        except Exception as exc:  # noqa: BLE001
            return await self._json(send, 500, {"error": {"message": f"{type(exc).__name__}"}})
        await self._json(send, 200, {"model": self.model, "answers": answers,
                                     "usage": {"input_tokens": n, "output_tokens": 0}})

    @staticmethod
    async def _json(send, code, obj):
        raw = json.dumps(obj).encode()
        await send({"type": "http.response.start", "status": code,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode())]})
        await send({"type": "http.response.body", "body": raw})


# ------------------------------------------------------------------------------------------------ manifest check
def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_weights(model_dir: str, manifest_path: str) -> None:
    """Check the sha256 of the pinned weight files against the release manifest (reax-s1-manifest/1)."""
    files = json.loads(Path(manifest_path).read_text())["weights"]["files"]
    out: dict = {}

    def one(name, digest):
        p = Path(model_dir) / name
        out[name] = p.is_file() and _sha(p) == digest
    ts = [threading.Thread(target=one, args=kv) for kv in files.items()]
    [t.start() for t in ts]
    [t.join() for t in ts]
    bad = [n for n, ok in out.items() if not ok]
    if bad:
        sys.exit(f"weights mismatch or missing: {', '.join(sorted(bad))}")


# ------------------------------------------------------------------------------------------------ main
def load_backend(name: str, a):
    if name == "plumb-4b":
        from plumb_engine import PlumbBackend
        kw = {}
        if a.batch_tokens:
            kw["max_batch_tokens"] = a.batch_tokens
        return PlumbBackend(a.model_dir, compile=a.compile, **kw)
    raise SystemExit(f"unknown model {name!r} (known: plumb-4b)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="plumb-4b")
    ap.add_argument("--model-dir", default=os.environ.get("PLUMB_MODEL_DIR", "/model"))
    ap.add_argument("--manifest", default=os.environ.get("REAX_MANIFEST", ""), help="release manifest to verify weights against")
    ap.add_argument("--compile", default=os.environ.get("REAX_COMPILE", "1"))
    ap.add_argument("--batch-tokens", type=int, default=0)
    ap.add_argument("--max-wait-ms", type=float, default=float(os.environ.get("REAX_MAX_WAIT_MS", "0")))
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args(argv)

    token = _token()
    if a.manifest and Path(a.manifest).is_file():
        verify_weights(a.model_dir, a.manifest)
    backend = load_backend(a.model, a)
    sched = Scheduler(backend, a.max_wait_ms)
    app = App(backend, sched, token, "s1-fast" if a.model == "plumb-4b" else a.model)

    def warm():
        t0 = time.time()
        if not a.no_warmup:
            backend.warmup()
        app.ready = True
        print(f"ready: warm-up {time.time() - t0:.0f}s", flush=True)

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
