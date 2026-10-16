"""Minimal decision server for the GGUF CPU backend (``POST /internal/v1/decide``, ``GET /health``).

Same wire contract as the s1 runtime sidecar: header ``X-S1M-Internal-Token``, body ``{"state", "questions": {id: q}}``,
answer ``{"model", "answers": {id: {...}}, "usage": {...}}``. Requests are tokenised in the request thread, a single
scheduler thread packs queued items (up to ``backend.max_batch_tokens``) and calls ``backend.run``.
This stand-in exists until the multi-model ``engine/serve.py`` takes over; it only knows the GGUF CPU models.

  python engine/cpu_serve.py --model gutsy-0.8b --model-dir /model

Environment: RUNTIME_HOST / RUNTIME_PORT (default 127.0.0.1:8103) or RUNTIME_UDS, S1M_INTERNAL_TOKEN or
S1M_INTERNAL_TOKEN_FILE (RUNTIME_ALLOW_NO_AUTH=1 to disable), GGUF_CONTEXTS, GGUF_THREADS, GGUF_N_BATCH.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends.gguf_cpu import SPECS, GGUFCPUBackend  # noqa: E402


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _token() -> str:
    tok = os.environ.get("S1M_INTERNAL_TOKEN", "")
    path = os.environ.get("S1M_INTERNAL_TOKEN_FILE", "")
    if not tok and path and os.path.exists(path):
        tok = open(path).read().strip()
    if not tok and os.environ.get("RUNTIME_ALLOW_NO_AUTH") != "1":
        sys.exit("cpu_serve: no token (S1M_INTERNAL_TOKEN or S1M_INTERNAL_TOKEN_FILE; RUNTIME_ALLOW_NO_AUTH=1 to disable)")
    return tok


class Scheduler:
    def __init__(self, backend, max_items: int = 64):
        self.be, self.max_items = backend, max_items
        self.q: queue.SimpleQueue = queue.SimpleQueue()
        self.thread = threading.Thread(target=self._loop, name="scheduler", daemon=True)

    def submit(self, enc, loop, fut):
        self.q.put((enc, loop, fut))

    def _loop(self):
        limit, carry = self.be.max_batch_tokens, None
        while True:
            first = carry or self.q.get()
            carry, batch, n = None, [first], first[0].n_tokens
            while limit and len(batch) < self.max_items:
                try:
                    item = self.q.get_nowait()
                except queue.Empty:
                    break
                if n + item[0].n_tokens > limit:
                    carry = item
                    break
                batch.append(item)
                n += item[0].n_tokens
            try:
                res, err = self.be.run([b[0] for b in batch]), None
            except Exception as exc:  # noqa: BLE001
                res, err = None, exc
            for i, (_, loop, fut) in enumerate(batch):
                loop.call_soon_threadsafe(_deliver, fut, None if err else res[i], err)


def _deliver(fut, val, err):
    if not fut.done():
        fut.set_exception(err) if err is not None else fut.set_result(val)


class App:
    def __init__(self, backend, sched, token: str, model: str):
        self.be, self.sched, self.token, self.model = backend, sched, token.encode(), model

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                await send({"type": msg["type"] + ".complete"})
                if msg["type"] == "lifespan.shutdown":
                    return
        path, method = scope["path"], scope["method"]
        if path == "/health" and method == "GET":
            return await self._json(send, 200, {"ok": True, "model": self.model, "kind": self.be.kind})
        if path != "/internal/v1/decide" or method != "POST":
            return await self._json(send, 404, {"error": {"message": "not found"}})
        given = dict(scope["headers"]).get(b"x-s1m-internal-token", b"")
        if self.token and not hmac.compare_digest(given, self.token):
            return await self._json(send, 401, {"error": {"message": "unauthorized"}})
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        try:
            req = json.loads(body)
            state, questions = req["state"], req["questions"]
            if not isinstance(questions, dict) or not questions:
                raise ValueError("questions")
        except Exception as exc:  # noqa: BLE001
            return await self._json(send, 400, {"error": {"message": f"bad request: {type(exc).__name__}"}})
        loop = asyncio.get_running_loop()
        try:
            encs = {qid: await loop.run_in_executor(None, self.be.encode, state, q) for qid, q in questions.items()}
            futs = {}
            for qid, enc in encs.items():
                futs[qid] = loop.create_future()
                self.sched.submit(enc, loop, futs[qid])
            answers = {qid: self.be.answer(questions[qid], await fut) for qid, fut in futs.items()}
        except ValueError as exc:
            return await self._json(send, 400, {"error": {"message": str(exc)}})
        usage = {"input_tokens": sum(e.n_tokens for e in encs.values()), "output_tokens": 0}
        await self._json(send, 200, {"model": self.model, "answers": answers, "usage": usage})

    @staticmethod
    async def _json(send, code, obj):
        raw = json.dumps(obj).encode()
        await send({"type": "http.response.start", "status": code,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode())]})
        await send({"type": "http.response.body", "body": raw})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gutsy-0.8b", choices=sorted(SPECS))
    ap.add_argument("--model-dir", default=os.environ.get("GGUF_MODEL_DIR", "/model"))
    ap.add_argument("--no-verify", action="store_true", help="skip the sha256 check of the pinned weight file")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args(argv)
    spec = SPECS[a.model]
    token = _token()
    if not a.no_verify and _sha(Path(a.model_dir) / spec.file) != spec.sha256:
        sys.exit(f"weights mismatch or missing: {spec.file}")
    backend = GGUFCPUBackend(a.model_dir, spec)
    if not a.no_warmup:
        t0 = time.time()
        backend.warmup()
        print(f"ready: {a.model} contexts={backend.contexts} threads={backend.threads} warm-up {time.time() - t0:.1f}s",
              flush=True)
    sched = Scheduler(backend)
    sched.thread.start()
    import uvicorn
    kw = dict(log_level="warning", access_log=False, lifespan="on", timeout_keep_alive=75)
    if os.environ.get("RUNTIME_UDS"):
        kw["uds"] = os.environ["RUNTIME_UDS"]
    else:
        kw.update(host=os.environ.get("RUNTIME_HOST", "127.0.0.1"), port=int(os.environ.get("RUNTIME_PORT", "8103")))
    uvicorn.run(App(backend, sched, token, a.model), **kw)


if __name__ == "__main__":
    main()
