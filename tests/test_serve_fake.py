"""serve.py contract tests with a fake CPU backend (no GPU): auth, 400s, health gating, batching, multi-question.
Run: python tests/test_serve_fake.py"""
import asyncio
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
import aiohttp  # noqa: E402
import uvicorn  # noqa: E402
from backends.base import Encoded  # noqa: E402
from serve import App, Scheduler  # noqa: E402


class Fake:
    name, kind, max_input_tokens, max_batch_tokens = "fake", "native-gpu", 50, 100
    batches = []

    def options(self, q):
        return list(q["criteria"]) if q["type"] == "choice" else (["true", "false"] if q["type"] == "noul" else [str(i) for i in range(len(q["criteria"]))])

    def encode(self, state, q):
        n = len(json.dumps(state))
        if n > self.max_input_tokens:
            raise ValueError("too long")
        return Encoded(payload=n, n_tokens=n, n_options=len(self.options(q)))

    def run(self, batch):
        time.sleep(0.01)
        self.batches.append(len(batch))
        return [[1.0 / e.n_options] * e.n_options for e in batch]


async def main():
    d = tempfile.mkdtemp()
    sock = os.path.join(d, "e.sock")
    be = Fake()
    sched = Scheduler(be)
    app = App(be, sched, "tok", "s1-fast")
    sched.start(on_ready=lambda: setattr(app, "ready", True))
    srv = uvicorn.Server(uvicorn.Config(app, uds=sock, log_level="warning", lifespan="on"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not os.path.exists(sock):
        await asyncio.sleep(0.05)
    os.chmod(sock, 0o666)
    H = {"X-S1M-Internal-Token": "tok"}
    q = {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": "B"}}
    async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=sock)) as s:
        async def post(path, body, hdr=H):
            async with s.post("http://e" + path, data=json.dumps(body), headers=hdr) as r:
                return r.status, await r.json()
        async with s.get("http://e/health") as r:
            assert r.status == 200
        assert (await post("/internal/v1/decide", {"state": "s", "questions": {"q": q}}, {}))[0] == 401
        assert (await post("/internal/v1/decide", {"state": "s", "questions": {}}))[0] == 400
        assert (await post("/internal/v1/decide", {"state": "s", "questions": {"a": q, "b": q}}))[0] == 400
        assert (await post("/internal/v1/decide", {"questions": {"q": q}}))[0] == 400
        assert (await post("/internal/v1/decide", {"state": "s", "questions": {"q": {"type": "x"}}}))[0] == 400
        assert (await post("/internal/v1/decide", {"state": "z" * 80, "questions": {"q": q}}))[0] == 400
        code, js = await post("/internal/v1/decide", {"state": "s", "questions": {"q": q}})
        assert code == 200 and js["answers"]["q"]["probabilities"] == {"a": 0.5, "b": 0.5}, js
        assert js["answers"]["q"]["choice"] == "a" and js["usage"]["output_tokens"] == 0
        noul = {"type": "noul", "instructions": "x"}
        code, js = await post("/internal/v1/decide", {"state": {"k": 1}, "questions": {"n": noul}})
        assert js["answers"]["n"]["noul"] == 0.5
        score = {"type": "score", "instructions": "x", "criteria": ["l0", "l1", "l2"]}
        code, js = await post("/v1/systemone", {"state": "s", "questions": {"a": q, "n": noul, "s": score}})
        assert code == 200 and abs(js["answers"]["s"]["score"] - 1.0) < 1e-9, js
        Fake.batches.clear()
        rs = await asyncio.gather(*(post("/internal/v1/decide", {"state": "s", "questions": {"q": q}}) for _ in range(40)))
        assert all(c == 200 for c, _ in rs)
        print("batches", Fake.batches)
        assert max(Fake.batches) > 1
    print("ok")
    srv.should_exit = True


asyncio.run(main())
