"""engine/serve.py over one fake backend per kind (native-gpu stub, vLLM via a fake vLLM server, gguf-cpu stub):
routes, auth, error codes, batching, and the answer shape of /internal/v1/decide and /v1/systemone per backend.
The ASGI app is called directly (no sockets); the scheduler thread is the real one."""
import asyncio
import json
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from engine import registry, serve
from engine.backends import vllm_readout as V
from engine.backends.base import Encoded
from tests.fake_tok import FakeTok
from tests.test_vllm_readout import FakeVllm

ROOT = Path(__file__).resolve().parent.parent
CHOICE = {"type": "choice", "instructions": "i", "criteria": {"a": "x", "b": "y"}}
NOUL = {"type": "noul", "instructions": "i"}
SCORE = {"type": "score", "instructions": "i", "criteria": ["l0", "l1", "l2"]}


class Stub:
    max_input_tokens, max_batch_tokens = 100, 10

    def __init__(self):
        self.batches = []

    def options(self, q):
        if q["type"] == "noul":
            return ["false", "true"]
        return list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]

    def encode(self, state, q):
        if state == "toolong":
            raise ValueError("maximum context length exceeded")
        if q["type"] not in ("choice", "noul", "score"):
            raise ValueError("bad type")
        return Encoded(state, 4, len(self.options(q)))

    def run(self, batch):
        self.batches.append(len(batch))
        return [[1.0 / e.n_options] * e.n_options for e in batch]


class NativeStub(Stub):          # a pplx/d3-like backend: its own systemone answer shape, no encode_inline
    name, kind = "native-stub", "native-gpu"

    def answer(self, q, probs):
        return {"type": q["type"], "own": True, "probabilities": dict(zip(self.options(q), probs))}


class PlumbStub(Stub):           # no own shape: the s1 sidecar shape on both routes
    name, kind, encode_inline = "plumb-4b", "native-gpu", True


class GgufStub(Stub):            # the backend's own shape is the s1 shape on both routes
    name, kind = "gguf-stub", "gguf-cpu"

    def answer(self, q, probs):
        out = {"type": q["type"], "probabilities": dict(zip(self.options(q), probs)), "confidence": max(probs), "gguf": True}
        if q["type"] == "noul":
            out["noul"] = probs[1]
        elif q["type"] == "choice":
            out["choice"] = self.options(q)[probs.index(max(probs))]
        else:
            out["score"] = sum(i * p for i, p in enumerate(probs))
        return out


async def call(app, path, body=None, token="tok", method="POST"):
    raw = json.dumps(body).encode() if body is not None else b""
    sent = []
    hdrs = [(b"x-s1m-internal-token", token.encode())] if token else []

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    async def send(msg):
        sent.append(msg)
    await app({"type": "http", "method": method, "path": path, "headers": hdrs}, receive, send)
    return sent[0]["status"], json.loads(sent[1]["body"])


def make_app(backend, token="tok", ready=True):
    sched = serve.Scheduler(backend)
    app = serve.App(backend, sched, token, "s1-fast" if backend.name == "plumb-4b" else backend.name)
    sched.start(on_ready=lambda: setattr(app, "ready", ready))
    for _ in range(200):
        if app.ready == ready and (ready or sched.thread.is_alive()):
            break
        time.sleep(0.01)
    return app


@pytest.mark.parametrize("cls", [NativeStub, PlumbStub, GgufStub])
def test_contract_and_errors(cls):
    async def go():
        app = make_app(cls())
        code, out = await call(app, "/internal/v1/decide", {"state": "s", "questions": {"q1": CHOICE}})
        a = out["answers"]["q1"]
        assert code == 200 and a["probabilities"] == {"a": 0.5, "b": 0.5} and a["choice"] == "a" and a["confidence"] == 0.5
        assert out["model"] == app.model and out["usage"] == {"input_tokens": 4, "output_tokens": 0}
        assert ("gguf" in a) == (cls is GgufStub) and "own" not in a        # the decide route never uses the systemone shape
        code, out = await call(app, "/internal/v1/decide", {"state": "s", "questions": {"n": NOUL, }})
        assert out["answers"]["n"]["noul"] == 0.5
        code, out = await call(app, "/internal/v1/decide", {"state": "s", "questions": {"s": SCORE}})
        assert out["answers"]["s"]["score"] == pytest.approx(1.0)
        bad = [({"state": "s", "questions": {"q1": CHOICE}}, "wrong", 401),
               ({"state": "s", "questions": {"q1": CHOICE, "q2": CHOICE}}, "tok", 400),
               ({"state": "toolong", "questions": {"q1": CHOICE}}, "tok", 422),
               ({"state": "s", "questions": {"q1": {"type": "x"}}}, "tok", 400),
               ({"state": "s"}, "tok", 400), ({"questions": {"q1": CHOICE}}, "tok", 400)]
        for body, tok, want in bad:
            assert (await call(app, "/internal/v1/decide", body, tok))[0] == want, body
        assert (await call(app, "/nope", {}))[0] == 404
        assert (await call(app, "/health", method="GET"))[1]["kind"] == app.be.kind
    asyncio.run(go())


def test_systemone_shapes_per_backend():
    async def go():
        qs = {"c": CHOICE, "n": NOUL, "s": SCORE}
        body = {"state": "s", "questions": qs}
        code, out = await call(make_app(NativeStub()), "/v1/systemone", body)
        assert code == 200 and all(a.get("own") for a in out["answers"].values())          # the model's own shape
        for cls, flag in ((PlumbStub, None), (GgufStub, "gguf")):
            code, out = await call(make_app(cls()), "/v1/systemone", body)
            assert code == 200 and out["answers"]["c"]["choice"] == "a" and out["answers"]["n"]["noul"] == 0.5
            assert all((flag in a) == bool(flag) for a in out["answers"].values())
        assert (await call(make_app(PlumbStub()), "/v1/systemone", body, token="wrong"))[0] == 401
    asyncio.run(go())


def test_health_gates_on_warmup_and_no_token_means_open():
    async def go():
        app = make_app(PlumbStub(), token="", ready=False)
        assert (await call(app, "/health", method="GET"))[0] == 503
        assert (await call(app, "/internal/v1/decide", {"state": "s", "questions": {"q": CHOICE}}, token=""))[0] == 503
        app.ready = True
        assert (await call(app, "/health", method="GET"))[0] == 200
        assert (await call(app, "/internal/v1/decide", {"state": "s", "questions": {"q": CHOICE}}, token=""))[0] == 200
    asyncio.run(go())


def test_batching_packs_queued_items():
    async def go():
        be = NativeStub()
        app = make_app(be)
        rs = await asyncio.gather(*(call(app, "/internal/v1/decide", {"state": "s", "questions": {"q": CHOICE}}) for _ in range(30)))
        assert all(c == 200 for c, _ in rs) and sum(be.batches) == 30
        assert max(be.batches) > 1 and max(be.batches) <= 2        # max_batch_tokens 10 / 4 tokens per item
    asyncio.run(go())


def test_vllm_backend_through_serve():
    FakeVllm.seen.clear()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeVllm)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        be = V.VllmReadoutBackend("unused", family="quyet", name="quyet-1.0-large", base_url=f"http://127.0.0.1:{srv.server_port}",
                                  tokenizer=FakeTok())
        crit = {"type": "choice", "instructions": "Pick.", "criteria": {"x": "first", "y": "second"}}

        async def go():
            app = make_app(be)
            body = {"state": "s", "questions": {"c": crit, "n": {"type": "noul", "instructions": "Is it?", "criteria": {"true": "yes"}}}}
            code, out = await call(app, "/v1/systemone", body)
            assert code == 200 and out["model"] == "quyet-1.0-large"
            assert out["answers"]["c"]["choice"] == "x" and set(out["answers"]["c"]["probabilities"]) == {"x", "y"}
            assert out["answers"]["n"]["type"] == "noul" and "confidence" in out["answers"]["n"]     # Quyet's own shape (rounded)
            code, out = await call(app, "/internal/v1/decide", {"state": "s", "questions": {"c": crit}})
            a = out["answers"]["c"]                                                              # s1 shape: unrounded probabilities
            assert code == 200 and a["choice"] == "x" and a["confidence"] == max(a["probabilities"].values())
            assert a["probabilities"]["x"] != round(a["probabilities"]["x"], 4)
            toolong = {"state": "w" * 100000, "questions": {"c": crit}}
            assert (await call(app, "/internal/v1/decide", toolong))[0] in (200, 422)
            many = {"type": "choice", "instructions": "q", "criteria": {f"k{i}": None for i in range(11)}}
            assert (await call(app, "/internal/v1/decide", {"state": "s", "questions": {"c": many}}))[0] == 400
        asyncio.run(go())
    finally:
        srv.shutdown()
        srv.server_close()
        be.close()


def test_load_backend_uses_registry_and_passes_opts(monkeypatch):
    got = {}
    monkeypatch.setattr(registry, "build", lambda mid, d, **o: got.update(id=mid, dir=d, opts=o) or "backend")
    a = type("A", (), dict(opt=["preset=fp8-kv", "max_options=255", "flag=true"], model_dir="/m", no_verify=False,
                           batch_tokens=0, compile="1"))
    assert serve.load_backend("quyet", a) == "backend"                    # alias resolves to the registry id
    assert got == {"id": "quyet-1.0-large", "dir": "/m", "opts": {"preset": "fp8-kv", "max_options": 255, "flag": True}}
    with pytest.raises(SystemExit, match="unknown model"):
        serve.load_backend("nope", a)
    with pytest.raises(SystemExit):                                          # documented-only entry has no backend
        monkeypatch.undo()
        serve.load_backend("s1-llm-auto-router", a)


def test_gguf_weight_check(tmp_path):
    a = type("A", (), dict(opt=[], model_dir=str(tmp_path), no_verify=False, batch_tokens=0, compile="1"))
    with pytest.raises(SystemExit, match="weights mismatch or missing"):
        serve.load_backend("gutsy-0.8b", a)


@pytest.mark.parametrize("cmd", [[sys.executable, "engine/serve.py"], [sys.executable, "-m", "engine.serve"]])
def test_cli_both_entry_points(cmd):
    r = subprocess.run([*cmd, "--help"], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "--opt" in r.stdout and "--model-dir" in r.stdout and "--port" in r.stdout
    r = subprocess.run([*cmd, "--model", "nope"], cwd=ROOT, capture_output=True, text=True, timeout=60,
                       env={"PATH": "/usr/bin", "RUNTIME_ALLOW_NO_AUTH": "1"})
    assert r.returncode != 0 and "unknown model" in r.stderr
