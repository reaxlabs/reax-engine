import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from engine import mini_serve
from engine.backends.base import Encoded


class Fake:
    name, kind, max_input_tokens, max_batch_tokens = "fake", "vllm", 100, 10
    batches = []

    def options(self, q):
        return ["false", "true"] if q["type"] == "noul" else list(q["criteria"])

    def encode(self, state, q):
        if state == "toolong":
            raise ValueError("maximum context length exceeded")
        if q["type"] == "x":
            raise ValueError("bad type")
        return Encoded(state, 4, len(self.options(q)))

    def run(self, batch):
        Fake.batches.append(len(batch))
        return [[1.0 / e.n_options] * e.n_options for e in batch]


@pytest.fixture()
def server():
    be = Fake()
    sched = mini_serve.Scheduler(be)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mini_serve.make_handler(be, sched, "tok"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


def post(url, path, body, token=None):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **({"X-S1M-Internal-Token": token} if token else {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_decide_contract_and_errors(server):
    q = {"type": "choice", "instructions": "i", "criteria": {"a": "x", "b": "y"}}
    code, out = post(server, "/internal/v1/decide", {"state": "s", "questions": {"q1": q}}, "tok")
    assert code == 200 and out["answers"]["q1"]["probabilities"] == {"a": 0.5, "b": 0.5} and out["answers"]["q1"]["choice"] == "a"
    assert out["usage"] == {"input_tokens": 4, "output_tokens": 0}
    assert post(server, "/internal/v1/decide", {"state": "s", "questions": {"q1": q}}, "wrong")[0] == 401
    assert post(server, "/internal/v1/decide", {"state": "s", "questions": {"q1": q, "q2": q}}, "tok")[0] == 400
    assert post(server, "/internal/v1/decide", {"state": "toolong", "questions": {"q1": q}}, "tok")[0] == 422
    assert post(server, "/internal/v1/decide", {"state": "s", "questions": {"q1": {"type": "x"}}}, "tok")[0] == 400
    assert post(server, "/internal/v1/decide", {"state": "s"}, "tok")[0] == 400
    assert post(server, "/nope", {})[0] == 404


def test_systemone_multi_and_health(server):
    qs = {"u": {"type": "choice", "criteria": {"a": None, "b": None, "c": None}}, "v": {"type": "noul"}}
    code, out = post(server, "/v1/systemone", {"state": "s", "questions": qs})
    assert code == 200 and set(out["answers"]) == {"u", "v"} and out["answers"]["v"]["noul"] == 0.5
    with urllib.request.urlopen(server + "/health") as r:
        assert json.load(r)["ok"]
