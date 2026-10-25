"""Production template goldens and reference-image contract, CPU only."""
import asyncio
import base64
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from engine import registry, serve
from engine.backends import vllm_readout as V
from engine.backends.common import softmax_t
from tests.fake_tok import FakeTok
from tests.test_serve_backends import call, make_app, NativeStub

GOLD = json.loads((Path(__file__).parent / "fixtures/node_letters_golden.json").read_text())
PNG = GOLD["image"]
Q = {"type": "choice", "instructions": "What color?", "criteria": {"red": "red", "blue": "blue"}}


def backend(**kw):
    return V.VllmReadoutBackend("unused", family="blink", name="blink-v0.3-26b-a4b", preset="ref-prod",
                              max_input_tokens=8000, tokenizer=FakeTok(), base_url="http://127.0.0.1:1", **kw)


@pytest.mark.parametrize("row", GOLD["rows"])
def test_production_goldens_and_unchanged_blink_text(row):
    tok = FakeTok()
    d = V.parse_quyet_question(row["question"], 10)
    assert V.build_quyet_prompt(tok, row["state"], d, V.LETTERS, 10**9)[0] == row["quyet_prompt"]
    assert V.build_blink_image_messages(row["state"], row["question"], V.LETTERS, PNG) == row["vision_messages"]
    e = V.ReadoutEncoder(tok, V.load_family_config("blink"))
    assert tok.decode(e.encode(row["state"], row["question"]).payload.ids) == row["blink_text_prompt_4230e50"]


def test_quyet_reference_registry_and_argv(monkeypatch):
    spec = registry.get("quyet-1-large")
    assert spec.id == "quyet-1.0-large"
    assert spec.revision == "d997d58951e3c0da4486cf0884ca9d4d916e95dc"
    assert spec.production_opts == {"preset": "ref-prod", "max_input_tokens": 8000, "max_options": 10,
                                    "no_truncation": True}
    monkeypatch.setenv("REAX_PROFILE", "reference")
    got = {}
    monkeypatch.setattr(registry, "build", lambda name, model_dir, **kw: got.update(kw))
    from argparse import Namespace
    serve.load_backend("quyet-1-large", Namespace(opt=[], model_dir="/model"))
    assert got == spec.production_opts
    argv = V.vllm_argv(V.preset_for("quyet", "ref-prod"), "/model")
    for flag, value in {"--gpu-memory-utilization": "0.45", "--max-model-len": "8192",
                        "--max-num-seqs": "64", "--max-logprobs": "256", "--quantization": "fp8",
                        "--kv-cache-dtype": "fp8"}.items():
        assert argv[argv.index(flag) + 1] == value
    assert "--enable-prefix-caching" in argv
    assert V.load_family_config("quyet").temperatures == {"choice": 1.3007, "score": 1.3159, "noul": 1.4957}
    e = V.ReadoutEncoder(FakeTok(), V.load_family_config("quyet", no_truncation=True))
    assert "x" * 6100 in FakeTok().decode(e.encode("x" * 6100, Q).payload.ids)
    with pytest.raises(ValueError, match="maximum context length"):
        e.encode("x" * 8100, Q)
    argv = V.vllm_argv(V.preset_for("blink", "ref-prod"), "/model")
    assert argv[argv.index("--limit-mm-per-prompt") + 1] == '{"image":2,"audio":0,"video":0}'
    assert "--chat-template-content-format" not in argv


@pytest.mark.parametrize("images", ["bad", {}, False, 0, [123], [PNG, PNG],
    ["https://example.com/image.png"], ["data:image/png;base64,%%%"],
    ["data:image/png;base64," + base64.b64encode(b"not an image").decode()],
    [PNG.replace("image/png", "image/jpeg")]])
def test_image_refusals_are_422(images):
    be = backend()
    try:
        app = make_app(be)
        status, out = asyncio.run(call(app, "/internal/v1/decide", {"state": {}, "questions": {"q": Q}, "images": images}))
        assert status == 422 and out["code"] == "invalid_images"
    finally:
        be.close()


def test_text_only_images_retain_unsupported_style():
    status, out = asyncio.run(call(make_app(NativeStub()), "/internal/v1/decide",
                                   {"state": {}, "questions": {"q": Q}, "images": [PNG]}))
    assert status == 422 and out["code"] == "unsupported_images" and out["fallback"] == "author"


def test_vision_chat_wire_and_usage_and_context_refusal():
    seen = []
    class FakeChat(BaseHTTPRequestHandler):
        count = 777
        def log_message(self, *_): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append((self.path, body))
            ids = body["allowed_token_ids"]
            response = {"choices": [{"logprobs": {"content": [{"token": f"token_id:{ids[0]}", "logprob": -0.2,
                        "top_logprobs": [{"token": f"token_id:{ids[0]}", "logprob": -0.2},
                                         {"token": f"token_id:{ids[1]}", "logprob": -2.0}]}]}}],
                        "usage": {"prompt_tokens": self.count}}
            raw = json.dumps(response).encode()
            self.send_response(200);self.send_header("Content-Length", str(len(raw)));self.end_headers();self.wfile.write(raw)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeChat)
    thread = threading.Thread(target=srv.serve_forever, daemon=True);thread.start()
    be = backend();be.client = V.VllmClient(f"http://127.0.0.1:{srv.server_port}")
    try:
        app = make_app(be)
        body = {"state": "raw string", "questions": {"q": Q}, "images": [PNG]}
        code, out = asyncio.run(call(app, "/internal/v1/decide", body))
        assert code == 200 and out["usage"] == {"input_tokens": 777, "output_tokens": 0}
        assert out["answers"]["q"]["probabilities"]["red"] == pytest.approx(softmax_t([-0.2, -2], 1.3)[0])
        path, wire = seen[0]
        assert path == "/v1/chat/completions" and wire["temperature"] == 1 and wire["top_p"] == 1
        assert wire["logprobs"] is True and wire["top_logprobs"] == 2 and wire["max_tokens"] == 1
        assert wire["messages"] == V.build_blink_image_messages("raw string", Q, V.LETTERS, PNG)
        FakeChat.count = 8001
        assert asyncio.run(call(app, "/internal/v1/decide", body))[0] == 422
        body["questions"]["q2"] = Q
        assert asyncio.run(call(app, "/v1/systemone", body))[0] == 422
    finally:
        be.close();srv.shutdown();srv.server_close();thread.join()
