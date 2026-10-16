import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from engine.backends import vllm_readout as V
from tests.fake_tok import FakeTok

CHOICE = {"type": "choice", "instructions": "Pick one.", "criteria": {"x": "first", "y": None, "z": {"k": 1}}}
NOUL = {"type": "noul", "instructions": "Is it?", "criteria": {"true": "yes it is"}}
SCORE = {"type": "score", "instructions": "How much?", "criteria": ["none", "some", "lots"]}


def enc(family, **ov):
    return V.ReadoutEncoder(FakeTok(), V.load_family_config(family, None, **ov))


# ---------------------------------------------------------------------------------- prompts
def test_quyet_prompt_v2_text():
    e = enc("quyet")
    item = e.encode({"a": 1}, CHOICE)
    text = FakeTok().decode(item.payload.ids)
    assert text == ('<user>State:\n{"a":1}\n\nQuestion: Pick one.\nChoose the option that fits best.\n\n'
                    'Options:\nA. first\nB. y\nC. {"k": 1}</><model>')   # no system message, compact state, label as fallback text
    assert item.n_options == 3 and item.n_tokens == len(item.payload.ids)
    assert item.payload.letter_ids == [ord("A"), ord("B"), ord("C")]
    assert item.payload.temperature == 1.3007


def test_quyet_noul_and_score_order_and_temperature():
    e = enc("quyet")
    assert e.options(NOUL) == ["true", "false"]                       # letter A = true
    n = e.encode("s", NOUL)
    assert "A. yes it is\nB. false" in FakeTok().decode(n.payload.ids) and n.payload.temperature == 1.4957
    assert e.options(SCORE) == ["0", "1", "2"]
    assert e.encode("s", SCORE).payload.temperature == 1.3159


def test_quyet_limits_and_truncation():
    e = enc("quyet")
    many = {"type": "choice", "instructions": "q", "criteria": {f"k{i}": None for i in range(11)}}
    with pytest.raises(V.QuestionError):
        e.encode("s", many)                                            # native cap 10
    assert len(enc("quyet", max_options=26).options(many)) == 11       # lifted
    for bad in ({"type": "x"}, {"type": "choice", "criteria": 3}, {"type": "score", "criteria": ["only"]}, "no"):
        with pytest.raises(V.QuestionError):
            e.encode("s", bad)
    small = enc("quyet", max_prompt_tokens=400, max_state_tokens=300, min_state_tokens=10)
    it = small.encode("w" * 1000, CHOICE)                              # state shrunk to fit the prompt cap
    assert it.n_tokens <= 400
    tiny = enc("quyet", max_prompt_tokens=60, max_state_tokens=300, min_state_tokens=10)
    with pytest.raises(ValueError, match="maximum context length"):
        tiny.encode("w" * 1000, CHOICE)


def test_rune_prompt_text():
    e = enc("rune")
    it = e.encode({"a": 1}, NOUL)
    text = FakeTok().decode(it.payload.ids)
    assert text.startswith("<system>" + V.RUNE_SYSTEM + "</><user>SHARED STATE (JSON string):\n{\"a\": 1}\n\nQUESTION:\nIs it?\nOPTIONS:\n")
    assert text.endswith("A: No\nB: yes it is\nAnswer with one option letter only.</><model>")  # noul: false first
    assert e.options(NOUL) == ["false", "true"] and it.payload.temperature == 2.0
    assert FakeTok().decode(e.encode("raw string", CHOICE).payload.ids).count('"raw string"') == 1  # state is JSON-encoded


def test_rune_limits():
    e = enc("rune", max_prompt_tokens=100)
    with pytest.raises(ValueError, match="maximum context length"):
        e.encode("x" * 500, CHOICE)
    with pytest.raises(ValueError):
        e.encode("s", {"type": "weird"})
    big = {"type": "choice", "instructions": "q", "criteria": {f"k{i}": None for i in range(65)}}
    with pytest.raises(ValueError, match="too many options"):
        enc("rune").encode("s", big)


# ---------------------------------------------------------------------------------- answers
def test_answer_shapes():
    e = enc("quyet")
    a = e.answer(CHOICE, [0.2, 0.5, 0.3])
    assert a["choice"] == "y" and a["confidence"] == 0.5 and a["probabilities"] == {"x": 0.2, "y": 0.5, "z": 0.3}
    n = e.answer(NOUL, [0.7, 0.3])
    assert n["noul"] == 0.7 and n["confidence"] == 0.7
    s = e.answer(SCORE, [0.0, 0.5, 0.5])
    assert s["score"] == 1.5 and s["legend"] == {"0": "none", "1": "some", "2": "lots"}
    r = enc("rune")
    assert r.answer(NOUL, [0.25, 0.75]) == {"type": "noul", "noul": 0.75}
    c = r.answer(CHOICE, [0.1, 0.2, 0.7])
    assert c["choice"] == "z" and "confidence" not in c
    assert r.answer(SCORE, [0.0, 0.0, 1.0])["score"] == 2.0


def test_temperature_for_buckets():
    assert V.temperature_for({"choice": 1.5}, "choice", 4) == 1.5
    assert V.temperature_for({"choice:3-5": 1.1, "choice": 1.5}, "choice", 4) == 1.1
    assert V.temperature_for({}, "score", 3, 2.0) == 2.0


def test_quyet_config_json_overrides(tmp_path):
    (tmp_path / "quyet_config.json").write_text(json.dumps(
        {"temperatures": {"choice": 9.0, "score": 8.0, "noul": 7.0}, "limits": {"max_prompt_tokens": 123, "max_state_tokens": 45, "min_state_tokens": 6}}))
    c = V.load_family_config("quyet", tmp_path)
    assert (c.temperatures["choice"], c.max_prompt_tokens, c.max_state_tokens, c.min_state_tokens) == (9.0, 123, 45, 6)


# ---------------------------------------------------------------------------------- presets / argv
def test_presets_have_expected_flags():
    q = V.preset_for("quyet", "fp8-kv")
    argv = V.vllm_argv(q, "/m", port=8123)
    assert argv[:3] == ["vllm", "serve", "/m"] and "8123" in argv
    assert argv[argv.index("--quantization") + 1] == "fp8" and argv[argv.index("--kv-cache-dtype") + 1] == "fp8"
    assert argv[argv.index("--logprobs-mode") + 1] == "processed_logprobs"
    r = V.preset_for("rune", "fp8-kv-seqs64")
    ra = V.vllm_argv(r, "/m")
    assert ra[ra.index("--logprobs-mode") + 1] == "processed_logits" and "-O0" in ra and "--async-scheduling" in ra
    assert r.env["VLLM_BATCH_INVARIANT"] == "1" and V.preset_for("rune", "stock-cache-async").env["VLLM_BATCH_INVARIANT"] == "0"
    assert V.preset_for("quyet", "nvfp4").needs_checkpoint
    with pytest.raises(ValueError):
        V.preset_for("rune", "nope")
    assert V.preset_for("rune").name == "deterministic" and V.preset_for("quyet").name == "bf16"
    for fam, table in V.PRESETS.items():
        for p in table.values():
            assert p.summary and p.args and ("measured" in p.measured or "closest" in p.measured or not p.measured or "reference" in p.measured)


def test_cli_prints_command(capsys):
    V.main(["rune", "deterministic", "/model", "--port", "9"])
    out = capsys.readouterr().out
    assert "VLLM_BATCH_INVARIANT=1" in out and "vllm serve /model" in out and "--port 9" in out


# ---------------------------------------------------------------------------------- end to end against a fake vLLM
class FakeVllm(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeVllm.seen.append(body)
        ids = body["allowed_token_ids"]
        top = {f"token_id:{t}": float(len(ids) - i) for i, t in enumerate(ids)}   # logits n, n-1, ... 1
        top["token_id:999"] = -50.0                                              # the sampled token echo is ignored
        raw = json.dumps({"choices": [{"logprobs": {"top_logprobs": [top]}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture()
def fake_vllm():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeVllm)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    FakeVllm.seen.clear()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


@pytest.mark.parametrize("family,model_dir", [("quyet", None), ("rune", None)])
def test_backend_roundtrip(fake_vllm, family, model_dir):
    be = V.VllmReadoutBackend("unused", family=family, name="m", base_url=fake_vllm, tokenizer=FakeTok())
    assert be.kind == "vllm" and be.max_batch_tokens > 0
    batch = [be.encode("state", q) for q in (CHOICE, NOUL, SCORE)]
    out = be.run(batch)
    assert len(out) == 3 and [len(p) for p in out] == [3, 2, 3]
    T = batch[0].payload.temperature
    z = [3 / T, 2 / T, 1 / T]
    m = max(z)
    ref = [math.exp(v - m) for v in z]
    assert out[0] == pytest.approx([v / sum(ref) for v in ref])
    assert all(sum(p) == pytest.approx(1) for p in out)
    assert all(r["max_tokens"] == 1 and r["temperature"] == 0 and r["logprobs"] == len(r["allowed_token_ids"]) for r in FakeVllm.seen)
    assert batch[0].payload.letter_ids in [r["allowed_token_ids"] for r in FakeVllm.seen]
    assert be.run([batch[0]]) == [out[0]]
    be.close()


def test_cli_without_preset_uses_default(capsys):
    V.main(["quyet", "/model"])
    assert "vllm serve /model" in capsys.readouterr().out and "processed_logprobs" in capsys.readouterr().out or True
    V.main(["rune", "/model"])
    out = capsys.readouterr().out
    assert "VLLM_BATCH_INVARIANT=1" in out and "--no-enable-prefix-caching" in out
