"""CPU test of the v1.2 request path of engine/server.py with a fake GPU engine (no GPU, no weights).

The fake engine's logits at each readout row hash the 64 tokens before the row, so a wrong marker position, chunk-to-key mapping
or sequence offset in a packed step changes the answer. Expected answers come from the author's own code
(pplx_decider.protocol: Encoder.plan + answer).

Needs a v1.2 checkpoint directory with the small files only (joint_config.json, decision_config.json, tokenizer/processor files,
chat_template.jinja, source/src/pplx_decider/), given as $PPLX_V12_MODEL_DIR (27B or Flash); skipped when torch, transformers or
that directory are missing. Python 3.12, torch (cpu), transformers 5.x.
"""
import hashlib, json, math, os, sys, threading, types, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
import torch  # noqa: E402

MODEL = Path(os.environ.get("PPLX_V12_MODEL_DIR", "/nonexistent"))
if not (MODEL / "joint_config.json").exists() or not (MODEL / "source" / "src" / "pplx_decider").exists():
    pytest.skip("set PPLX_V12_MODEL_DIR to a v1.2 checkpoint dir with source/src/pplx_decider", allow_module_level=True)

ENGINE_DIR = Path(__file__).resolve().parent.parent / "engine"
CALLS = []
TEMP = json.loads((MODEL / "decision_config.json").read_text())["temperature"]


def fake_row(seq_ids, p):
    h = hashlib.sha256(bytes(json.dumps([int(t) for t in seq_ids[max(0, p - 64):p + 1]]), "ascii")).digest()
    g = torch.Generator().manual_seed(int.from_bytes(h[:8], "little"))
    return torch.randn(255, generator=g) * 3


class Ctx:
    pass


@pytest.fixture(scope="module")
def ctx():
    sys.path.insert(0, str(ENGINE_DIR))
    sys.path.insert(0, str(MODEL / "source" / "src"))
    import pplx_engine, server
    from pplx_decider.protocol import Encoder, answer as author_answer
    from transformers import AutoProcessor

    class FakeEngine:
        def __init__(self, *a, **k):
            self.temperature = TEMP
            self.shared_until, self.n_layers, self.shared_mixq = 16, 64, False

        def forward_marked(self, ids, lens, reads):
            CALLS.append((list(lens), [list(r) for r in reads]))
            out, off = [], 0
            for n, r in zip(lens, reads):
                seq = ids[off:off + n].tolist()
                assert r == sorted(r) and all(0 <= p < n for p in r)
                out += [fake_row(seq, p) for p in r]
                off += n
            assert off == ids.numel()
            return torch.stack(out)

        probs = pplx_engine.PplxEngine.probs

    real_engine, real_torch = pplx_engine.PplxEngine, server.torch
    pplx_engine.PplxEngine = FakeEngine
    server.torch = types.SimpleNamespace(tensor=lambda d, device=None, **k: torch.tensor(d, **k))
    c = Ctx()
    c.server, c.author_answer = server, author_answer
    c.args = lambda **kw: types.SimpleNamespace(**{**dict(
        model=str(MODEL), linear="bf16", compile="1", fp8_skip="", gva="1", nvfp4="", nvfp4_scales="", budget=16384, max_wait_ms=0.0,
        shared_state="0", shared_min=64, shared_until=None, shared_mixq=None, v12_split="joint", max_length=8192,
        v12_overlength="reject", served_name=None, warm_steps=3), **kw})
    c.enc = Encoder(AutoProcessor.from_pretrained(str(MODEL)), json.loads((MODEL / "decision_config.json").read_text())["codes"])
    c.servers = []

    def start(**kw):
        backend = server.EngineBackend(c.args(**kw))
        srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.handler(backend))
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        c.servers.append(srv)
        return backend, f"http://127.0.0.1:{srv.server_address[1]}"

    c.start = start
    c.backend, c.base = start()
    c.url = c.base + "/v1/systemone"
    yield c
    for s in c.servers:
        s.shutdown()
    pplx_engine.PplxEngine, server.torch = real_engine, real_torch


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def expected(c, request, max_length=8192):
    out, tokens = {}, 0
    chunks = c.enc.plan(request, max_length)
    for ch in chunks:
        seq = ch.input_ids.tolist()
        for j, key in enumerate(ch.keys):
            lg = fake_row(seq, int(ch.markers[j]))[: ch.counts[j]].float()
            out[key] = c.author_answer(request["questions"][key], (lg / TEMP).softmax(-1).tolist())
        tokens += ch.length
    return out, tokens, len(chunks)


def close(a, b):
    if isinstance(a, dict):
        return set(a) - {"probabilities"} <= set(b) and all(close(a[k], b[k]) for k in a)
    if isinstance(a, float):
        return math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)
    return a == b


def words(n, seed):
    return " ".join(f"w{(seed * 7919 + i * 104729) % 9973}" for i in range(n))


def make_request(n_q, state_words, opt_words, seed=0):
    qs = {}
    for i in range(n_q):
        kind = ("choice", "score", "noul")[i % 3]
        if kind == "choice":
            qs[f"q{seed}_{i}"] = {"type": "choice", "instructions": f"pick {i} " + words(5, seed + i),
                                  "criteria": {f"opt{j}": words(opt_words, seed * 100 + i * 10 + j) for j in range(2 + i % 5)}}
        elif kind == "score":
            qs[f"q{seed}_{i}"] = {"type": "score", "criteria": [words(opt_words, seed * 100 + i * 10 + j) for j in range(3 + i % 4)]}
        else:
            qs[f"q{seed}_{i}"] = {"type": "noul", "instructions": words(opt_words, seed + i)}
    return {"state": words(state_words, seed), "questions": qs}


def matches(c, r, body):
    exp = expected(c, r)[0]
    return all(close(exp[k], body["answers"][k]) for k in exp)


def test_joint_request_matches_author(ctx):
    r = make_request(5, 300, 20, seed=1)
    code, body = post(ctx.url, r)
    exp, tok, nseq = expected(ctx, r)
    assert code == 200 and nseq == 1
    assert matches(ctx, r, body)
    assert list(body["answers"]) == list(r["questions"])
    assert body["usage"] == {"input_tokens": tok, "output_tokens": 0, "sequences": 1}


def test_overlength_rejected_by_default_with_code(ctx):
    r = make_request(12, 500, 60, seed=2)
    assert expected(ctx, r)[2] > 1
    code, body = post(ctx.url, r)
    assert code == 422 and "answers" not in body
    assert "maximum context length exceeded" in body["error"] and "one sequence" in body["error"]
    assert body["code"] == "context_length" and body["fallback"] == "author"


def test_overlength_split_mode_reports_sequences(ctx):
    _, base = ctx.start(v12_overlength="split")
    r = make_request(12, 500, 60, seed=2)
    code, body = post(base + "/v1/systemone", r)
    exp, tok, nseq = expected(ctx, r)
    assert code == 200 and nseq > 1
    assert matches(ctx, r, body) and list(body["answers"]) == list(r["questions"])
    assert body["usage"] == {"input_tokens": tok, "output_tokens": 0, "sequences": nseq}


def _single(n_words):
    return {"state": words(n_words, 5), "questions": {"only": {"type": "noul"}}}


def test_8192_boundary(ctx):
    lo, hi = 1000, 9000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if ctx.enc.encode(_single(mid)).length <= 8192 else (lo, mid - 1)
    req, pad = _single(lo), 0
    while ctx.enc.encode(req).length < 8192:
        pad += 1
        req = {"state": words(lo, 5) + " ." * pad, "questions": {"only": {"type": "noul"}}}
    assert ctx.enc.encode(req).length == 8192
    CALLS.clear()
    code, body = post(ctx.url, req)
    assert code == 200 and body["usage"]["sequences"] == 1 and CALLS[-1][0] == [8192]
    req2 = {"state": req["state"] + " .", "questions": req["questions"]}
    assert ctx.enc.encode(req2).length == 8193
    code, body = post(ctx.url, req2)
    assert code == 422 and "maximum context length" in body["error"] and body["code"] == "context_length"


def test_one_overlong_question_rejects_whole_request(ctx):
    r = make_request(4, 300, 20, seed=3)
    r["questions"]["huge"] = {"type": "choice", "criteria": {"a": words(9000, 9), "b": "x"}}
    for srv_url in (ctx.url, ctx.start(v12_overlength="split")[1] + "/v1/systemone"):
        code, body = post(srv_url, r)
        assert code == 422 and "answers" not in body and body["code"] == "context_length"


def test_images_422_and_bad_requests_400(ctx):
    code, body = post(ctx.url, {"state": "s", "questions": {"a": {"type": "noul"}}, "images": ["data:image/png;base64,AAAA"]})
    assert code == 422 and "images" in body["error"] and body["code"] == "unsupported_images" and body["fallback"] == "author"
    assert post(ctx.url, {"state": "s", "questions": {"a": {"type": "bogus"}}})[0] == 400
    assert post(ctx.url, {"state": "s", "questions": {"a": {"type": "choice", "criteria": {f"o{i}": "x" for i in range(256)}}}})[0] == 400
    assert post(ctx.url, {"state": "s", "questions": {}})[0] == 400
    code, body = post(ctx.url, {"questions": {"a": {"type": "noul"}}})  # missing state, like the author's RequestError
    assert code == 400 and "state" in body["error"]


def test_served_name_and_health(ctx):
    code, body = post(ctx.url, make_request(2, 50, 5, seed=4))
    assert code == 200 and body["model"] == ctx.backend.served
    with urllib.request.urlopen(ctx.base + "/health") as r:
        h = json.loads(r.read())
    assert h["status"] == "ready" and h["linear"] == "bf16" and h["served_name"] == h["model"] == ctx.backend.served
    assert h["v12_overlength"] == "reject" and h["max_length"] == 8192
    b, base = ctx.start(served_name="pplx-decider-flash-v1.2", linear="fp8")
    code, body = post(base + "/v1/systemone", make_request(2, 50, 5, seed=4))
    with urllib.request.urlopen(base + "/health") as r:
        h = json.loads(r.read())
    assert body["model"] == h["model"] == "pplx-decider-flash-v1.2" and h["linear"] == "fp8"


def test_served_name_default_from_manifest(ctx, tmp_path):
    (tmp_path / "release_manifest.json").write_text(json.dumps({"hf_id": "perplexity-ai/pplx-decider-flash-v1.2"}))
    assert ctx.server.served_name(tmp_path) == "pplx-decider-flash-v1.2"
    assert ctx.server.served_name(tmp_path, "x") == "x"
    assert ctx.server.served_name(MODEL) in (MODEL.resolve().name, "pplx-decider-flash-v1.2", "pplx-decider-v1.2")


def test_concurrent_requests_packed_steps(ctx):
    _, base = ctx.start(v12_overlength="split")
    reqs = [make_request(1 + i % 7, 50 + 37 * i, 10 + i % 9, seed=100 + i) for i in range(20)]
    reqs += [make_request(10, 500, 60, seed=200 + i) for i in range(4)]
    CALLS.clear()
    with ThreadPoolExecutor(12) as ex:
        outs = list(ex.map(lambda r: post(base + "/v1/systemone", r), reqs))
    for r, (code, body) in zip(reqs, outs):
        _, tok, nseq = expected(ctx, r)
        assert code == 200 and matches(ctx, r, body)
        assert body["usage"]["input_tokens"] == tok and body["usage"]["sequences"] == nseq
    assert max(sum(l) for l, _ in CALLS) <= 16384


def test_v12_split_single(ctx):
    _, base = ctx.start(v12_split="single")
    r = make_request(4, 200, 15, seed=7)
    code, body = post(base + "/v1/systemone", r)
    assert code == 200
    for k, q in r["questions"].items():
        e = expected(ctx, {"state": r["state"], "questions": {k: q}})[0]
        assert close(e[k], body["answers"][k])


def test_shared_state_refused_and_protocol_asserted(ctx, tmp_path):
    with pytest.raises(SystemExit):
        ctx.server.EngineBackend(ctx.args(shared_state="1", warm_steps=0))
    for f in MODEL.iterdir():  # a checkpoint whose joint_config differs from the author's PROTOCOL is refused
        if f.name != "source":
            (tmp_path / f.name).symlink_to(f)
    (tmp_path / "source").symlink_to(MODEL / "source")
    (tmp_path / "joint_config.json").unlink()
    bad = json.loads((MODEL / "joint_config.json").read_text())
    bad["marker"] = "<|other|>"
    (tmp_path / "joint_config.json").write_text(json.dumps(bad))
    with pytest.raises(AssertionError, match="PROTOCOL"):
        ctx.server.EngineBackend(ctx.args(model=str(tmp_path), warm_steps=0))
