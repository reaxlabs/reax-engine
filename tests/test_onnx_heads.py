"""classifier_heads path (engine/backends/onnx_heads.py + the classifier routes of engine/serve.py) on CPU.

A tiny generated ONNX model and tokenizer stand in for the real one (needs numpy, onnxruntime, tokenizers, onnx; skipped
when one is missing). The real s1-llm-auto-router is tested against the reference implementation when its weights are
present: set REAX_ROUTER_DIR to the model directory.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")
onnx = pytest.importorskip("onnx")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine.backends import onnx_heads  # noqa: E402
from engine.backends.onnx_heads import S1_LLM_AUTO_ROUTER, OnnxHeadsBackend  # noqa: E402

OUT_DIMS = (9, 5, 4, 4)
VOCAB = 40
MAX_LEN = 16
TEMPS = {"category": 0.5, "difficulty": 0.8, "stakes": 0.75, "flags": 0.6}


def build_tiny_model(root: Path):
    from onnx import TensorProto as T, helper as h, numpy_helper as nh
    from tokenizers import Tokenizer, models, pre_tokenizers

    rng = np.random.default_rng(0)
    D = 8
    inits = [nh.from_array(rng.normal(size=(VOCAB, D)).astype(np.float32), "emb")]
    nodes = [
        h.make_node("Gather", ["emb", "input_ids"], ["e"]),                      # [B,L,D]
        h.make_node("Cast", ["attention_mask"], ["m"], to=T.FLOAT),               # [B,L]
        h.make_node("Unsqueeze", ["m", "ax"], ["m3"]),                            # [B,L,1]
        h.make_node("Mul", ["e", "m3"], ["em"]),
        h.make_node("ReduceSum", ["em", "ax1"], ["s"], keepdims=0),               # [B,D]
        h.make_node("ReduceSum", ["m", "ax1"], ["n"], keepdims=1),                # [B,1]
        h.make_node("Div", ["s", "n"], ["pooled"]),
    ]
    inits += [nh.from_array(np.array([2], dtype=np.int64), "ax"), nh.from_array(np.array([1], dtype=np.int64), "ax1")]
    outs = []
    for i, d in enumerate(OUT_DIMS):
        inits.append(nh.from_array(rng.normal(size=(D, d)).astype(np.float32), f"w{i}"))
        nodes.append(h.make_node("MatMul", ["pooled", f"w{i}"], [f"o{i}"]))
        outs.append(h.make_tensor_value_info(f"o{i}", T.FLOAT, ["B", d]))
    g = h.make_graph(nodes, "tiny", [h.make_tensor_value_info("input_ids", T.INT64, ["B", "L"]),
                                     h.make_tensor_value_info("attention_mask", T.INT64, ["B", "L"])], outs, inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(root / "model.gq.onnx"))
    vocab = {"[UNK]": 0, **{f"w{i}": i for i in range(1, VOCAB)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    (root / "tokenizer").mkdir()
    tok.save(str(root / "tokenizer" / "tokenizer.json"))
    (root / "meta.json").write_text(json.dumps({"max_len": MAX_LEN, "temperatures": TEMPS, "name": "s1-llm-auto-router"}))


@pytest.fixture(scope="module")
def be(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    build_tiny_model(root)
    b = OnnxHeadsBackend(root, threads=1, verify=False)
    b.root = root
    return b


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def test_probabilities_match_manual_reference(be):
    probs, n = be.classify("w3 w4 w5", "w1 w2")
    ids = be.tok.encode("w1 w2", "w3 w4 w5").ids
    assert n == len(ids)
    arr = np.asarray([ids], dtype=np.int64)
    cat, dif, stk, flg = (o[0] for o in be.sess.run(None, {"input_ids": arr, "attention_mask": np.ones_like(arr)}))
    assert np.array_equal(probs["category"], softmax(cat.astype(np.float64) / TEMPS["category"]))
    assert np.array_equal(probs["difficulty"], softmax(dif.astype(np.float64) / TEMPS["difficulty"]))
    assert np.array_equal(probs["stakes"], softmax(stk.astype(np.float64) / TEMPS["stakes"]))
    for i, f in enumerate(onnx_heads.ROUTER_FLAGS):
        p = 1 / (1 + np.exp(-float(flg[i]) / TEMPS["flags"]))
        assert probs[f][1] == p and probs[f][0] == 1.0 - p
    for h in S1_LLM_AUTO_ROUTER.heads:
        assert abs(probs[h.name].sum() - 1) < 1e-12


def test_truncation_and_context_cap(be, monkeypatch):
    long = " ".join(f"w{i % 30 + 1}" for i in range(200))
    assert be.tokenize(long).n_tokens == MAX_LEN
    # only the first 600 characters of the context are read
    monkeypatch.setattr(onnx_heads, "CONTEXT_CHARS", 9)   # the cap is 600 characters for the real model
    assert be.tokenize("w1", "w2 " * 200).payload == be.tokenize("w1", "w2 w2 w2 ").payload
    monkeypatch.undo()
    with pytest.raises(ValueError):   # context alone fills max_len: the request cannot be truncated into it
        be.encode({"request": "w1", "context": "w2 " * 200})


def test_input_validation(be):
    for bad in ("", "   \n", {"request": ""}, 5, None):
        with pytest.raises(ValueError):
            be.encode(bad)
    with pytest.raises(ValueError):
        be.encode("w1 " * 9000)       # more than 16 KiB of state
    assert be.encode("w1 w2").n_tokens == 2
    assert be.encode({"request": "w1", "context": {"a": 1}}).n_tokens > 1   # object context is serialised
    assert be.encode(["w1"]).n_tokens > 0                                       # non-text state is serialised


def test_padded_batch_equals_single_runs(be):
    encs = [be.tokenize(t) for t in ("w1", "w1 w2 w3 w4 w5 w6", "w7 w8 w9")]
    single = [be.run([e])[0] for e in encs]
    batched = be.run(encs)
    for s, b in zip(single, batched):
        for h in S1_LLM_AUTO_ROUTER.heads:
            np.testing.assert_allclose(s[h.name], b[h.name], rtol=0, atol=1e-6)


def test_heads_answer_shape(be):
    probs, _ = be.classify("w3 w4")
    ans = be.heads_answer(probs)
    assert list(ans) == ["category", "difficulty", "stakes", *onnx_heads.ROUTER_FLAGS]
    assert ans["category"]["label"] in onnx_heads.ROUTER_CATEGORIES
    assert set(ans["category"]["probabilities"]) == set(onnx_heads.ROUTER_CATEGORIES)
    assert ans["difficulty"]["label"] in onnx_heads.ROUTER_DIFFICULTY and 0 <= ans["difficulty"]["score"] <= 4
    assert set(ans["needs_tools"]["probabilities"]) == {"false", "true"}
    json.dumps(ans)


QUESTIONS = {
    "category": {"type": "choice", "criteria": {c: "" for c in onnx_heads.ROUTER_CATEGORIES}},
    "difficulty": {"type": "score", "criteria": list(onnx_heads.ROUTER_DIFFICULTY)},
    "stakes": {"type": "score", "criteria": list(onnx_heads.ROUTER_STAKES)},
    **{f: {"type": "noul"} for f in onnx_heads.ROUTER_FLAGS},
}


def test_fixed_question_schema(be):
    be.check_questions(QUESTIONS)
    for bad in ({"colour": {"type": "choice"}}, {"category": {"type": "noul"}},
                {"category": {"type": "choice", "criteria": {"a": ""}}},
                {"difficulty": {"type": "score", "criteria": ["a", "b"]}}):
        with pytest.raises(ValueError):
            be.check_questions(bad)
    probs, _ = be.classify("w3 w4")
    ans = be.sidecar_answers(QUESTIONS, probs)
    assert ans["category"]["confidence"] == max(ans["category"]["probabilities"].values())
    assert ans["needs_tools"]["probabilities"]["true"] == ans["needs_tools"]["noul"]
    assert ans["stakes"]["legend"] == {"0": "negligible", "1": "low", "2": "medium", "3": "high"}


# ------------------------------------------------------------------------------------------------ serve.py routes
async def call(app, method, path, body=None, headers=()):
    raw = json.dumps(body).encode() if body is not None else b""
    sent, state = [], {"done": False}

    async def receive():
        if state["done"]:
            return {"type": "http.disconnect"}
        state["done"] = True
        return {"type": "http.request", "body": raw, "more_body": False}

    async def send(msg):
        sent.append(msg)
    await app({"type": "http", "method": method, "path": path, "headers": list(headers)}, receive, send)
    return sent[0]["status"], json.loads(sent[1]["body"])


def test_serve_routes(be):
    sys.path.insert(0, str(ROOT / "engine"))
    import serve

    sched = serve.Scheduler(be)
    app = serve.App(be, sched, "tok", be.name)
    app.aliases = ("llm-auto-router",)
    sched.start(on_ready=lambda: setattr(app, "ready", True))
    H = [(b"x-s1m-internal-token", b"tok")]

    async def main():
        while not app.ready:
            await asyncio.sleep(0.01)
        st, out = await call(app, "POST", "/v1/classify", {"model": "s1-llm-auto-router", "request": "w3 w4", "context": "w1"}, H)
        assert st == 200 and out["model"] == "s1-llm-auto-router" and out["usage"] == {"input_tokens": 3, "output_tokens": 0}
        assert set(out["heads"]) == {h.name for h in S1_LLM_AUTO_ROUTER.heads}
        assert out["heads"]["category"]["label"] in out["heads"]["category"]["probabilities"]
        # same numbers through the production-compatible route
        st, prod = await call(app, "POST", "/v1/systemone", {"state": {"request": "w3 w4", "context": "w1"}, "questions": QUESTIONS}, H)
        assert st == 200 and prod["usage"]["input_tokens"] == 3
        assert prod["answers"]["category"]["choice"] == out["heads"]["category"]["label"]
        assert prod["answers"]["category"]["probabilities"] == out["heads"]["category"]["probabilities"]
        st, one = await call(app, "POST", "/internal/v1/decide", {"state": "w3 w4", "questions": {"needs_tools": QUESTIONS["needs_tools"]}}, H)
        assert st == 200 and list(one["answers"]) == ["needs_tools"]
        # errors
        assert (await call(app, "POST", "/v1/classify", {"request": "w3"}))[0] == 401
        assert (await call(app, "POST", "/v1/classify", {"request": ""}, H))[0] == 400
        assert (await call(app, "POST", "/v1/classify", {"model": "other", "request": "w3"}, H))[0] == 400
        assert (await call(app, "POST", "/v1/classify", {"model": "llm-auto-router", "request": "w3"}, H))[0] == 200
        assert (await call(app, "POST", "/v1/classify", {"context": "w3"}, H))[0] == 400
        assert (await call(app, "POST", "/v1/systemone", {"state": "w3", "questions": {"colour": {"type": "choice"}}}, H))[0] == 400
        assert (await call(app, "GET", "/v1/classify", None, H))[0] == 404
        # concurrent requests all answer
        rs = await asyncio.gather(*[call(app, "POST", "/v1/classify", {"request": f"w{i % 30 + 1} w2"}, H) for i in range(20)])
        assert all(s == 200 for s, _ in rs)
    asyncio.run(main())


def test_classify_route_absent_for_decision_models():
    sys.path.insert(0, str(ROOT / "engine"))
    import serve

    class Fake:
        name, kind, max_input_tokens, max_batch_tokens = "fake", "native-gpu", 10, 0
    app = serve.App(Fake(), serve.Scheduler(Fake()), "", "fake")
    app.ready = True
    assert asyncio.run(call(app, "POST", "/v1/classify", {"request": "x"}))[0] == 404


# ------------------------------------------------------------------------------------------------ the real model
REAL = Path(os.environ.get("REAX_ROUTER_DIR", ROOT / "weights" / "router"))


@pytest.mark.skipif(not (REAL / "model.gq.onnx").is_file(), reason="s1-llm-auto-router weights not present (REAX_ROUTER_DIR)")
def test_real_router_equals_reference():
    from bench.router_equiv import make_samples
    from bench.router_reference import Reference

    be = OnnxHeadsBackend(REAL, threads=2)       # also checks the pinned sha256 of the three files
    ref = Reference(REAL, threads=2)
    n_checked = 0
    for req, ctx in make_samples(40):
        try:
            pr, nr = ref.predict(req, ctx)
        except Exception:
            with pytest.raises(ValueError):
                be.classify(req, ctx)
            continue
        if not req.strip():
            continue                              # the engine rejects blank requests, the reference does not
        pe, ne = be.classify(req, ctx)
        assert ne == nr
        for h in pr:
            assert np.array_equal(pr[h], pe[h]), (h, req[:40])
        n_checked += 1
    assert n_checked > 25
