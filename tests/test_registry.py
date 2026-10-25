import json

import pytest

from engine import registry
from engine.backends.base import Backend, Encoded
from engine.backends.common import s1_answer, softmax_t


def test_all_models_present_and_consistent():
    expected = {"plumb-4b", "pplx-decider-v1.1-27b", "d3", "d3-turbo", "reax-1", "quyet-1.0-large", "jeffan-31b-1710", "rune-26b-a4b", "blink-v0.3-26b-a4b", "gutsy-0.8b", "plumb-4b-q8",
                "s1-llm-auto-router"}
    assert set(registry.REGISTRY) == expected
    for s in registry.REGISTRY.values():
        assert s.status in registry.STATUSES and s.backend_kind in registry.BACKEND_KINDS
        assert s.licence and s.licence_notice and s.hf_repo and s.arch and s.readout and s.hardware
        assert len(s.revision) == 40 and set(s.revision) <= set("0123456789abcdef"), s.id
        json.dumps(s.record())
        assert (s.factory is None) == (s.status == "documented-only")


def test_status_by_model():
    st = {i: s.status for i, s in registry.REGISTRY.items()}
    assert st["pplx-decider-v1.1-27b"] == st["d3"] == st["d3-turbo"] == st["reax-1"] == st["plumb-4b"] == "native"
    assert st["quyet-1.0-large"] == st["rune-26b-a4b"] == "vllm-backed"
    assert st["gutsy-0.8b"] == st["plumb-4b-q8"] == "cpu"
    assert st["s1-llm-auto-router"] == "classifier"
    assert registry.get("s1-llm-auto-router").model_kind == "classifier_heads"


def test_aliases_and_errors():
    assert registry.get("s1-fast").id == "plumb-4b"
    assert registry.get("rune").id == "rune-26b-a4b"
    assert registry.get("quyet").id == "quyet-1.0-large"
    with pytest.raises(KeyError):
        registry.get("nope")
    with pytest.raises(FileNotFoundError):          # the classifier_heads factory loads (and verifies) the weights
        registry.build("s1-llm-auto-router", "/nowhere")
    ids = [a for s in registry.REGISTRY.values() for a in (s.id, *s.aliases)]
    assert len(ids) == len(set(ids)), "ids and aliases must be unique"


def test_licence_facts():
    assert "Quyet by Chinh Nguyen" in registry.get("quyet").licence_notice
    assert registry.get("rune").gated and not registry.get("d3").gated
    assert registry.get("s1-llm-auto-router").licence == "MIT"


def test_table_lists_every_model():
    t = registry.table()
    for i in registry.REGISTRY:
        assert i in t
    registry.main([])
    registry.main(["--json"])


def test_s1_answer_shapes():
    c = s1_answer("choice", ["a", "b", "c"], [1, 3, 1])
    assert c["choice"] == "b" and c["confidence"] == pytest.approx(0.6) and sum(c["probabilities"].values()) == pytest.approx(1)
    n = s1_answer("noul", ["false", "true"], [0.25, 0.75])
    assert n["noul"] == 0.75 and n["confidence"] == 0.75
    n2 = s1_answer("noul", ["true", "false"], [0.9, 0.1])     # quyet order: letter A = true
    assert n2["noul"] == 0.9
    s = s1_answer("score", ["0", "1", "2"], [0.0, 0.5, 0.5])
    assert s["score"] == pytest.approx(1.5)
    with pytest.raises(ValueError):
        s1_answer("choice", ["a", "b"], [1.0])
    with pytest.raises(ValueError):
        s1_answer("choice", ["a", "b"], [float("nan"), 1.0])
    with pytest.raises(ValueError):
        s1_answer("rank", ["a"], [1.0])


def test_softmax_t():
    p = softmax_t([0.0, 0.0], 2.0)
    assert p == [0.5, 0.5]
    assert softmax_t([1000.0, 0.0], 1.0)[0] == pytest.approx(1.0)
    assert softmax_t([2.0, 0.0], 2.0)[0] == pytest.approx(softmax_t([1.0, 0.0], 1.0)[0])


def test_backend_protocol_shape():
    e = Encoded(payload=[1, 2], n_tokens=2, n_options=2)
    assert (e.n_tokens, e.n_options) == (2, 2)
    assert {"name", "kind", "max_input_tokens", "max_batch_tokens", "encode", "run", "options"} <= set(Backend.__annotations__) | set(dir(Backend))
