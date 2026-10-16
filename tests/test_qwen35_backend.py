"""Qwen35Backend encode/options/answer with the engine and tokenizer stubbed out (no GPU, no weights)."""
import pytest

pytest.importorskip("torch")
from engine import d3_contract  # noqa: E402
from engine.backends import qwen35  # noqa: E402
from tests.fake_tok import FakeTok  # noqa: E402

CH = {"type": "choice", "instructions": "i", "criteria": {"a": "x", "b": None}}


def bare(family, max_tokens):
    be = object.__new__(qwen35.Qwen35Backend)
    be.family, be.name, be.max_input_tokens, be.max_batch_tokens = family, family, max_tokens, 1000
    be._pp = qwen35._pplx_engine_module()
    be._d3, be._tok, be._codes = d3_contract, FakeTok(), [chr(65 + i) for i in range(26)]
    be._prompter = type("P", (), {"ids": staticmethod(lambda state, q: ([7] * 5, len(be._pp.options(q)[0])))})()
    return be


def test_d3_encode_options_answer():
    be = bare("d3", 10000)
    e = be.encode("state", CH)
    assert e.n_options == 2 and e.n_tokens == len(e.payload) and be.options(CH) == ["a", "b"]
    assert be.options({"type": "noul"}) == ["false", "true"]
    a = be.answer(CH, [0.75, 0.25])
    assert a["choice"] == "a" and a["probabilities"] == {"a": 0.75, "b": 0.25} and 0 < a["confidence"] < 1
    with pytest.raises(ValueError, match="maximum context length"):
        bare("d3", 20).encode("state" * 20, CH)
    with pytest.raises(ValueError):
        be.encode("s", {"type": "choice", "criteria": {}})


def test_pplx_encode_options_answer():
    be = bare("pplx", 8192)
    e = be.encode("state", CH)
    assert e.payload == [7] * 5 and e.n_options == 2 and be.options(CH) == ["a", "b"]
    n = be.answer({"type": "noul"}, [0.4, 0.6])
    assert n["noul"] == pytest.approx(0.6) and n["probabilities"]["false"] == pytest.approx(0.4)
    with pytest.raises(ValueError):
        be.encode("s", {"type": "weird"})
    with pytest.raises(ValueError, match="maximum context length"):
        bare("pplx", 3).encode("s", CH)


def test_family_validation():
    with pytest.raises(ValueError):
        qwen35.Qwen35Backend("/nowhere", family="gemma")
