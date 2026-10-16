"""pplx answer shaping / question helpers (needs torch only to import pplx_engine; no weights, no GPU)."""
import pytest

pytest.importorskip("torch")
from engine.backends import qwen35  # noqa: E402


@pytest.fixture(scope="module")
def pp():
    return qwen35._pplx_engine_module()


def test_options_and_answers(pp):
    ch = {"type": "choice", "instructions": "i", "criteria": {"a": "x", "b": "y"}}
    assert pp.options(ch)[0] == ["a", "b"]
    a = pp.answer(ch, [3.0, 1.0])
    assert a["choice"] == "a" and a["confidence"] == pytest.approx(0.5)
    n = pp.answer({"type": "noul"}, [0.3, 0.7])
    assert n == {"type": "noul", "noul": pytest.approx(0.7)}
    s = pp.answer({"type": "score", "criteria": ["l", "m", "h"]}, [0.0, 0.5, 0.5])
    assert s["score"] == pytest.approx(1.5) and s["legend"] == {"0": "l", "1": "m", "2": "h"}


def test_validate_pplx_question():
    ok = {"type": "choice", "criteria": {"a": "x"}}
    qwen35.validate_pplx_question(ok)
    for bad in ({"type": "rank"}, {"type": "choice", "criteria": ["a"]}, {"type": "score", "criteria": ["a"]}, "x"):
        with pytest.raises(ValueError):
            qwen35.validate_pplx_question(bad)


def test_decision_messages_match_author_prompt(pp):
    q = {"type": "choice", "instructions": "Which?", "criteria": {"a": "first", "b": None}}
    m = pp.decision_messages("S", q, ["A", "B"])
    assert m[0]["content"] == pp.SYSTEM
    assert m[1]["content"][0]["text"] == "State:\nS\n\nQuestion:\nWhich?\n\nOptions:\nA: a: first\nB: b\n\nReturn only the letter code of the best option."
