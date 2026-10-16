import pytest

from engine import d3_contract as D
from tests.fake_tok import FakeTok

CODES = [chr(65 + i) for i in range(26)]


def test_normalize_question_kinds():
    c = D.normalize_question({"type": "choice", "instructions": "i", "criteria": {"a": "x", "b": None}})
    assert c.keys == ["a", "b"] and c.rendered["type"] == "choice"
    n = D.normalize_question({"type": "noul", "instructions": "i"})
    assert n.keys == ["false", "true"] and n.descriptions == ["No / false", "Yes / true"]
    s = D.normalize_question({"type": "score", "instructions": "i", "criteria": ["lo", {"v": 1}, "hi"]})
    assert s.keys == ["0", "1", "2"] and list(s.rendered["criteria"]) == ["lo", '{"v": 1}', "hi"]
    for bad in ({"type": "choice", "criteria": {}}, {"type": "noul", "criteria": {"maybe": "x"}}, {"type": "score", "criteria": ["a"]},
                {"type": "score", "criteria": ["a", "a"]}, {"type": "rank"}, "x", {"type": "choice", "criteria": {"a": float("nan")}}):
        with pytest.raises(ValueError):
            D.normalize_question(bad)


def test_prompt_bytes():
    q = D.normalize_question({"type": "choice", "instructions": "Pick", "criteria": {"a": "first", "b": None}})
    text = D.render(FakeTok(), {"k": 1}, q.rendered, CODES)
    assert text == ("<system>" + D.SYSTEM_PROMPT + '</><user>State:\n{"k": 1}\n\nQuestion:\nPick\n\nOptions:\nA: a: first\nB: b\n\n'
                    "Reply with only the code of the best option.</><model>")
    assert "State:\n(empty)" in D.render(FakeTok(), "", q.rendered, CODES)
    assert "Choose the best matching option." in D.render(FakeTok(), "s", D.normalize_question({"type": "noul"}).rendered, CODES)


def test_product_answers():
    c = D.normalize_question({"type": "choice", "instructions": "i", "criteria": {"a": "x", "b": "y"}})
    a = D.product_answer(c, [3.0, 1.0])
    assert a["choice"] == "a" and a["probabilities"] == {"a": 0.75, "b": 0.25} and 0 < a["confidence"] < 1
    n = D.product_answer(D.normalize_question({"type": "noul"}), [0.2, 0.8])
    assert n == {"type": "noul", "noul": 0.8}
    s = D.product_answer(D.normalize_question({"type": "score", "criteria": ["lo", "hi"]}), [0.25, 0.75])
    assert s["score"] == 0.75 and s["legend"] == {"0": "lo", "1": "hi"}
    uniform = D.product_answer(c, [1.0, 1.0])
    assert uniform["confidence"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        D.product_answer(c, [1.0])
    with pytest.raises(ValueError):
        D.product_answer(c, [0.0, 0.0])
