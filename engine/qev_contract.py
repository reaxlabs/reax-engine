"""qev prompt / answer contract for baselabs mammoth-1 and elephant-1 (families "mammoth", "elephant").

Ported from OneJev qev @81ce62f + the checkpoints' serve.patch (qev/prompt.py, qev/answers.py, qev/engine.py
_prefix_and_suffix_ids / decide; Apache License 2.0, (c) the OneJev authors). Same rendered text, token ids, state/question
split and answer formulas as the authors' DecisionEngine; nothing here imports author code.

Checkpoint contract (config.json modeling_overrides + readout.json):
  attention  full_attention = block_bidirectional: block 0 = system turn + <state> block, block 1 = the question suffix;
             tokens attend bidirectionally within their block, the suffix also reads block 0; the prefix never reads the
             suffix. GatedDeltaNet layers stay causal. This is exactly the engine's prefix_bidir mode with
             state = the shared-prefix token count.
  readout    slot_lm: softmax over the LM-head logits (fp32 head) of the slot-label tokens at the last position,
             temperature 1.0 unless calibration.json exists (none in the pinned revisions).
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PROMPT_VERSION = "qev-labels-v2"
SYSTEM_PROMPT = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_NOT_ONE_TOKEN = {"BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ", "GJ", "GK", "GQ", "GZ",
                  "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ"}
SLOT_LABELS = tuple(list(LETTERS) + [a + b for a in LETTERS for b in LETTERS if a + b not in _NOT_ONE_TOKEN][: 256 - 26])
MAX_OPTIONS = 255  # engine readout width; the author schema allows 255 choice options, 10 score levels
MAX_SCORE_LEVELS = 10
NOUL_DEFAULT_INSTRUCTIONS = "Is the statement true, or is the answer to the question yes?"
NOUL_DEFAULT_TRUE = "the statement is true / the answer is yes"
NOUL_DEFAULT_FALSE = "the statement is false / the answer is no"
QUESTION_HEADER, SCORE_HEADER = "Question:", "Rate the state:"
ANSWER_LINE, ANSWER_LINE_MANY = "Answer with one letter: {letters}.", "Answer with one label: {letters}."
PREFIX_MARK = "QEV_PREFIX_MARK"


def render_entry(entry: Any) -> str:
    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry.strip()
    return json.dumps(entry, ensure_ascii=False, indent=2)


def render_state(state: Any) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def state_block(state: Any) -> str:
    return f"<state>\n{render_state(state)}\n</state>\n\n"


def slot_ids(tokenizer) -> list[int]:
    """Token ids of the slot labels (qev DecisionEngine._verify_slots): A..Z required, two-letter labels all-or-nothing."""
    def one(label):
        enc = tokenizer.encode(label, add_special_tokens=False)
        return enc[0] if len(enc) == 1 and tokenizer.decode(enc) == label else None

    ids = [one(l) for l in LETTERS]
    if any(i is None for i in ids):
        raise ValueError("answer slots A..Z are not single round-trip tokens")
    extra = [one(l) for l in SLOT_LABELS[26:]]
    if all(i is not None for i in extra):
        ids += extra
    if len(set(ids)) != len(ids):
        raise ValueError("answer slot tokens collide")
    return ids


@dataclass
class Question:
    kind: str  # noul | choice | score
    keys: list[str]  # answer keys in slot order (noul: ["yes", "no"])
    descriptions: list[Any]
    suffix: str


def _json_value(value: Any) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not JSON data") from exc


def normalize_question(q: Any, max_options: int = MAX_OPTIONS) -> Question:
    """Validate one System One question (qev schema rules) and render its suffix (qev render_question, tagged layout)."""
    if not isinstance(q, Mapping):
        raise ValueError("a question is an object with a type")
    kind, crit = q.get("type"), q.get("criteria")
    if q.get("instructions") is not None:
        _json_value(q.get("instructions"))
    if kind == "noul":
        if crit is not None and (not isinstance(crit, Mapping) or set(crit) - {"true", "false"}):
            raise ValueError('noul criteria may only describe "false" and "true"')
        instructions = render_entry(q.get("instructions")) or NOUL_DEFAULT_INSTRUCTIONS
        t = render_entry(crit["true"]) if crit and crit.get("true") is not None else NOUL_DEFAULT_TRUE
        f = render_entry(crit["false"]) if crit and crit.get("false") is not None else NOUL_DEFAULT_FALSE
        keys, names, descs = ["yes", "no"], ["yes", "no"], [t, f]
    elif kind == "choice":
        if not isinstance(crit, Mapping) or not 1 <= len(crit) <= max_options:
            raise ValueError(f"choice criteria must map 1 to {max_options} option keys to descriptions")
        if any(not isinstance(k, str) or not k for k in crit):
            raise ValueError("choice option keys must be nonempty strings")
        _json_value(dict(crit))
        instructions = render_entry(q.get("instructions")) or "Which option applies to the state?"
        keys = list(crit)
        names, descs = list(keys), [crit[k] for k in keys]
    elif kind == "score":
        if not isinstance(crit, (list, tuple)) or not 1 <= len(crit) <= MAX_SCORE_LEVELS:
            raise ValueError(f"score criteria must be an ordered list of 1 to {MAX_SCORE_LEVELS} levels")
        _json_value(list(crit))
        instructions = render_entry(q.get("instructions")) or "Which level describes the state?"
        keys = [str(i) for i in range(len(crit))]
        names, descs = [f"level {i}" for i in range(len(crit))], list(crit)
    else:
        raise ValueError(f"unsupported question type {kind!r} (choice, noul or score)")
    letters = list(SLOT_LABELS[: len(keys)])
    lines = []
    for letter, name, desc in zip(letters, names, descs):
        text = render_entry(desc)
        lines.append(f"{letter}. {name}: {text}" if text else f"{letter}. {name}")
    head = f"{QUESTION_HEADER if kind != 'score' else SCORE_HEADER} {instructions}\n\nOptions:\n"
    suffix = head + "\n".join(lines) + "\n\n" + (ANSWER_LINE if len(letters) <= 26 else ANSWER_LINE_MANY).format(
        letters=", ".join(letters))
    return Question(kind, keys, descs, suffix)


def render(tokenizer, state: Any, suffix: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": state_block(state) + suffix}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode_request(tokenizer, state: Any, questions: list[Question]) -> tuple[list[list[int]], int]:
    """Full token ids of every branch and the shared-prefix length P (qev _prefix_and_suffix_ids): the longest common
    token prefix of all branches that ends at or before the end of the state block. P is the block-0 length."""
    full = [tokenizer.encode(render(tokenizer, state, q.suffix), add_special_tokens=False) for q in questions]
    text = render(tokenizer, state, PREFIX_MARK)
    prefix = tokenizer.encode(text[: text.index(PREFIX_MARK)], add_special_tokens=False)
    n = len(prefix)
    while n > 0 and not all(ids[:n] == prefix[:n] for ids in full):
        n -= 1
    if n <= 0 or any(len(ids) <= n for ids in full):
        raise ValueError("a branch has an empty suffix; state block boundary mismatch")
    return full, n


def _normalize(p):
    s = float(sum(p))
    return [1.0 / len(p)] * len(p) if s <= 0 else [float(x) / s for x in p]


def choice_confidence(p):
    if len(p) == 1:
        return 1.0
    p = _normalize(p)
    u = 1.0 / len(p)
    return (max(p) - u) / (1.0 - u)


def score_confidence(p):
    if len(p) == 1:
        return 1.0
    p = _normalize(p)
    k = len(p)
    mode = max(range(k), key=p.__getitem__)
    dist = sum(pi * abs(i - mode) for i, pi in enumerate(p))
    c = (k - 1) / 2
    return max(0.0, 1.0 - dist / (sum(abs(i - c) for i in range(k)) / k))


def _canonical(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def product_answer(q: Question, probs) -> dict[str, Any]:
    """System One answer (s1contract field sets) with the authors' formulas (qev answers.py), unrounded."""
    p = _normalize(probs)
    if q.kind == "noul":
        y = min(1.0, max(0.0, p[0]))
        return {"type": "noul", "noul": y, "probabilities": {"false": 1.0 - y, "true": y}}
    if q.kind == "choice":
        best = max(range(len(p)), key=p.__getitem__)
        return {"type": "choice", "choice": q.keys[best], "probabilities": dict(zip(q.keys, p)),
                "confidence": choice_confidence(p)}
    return {"type": "score", "score": sum(i * x for i, x in enumerate(p)), "probabilities": dict(zip(q.keys, p)),
            "confidence": score_confidence(p), "legend": {k: _canonical(d) for k, d in zip(q.keys, q.descriptions)}}
