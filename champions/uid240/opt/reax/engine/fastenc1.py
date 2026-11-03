"""kai lever I, front end (KAI_ENC1, default off; acts only with MAM_FASTENC=1, mammoth family): rgh's FASTENC encode of
one question (backends/qwen35._encode_request_fast: TWO chat-template renders and TWO tokenisations, the full text and the
state-block prefix) done with ONE render by string concatenation and ONE tokenisation.

Between two steps of a closed-loop load every answered client sends its next request, and the step closes only when all
are encoded: the encode is on the GPU's idle path. Most of it holds the GIL (the jinja render), so the encode threads
serialise on it.

* Render: mammoth's chat template emits, for [system, user] + generation prompt with thinking off,
  HEAD + (user content)|trim + TAIL, where HEAD / TAIL are fixed (they hold the system prompt). They are taken once from
  the template itself (one render with a marker as the content); a start-up probe and the first `check` calls compare the
  concatenation with the jinja render.
* Tokenise: the same Rust call FASTENC makes for the full text (tokenizer._tokenizer.encode_batch, add_special_tokens
  False, GIL released) gives the ids and the char offsets. P = the number of tokens that end at or before the suffix start
  k = len(HEAD) + len(state block); used only when a token boundary falls exactly at k (else FASTENC serves the call).
  At k the text is the fixed "</state>\\n\\n" + "Question:" / "Rate the state:": the pre-token ">\\n\\n" ends there in the
  full text and in the prefix text alike, so the prefix tokenises to exactly those P ids (FASTENC's common-prefix rule
  then also gives P).
* Self-check: the first `check` calls (default 256) also run FASTENC; one difference turns the lever off for good (and
  that call is served by FASTENC). Same ids, same P, same errors otherwise.
"""
from __future__ import annotations

import os
import sys
import threading

from . import qev_contract

_MARK = "KAI\x1fENC1\x1fMARK"


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    v = e.get("KAI_ENC1", "").strip()
    if v not in ("", "0", "1"):
        raise SystemExit(f"KAI_ENC1 must be 0 or 1 or unset, not {v!r}")
    return v == "1"


def _template_parts(tok):
    msgs = [{"role": "system", "content": qev_contract.SYSTEM_PROMPT}, {"role": "user", "content": _MARK}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if text.count(_MARK) != 1:
        return None
    i = text.index(_MARK)
    return text[:i], text[i + len(_MARK):]


class Enc1:
    def __init__(self, tok, fast, check: int = 256):
        """tok: the HF tokenizer; fast: FASTENC's encode (tok, state, normalized question) -> ([ids], P)"""
        self.tok, self.fast, self.check = tok, fast, int(check)
        self.rt = getattr(tok, "_tokenizer", None)
        self.seen = 0
        self.on = False
        self.why = ""
        self._lock = threading.Lock()
        parts = _template_parts(tok) if self.rt is not None else None
        if parts is None:
            self.why = "no Rust tokenizer or the template does not echo the content once"
            return
        self.head, self.tail = parts
        for st, suf in (({"k": 1}, "Question: x"), ("  pad  ", "Rate the state: y \n"), (None, "Question:\n\nA. a")):
            if self.render(st, suf) != qev_contract.render(tok, st, suf):
                self.why = "start-up probe: concatenation differs from the template render"
                return
        self.on = True

    def render(self, state, suffix: str) -> str:
        """== qev_contract.render(tok, state, suffix) (jinja: HEAD + content|trim + TAIL)"""
        return self.head + (qev_contract.state_block(state) + suffix).strip() + self.tail

    def _one(self, state, q):
        sb = qev_contract.state_block(state)
        text = self.head + (sb + q.suffix).strip() + self.tail
        k = len(self.head) + len(sb)
        if not sb or sb[0].isspace():
            return None
        enc = self.rt.encode_batch([text], add_special_tokens=False)[0]
        ids, offs = enc.ids, enc.offsets
        if len(offs) != len(ids):
            return None
        n = 0
        for _, end in offs:
            if end > k:
                break
            n += 1
        if n <= 0 or n >= len(ids) or offs[n - 1][1] != k or offs[n][0] < k:
            return None
        return list(ids), n

    def encode(self, state, q):
        """([full ids], P) as FASTENC gives it; FASTENC itself when the lever is off, cannot prove P, or is self-checking"""
        if not self.on:
            return self.fast(self.tok, state, q)
        got = encode1(self, state, q)
        if got is None:
            return self.fast(self.tok, state, q)
        if self.seen >= self.check:
            return [got[0]], got[1]
        with self._lock:
            self.seen += 1
        full, p = self.fast(self.tok, state, q)
        if (full[0], p) != got:
            self.on = False
            self.why = f"differs from FASTENC (P {got[1]} vs {p}, {len(got[0])} vs {len(full[0])} ids)"
            print(f"KAI_ENC1: {self.why}; turned off", file=sys.stderr, flush=True)
        return full, p


def encode1(e: Enc1, state, q):
    """the single render + tokenisation (module function: kprofile stamps it); None = cannot prove P"""
    return e._one(state, q)
