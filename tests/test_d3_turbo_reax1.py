"""d3-turbo state-once routing and the REAX 1 joint backend of engine/backends/qwen35.py with the engine, tokenizer and
author Encoder stubbed out (no GPU, no weights), plus the registry entries and the REAX_PROFILE=reference options."""
import sys
import types

import pytest

from engine import d3_contract as D
from engine import registry, serve
from engine.backends.base import Encoded
from engine.backends import qwen35
from tests.fake_tok import FakeTok

CH = {"type": "choice", "instructions": "i", "criteria": {"a": "x", "b": None}}


class OffsetTok(FakeTok):
    """FakeTok with offset mapping (one token per character), like a fast HF tokenizer's return_offsets_mapping."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        out = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


def turbo(shared=True, group=True, shared_min=4):
    be = object.__new__(qwen35.Qwen35Backend)
    be.family, be.name, be.max_input_tokens, be.max_batch_tokens = "d3", "d3-turbo", 10000, 1000
    be._d3, be._tok, be._codes = D, OffsetTok(), [chr(65 + i) for i in range(26)]
    be.prefix_bidir, be.shared, be.shared_group, be.shared_min = True, shared, group, shared_min
    return be


# ------------------------------------------------------------------------------------------------ encode_split
def test_encode_split_state_block():
    be = turbo()
    e = be.encode("STATE", CH)
    text = D.render(be._tok, "STATE", D.normalize_question(CH).rendered, be._codes)
    assert e.payload == [ord(c) for c in text] and e.n_tokens == len(text)
    body_end = text.index("State:\nSTATE") + len("State:\nSTATE")
    assert e.aux == body_end and text[e.aux:].startswith("\n\nQuestion:\n")      # state block = everything up to the end of the state text


def test_encode_split_token_straddling_the_boundary_belongs_to_the_question_block():
    class Straddle(OffsetTok):
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            out = super().__call__(text, add_special_tokens, return_offsets_mapping)
            end = text.index("\n\nQuestion:\n")
            out["offset_mapping"][end - 1] = (end - 1, end + 2)      # the last state token swallows "\n\n"
            return out

    be = turbo()
    be._tok = Straddle()
    text = D.render(be._tok, "STATE", D.normalize_question(CH).rendered, be._codes)
    assert be.encode("STATE", CH).aux == text.index("\n\nQuestion:\n") - 1


def test_encode_split_rejects_missing_segment():
    with pytest.raises(ValueError, match="no state segment"):
        D.encode_split(OffsetTok(), "no marks here", "s")
    one_token = lambda text, add_special_tokens=False, return_offsets_mapping=False: {"input_ids": [1], "offset_mapping": [(0, len(text))]}  # noqa: E731
    with pytest.raises(ValueError, match="empty state or question block"):      # a single token spanning state and question
        D.encode_split(one_token, "State:\ns\n\nQuestion:\nx", "s")


def test_plain_d3_encode_has_no_state_block():
    be = turbo()
    be.prefix_bidir = False
    assert be.encode("s", CH).aux is None


# ------------------------------------------------------------------------------------------------ routing
def item(prefix, suffix, sb=None, n=2):
    ids = list(prefix) + list(suffix)
    return Encoded(ids, len(ids), n, len(prefix) if sb is None else sb)


STATE_A = list(range(100, 110))   # 10 tokens
STATE_B = list(range(200, 210))


def test_single_decision_is_a_group_of_state_block_plus_one_suffix():
    be = turbo(group=False)
    batch = [item(STATE_A, [1, 2, 3]), item(STATE_A, [4, 5, 6, 7])]
    plain, groups = be.plan_shared(batch)
    assert plain == []
    assert groups == [(STATE_A, [([1, 2, 3], 0)]), (STATE_A, [([4, 5, 6, 7], 1)])]      # without shared_group: never merged


def test_siblings_with_identical_state_block_merge_into_one_group():
    be = turbo()
    batch = [item(STATE_A, [1, 2, 3]), item(STATE_B, [9, 9, 9]), item(STATE_A, [4, 5, 6, 7]), item(STATE_A, [8, 8])]
    plain, groups = be.plan_shared(batch)
    assert plain == []
    merged = [g for g in groups if len(g[1]) > 1]
    assert merged == [(STATE_A, [([1, 2, 3], 0), ([4, 5, 6, 7], 2), ([8, 8], 3)])]
    assert (STATE_B, [([9, 9, 9], 1)]) in groups                                         # a different state stays alone
    assert sorted(i for _, ss in groups for _, i in ss) == [0, 1, 2, 3]


def test_same_tokens_but_different_state_block_length_do_not_merge():
    be = turbo()
    ids = STATE_A + [1, 2, 3, 4]
    batch = [Encoded(list(ids), len(ids), 2, 10), Encoded(list(ids), len(ids), 2, 8)]
    _, groups = be.plan_shared(batch)
    assert [len(ss) for _, ss in groups] == [1, 1] and sorted(len(p) for p, _ in groups) == [8, 10]


def test_common_prefix_longer_than_state_block_still_splits_at_the_state_block():
    be = turbo()
    batch = [item(STATE_A, [1, 2, 3, 4]), item(STATE_A, [1, 2, 3, 5])]                  # questions share a head too
    _, groups = be.plan_shared(batch)
    assert groups == [(STATE_A, [([1, 2, 3, 4], 0), ([1, 2, 3, 5], 1)])]


def test_state_shorter_than_min_or_conv_window_uses_the_plain_path_or_single_group():
    be = turbo(shared_min=4)
    tiny = Encoded([1, 2, 3, 4, 5, 6], 6, 2, 2)                                          # state block of 2 tokens < conv window 3
    ok3 = Encoded([1, 2, 3, 4, 5, 6], 6, 2, 3)
    plain, groups = be.plan_shared([tiny, ok3])
    assert plain == [0] and groups == [([1, 2, 3], [([4, 5, 6], 1)])]


def test_answer_does_not_depend_on_co_arrival_structure():
    """The same decision is planned as prefix=state block with ITS OWN suffix whether it arrives alone or with siblings."""
    be = turbo()
    alone = be.plan_shared([item(STATE_A, [4, 5, 6, 7])])[1]
    together = be.plan_shared([item(STATE_A, [1, 2, 3]), item(STATE_A, [4, 5, 6, 7])])[1]
    assert alone == [(STATE_A, [([4, 5, 6, 7], 0)])]
    assert together == [(STATE_A, [([1, 2, 3], 0), ([4, 5, 6, 7], 1)])]


# ------------------------------------------------------------------------------------------------ run (fake engine)
try:
    import torch
except ImportError:  # CPU CI without torch: the routing tests above still run
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed")


class FakeEngine:
    prefix_bidir, shared_until, n_layers, shared_mixq = True, 64, 64, False

    def __init__(self):
        self.calls = []

    def forward(self, ids, lens, states=None):
        self.calls.append(("forward", list(lens), list(states) if states is not None else None))
        return torch.tensor([[float(n)] for n in lens])

    def forward_groups(self, groups, until=None):
        self.calls.append(("groups", [(len(p), [len(s) for s in ss]) for p, ss in groups]))
        return torch.tensor([[float(len(p) + len(s))] for p, ss in groups for s in ss])

    def probs(self, logits, counts):
        return [(float(l[0]), n) for l, n in zip(logits, counts)]


@needs_torch
def test_run_shared_maps_results_back_in_batch_order():
    be = turbo()
    be.engine, be._device, be.steps = FakeEngine(), "cpu", 0
    tiny = Encoded([1, 2, 3, 4, 5, 6], 6, 7, 2)
    batch = [item(STATE_A, [1, 2, 3], n=3), tiny, item(STATE_B, [9], n=4), item(STATE_A, [4, 5], n=5)]
    out = be.run(batch)
    assert out == [(13.0, 3), (6.0, 7), (11.0, 4), (12.0, 5)]
    kinds = [c[0] for c in be.engine.calls]
    assert kinds == ["forward", "groups"] and be.engine.calls[0][2] == [2]


@needs_torch
def test_run_without_shared_passes_state_blocks_to_forward():
    be = turbo(shared=False, group=False)
    be.engine, be._device, be.steps = FakeEngine(), "cpu", 0
    be.run([item(STATE_A, [1, 2, 3]), item(STATE_B, [4])])
    assert be.engine.calls == [("forward", [13, 11], [10, 10])]


@needs_torch
def test_warmup_shared_runs_every_path_on_the_fake_engine():
    be = turbo()
    be.engine, be._device, be.steps = FakeEngine(), "cpu", 0
    be.max_batch_tokens, be.max_input_tokens = 1024, 8192
    be.shared_min = 64
    be.warmup(steps=4)
    kinds = {c[0] for c in be.engine.calls}
    assert kinds == {"groups"} and be.steps > 50                      # every synthetic item has a state block >= 3 tokens
    assert any(len(x[1]) > 1 for c in be.engine.calls for x in c[1])      # shared groups with several suffixes ran


# ------------------------------------------------------------------------------------------------ REAX 1
class FakeChunk:
    def __init__(self, ids, marker, count):
        self.input_ids, self.markers, self.counts, self.keys, self.length = torch.tensor(ids), torch.tensor([marker]), [count], ["q"], len(ids)


class LengthError(Exception):
    pass


class RequestError(Exception):
    pass


class FakeEncoder:
    def plan(self, request, max_length):
        q = request["questions"]["q"]
        if q.get("criteria") == "bad":
            raise RequestError("bad criteria")
        ids = [5] * (3 + len(str(request["state"])) )
        if len(ids) > max_length:
            raise LengthError(f"{len(ids)} > {max_length}")
        return [FakeChunk(ids, len(ids) - 2, 2)]


def reax(max_tokens=20):
    be = object.__new__(qwen35.Qwen35Backend)
    be.family, be.name, be.max_input_tokens, be.max_batch_tokens = "reax1", "reax-1", max_tokens, 8192
    be._pp = qwen35._pplx_engine_module()
    be._encoder, be._v12_errors = FakeEncoder(), (LengthError, RequestError)
    return be


@needs_torch
def test_reax1_encode_marker_options_and_errors():
    be = reax()
    e = be.encode("abc", CH)
    assert e.payload == [5] * 6 and e.n_tokens == 6 and e.n_options == 2 and e.aux == 4
    assert be.options(CH) == ["a", "b"] and be.options({"type": "noul"}) == ["false", "true"]
    with pytest.raises(ValueError, match="maximum context length"):
        be.encode("x" * 40, CH)
    with pytest.raises(ValueError, match="bad criteria"):
        be.encode("s", {"type": "choice", "criteria": "bad"})
    with pytest.raises(ValueError):
        be.encode("s", {"type": "weird"})


@needs_torch
def test_reax1_run_reads_the_marker_row_of_every_sequence():
    be = reax()
    calls = []

    class Eng:
        def forward_marked(self, ids, lens, reads):
            calls.append((ids.tolist(), lens, reads))
            return torch.tensor([[float(r[0])] for r in reads])

        def probs(self, logits, counts):
            return [(float(l[0]), n) for l, n in zip(logits, counts)]

    be.engine, be._device, be.steps = Eng(), "cpu", 0
    out = be.run([Encoded([1, 2, 3, 4], 4, 2, 2), Encoded([7, 8, 9], 3, 5, 1)])
    assert calls == [([1, 2, 3, 4, 7, 8, 9], [4, 3], [[2], [1]])] and out == [(2.0, 2), (1.0, 5)]


# ------------------------------------------------------------------------------------------------ registry / profile
def test_registry_has_the_reference_slots():
    t, r = registry.get("d3-turbo"), registry.get("reax1")
    assert (t.hf_repo, t.revision) == ("vllm-sr/d3-turbo", "cfa0d41d253467e4742d5aedb46d4f0ef3b84b41")
    assert (r.id, r.hf_repo, r.revision) == ("reax-1", "decisionmodels/reax-1", "ca2a0cc54a537c1f5cf24e5b09387cfb357b6518")
    assert registry.get("d3turbo").id == "d3-turbo" and registry.get("decisionmodels-reax-1").id == "reax-1"
    assert t.production_opts == {"budget": 16384, "max_input_tokens": 8192, "warm_steps": 200, "shared_state": True, "shared_group": True,
                                 "shared_until": 64, "shared_mixq": False}
    assert r.production_opts == {"budget": 8192, "max_input_tokens": 8192, "warm_steps": 200}
    assert registry.get("pplx").production_opts == {"budget": 4096, "warm_steps": 200}
    assert registry.get("blink").production_opts == {"preset": "ref-prod", "max_input_tokens": 8000}
    assert registry.get("d3").production_opts == {}                      # the plain d3 entry keeps its old behaviour


def test_d3_turbo_backend_refuses_sharing_on_a_noncausal_checkpoint(monkeypatch):
    fake_pp = types.SimpleNamespace(PplxEngine=lambda *a, **k: types.SimpleNamespace(prefix_bidir=False, shared_until=64, n_layers=64, shared_mixq=False))
    monkeypatch.setattr(qwen35, "_pplx_engine_module", lambda: fake_pp)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda d: OffsetTok())))
    import json, tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "decision_config.json").write_text(json.dumps({"codes": ["A", "B"]}))
    with pytest.raises(ValueError, match="prefix_bidir"):
        qwen35.Qwen35Backend(d, family="d3", shared_state=True)


def test_reference_profile_applies_production_opts(monkeypatch):
    seen = {}

    def fake_build(model_id, model_dir, **opts):
        seen["opts"] = opts
        return object()

    monkeypatch.setattr(registry, "build", fake_build)
    a = types.SimpleNamespace(opt=["budget=2048"], model_dir="/m", no_verify=True, batch_tokens=0, compile="1")
    monkeypatch.delenv("REAX_PROFILE", raising=False)
    serve.load_backend("d3-turbo", a)
    assert seen["opts"] == {"budget": 2048}
    monkeypatch.setenv("REAX_PROFILE", "reference")
    serve.load_backend("d3-turbo", a)
    assert seen["opts"]["shared_state"] is True and seen["opts"]["budget"] == 2048 and seen["opts"]["shared_until"] == 64
    monkeypatch.setenv("REAX_PROFILE", "nope")
    with pytest.raises(SystemExit):
        serve.load_backend("d3-turbo", a)
