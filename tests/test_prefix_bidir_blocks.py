"""prefix_bidir (vllm-sr/d3-turbo) as varlen segments (PplxEngine._blocks + _attend) equals the dense block mask: the state
block attends to itself only, the question block reads the whole row. CPU, no weights."""
import types

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
from engine.backends import qwen35  # noqa: E402

pp = qwen35._pplx_engine_module()


def run(lens, states, nh=8, nkv=2, hd=16, seed=0):
    torch.manual_seed(seed)
    offs = [0]
    for n in lens:
        offs.append(offs[-1] + n)
    T = offs[-1]
    q, k, v = torch.randn(T, nh, hd), torch.randn(T, nkv, hd), torch.randn(T, nkv, hd)
    eng = types.SimpleNamespace(_varlen=None, attn_impl="sdpa", nh=nh, nkv=nkv, hd=hd)
    kidx, cu_q, cu_k, mq, mk, oq, ok = pp.PplxEngine._blocks(offs, lens, states, "cpu")
    o = pp.PplxEngine._attend(eng, q, k.index_select(0, kidx), v.index_select(0, kidx), cu_q, cu_k, mq, mk, oq, ok)
    ref = []
    for a, n, s in zip(offs[:-1], lens, states):
        m = torch.ones(n, n, dtype=torch.bool)
        m[:s, s:] = False  # state queries do not see question keys
        qi = q[a:a + n].transpose(0, 1)[None]
        ki = k[a:a + n].repeat_interleave(nh // nkv, 1).transpose(0, 1)[None]
        vi = v[a:a + n].repeat_interleave(nh // nkv, 1).transpose(0, 1)[None]
        ref.append(F.scaled_dot_product_attention(qi, ki, vi, attn_mask=m, scale=hd ** -0.5)[0].transpose(0, 1))
    return o, torch.cat(ref)


def test_blocks_equal_dense_block_mask():
    o, ref = run([9, 5, 13], [4, 2, 12])
    assert (o - ref).abs().max() < 1e-5


def test_blocks_single_row_and_edges():
    for lens, states in (([7], [1]), ([7], [6]), ([30, 3, 17, 40], [29, 1, 8, 20])):
        o, ref = run(lens, states, seed=len(lens))
        assert (o - ref).abs().max() < 1e-5


def test_blocks_metadata():
    kidx, cu_q, cu_k, mq, mk, oq, ok = pp.PplxEngine._blocks([0, 9, 14], [9, 5], [4, 2], "cpu")
    assert cu_q.tolist() == [0, 4, 9, 11, 14] and cu_k.tolist() == [0, 4, 13, 15, 20]
    assert kidx.tolist() == list(range(4)) + list(range(9)) + [9, 10] + list(range(9, 14))
    assert mq == 5 and mk == 9


def test_noncausal_modes_are_the_same_code_path():
    """d3 / pplx (noncausal_full_attention) never build blocks: forward() without states keeps the old call."""
    import inspect
    src = inspect.getsource(pp.PplxEngine._forward_gen)
    assert "if self.prefix_bidir:" in src and "blocks = None" in src
    src = inspect.getsource(pp.PplxEngine._attention)
    assert "if blocks is None:" in src and "causal=self.causal" in src


def test_prefix_bidir_flag_in_decision_config():
    import inspect
    src = inspect.getsource(pp.PplxEngine._read_weights)
    assert '("noncausal_full_attention", "prefix_bidir")' in src and 'self.prefix_bidir = dc.get("attention_mode") == "prefix_bidir"' in src
