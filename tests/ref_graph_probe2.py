"""Layer-3 (first attention) sub-module outputs: HF eager vs HF CUDA-graph replay, via forward hooks."""
import sys

import torch
from jevk5.runtime import JevK5

ref = JevK5(sys.argv[1], graphs=False)
m = ref.model.model
full = list(m.layers)
m.layers = torch.nn.ModuleList(full[:4])
m.config.num_hidden_layers = 4
n = 256
x = torch.randint(1000, 100000, (1, n), device="cuda")
L = full[3]
names = {"in_norm": L.input_layernorm, "q_proj": L.self_attn.q_proj, "k_proj": L.self_attn.k_proj, "v_proj": L.self_attn.v_proj,
         "q_norm": L.self_attn.q_norm, "k_norm": L.self_attn.k_norm, "o_proj": L.self_attn.o_proj, "self_attn": L.self_attn,
         "post_norm": L.post_attention_layernorm, "mlp": L.mlp}
store = {}
cur = {}


def mk(name):
    def f(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        cur[name] = o.clone()
    return f


hs = [mod.register_forward_hook(mk(k)) for k, mod in names.items()]
with torch.inference_mode():
    m(input_ids=x, use_cache=False)
    eager = dict(cur)
    sid = torch.zeros((1, n), dtype=torch.long, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            m(input_ids=sid, use_cache=False)
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        m(input_ids=sid, use_cache=False)
    gr = dict(cur)
    sid.copy_(x)
    g.replay()
    torch.cuda.synchronize()
    for k in names:
        d = (gr[k].float() - eager[k].float()).abs()
        print(f"{k:10s} max|d| {d.max().item():.5f} equal frac {(d == 0).float().mean().item():.4f}")
