import sys
import torch
import torch.nn.functional as F
from jevk5.runtime import JevK5

ref = JevK5(sys.argv[1], graphs=False)
m = ref.model.model
full = list(m.layers)
m.layers = torch.nn.ModuleList(full[:4])
m.config.num_hidden_layers = 4
n = 256
orig = F.scaled_dot_product_attention
log = []


def spy(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    log.append((None if attn_mask is None else (attn_mask.dtype, tuple(attn_mask.shape), attn_mask.stride(), attn_mask), k.stride(), tuple(q.shape), q.stride(), tuple(k.shape), attn_mask is None, is_causal, scale, enable_gqa, torch.cuda.is_current_stream_capturing()))
    return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)


F.scaled_dot_product_attention = spy
torch.nn.functional.scaled_dot_product_attention = spy
x = torch.randint(1000, 100000, (1, n), device="cuda")
with torch.inference_mode():
    m(input_ids=x, use_cache=False)
    print("eager:", log[-1]); log.clear()
    sid = torch.zeros((1, n), dtype=torch.long, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            m(input_ids=sid, use_cache=False)
    torch.cuda.current_stream().wait_stream(stream)
    log.clear()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        m(input_ids=sid, use_cache=False)
    e = log[-1]
    mk = e[0][3]
    g.replay(); torch.cuda.synchronize()
    print("capture:", e[0][:3], e[1:])
    mm = mk[0, 0]
    print("mask is lower-triangular all-true:", bool(torch.equal(mm, torch.ones_like(mm).tril())), mm.dtype, "sum", int(mm.sum()))
