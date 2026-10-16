"""Per-op comparison of layer 0 (DeltaNet) and layer 3 (attention) of the native engine against HF modules fed with the
same inputs. Debug tool; GPU. usage: op_diff.py MODEL_DIR ITEMS.jsonl [idx]"""
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import transformers.models.qwen3_5.modeling_qwen3_5 as M  # noqa: E402
from jevk5.runtime import JevK5  # noqa: E402
from plumb_engine import PlumbBackend  # noqa: E402
import pplx_engine as P  # noqa: E402

model_dir, items = sys.argv[1], sys.argv[2]
idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
row = [json.loads(line) for line in open(items)][idx]
q = row["question"]
q.setdefault("instructions", "")
ref = JevK5(model_dir, graphs=False)
be = PlumbBackend(model_dir, compile="0")
eng = be.engine
enc = be.encode(row["state"], q)
ids = torch.tensor([enc.payload.tolist()], device="cuda")
T = ids.shape[1]

rec = {}
orig = M.torch_chunk_gated_delta_rule


def wrapped(*a, **kw):
    out = orig(*a, **kw)
    rec.setdefault("chunk", []).append((a, kw, out))
    return out


M.torch_chunk_gated_delta_rule = wrapped
lay = ref.model.model.layers
hooks = {}


def hook(name):
    def f(mod, inp, out):
        rec.setdefault(name, []).append((inp, out))
    return f


for i in (0, 3):
    L = lay[i]
    L.input_layernorm.register_forward_hook(hook(f"in_norm{i}"))
    L.post_attention_layernorm.register_forward_hook(hook(f"post_norm{i}"))
    L.mlp.register_forward_hook(hook(f"mlp{i}"))
    if i == 0:
        la = L.linear_attn
        for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
            getattr(la, n).register_forward_hook(hook(n))
        la.norm.register_forward_hook(hook("gnorm"))
        la.register_forward_hook(hook("linattn"))
    else:
        sa = L.self_attn
        sa.register_forward_hook(hook("selfattn"))
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(sa, n).register_forward_hook(hook(n))
with torch.inference_mode():
    ref.model.model(input_ids=ids, use_cache=False)


def cmp(name, a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    d = (a - b).abs()
    print(f"{name:28s} max|d| {d.max().item():9.5f}  rel {((a - b).norm() / b.norm()).item():.6f}  exact-equal frac {(d == 0).float().mean().item():.4f}")


with torch.inference_mode():
    h0 = F.embedding(ids[0], eng.embed)
    L0 = eng.layers[0]
    hx = rec["in_norm0"][0][0][0][0]  # HF input to layer 0 norm [T, H]
    x = rec["in_norm0"][0][1][0]
    cmp("embed", h0, hx)
    cmp("in_norm0", P.rms_norm(hx, L0["in_norm"], eng.eps), x)
    qkvz = L0["in_qkvz"](x)
    cmp("in_proj_qkv", qkvz[:, :8192], rec["in_proj_qkv"][0][1][0])
    cmp("in_proj_z", qkvz[:, 8192:], rec["in_proj_z"][0][1][0].reshape(T, -1))
    ba = L0["in_ba"](x)
    cmp("in_proj_b", ba[:, :32], rec["in_proj_b"][0][1][0])
    cmp("in_proj_a", ba[:, 32:], rec["in_proj_a"][0][1][0])
    (cq, ckw, cout) = rec["chunk"][0]
    qq, kk, vv = cq[0], cq[1], cq[2]
    g, beta = ckw["g"], ckw["beta"]
    o_hf = cout[0]
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    o_fla, _ = chunk_gated_delta_rule(qq, kk, vv, g=g, beta=beta, initial_state=None, output_final_state=False,
                                      use_qk_l2norm_in_kernel=True)
    cmp("delta rule (fla vs hf-torch)", o_fla, o_hf)
    # gated norm
    z = rec["in_proj_z"][0][1][0].reshape(-1, 128)
    gn_in = rec["gnorm"][0][0][0]
    cmp("gated norm", P.rms_norm_gated(gn_in, z, L0["gnorm"], eng.eps), rec["gnorm"][0][1])
    lin_out = rec["linattn"][0][1]
    cmp("out_proj", L0["out"](rec["out_proj"][0][0][0].reshape(T, -1)), rec["out_proj"][0][1][0])
    # mlp
    m_in = rec["mlp0"][0][0][0][0]
    gu = L0["gu"](m_in)
    mid = P.silu_mul(gu, eng.inter)
    cmp("mlp (fused gu)", L0["down"](mid), rec["mlp0"][0][1][0])
