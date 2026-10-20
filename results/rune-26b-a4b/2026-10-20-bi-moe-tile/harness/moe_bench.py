"""Batch-invariant fused-MoE microbench at the shapes of a Gemma-4 MoE checkpoint (reads config.json).
For each candidate fixed tile: time at several M, and a needle test (rows of a big batch == the same rows run alone, bitwise)."""
import json, os, sys, time, itertools
os.environ["VLLM_BATCH_INVARIANT"] = "1"
import torch
from vllm.model_executor.layers.fused_moe import fused_moe as fm
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
cfg = json.load(open(sys.argv[1]))
tc = cfg.get("text_config", cfg)
H = tc["hidden_size"]; E = tc.get("num_experts") or tc.get("num_local_experts"); I = tc.get("moe_intermediate_size") or tc.get("expert_intermediate_size")
TOPK = tc.get("num_experts_per_tok") or tc.get("top_k_experts")
print("shapes H", H, "E", E, "I", I, "topk", TOPK, flush=True)
torch.manual_seed(0); dev = "cuda"
w1 = (torch.randn(E, 2 * I, H, device=dev, dtype=torch.bfloat16) * 0.02)
w2 = (torch.randn(E, H, I, device=dev, dtype=torch.bfloat16) * 0.02)
def inputs(M, seed=1):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
    logits = torch.randn(M, E, device=dev, generator=g)
    w, ids = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)
    return x, (w / w.sum(-1, keepdim=True)).float(), ids.to(torch.int32)
def run(x, w, ids):
    return fm.fused_experts(x, w1, w2, w, ids, activation=MoEActivation.GELU_TANH)
CANDS = {
    "default_64_64_32": None,
    "128_128_64_w4_s3": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
    "128_128_64_w8_s3": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "SPLIT_K": 1, "num_warps": 8, "num_stages": 3},
    "64_128_64_w4_s3": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
    "64_128_64_w4_s4": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "SPLIT_K": 1, "num_warps": 4, "num_stages": 4},
    "128_64_64_w4_s3": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
    "128_256_64_w8_s3": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "SPLIT_K": 1, "num_warps": 8, "num_stages": 3},
    "64_64_64_w4_s3": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3},
}
Ms = [int(m) for m in os.environ.get("MS", "64,512,2048,8192,32768").split(",")]
res = {}
for name, c in CANDS.items():
    if c is None: os.environ.pop("REAX_BI_MOE_CONFIG", None)
    else: os.environ["REAX_BI_MOE_CONFIG"] = json.dumps(c)
    row = {}
    try:
        for M in Ms:
            x, w, ids = inputs(M)
            for _ in range(3): run(x, w, ids)
            torch.cuda.synchronize(); n = max(3, int(20 * 8192 / max(M, 1024)))
            t = time.perf_counter()
            for _ in range(n): run(x, w, ids)
            torch.cuda.synchronize(); row[M] = (time.perf_counter() - t) / n * 1e3
        # needle: rows 5..21 of M=8192 computed in a batch of 17
        x, w, ids = inputs(8192)
        big = run(x, w, ids)[5:22]
        small = run(x[5:22].contiguous(), w[5:22].contiguous(), ids[5:22].contiguous())
        row["invariant"] = bool(torch.equal(big, small))
        x1 = run(x[7:8].contiguous(), w[7:8].contiguous(), ids[7:8].contiguous())
        row["invariant_m1"] = bool(torch.equal(big[2:3], x1))
    except Exception as e:  # noqa
        row["error"] = repr(e)[:300]
    res[name] = row
    print(name, json.dumps(row), flush=True)
json.dump(res, open(os.environ.get("OUT", "moe_bench.json"), "w"), indent=1)
