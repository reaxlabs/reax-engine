#!/usr/bin/env python3
"""Probe GEMM throughput on this GPU for the model's projection shapes: bf16 vs FP8 rowwise vs NVFP4 (if available)."""
import json, sys, time

import torch
import torch.nn.functional as F

M = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
shapes = {"dn_qkvz": (16384, 5120), "dn_out": (5120, 6144), "attn_qkv": (14336, 5120), "attn_o": (5120, 6144),
          "mlp_gu": (34816, 5120), "mlp_down": (5120, 17408)}
dev = "cuda"
res = {"gpu": torch.cuda.get_device_name(), "cap": torch.cuda.get_device_capability(), "torch": torch.__version__, "M": M}


def timeit(fn, n=20):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


tot = {"bf16": 0.0, "fp8": 0.0, "fp8_q": 0.0, "nvfp4": 0.0}
for name, (N, K) in shapes.items():
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) / K ** 0.5
    flop = 2 * M * N * K
    t_bf = timeit(lambda: F.linear(x, w))
    r = {"bf16_tflops": flop / t_bf / 1e12}
    tot["bf16"] += t_bf
    try:
        ws = w.float().abs().amax(1, keepdim=True) / 448
        w8 = (w.float() / ws).to(torch.float8_e4m3fn).t()
        xs = x.float().abs().amax(1, keepdim=True) / 448
        x8 = (x.float() / xs).to(torch.float8_e4m3fn)
        wsr = ws.view(1, -1).contiguous()
        f = lambda: torch._scaled_mm(x8, w8, scale_a=xs, scale_b=wsr, out_dtype=torch.bfloat16)  # noqa: E731
        t8 = timeit(f)
        y = f()
        r["fp8_tflops"] = flop / t8 / 1e12
        r["fp8_relerr"] = ((y.float() - F.linear(x, w).float()).norm() / F.linear(x, w).float().norm()).item()
        tot["fp8"] += t8

        def fq():
            s = x.abs().amax(1, keepdim=True).float() / 448
            q = (x / s).to(torch.float8_e4m3fn)
            return torch._scaled_mm(q, w8, scale_a=s, scale_b=wsr, out_dtype=torch.bfloat16)
        tq = timeit(fq)
        r["fp8_with_quant_tflops"] = flop / tq / 1e12
        tot["fp8_q"] += tq
    except Exception as e:  # noqa: BLE001
        r["fp8_error"] = f"{type(e).__name__}: {e}"[:200]
    res[name] = r
res["total_ms"] = {k: v * 1e3 for k, v in tot.items()}
# NVFP4 probe
try:
    from torch.nn.functional import ScalingType, SwizzleType, scaled_mm  # noqa: F401
    res["nvfp4_api"] = "torch.nn.functional.scaled_mm present"
except Exception as e:  # noqa: BLE001
    res["nvfp4_api"] = f"absent: {type(e).__name__}"
print(json.dumps(res, indent=1))
