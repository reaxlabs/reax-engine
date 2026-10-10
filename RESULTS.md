# Results (10 Oct 2026)

Model: perplexity-ai/pplx-decider-v1.1-27b @ 6195aa55. GPU: 1x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB).
Reference = the model's own code (`autojev.model.DecisionModel`, transformers + SDPA, bf16) served one request at a time,
exactly like the model's reference server. Both servers were measured on the same machine with the same harness:
the JevBench public set (300 items) and a fixed 1,000-row Decision Index sample (5,129 decisions), scored with the
public scorers, paired bootstrap 95 % CIs for the score differences.

| | reference server | REAX Engine (FP8) | ratio |
|---|---|---|---|
| Decision-Index-style traffic (multi-question requests, 8 clients) | 5.00 dec/s | 15.70 dec/s | 3.1x |
| JevBench public set (single-question requests, 16 clients) | 7.83 items/s | 20.62 items/s | 2.6x |
| single-request latency, p50 | 121 ms | 84 ms | 1.4x lower |
| Decision Index sample score | 58.02 | 58.23 (+0.21, CI -0.67..+1.32) | same within noise |
| JevBench public capability | 84.70 | 84.61 (-0.09, CI -1.46..+1.35) | same within noise |

Repeatability: at concurrency 1 the engine is bit-reproducible (4 x 199 repeats, zero drift). Under concurrent load,
probabilities vary slightly with co-batched requests (mean |dp| ~1e-3), as with any batched server.

## How we got there (one host, offline packed throughput on a fixed 669-decision validation mix)
| step | decisions/s |
|---|---|
| reference code, one decision at a time | 4.8 |
| own engine, bf16 (packed varlen batches, fused projections, torch.compile fusions) | 8.9 |
| + FP8 W8A8 GEMMs | 13.7 |
| + activation quantisation fused into norm/activation kernels | 14.0 |
| + native grouped-value heads in the DeltaNet kernels | 14.4 |
| + autoresearch round (strided conv inputs, fused Triton gated-norm+quant, tensor-scaled gate/up GEMM) | 15.5 |

## What did not help (measured)
- A C++ host engine: at concurrency 1 host overhead is ~1/3 of wall time, but capturing the whole forward in a CUDA graph
  (the upper bound of what a C++ host loop can remove) only cut mean single-request latency 61 -> 53 ms and gave no
  throughput gain; all heavy kernels are CUTLASS/Triton either way.
- Mature engines on the same architecture (causal attention, i.e. less work and not usable for this model): vLLM 0.31
  1.02-1.085x, SGLang 0.98x, TensorRT-LLM 0.81x of our throughput.
- Naive (uncalibrated, dynamic-scale) NVFP4 on the MLPs: only +8 %; calibrated static scales are what makes 4-bit pay off (below).
- A short "healing" LoRA distillation from the bf16 model on top of the 4-bit variant: no measurable score or agreement gain,
  and the unmerged adapter cost 22 % throughput.

## Experimental: calibrated NVFP4 (4-bit) variant
Static activation scales calibrated on 512 training-split decisions (not benchmark items; file
engine/scales/nvfp4-act-amax-max512.json); NVFP4 for the MLP and DeltaNet projections of layers 4-59, FP8 elsewhere.
Same harness and pairing as above (16-core host):
| | decisions/s (DI sample) | Decision Index sample | JevBench public capability | answers identical to reference |
|---|---|---|---|---|
| reference server (bf16) | 5.5 | 58.02 | 84.70 | - |
| REAX Engine FP8 | 15.5 | 57.98 | 84.39 | 99.1 % / 99.7 % |
| REAX Engine NVFP4 mid-layers | 21.8 | 58.30 (CI of delta -1.13..+2.22) | 84.17 (CI -1.91..+1.20) | ~97.5 % / ~98.5 % |
Scores stay within measurement noise; about 2-3 % of individual answers differ from the bf16 model. Run with
`--nvfp4 gu@4-59,down@4-59,in_qkvz@4-59,out@4-59 --nvfp4-scales engine/scales/nvfp4-act-amax-max512.json` (Blackwell GPUs only).

## Other GPUs (FP8, same engine)
B200 33.3, H200 22.5, RTX PRO 6000 15.5, RTX 5090 (32 GB, lazy weight loading) 10.3 decisions/s.
