# Results (17 Oct 2026): shared-state mode, 6.0x at equal quality

Model: perplexity-ai/pplx-decider-v1.1-27b @ 6195aa55. GPU: 1x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, 600 W), RunPod,
128 cores. Reference = the author's `DecisionModel` (bf16, SDPA) served serially, questions in chunks of 8 (`engine/server.py --backend ref`).
Harness: the official owo runners (`run_di.py --workers 8 --limit 1000` = the DI-1000 sample, 1,000 requests / 5,129 decisions;
`run_p300.py --workers 16`), paired bootstrap 95 % CIs vs the reference run on the same pod. Every run starts with fresh Triton /
Inductor caches; reference and candidates were interleaved (5 reference runs: 5.242-5.281 dec/s, answers bit-identical across runs).
Ratio = worst case: slowest candidate run / fastest reference run. Full DI-5k (5,000 requests / 25,509 decisions) quality on a second pod.

| | dec/s DI-1000 (3 runs) | ratio | DI-1000 delta [CI] | DI-5k delta [CI] | P300 capability delta [CI] | top-1 flips vs ref (multi-q / single-q, DI-5k) |
|---|---|---|---|---|---|---|
| reference (author, bf16) | 5.24-5.28 | 1.0x | - | - | - | - |
| REAX FP8 (published) | 16.21 | 3.07x | | | | |
| **REAX FP8 + shared state** | **31.73-31.98** | **6.0x** | +0.28 / +0.45 / +0.45 | **-0.04 [-0.54, +0.41]** | +0.32 [-1.27, +1.68] | 2.1 % / 2.3 % |
| REAX NVFP4 mid (published) | 22.49-22.93 | 4.3x | -1.19 [-3.01, +0.56] | -0.70 [-1.45, 0.00] | -0.26 [-1.68, +1.49] | 1.5 % / 7.2 % |
| REAX NVFP4 mid + shared state | 42.94-43.62 | 8.1x | +0.57 / +0.14 / -0.93 | -0.50 [-1.25, +0.20] | -0.26 [-1.68, +1.49] | 2.5 % / 7.0 % |
On the DI-5k traffic mix (78 % single-question requests, where sharing does not apply) the ratios are 5.6x (FP8 + shared, 29.5 vs
5.23 dec/s) and 7.8x (NVFP4 + shared). Single-request latency (P300, c1): 50 ms vs 112-115 ms. Throughput is the same at 4, 8 and 16
clients; answers are bit-identical across repeats at concurrency 1 (60 DI requests x 3); run-to-run top-1 differences under load 0.3-0.5 %.

**Shared-state mode is an approximation** (`--shared-state 1 --shared-until 64 --shared-mixq 1`, see docs/models/pplx-shared-state.md).
The questions of one request share the encoded state: the state is encoded once per request instead of once per question. In the
model's noncausal attention layers the state tokens then attend to the state and all questions of the request instead of the
state and their own question, so an answer can depend on the other questions of its request. It roughly doubles the top-1
differences on multi-question decisions (FP8 0.5 % -> 2.1 %), with no measurable score change (DI-5k CI -0.54..+0.41). Single-question
requests are unaffected (same path as without the mode). The speed-up depends on the traffic: it comes from requests with several
questions about one state.

What did not hold up: NVFP4 on every layer ("a4", +9 % throughput) fails the DI-5k gate (-0.91 [-1.72, -0.14]); sharing only the first
16 layers is lossless-er but only ~1.1x; streams and larger batches give nothing. A cold server first stalled for tens of seconds on step
sizes it had not seen (fla conv kernels specialise per 1024-token bucket): the warm-up now sweeps every bucket and oversized
requests are split. Raw runs: results/pplx-decider-v1.1-27b/2026-10-17-shared-state/.

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
