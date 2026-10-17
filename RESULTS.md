# Results (17 Oct 2026): pplx-decider v1.2 (joint format), FP8 2.4-2.5x the author's fast path, scores within noise

Models: perplexity-ai/pplx-decider-v1.2 (27B) @225c913d and perplexity-ai/pplx-decider-flash-v1.2 (4B) @6d771ad7. GPU: 1x RTX PRO
6000 Blackwell Server Edition (96 GB), RunPod SECURE (pod A) unless marked. **Reference = the author's v1.2 fast path**
(`Decider.from_pretrained(..., fast=True)`: packing, whole-layer `torch.compile`, CUDA graphs, last-layer pruning; bf16, its
default), behind a thin batching server (results/pplx-decider-v1.2/2026-10-17/harness/v12_server.py: a GPU thread passes every queued request to one
`decide(list)` call so the fast path can pack concurrent requests). Same harness as below (official owo runners, DI-1000 sample =
1,000 requests / 5,129 decisions, 8 clients; P300 with 16 clients; c1 latency = 48 sequential P300 requests), paired bootstrap 95 % CIs,
fresh Triton/Inductor caches every run, author and engine interleaved on the same pod. Both servers use the same author `Encoder`
(identical prompts: 1,777,408 input tokens on DI-1000 in joint mode, 3,593,463 one question per request) and the same length rule
(a single question over 8,192 tokens -> 422; 3 of 300 P300 items, none of the DI-1000 or DI-5k rows).

**The old 6x does not carry over.** It was measured against the v1.1 serial research server. v1.2 already shares the state across
the questions of a request (joint format) and ships a compiled fast path, so the headroom is smaller: **2.4-2.5x** with FP8 in
joint mode (27B, 8 clients, DI-1000), 2.1x one question per request, 1.5x at the same precision (bf16), 1.6x against the author's
own off-by-default `fp8=True`. Quality: the paired DI-1000 deltas of the FP8 runs lie within about +-1 point (CIs include 0; bf16
within +-0.5), and 99.3 % of the decisions keep the author's top answer. All numbers are steady state: the engine warms up for
about 5 minutes before serving (the author's fast path about 40 s).

| 27B, DI-1000 | dec/s | x author fast (bf16) | c1 p50 / p95 | DI delta [95 % CI] vs author | answers same as author | P300 capability delta |
|---|---|---|---|---|---|---|
| author fast path, joint (bf16, default) | 12.53-12.55 (2 runs; 32 clients 12.93) | 1.0x | 305 / 308 ms | - | - | - |
| author fast path, joint, its `fp8=True` option | 18.93-19.11 (2 runs) | 1.5x | 191 / 195 ms | +0.30 [-0.18, +1.04]; r2 +0.96 [+0.14, +2.41] (vs its bf16) | 99.4 % | |
| REAX bf16, joint | 18.97 | 1.51x | 82 / 142 ms | -0.06 [-0.47, +0.26] | 99.7 % | +0.01 [-0.28, +0.82] |
| **REAX FP8, joint** | **30.94-31.10 (2 runs)** | **2.47x** (slowest REAX / fastest author run; 2.39x vs the author at 32 clients, REAX at 8) | **51-53 / 87-88 ms** | **+0.63 [-0.16, +1.78]; r2 +0.10 [-0.73, +1.33]** | 99.3 % | +1.13 [-0.74, +2.22] |
| REAX NVFP4 mid (v1.1 scales), joint - experimental | 42.77 | 3.41x | 50 / 63 ms | -0.80 [-2.33, +0.72] | 98.1 % | +0.13 [-2.33, +1.78] |
| author fast path, one question per request | 7.59 | 1.0x (own mode) | 305 / 308 ms | (joint vs single: -0.11 [-0.92, +0.73]) | | |
| **REAX FP8, one question per request** | **15.99** | **2.11x** | 52 / 90 ms | +0.42 [-0.69, +1.77] | 99.2 % | +1.03 [-1.00, +2.17] |

- vs the author's own (off-by-default) FP8 option: REAX FP8 is 1.62x faster (30.94 / 19.11); both FP8 variants keep about the
  same share of the bf16 top answers (author FP8 vs author bf16 99.4 %, REAX FP8 vs author bf16 99.3 %).
- The author's static single-bucket mode `FastOptions(buckets=(32768,))` (card: about +5 % offline throughput on GB200, "for
  offline throughput only") is much slower under this serving load, because every step is padded to 32k tokens: on pod B P300
  1.41 vs 4.05 items/s for the default buckets, c1 p50 5.5 s, DI-1000 2.8 dec/s over the first 479 rows (stopped early) vs 11.2
  for the default buckets on the same pod. The default buckets are therefore the favourable author configuration here. Not run: REAX at
  32 clients (v1.1 engine: within 3 % at 4/8/16 clients).
- Latency: the author's default buckets pad every pack to at least 2,048 tokens, which sets its flat ~305 ms c1 latency on this
  GPU; c1 and P300 ratios are reported, not used as speed-up headlines.
- Run-to-run noise: two identical REAX FP8 runs differ by -0.65 DI points [-1.24, -0.14] (99.6 % same answers; batch
  composition under load), so single DI-1000 CIs that just exclude 0 (e.g. P300 intelligence of REAX FP8 r2, +1.03 [+0.05, +2.32],
  or the author FP8 r2 above) are within run-to-run variation. At concurrency 1 the engine is deterministic: 60 DI requests
  (15 multi-question) x 3 repeats, 304/304 answers bit-identical.
- Exactness of the port (tools/exact_v12.py, 40 multi-question joint requests / 178 questions, vs the author's exact reference
  path `fast=False`, one request per call): bf16 100 % same top answer, mean over questions of the max |dp| over options 0.00025;
  FP8 100 %, 0.0011-0.0016. (For scale only, different metric and GPU: the model card reports mean |dp| 0.002 for the author's
  fast path vs its reference.) Small sample; the DI-1000 runs above are the larger check.
- Cost at USD 2.49/h (RunPod SECURE): author fast joint USD 0.055 per 1k decisions, REAX FP8 joint USD 0.022; one question per
  request USD 0.091 vs 0.043.
- **Full Decision-Index set (5,000 requests / 25,509 decisions, pod B = RunPod COMMUNITY RTX PRO 6000, both runs on that pod):**
  REAX FP8 joint 28.05 dec/s vs author fast joint 11.76 = **2.38x**; DI **+0.23 [-0.36, +0.75]**, 99.2 % same top answer; four
  areas' CIs include 0, tools +0.10 [+0.03, +0.17]. Same pod, DI-1000: 27.58 vs 11.20 = 2.46x, DI +0.57 [-0.38, +1.91], 99.4 %
  same top answer. (Pod B is a slower host than pod A; ratios are only formed within a pod.)

| Flash 4B, DI-1000 | dec/s | x author fast | c1 p50 | DI delta [CI] vs author | answers same |
|---|---|---|---|---|---|
| author fast path, joint (bf16) | 56.52 | 1.0x | 59 ms | - | - |
| author fast path, joint, `fp8=True` | 72.23 | 1.28x | 40 ms | -0.40 [-1.94, +0.97] | 96.9 % |
| **REAX FP8, joint** | **137.06** | **2.43x** (1.90x vs author FP8) | **27 ms** | -1.05 [-2.33, +0.39] | 96.9 % |
| author fast path, one question per request | 35.91 | 1.0x | 59 ms | | |
| **REAX FP8, one question per request** | **77.83** | **2.17x** | 27 ms | -0.19 [-1.32, +1.06] | 98.5 % |
| pod B: author fast path, joint (bf16) | 51.67 | 1.0x | 63 ms | - | - |
| **pod B: REAX bf16, joint (quality-safe)** | **90.66** | **1.75x** | 42 ms | **+0.24 [-0.09, +0.58]** | 99.5 % |

FP8 moves the small model more than the 27B: the author's own FP8 option changes as many Flash answers (96.9 %) as ours, and the
FP8 DI-1000 delta is the weakest of all rows (-1.05, CI includes 0). For Flash the quality-safe choice is REAX bf16 (1.75x, 99.5 %
same answers); FP8 (2.4x) only after a full-set check. The bf16 port is exact on Flash too (100 % / 99.4 % same top answer on the
178-question check, mean max |dp| 0.0025).

**Caveats.** Text only (vision requests -> 422; the author's library handles images). Requests are capped at 8,192 tokens per
sequence (author default 262,144): longer joint requests are split into question chunks like the author's `Encoder.plan`, a single
longer question is refused. The author's fast path needed one fix to run as intended inside a server thread: torch 2.14 keeps
`torch._dynamo.config` per thread, so its `cache_size_limit = 256` set at load time does not reach a worker thread and the layer
compile falls back to eager after 8 recompiles (as shipped in a thread: 10.43 dec/s joint, 5.22 one per request). All author
numbers above use the limit re-applied in the serving thread; that is the favourable baseline. NVFP4 reuses the v1.1 activation
scales; it is not recommended without a v1.2 calibration and a full-set check (the v1.1 equivalent failed the DI-5k gate).
Details, raw runs and paired comparisons: results/pplx-decider-v1.2/2026-10-17/, docs/models/pplx-decider-v1.2.md.

# Results (17 Oct 2026): shared-state mode, 5.8-6.0x the reference, no measurable score change

Model: perplexity-ai/pplx-decider-v1.1-27b @ 6195aa55. GPU: 1x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, 600 W), RunPod,
128 cores. Reference = the author's `DecisionModel` (bf16, SDPA), the author's serial research server: one request at a time,
questions in chunks of 8 (`engine/server.py --backend ref`). Harness: the official owo runners (`run_di.py --workers 8 --limit 1000`
= the DI-1000 sample, 1,000 requests / 5,129 decisions, 8 clients; `run_p300.py --workers 16`), paired bootstrap 95 % CIs vs the
reference run. Every run starts with fresh Triton/Inductor caches; reference and candidates were interleaved on the same pod
(5 reference runs: 5.242-5.281 dec/s, answers bit-identical across all of them). Ratio = slowest candidate run / fastest reference run.

**Result.** With FP8 weights and shared-state encoding the engine answers 31.7-32.0 decisions/s on the DI-1000 sample against
5.24-5.28 for the author's reference on the same pod: **6.0x** (worst case, 4 candidate runs vs the fastest of 5 reference runs). On
the full Decision-Index set (5,000 requests / 25,509 decisions, same pod, one run each) 32.3 vs 5.61 decisions/s = **5.8x** (the
reference is relatively faster on the full set; the request mix is the same: 78 % single-question requests, 85 % of decisions in
multi-question requests). Plain FP8 without sharing: 3.1x; bf16 engine: 1.8x.
**Quality:** on the full set the Decision Index differs by -0.18 points (95 % CI -0.69 to +0.33; repeat on a second pod -0.04
[-0.54, +0.41]; FP8 without sharing: -0.20 [-0.64, +0.24]); no area differs significantly. Individual answers do change: 2.1 % of multi-question decisions pick a different top
option (0.6 % for FP8 alone), and a question's probabilities can depend on the other questions in its request (top-1 changed for 10
of 108 questions when asked alone, test of 40 requests; FP8 without sharing: 2 of 108 from batch-composition noise). This is an approximation, not a lossless speed-up.

| (DI-1000 unless marked) | dec/s | ratio | DI-5k delta [CI] | top-1 flips vs ref, DI-5k (multi-q / single-q) | P300 capability delta [CI] |
|---|---|---|---|---|---|
| reference (author, bf16, serial) | 5.24-5.28 (5 runs) | 1.0x | - | - | - |
| REAX bf16, no sharing (1 run) | 9.72 | 1.8x | | 0.09 % / 0.8 % (DI-1000) | |
| REAX FP8, no sharing (1 run) | 16.21 | 3.1x | -0.20 [-0.64, +0.24] (2nd pod) | 0.6 % / 2.7 % | |
| **REAX FP8 + shared state** | **31.73-31.98 (4 runs)** | **6.0x** (DI-5k 5.8x) | **-0.18 [-0.69, +0.33]**, 2nd pod -0.04 [-0.54, +0.41] | 2.2 % / 2.7 % | +0.32 [-1.27, +1.68] * |
| REAX NVFP4 mid (published 4.4x mode), no sharing | 22.49-22.93 | 4.3x | -0.70 [-1.45, 0.00] (2nd pod) | 1.5 % / 7.2 % | -0.26 [-1.68, +1.49] * |
| REAX NVFP4 mid + shared state (faster, less safe) | 42.94-43.62 (3 runs) | 8.1x | -0.50 [-1.25, +0.20] (2nd pod) | 2.5 % / 7.0 % | -0.26 [-1.68, +1.49] * |
\* P300 is single-question traffic: shared state is not exercised there; the P300 row measures the quantization only.
DI-5k flips by questions per request (FP8 + shared vs FP8 alone): 2-4 questions 4.3 % vs 1.3 %, 5-16: 2.7 % vs 0.9 %, 17+: 2.0 % vs 0.5 %.
Controls and NVFP4 rows on DI-5k ran on a second pod against its own reference run (5.23 dec/s).
Other measurements: throughput within 3 % at 4, 8 and 16 clients (31.0 / 31.8 / 31.8); answers bit-identical across repeats at
concurrency 1 (60 DI requests x 3); under load run-to-run top-1 differences 0.3 % (co-batching). No request over 8 s (reference: up to
68 s, queueing behind a serial server). Single-request latency (P300, c1): 50 ms vs 112-115 ms (2.2x). Cold start: 6.5-7.5 min warm-up
with fresh kernel caches. dec/s = answered decisions per second; the `usage` tokens are full tokens, the engine executes ~2.5x fewer.

**Shared-state mode** (`--shared-state 1 --shared-until 64 --shared-mixq 1`, docs/models/pplx-shared-state.md): the questions of
one request share the encoded state (system prompt + state are encoded once per request instead of once per question). In the
model's 16 noncausal attention layers the state tokens then attend to the state and all questions of the request instead of the
state and their own question; each question attends to the state and itself. Single-question requests take the normal path. The
speed-up comes from requests with several questions about one state; traffic of single-question requests gets the plain FP8 3.1x.
Candidate history (all measured, nothing dropped): NVFP4 on every layer ("a4", +9 %) fails the DI-5k gate (-0.91 [-1.72, -0.14]);
NVFP4 mid + sharing passes but its own no-sharing control is at the gate's edge (-0.70 [-1.45, 0.00]); FP8 + sharing was measured
after those results. Sharing only the first 16 layers loses less but gains only ~1.1x; state tokens that see no question at all
(instead of all questions) lose more (bf16 top-1 95.6 % vs 97.6 %). Streams and larger batches gave nothing. A cold server first
stalled for tens of seconds on step sizes it had not seen (fla conv kernels specialise per 1024-token bucket): the warm-up now sweeps
every bucket and requests larger than the step budget are split. Raw runs and a summary of every run:
results/pplx-decider-v1.1-27b/2026-10-17-shared-state/.

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
