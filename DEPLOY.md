# Deploying the pplx-decider-v1.1-27b engine (production recipe)

Validated config (evaluation harness, paired against the model's reference server on the same GPU): FP8 W8A8,
`python engine/server.py --model <weights dir> --backend engine --linear fp8 --compile 1 --gva 1 --budget 16384 --port 8741`
-> /v1/systemone (author-compatible answers + noul probabilities), GET /health. 15.7 dec/s on a 1,000-request Decision Index sample,
single-question c1 p50 ~55 ms on one RTX PRO 6000 Blackwell (sm_120). Quality (paired vs reference): Decision Index sample +0.21 [-0.67, 1.32], JevBench public capability -0.09 [-1.46, 1.35].

- Weights: perplexity-ai/pplx-decider-v1.1-27b @ 6195aa55a72ae21e503c8a60198f7b390d42adb5 (Apache-2.0; keep LICENSE + NOTICE).
  bf16 on disk (~49 GiB); FP8 conversion happens at load (~10 s).
- Software (pinned, as validated): Python 3.12, torch 2.14.0+cu130 (CUDA 13 driver >= 580), transformers 5.17.0,
  flash-linear-attention 0.5.2 (fla-core 0.5.2), safetensors 0.8.0, huggingface-hub 1.31.0. See tools/pod_setup.sh.
- GPU: needs FP8 tensor cores (sm_89+; validated sm_120). Memory: FP8 weights ~27 GiB + bf16 embedding 2.5 GiB; peak during
  load ~50 GiB (bf16 shard then convert); activations bounded by --budget tokens per step (16k), not by concurrency.
  Budget for a 96 GB card: leave >= 55 GiB free for this process at start (steady state ~32-36 GiB).
- 32 GB cards (RTX 5090): set CIE_LAZY_LOAD=1 (stream+quantize weights per tensor; ~28 GiB in FP8); measured 10.3 dec/s.
- Requests > 8192 tokens per question branch -> HTTP 422 "maximum context length ..." (same limit as the author).
- Batch-variance: under concurrent load a decision's probabilities vary slightly with co-batched requests (mean |dp| ~1e-3,
  like any batched server); at c1 results are bit-reproducible (4x199 repeats, 0 drift).
- First requests trigger torch.compile + Triton autotune (~1-2 min warm-up inside the server start; /health only answers
  after warm-up).
