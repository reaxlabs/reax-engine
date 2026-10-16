# Plumb-4B (s1-fast) on the REAX Engine

`engine/plumb_engine.py` + `engine/serve.py` serve [`crh225/plumb-4b`](https://huggingface.co/crh225/plumb-4b)
(Qwen3.5-4B hybrid, 24 Gated-DeltaNet + 8 causal full-attention layers, JevK5 decision readout) with the REAX Engine's
kernels and a continuous batcher, behind the s1 runtime sidecar contract:

- `POST /internal/v1/decide` (header `X-S1M-Internal-Token`, one question per call), `POST /v1/systemone`
  (any number of questions), `GET /health` (200 only after warm-up).
- Listens on `$RUNTIME_UDS` (mode 0666) if set, else `RUNTIME_HOST:RUNTIME_PORT` (127.0.0.1:8101).
- Token from `S1M_INTERNAL_TOKEN` or the file `S1M_INTERNAL_TOKEN_FILE` (`RUNTIME_ALLOW_NO_AUTH=1` to disable).
- Weights read-only at `$PLUMB_MODEL_DIR`; if `$REAX_MANIFEST` exists the sha256 of the weight files is checked first.

```bash
docker build -f docker/s1-fast/Dockerfile -t reax-engine-s1-fast .
docker run --rm --gpus device=0 --network none --read-only --tmpfs /tmp:rw,exec,size=16g --cap-drop ALL \
  -v /path/to/plumb-4b:/model:ro -v /run/reax:/run/reax -e RUNTIME_UDS=/run/reax/engine.sock \
  -e S1M_INTERNAL_TOKEN_FILE=/run/reax/token reax-engine-s1-fast
```

## Two modes (`PLUMB_MODE`)

**`exact` (default)** reproduces the reference runtime's numbers *bit for bit*, at any concurrency, and is still faster:

- The reference (jevk5) answers from CUDA graphs captured per padded length (128, 192, ..., 4096). A decision's
  logits are therefore fixed by the GEMM shapes of its bucket (`M` = bucket) and by one more detail: inside a graph
  capture, Hugging Face's attention takes a different path than in eager mode (explicit bool causal mask + `repeat_kv`
  instead of `is_causal`), which changes the bf16 rounding of every full-attention layer.
- The engine runs the same op sequence per decision (separate projections, bf16 `conv1d` + `silu`, masked SDPA, fla
  `chunk_gated_delta_rule`, full last layer, no `torch.compile`) padded to the same bucket, in one CUDA graph per
  (stream, bucket). The decisions of a scheduler step are spread over `PLUMB_STREAMS` (default 4) CUDA streams and
  replayed concurrently, so the GPU overlaps the latency-bound kernels of several decisions. Answers do not depend on
  the batch they were scheduled in. Verified: letter logits equal the reference graph replay for 100 % of items
  (`tests/exact_diff.py`), the competition evaluator reports distance 0.0 / top agreement 100 % / determinism 0.0.
- One more source of nondeterminism had to be removed: fla autotunes its Triton kernels by timing, and the
  `l2norm_fwd_kernel` row reduction gives different last bits for some configurations, so about 1 process start in 10
  (reference runtime and this engine alike) computed slightly different answers (mean logit distance ~0.06 against
  the others). `pin_fla_autotune()` fixes one of the configs the majority of starts end up with.

**`fast`** (`PLUMB_MODE=fast`) packs all queued decisions into one varlen batch (fused projections, fla conv, flash
varlen attention, last-layer pruning, `torch.compile` fusions, CUDA graphs for small batches). It is 2x-3x faster than
`exact`, but it is not bit-identical: Plumb's letter logits are bf16 (steps of 0.0625-0.125 at |logit| 8-32, T = 2.07),
so any change in GEMM shapes moves some option by one step: mean logit distance 0.05-0.07 and 98-99.5 % top-choice
agreement against the reference, and results that depend on the co-batched requests. That fails the competition's
determinism gate (0.05) and is on the edge of the agreement gate (99 %). Use it only where answers need not
reproduce the reference.


## Measurements (one RTX PRO 6000 Blackwell, 16 Oct 2026; competition harness `reax-engine-eval`, track s1-fast)

Reference = the runtime built from `reax-subnet` runtimes/s1-fast (jevk5, batch-1 CUDA graphs behind a lock).

| | reference runtime | engine, `exact` (default) | engine, `fast` |
|---|---|---|---|
| decisions/s, generated s1 items, concurrency 1 | 67.4 (p50 14.8 ms) | 66.0 (p50 14.9 ms) | 100 (p50 9.8 ms) |
| concurrency 16 | 68.0 | 102.4 | 180-199 |
| concurrency 64 | 67.1 | 104.0 | 214-218 |
| P300 public set (long items), c1 / c16 | 34.8 / 35.1 | 34.1 / 41.7 | 48.7 / 66.5 |
| evaluator score vs reference (2 runs each) | - | 1.43-1.47, all gates passed, distance 0.0 | not valid: top agreement 98.0-98.6 %, one item distance 1.03 |
| GPU memory | 12.1 GiB | 15.2 GiB | ~15 GiB |
| start until `/health` 200 (read-only container, weights verified) | 36 s | 48 s | 104 s |

`exact` answers equal the reference's for every item tried: 600 + 1000 generated items, the 231-item public set and the
300-item P300 set (3 of them exceed 4096 tokens: HTTP 400 in both), at concurrency 1 and 16 (mean/max logit distance
0.0, top agreement 100 %); P300 accuracy 0.7306 for both. In reverse (reference = this engine, candidate = the reference
runtime) the evaluator also reports distance 0.0 and passes (score 0.67-0.70).

Why the large gap between the modes: see above. FP8 W8A8 (per-channel weights, per-token activations, all linears)
agrees on only 91-92 % of the items (mean distance 0.28, max 1.4) and was not pursued.
