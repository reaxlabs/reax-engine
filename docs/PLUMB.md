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

See `RESULTS-PLUMB.md` for the measurements.
