# REAX Engine

A prefill-only inference engine for **decision models**: models that answer a structured question about a state with
one forward pass and a readout over the answer options (no token generation). The first supported model is
[`perplexity-ai/pplx-decider-v1.1-27b`](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) (Qwen3.8-27B hybrid:
48 Gated-DeltaNet layers + 16 noncausal full-attention layers, 255-way readout), which stock vLLM/SGLang/TensorRT-LLM
cannot serve because its full-attention layers are noncausal.

**Result (one RTX PRO 6000 Blackwell, same GPU and same evaluation harness for both):** about **3.1x** the decisions/s of the
model's reference server on Decision-Index-style traffic and **2.6x** on the JevBench public set, single-request latency
1.4x lower, with the same benchmark scores within measurement noise. Details and method: RESULTS.md.

## How
- Packed variable-length batches (no padding): fla varlen kernels for the DeltaNet layers, varlen noncausal flash attention.
- FP8 W8A8 GEMMs (per-channel weights, per-token activations) with the activation quantisation fused into the
  preceding norm/activation kernels; fused projections; torch.compile + Triton fusions for all elementwise work.
- Last-layer pruning (only the readout position needs the final layer's attention/MLP outputs).
- Continuous-batching HTTP server (`/v1/systemone`), compatible with the reference server's request/answer format.

## Models
One registry, backend API and evaluation harness for several decision models: pplx-decider and d3 (native), Quyet, Rune and Blink
(vLLM-backed readout, not native kernels), Plumb-4B (native) and the CPU GGUF models. See `docs/MODELS.md`, `python -m engine.registry`,
`bench/README.md`.

`engine/serve.py` is the one server for all of them (`/internal/v1/decide`, `/v1/systemone`, `/health`):

```bash
S1M_INTERNAL_TOKEN=secret python engine/serve.py --model plumb-4b --model-dir ./plumb-4b       # or: python -m engine.serve ...
RUNTIME_ALLOW_NO_AUTH=1 python -m engine.serve --model quyet-1.0-large --model-dir ./quyet --port 8741 --opt preset=fp8-kv
```
`--model` takes a registry id or alias, `--opt key=value` passes backend options, `$RUNTIME_UDS` serves on a unix socket.
The token header (`X-S1M-Internal-Token`) is required on both routes unless `RUNTIME_ALLOW_NO_AUTH=1`.
The legacy `engine/server.py` below serves only pplx-decider.

## Model types
- **Readout decision models** (everything above): the question is part of the prompt, the answer is a softmax over the answer-option
  logits at the last prompt position. The engine's speed-ups (packed batches, FP8, fused kernels, vLLM or llama.cpp execution)
  apply to these.
- **Classifier-heads models** (`model_kind="classifier_heads"`, first: `s1-llm-auto-router`): an encoder with N fixed heads that
  answers a fixed set of questions about a request text in one pass, run with onnxruntime on CPU (`engine/backends/onnx_heads.py`,
  `POST /v1/classify`). This is an equivalent execution mode of the model's own path (bit-identical outputs, same latency), not a
  faster one: the engine contributes the registry, server contract and evaluation harness. See `docs/models/s1-llm-auto-router.md`.

## Quick start
```bash
MODEL_DIR=./model bash tools/setup.sh
python engine/server.py --model ./model --backend engine --linear fp8 --compile 1 --gva 1 --port 8741
curl -s localhost:8741/v1/systemone -H 'content-type: application/json' -d '{"model":"pplx","state":"Order #123 arrived broken.",
  "questions":{"team":{"type":"choice","instructions":"Which team handles this?","criteria":{"billing":"refunds","support":"technical issues","sales":"buying"}}}}'
```
An experimental calibrated NVFP4 (4-bit) mode reaches ~21.8 decisions/s on the same GPU at the same benchmark scores
within noise (RESULTS.md). See DEPLOY.md for hardware, memory and operational notes. Validated on RTX PRO 6000 Blackwell (sm_120); also measured
on H100, H200, B200 and RTX 5090 (32 GB cards: `CIE_LAZY_LOAD=1`).
