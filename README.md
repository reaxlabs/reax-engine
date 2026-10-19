# REAX Engine

A source-available **inference engine for decision models**. Free for individuals, small organisations and REAX subnet
participants; larger companies need a commercial licence (see [Licence](#licence)).

A decision model answers a structured question (choice, yes/no, score) about a state with **one forward pass and a readout over
the answer options** at the last prompt position: probabilities = softmax(option logits / T). There is no token generation.
The REAX Engine is built for exactly that workload: prefill-only, packed batches, readout of a few option logits, a decision API.

**Is "inference engine" the right word?** Yes, in the precise sense: it loads trained model weights and runs their forward pass to
produce predictions, with its own kernels, batching, scheduling and server. It is a *specialised* inference engine. It is **not** a
general LLM serving engine: no text generation, no chat/completions API, no KV cache across turns, no speculative decoding.
For three Gemma-4 models it delegates the forward pass to vLLM (marked "vLLM-backed" below) and adds the decision prompt,
readout, calibration, API and evaluation; for those models the REAX part is the decision layer, not the kernels.

## Models (one registry, one server, one evaluation harness)
`python -m engine.registry` prints the table; details and every measurement: [docs/MODELS.md](docs/MODELS.md).

| model | kind | speed vs the model's reference server (same GPU, same harness) | answers vs reference |
|---|---|---|---|
| Plumb-4B (s1-fast) | native | 1.43-1.47x (`exact` mode, SN92 engine-competition scorer, c16) | bit-identical |
| Perplexity Decider v1.1 27B | native, FP8 | 2.62x (P300, c32); 3.1x Decision-Index traffic (RESULTS.md) | 99.0 % top-1, score within noise |
| Perplexity Decider v1.1 27B, shared-state mode (experimental, `engine/server.py --shared-state 1`) | native, FP8 | 5.8-6.0x Decision-Index traffic (gain comes from multi-question requests about one state; single-question requests ~3x) | approximation: no measurable DI change (-0.18 [-0.69, +0.33]), 2.1 % of multi-question top-1 answers differ ([docs](docs/models/pplx-shared-state.md)) |
| d3 (vllm-sr/d3) | native, FP8 | 2.54x (P300, c32) | 100 % top-1 on P300, 99.4 % on DI-1000 |
| Quyet-1.0-Large | vLLM-backed, FP8 + FP8 KV | 3.43x (P300, c32) | 99.0 % top-1, score within noise |
| Rune 26B-A4B | vLLM-backed | 1.0x (`deterministic` preset) | bit-identical |
| Blink v0.3 26B-A4B | vLLM-backed, batch-invariant | 0.70x of stock vLLM (the price of batch invariance) | - |
| Gutsy 0.8B, Plumb-4B Q8 (GGUF) | CPU (llama.cpp) | 1.0x | bit-identical |
| s1-llm-auto-router | N-head classifier (onnxruntime, CPU) | equivalent execution mode | see docs |

Measured 16 Oct 2026 on one NVIDIA RTX PRO 6000 Blackwell (96 GB) unless stated; P300 = the 300-item public JevBench set,
DI-1000 = a fixed 1,000-request Decision Index sample. Ratios at high concurrency compare our batching server with each model's
own (often serial) reference server; they are serving-system gains, not pure kernel gains. Model weights are not part of this
repository; each model keeps its own licence (all listed models: Apache-2.0, the router: MIT).

## One server
`engine/serve.py --model <id>` serves every registry model with the same API:
- `POST /v1/systemone` — state + one or more questions → answers (the decision-model API).
- `POST /internal/v1/decide` — the REAX subnet runtime contract (TCP or a unix socket via `RUNTIME_UDS`).
- `POST /v1/classify` — classifier-heads models only (per-head labels + probabilities).
- `GET /health` — 200 after warm-up.

```bash
S1M_INTERNAL_TOKEN=secret python engine/serve.py --model plumb-4b --model-dir ./plumb-4b       # or: python -m engine.serve ...
RUNTIME_ALLOW_NO_AUTH=1 python -m engine.serve --model quyet-1.0-large --model-dir ./quyet --port 8741 --opt preset=fp8-kv
```
`--model` takes a registry id or alias, `--opt key=value` passes backend options. The token header (`X-S1M-Internal-Token`) is
required unless `RUNTIME_ALLOW_NO_AUTH=1`. The legacy `engine/server.py` serves only pplx-decider.

Images: `docker/s1-fast` (Plumb-4B, the SN92 engine-competition reference), `docker/qwen35` (pplx, d3), `docker/vllm-readout`
(Quyet, Rune, Blink), `docker/gguf-cpu`.

## Model types
- **Readout decision models** (all models above except the router): the question is part of the prompt, the answer is a softmax over the answer-option
  logits at the last prompt position. The engine's speed-ups (packed batches, FP8, fused kernels, vLLM or llama.cpp execution)
  apply to these.
- **Classifier-heads models** (`model_kind="classifier_heads"`, first: `s1-llm-auto-router`): an encoder with N fixed heads that
  answers a fixed set of questions about a request text in one pass, run with onnxruntime on CPU (`engine/backends/onnx_heads.py`,
  `POST /v1/classify`). This is an equivalent execution mode of the model's own path (bit-identical outputs, same latency), not a
  faster one: the engine contributes the registry, server contract and evaluation harness. See `docs/models/s1-llm-auto-router.md`.

## Plumb-4B and the REAX Engine competition
The SN92 Engine competition measures miner engines for Plumb-4B against a reference engine on the same GPU. The reference is this
engine in `exact` mode: the same answers bit for bit as the original runtime, deterministic across process starts, ~1.45x faster.
How it works and what did not pass the gates: [docs/PLUMB.md](docs/PLUMB.md).

## How (native models)
- Packed variable-length batches (no padding): fla varlen kernels for the Gated-DeltaNet layers, varlen flash attention
  (noncausal for pplx/d3, causal for Plumb).
- FP8 W8A8 GEMMs with the activation quantisation fused into the preceding norm/activation kernels; fused projections;
  torch.compile + Triton fusions (pplx, d3). Plumb `exact` mode instead replays the reference's op sequence in per-bucket CUDA
  graphs on several streams.
- Last-layer pruning: only the readout position needs the final layer's attention/MLP.
- Continuous batching: one GPU scheduler thread, tokenisation off the GPU thread.

## Quick start (Perplexity Decider, FP8)
```bash
MODEL_DIR=./model bash tools/setup.sh
RUNTIME_ALLOW_NO_AUTH=1 python -m engine.serve --model pplx-decider-v1.1-27b --model-dir ./model --port 8741 --opt linear=fp8
curl -s localhost:8741/v1/systemone -H 'content-type: application/json' -d '{"model":"pplx","state":"Order #123 arrived broken.",
  "questions":{"team":{"type":"choice","instructions":"Which team handles this?","criteria":{"billing":"refunds","support":"technical issues","sales":"buying"}}}}'
```
Evaluate any model against its reference: `bench/README.md`. Hardware and memory notes: DEPLOY.md. Earlier pplx results and the
experimental calibrated NVFP4 mode: RESULTS.md.

## Licence
The REAX Engine is source-available under the [REAX Engine License 1.0](LICENSE). In plain English (the LICENSE text is
what counts):
- **Free** for individuals for personal purposes, and for companies and other organisations (including non-profits and
  universities) with **no more than 10 employees and under USD 5 million annual revenue**, counted together with
  affiliates. Free use includes commercial use, changes and redistribution under the same terms, but not use for the
  benefit of an organisation above these limits (for example running or hosting the engine as a service for it).
- **Free for the REAX subnet** (Bittensor netuid 92), whatever your size: running miners and validators, serving the
  subnet's requests, building and submitting engine-competition entries based on the reference engine, and publishing
  them as the subnet rules require. Submitting an entry grants the licensor a licence to your changes in it, so winning
  entries can be adopted into the reference engine.
- **Free 30-day evaluation** once per company. A company that outgrows the free limits has 90 days to take a licence.
- **Everyone else needs a commercial licence.** Contact us at **info@decisionmodels.io**. Guide prices (not an offer,
  plus VAT where applicable): about **USD 1,000 per year** for a company above the free limits, about **USD 10,000 per
  year** from 1,000 employees or USD 500 million annual revenue.
- **Earlier releases stay Apache-2.0:** everything up to and including commit `ffa8398` was published under the Apache
  License 2.0 and remains available under it.
- **Third-party code** (prompt contracts adapted from model authors, all Apache-2.0) keeps its own licence; see
  [NOTICE](NOTICE) and the file headers. Model weights are not part of this repository and keep their own licences.
