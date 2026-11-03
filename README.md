# REAX Engine

An **inference engine for decision models**. Open source under the MIT License (see [Licence](#licence)).

A decision model answers a structured question (choice, yes/no, score) about a state with **one forward pass and a readout over
the answer options** at the last prompt position: probabilities = softmax(option logits / T). There is no token generation.
The REAX Engine is built for exactly that workload: prefill-only, packed batches, readout of a few option logits, a decision API.

**Is "inference engine" the right word?** Yes, in the precise sense: it loads trained model weights and runs their forward pass to
produce predictions, with its own kernels, batching, scheduling and server. It is a *specialised* inference engine. It is **not** a
general LLM serving engine: no text generation, no chat/completions API, no KV cache across turns, no speculative decoding.
For three Gemma-4 models it delegates the forward pass to vLLM (marked "vLLM-backed" below) and adds the decision prompt,
readout, calibration, API and evaluation; for those models the REAX part is the decision layer, not the kernels.

## REAX 1.3 (`reax13/`, two-part ensemble on one GPU)
[`reax13/`](reax13/) is the inference engine of [decisionmodels/REAX-1.3](https://huggingface.co/decisionmodels/REAX-1.3): both model
parts (a Gemma-4-31B line and a Qwen-line 27B) load in FP8 on one NVIDIA RTX PRO 6000 96 GB, every question goes to both, and their
per-option probabilities are averaged 50/50. It ships an HTTP server (`/v1/systemone`) and a custom engine for the
[Decision Index kit](https://github.com/apolinario/decision-index). Build, run and test steps: [reax13/README.md](reax13/README.md);
the model card gives the same steps pinned to a commit of this repository.

## Which engine for what
Images are pinned by digest; a digest named here or in the subnet's miner kit is never overwritten or deleted.

| use | engine | image or path | source |
|---|---|---|---|
| mammoth-1 (s1-pro), fastest: engine competition v2 dm-all champion UID 240 | UID 240 engine | `ghcr.io/reaxlabs/reax-engine-dm240@sha256:7941f51a96086cc6d40b527d8062c2b424335875462114aa36eee7cb9d8244d5` (same digest as `ghcr.io/masterdev777/dm240`) | [`champions/uid240/`](champions/uid240/) (miner source, MIT, [provenance](champions/uid240/PROVENANCE.md)) |
| mammoth-1 (s1-pro), reference | ref-mammoth | `ghcr.io/reaxlabs/reax-engine-ref-mammoth@sha256:3080a593060542bccbdaf7cf0c66593fc182d9753b3facd50206df86d93b8588` | `docker/ref-qwen35/Dockerfile`, `REAX_MODEL=mammoth-1` |
| Plumb-4B (s1-fast), fastest: engine competition champion UID 62 | s1-fast champion | `ghcr.io/reaxlabs/reax-engine-champion-s1-fast@sha256:8467e642812347a6c34f5c3f26eb7dd42678e4a4a532aad09f3d22d9fa0138a7` (same digest as `ghcr.io/masterdev777/s1r`) | `mirrors/champions.txt` |
| Plumb-4B (s1-fast), reference | s1-fast | `ghcr.io/reaxlabs/reax-engine-s1-fast@sha256:9bd41632e33bd90377f7e29bdb9410b18cfe1fb557200160b9ebe9222abcecea` (until reax-subnet 1.2.0rc9.post5: `ghcr.io/reaxlabs/reax-s1-fast-runtime@sha256:d857a90eaf168d1db3ca982c5111b10fb61a65cc5f7113ea9e391fe6fca17b71`) | `docker/s1-fast/Dockerfile` |
| d3-turbo, REAX 1, Perplexity Decider (27B Qwen line), reference | ref-d3turbo | `ghcr.io/reaxlabs/reax-engine-ref-d3turbo@sha256:503cd9e00114dada655691de44b5a4ff1a08a985bcaf0021121967a73e436eac` | `docker/ref-qwen35/Dockerfile` |
| Blink v0.3 text, reference | ref-blink | `ghcr.io/reaxlabs/reax-engine-ref-blink@sha256:ab02e141ffb5672dc68e137736be0a6fa15445c7cc824a51f6ad2a6664c0e10a` | `docker/ref-blink/Dockerfile` |
| Blink vision (one image), reference | ref-blink (eager) | `ghcr.io/reaxlabs/reax-engine-ref-blink@sha256:97d06a91453979d6756176aba4a1cb1d78b38390643c37ed1b83f79a4ff9f606` | `docker/ref-blink/Dockerfile` |
| Quyet-1.0-Large, reference | ref-quyet | `ghcr.io/reaxlabs/reax-engine-ref-quyet@sha256:da746016917b4f8a7adb3aeace03e8985278d277b8308ea380ec8f8318b8512f` | `docker/ref-quyet/Dockerfile` |
| Jeffan-31b-1710, reference | ref-jeffan | `ghcr.io/reaxlabs/reax-engine-ref-jeffan@sha256:fdebc6e594788aa48bc228083e71c5084d86ca71ce856f6dd1f59ce106db0f13` | `docker/ref-jeffan/Dockerfile` |
| REAX 1.3 (two-part ensemble) | reax13 | built from [`reax13/`](reax13/) | [reax13/README.md](reax13/README.md) |

**mammoth-1 on the UID 240 engine, tested settings** (RTX PRO 6000 96 GB; about 47 decisions/s at 2 requests in flight). Use the
`KAI_MODEL_ENV_MAMMOTH` value exactly as given: without it the image switches to FP4 for larger batches and fails the
self-check already at 2 requests in flight. The first start takes about 6 minutes (kernel compilation).
```bash
docker run -d --name s1-pro-mammoth --gpus device=0 --network host --init --read-only \
  --tmpfs /tmp:rw,exec,nosuid,nodev,size=16g --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 \
  --shm-size 2g -v <weights dir>:/model:ro -v <token file>:/run/s1m-token:ro \
  -e S1M_INTERNAL_TOKEN_FILE=/run/s1m-token -e RUNTIME_UDS= -e RUNTIME_HOST=127.0.0.1 -e RUNTIME_PORT=8102 \
  -e REAX_MODEL=mammoth-1 \
  -e KAI_MODEL_ENV_MAMMOTH=MAM_F8=1,MAM_F8_GU=1,MAM_FLA_PIN=1,MAM_SCONV=1,MAM_H2D1=1,MAM_ADAPT=1,KAI_MAX_WAIT_MS=16,MAM_ENCTHREAD=1,MAM_FASTENC=1,MAM_F8TAB=1,KAI_ACTQ=1,KAI_GDNPS=1,KAI_ENC1=1,KAI_HCI=1,KAI_RDASYNC=1,KAI_EARLY=1,KAI_N1=1,KAI_PDL=1,KAI_B11=1,KAI_GC_FREEZE=1,KAI_GDNFUSE=1,KAI_NOXSTORE=1 \
  ghcr.io/reaxlabs/reax-engine-dm240@sha256:7941f51a96086cc6d40b527d8062c2b424335875462114aa36eee7cb9d8244d5
reax-selfcheck --model s1-pro --url http://127.0.0.1:8102 --concurrency 2   # PASS, about 47/s
```
Cap the engine at 2 requests in flight: `REAX_RELAY_MAX_INFLIGHT=2` in the relay container (`reax-runtime-relay`, one relay per
engine, all requests through it), `"max_inflight": 2` for this worker in the gateway's workers file, or `REAX_RUNTIME_INFLIGHT=2`
when the miner runs on the same machine. `reax-selfcheck` and `reax-runtime-relay` come with
[reax-subnet](https://github.com/reaxlabs/reax-subnet). A second instance on the same GPU adds no throughput; scale with more GPUs,
each with its own relay.

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
The REAX Engine is licensed under the [MIT License](LICENSE).
- **Earlier releases:** everything up to and including commit `ffa8398` (tag `apache-2.0-final`) was published under the
  Apache License 2.0; later commits up to the MIT switch under the REAX Engine License 1.0/1.1, and they keep that licence.
- **Third-party code** (prompt contracts adapted from model authors, FP8 kernels from Hugging Face; Apache-2.0 or MIT) keeps
  its own licence; see [NOTICE](NOTICE) and the file headers. Model weights are not part of this repository and keep their
  own licences.
- **Miner engines** under `champions/` (first: UID 240) are the miners' published source, licensed here under the MIT License
  with their authors' attribution kept (`champions/uid240/LICENSE`; basis and the licences the authors first published under:
  `champions/uid240/PROVENANCE.md`). Third-party code inside them keeps its own licence.
