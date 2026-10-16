# s1-llm-auto-router: the `classifier_heads` model kind

`s1-llm-auto-router` (Hugging Face `decisionmodels/s1-llm-auto-router`, MIT, revision `572c5b6c`) is an encoder with seven
**fixed classification heads**. It is not a readout decision model, so it is served by its own backend and its own
route, but by the same engine: registry entry, server, scheduler and evaluation harness.

## Two kinds of model in this engine

| | readout decision models (`model_kind="decision"`) | classifier-heads models (`model_kind="classifier_heads"`) |
|---|---|---|
| input | `state` + a question (type, instructions, per-option criteria) in the prompt | `request` (+ `context`) text only |
| computation | decoder prefill, logits of the answer-option tokens at the last position | encoder forward, N heads in the same pass |
| questions | any question, options come with the request | fixed set, trained into the heads |
| probabilities | `softmax(option logits / T)` | per head: `softmax(logits / T_head)` or `sigmoid(logit / T_head)` |
| backends | native / vLLM / GGUF | `engine/backends/onnx_heads.py` (onnxruntime) |

The speed-ups of this engine (packed batches, FP8, fused kernels, vLLM or llama.cpp execution) apply to the readout models.
For the classifier-heads kind, the engine adds **no** speed-up: it is an equivalent execution mode of the model's own path
(same ONNX file, tokenizer, preprocessing, temperatures), with the registry, the server contract and the evaluation harness
around it.

## The model

An encoder (jhu-clsp/mmBERT-small pruned to 11 layers, int8 embeddings, 183 MB `model.gq.onnx`) reads the pair
`(context[:600], request)` truncated to 512 tokens (only the request side is truncated) and emits seven heads in one pass:

| head | type | classes |
|---|---|---|
| `category` | choice | coding, agentic, math, knowledge, long_context, tool_use, design, summarisation, general |
| `difficulty` | score | trivial, easy, moderate, hard, frontier |
| `stakes` | score | negligible, low, medium, high |
| `needs_tools`, `needs_vision`, `needs_long_context`, `follow_up` | noul (yes/no) | false, true |

Temperatures (`meta.json`): category 0.5, difficulty 0.8, stakes 0.75, flags 0.6. Pinned sha256 (checked at load):
`model.gq.onnx` `53a02107...0da80`, `meta.json` `b6689ba2...5649c2`, `tokenizer/tokenizer.json` `c01f8492...3ae230d`.

## Run it

```bash
pip install onnxruntime tokenizers numpy uvicorn
hf download decisionmodels/s1-llm-auto-router --revision 572c5b6cdb53b77fbb26ed33568ebbc5b71c69c0 --local-dir weights/router
RUNTIME_ALLOW_NO_AUTH=1 REAX_THREADS=4 python engine/serve.py --model s1-llm-auto-router --model-dir weights/router
```

`REAX_THREADS` sets the onnxruntime intra-op threads (default 4). CPU only by default; a GPU execution provider can be passed
to `OnnxHeadsBackend(providers=[...])` but is not tested or needed.

### `POST /v1/classify`

```json
{"model": "s1-llm-auto-router", "request": "Refactor this ABAP report to use CDS views", "context": "optional summary"}
```
`model` is optional (it must be `s1-llm-auto-router` or `llm-auto-router` if given); `context` is text or a JSON value (default empty).
Header `X-S1M-Internal-Token` as for the other routes. Response:

```json
{"model": "s1-llm-auto-router",
 "heads": {"category": {"label": "coding", "confidence": 0.93, "probabilities": {"coding": 0.93, "agentic": 0.01, "...": 0}},
           "difficulty": {"label": "moderate", "score": 2.1, "confidence": 0.6, "probabilities": {"trivial": 0.0, "...": 0}},
           "stakes": {"label": "low", "score": 1.0, "confidence": 0.7, "probabilities": {"negligible": 0.1, "...": 0}},
           "needs_tools": {"label": "false", "confidence": 0.98, "probabilities": {"false": 0.98, "true": 0.02}}},
 "usage": {"input_tokens": 12, "output_tokens": 0}}
```
(abridged: all seven heads are returned). `score` is the expected level, sum(i * p_i). Errors: 400 for blank request, a state over
16 KiB, or a context that leaves no room for the request in 512 tokens; 401 for a wrong token; 503 while warming up.

### Production-compatible shape

`POST /internal/v1/decide` (one question) and `POST /v1/systemone` (any subset of the seven) answer in the shape of the
reference router engine: `state` = `{"request", "context"}` (or text), `questions` = the fixed schema with the same option sets
(`category` criteria = the 9 categories, `difficulty` 5 levels, `stakes` 4 levels, flags `noul`), answers
`{"type", "choice"|"score"|"noul", "confidence", "probabilities", ...}`, `usage.input_tokens` = tokens of the single pass.
Any other question name, type or option set is a 400: the heads cannot answer other questions. `request_id` and
`deadline_unix_ms` of the reference server are accepted and ignored.

## Equivalence to the production path

Measured 16 Oct 2026 on a shared 12-core CPU host, 4 pinned cores, no network (bubblewrap), onnxruntime 1.23.2,
tokenizers 0.23.2, numpy 2.2.6 (the versions the reference server is pinned to).

Sample: 620 requests = 320 deterministic synthetic ones (empty and blank requests, 1 character, 5,000 characters, 4,000 words
and 3,000 emoji beyond 512 tokens (truncation), German/French/Spanish/Russian/Chinese/Japanese/Arabic/Hindi/Korean/Greek/Polish,
code, emoji and zero-width/RTL control characters, contexts of 599/600/601 characters and 5,000 characters, special-token text,
a context that cannot be truncated) plus 300 real held-out router test requests. `bench/router_equiv.py` makes the sample.

| comparison | answered | rejected by both | argmax identical | max abs probability difference |
|---|---|---|---|---|
| backend vs reference implementation (`bench/router_reference.py`), in process | 570 | 50 | 570 / 570 (100 %) | 0.0 (all bit-identical) |
| REAX server `/v1/systemone` vs the reference router engine server `/internal/v1/decide`, over HTTP | 563 | 57 | 563 / 563 (100 %) | 0.0 (all bit-identical, same `input_tokens`) |

"Rejected by both" are inputs neither side can answer: blank requests (HTTP only), state over 16 KiB, and contexts of 600 characters
that already fill 512 tokens (CJK, combining marks), where the tokenizer cannot truncate the pair; the reference raises, the
reference server answers 503 and this engine 400. Results: `results/s1-llm-auto-router/`.

Padded batches: running 2 or 8 requests as one padded batch gave bit-identical logits to single runs for 567 inputs (random
lengths up to 512 tokens, 4 threads). That is a property of the kernels on that CPU, so batching is **off by default**: the
scheduler runs one item per `run()`. `REAX_ROUTER_BATCH_TOKENS=4096` packs queued requests into one padded run; it did not
improve throughput in the measurement below.

## Latency and throughput (CPU, same threads)

Closed loop, 40 s per cell, requests from the same 300 real test requests plus the synthetic ones (long requests are
truncated to 512 tokens, so these are heavier than the model card's short-prompt figures), both servers with 4 onnxruntime
threads pinned to the same 4 cores, run one after the other, two rounds in opposite order. The host was heavily loaded
(load average 22 to 40 on 12 cores) by other jobs, so absolute numbers are inflated and noisy: only the comparison
between rows of the same round means anything.

| server | c | round 1 p50 / p95 ms, req/s | round 2 p50 / p95 ms, req/s |
|---|---|---|---|
| reference router engine server | 1 | 48.1 / 216.6, 14.0 | 58.2 / 319.9, 9.9 |
| REAX `classifier_heads` | 1 | 51.8 / 195.8, 14.1 | 52.9 / 257.5, 10.3 |
| REAX, batching on | 1 | 51.4 / 212.7, 13.7 | 55.9 / 225.4, 12.9 |
| reference router engine server | 4 | 296.7 / 746.1, 12.1 | 338.8 / 789.1, 11.0 |
| REAX `classifier_heads` | 4 | 264.8 / 544.1, 14.3 | 291.4 / 876.5, 11.2 |
| REAX, batching on | 4 | 291.9 / 871.6, 10.7 | 305.9 / 751.2, 11.5 |

Conclusion: indistinguishable within the noise, as expected for the same onnxruntime session on the same input. The
model card's own numbers (13.9 ms p50 on 4 pinned cores of an EPYC 9555, 26 ms on an older Xeon) were not re-measured.

## How it relates to the decision models

The router sits in front of them: a gateway classifies an incoming request (category, difficulty, stakes, tool and vision
needs) and picks the model that handles it, such as one of the decision models served by this engine or a general LLM. It
still cannot take an arbitrary question: heads are trained for the seven questions above, and this engine says so with a
400 rather than guessing.
