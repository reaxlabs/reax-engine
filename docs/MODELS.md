# Models in the REAX Engine

`python -m engine.registry` prints the table; `python -m engine.registry <id>` the full record (weights, pinned revision,
licence notice, hardware). One registry (`engine/registry.py`), one backend API (`engine/backends/base.py`), one evaluation
harness (`bench/`). Every number below says where and when it was measured; "not measured" means exactly that.

| id | status | backend | weights | note |
|---|---|---|---|---|
| `plumb-4b` (s1-fast) | native | `engine/plumb_engine.py` | `crh225/plumb-4b` | Qwen3.5-4B hybrid, causal, T=2.07 |
| `pplx-decider-v1.1-27b` | native | `engine/backends/qwen35.py` | `perplexity-ai/pplx-decider-v1.1-27b` | 255-way readout, noncausal full attention |
| `d3` | native | `engine/backends/qwen35.py` | `vllm-sr/d3` | same text stack as pplx-decider, d3 prompt contract |
| `quyet-1.0-large` | **vLLM-backed** | `engine/backends/vllm_readout.py` | `chinhnc/Quyet-1.0-Large` | Gemma-4 31B dense |
| `rune-26b-a4b` (s1-pro) | **vLLM-backed** | `engine/backends/vllm_readout.py` | `surogate/rune-26b-a4b-GGUF` (HF-gated) | Gemma-4 26B-A4B MoE |
| `gutsy-0.8b`, `plumb-4b-q8` | cpu | `engine/backends/gguf_cpu.py` | GGUF Q8_0 | llama.cpp |
| `s1-llm-auto-router` | documented-only | - | `decisionmodels/s1-llm-auto-router` | fixed-head encoder, not a letter readout |

**What "vLLM-backed" means.** Quyet and Rune are served by vLLM inside the REAX Engine: vLLM prefills the prompt and returns the
logprobs of the option-letter tokens for one generated token (`max_tokens=1`, `allowed_token_ids` = the option letters); the
engine builds the prompt, applies the model's temperature, shapes the answer and provides the server contract, registry and
evaluation harness. There are no native REAX kernels for these two models. The presets below are vLLM flag sets found by the
earlier serving studies.

## Readouts

| model | prompt | option tokens | temperature | max options | input limit |
|---|---|---|---|---|---|
| pplx-decider | system + State/Question/Options, 255 codes | 255 single-token codes | `decision_config.json` | 255 | 8192 tokens |
| d3 | `d3-code-readout-v1` (system + State/Question/Options) | 255 single-token codes (A..Z, AA..) | `decision_config.json` (1.0 at rev 225be8b), fp32 readout | 255 | none upstream; validated to 64k+ |
| Quyet | prompt v2: one user turn, compact JSON state, per-type "KIND" line | letters A.. | choice 1.3007 / score 1.3159 / noul 1.4957 (`quyet_config.json`) | 10 native (`max_options` opt lifts to 255 as the board runs did) | 8000 prompt tokens, state shrunk/truncated to 6000 (min 256) |
| Rune | Surogate decisions v1: system line + `SHARED STATE (JSON string)` user turn | A..Z, then two-letter single-token codes | 2.0, float64 softmax over soft-capped raw logits | 64 | 4096 prompt tokens (reference sidecar limit) |

Option order: Quyet `noul` = `[true, false]` (letter A is the true statement), Rune `noul` = `[false, true]`, others as pplx
(`false`, `true`). `Backend.options(question)` returns the keys in the order `run()` returns probabilities; `s1_answer` and the
models' own `answer()` shapes handle the rest.

## d3 and the noncausal attention question

d3's `decision_config.json` (revision 225be8b, 15 Oct 2026) says `"attention_mode": "noncausal_full_attention"`, `"pooling": "last"`,
`"readout_dtype": "float32"`, `"temperature": 1.0`. So **d3's 16 full-attention layers are noncausal** (as pplx-decider's; the
48 Gated-DeltaNet layers stay causal). `PplxEngine` asserts this mode at load. The modelling code in the HF repo was not read for this registry
entry. Practical evidence that the engine implements d3's attention correctly: the earlier private d3 engine (same code path) was
compared with the author runtime on the public sets on 12 Oct 2026 and all score CIs of the paired delta included 0; its live
P300 top-1 agreement with the previous production model was 0.963 (n=297). Per-decision agreement with the author runtime has to
be re-measured with `bench/pod_validate.sh d3`.

d3 port notes (from the private engine branch, ported with `engine/d3_contract.py` and additive edits to `pplx_engine.py` /
`fp4_triton.py`, pplx numerics unchanged):
- `engine/d3_contract.py` vendors d3's prompt and answer contract (`d3_format.py`, `normalize_question`, `product_answer`) from
  `vllm-sr/d3`; Apache-2.0, LICENSE/NOTICE in `upstream/d3/`. Revisions 5ce995d and 225be8b render the same prompt bytes and shape
  the same answers (checked by diff); the optional `permutation_average` mode of the author runtime (off by default) is not supported.
- fp32 readout matmul (`PPLX_READOUT_FP32=1`, set automatically for d3), int64 row indices in the fused FP8 MLP and NVFP4 Triton
  kernels (prompts above 64k tokens overflowed int32), opt-in `PPLX_EMBED_CPU=1` (bf16 embedding table in host RAM, 2.4 GiB less VRAM).

Measured with the private branch this port derives from (RTX PRO 6000 Blackwell, DI-1000 / P300, 12 Oct 2026): author bf16 5.40 DI
decisions/s; REAX FP8 15.43 (2.86x), c1 p50 52.6 ms / p95 93 ms, 28.7 GiB, c1 determinism 0 flips in 4 x 199 P300 repeats. **Not yet re-measured from this repo.**

## vLLM presets (named, `engine/backends/vllm_readout.py`)

All measured 11 Oct 2026 on one RTX PRO 6000 Blackwell, vLLM 0.30.0 (image `vllm/vllm-openai@sha256:8a69ffad...`), recipe ported.
"DI" = Decision-Index 1000-state sample (5,129 decisions), "P300" = the 300 public JevBench v1.6 items; decisions/s at the best
tested client concurrency (Quyet DI c8-c32, P300 c32; Rune DI/P300 c32). Quality columns are paired-bootstrap 95% CIs of the score
delta against the stock author arm of the same study; they say "not distinguishable from stock", not "identical".

**Quyet-1.0-Large** (`--opt preset=<name>`; default `bf16`)

| preset | what | DI / P300 dec/s | c1 p50 | quality / determinism |
|---|---|---|---|---|
| `bf16` | BF16, 64 seqs, prefix caching | measured arm was 1 seq: 9.66 / 8.44 (stock HF author 5.54 / 6.63) | 98 ms | reference arm; 64-seq config not separately measured |
| `fp8` | online FP8 weights | 24.85 / 17.34 | 59 ms | DI +0.40 [-1.11,+1.84], P300 cap +1.02 [-0.90,+2.95]; 0/300 c1 flips |
| `fp8-kv` | FP8 weights + FP8 KV | 28.65 / 19.08 | 54 ms | DI -0.68 [-1.96,+0.63], P300 +0.88 [-0.94,+2.80]; 0/300 c1 flips |
| `nvfp4` | calibrated full NVFP4 (needs a calibrated checkpoint, not public) | 36.71 / 26.26 | 43 ms | DI -0.12 [-2.52,+1.66], P300 -1.38 [-4.08,+1.32]; 0/300 c1 flips; agreement vs stock 96.7% DI / 95.7% P300; native DI Language -3.34 pt [-6.74,-0.33] |

Other 11 Oct arms that did **not** qualify: full NVFP4 + FP8 KV (46.5 / 31.4 dec/s) failed the determinism check (1/300 c1 argmax
change); scheduler variants with `max_num_seqs=32` or `max_num_batched_tokens=2048` gained < 3%. An earlier pilot saw 8/512 argmax
flips under batching at c8 (Quyet loaded-finish study, 9 Oct) - batched vLLM serving is not batch-invariant for Quyet.

**Rune 26B-A4B** (default `deterministic`)

| preset | what | DI / P300 dec/s | c1 p50 | quality / determinism |
|---|---|---|---|---|
| `deterministic` | REAX s1-pro recipe: `VLLM_BATCH_INVARIANT=1`, no chunked prefill, `-O0`, prefix cache off | closest 11 Oct control (32 seqs, util 0.65): 36.94 / 23.40; stock author arm 42.55 / 26.37 | 72 ms | bit-exact across c1/c16/c32, restarts and processes (7 Oct study) |
| `previous-f` | deterministic + prefix cache + async scheduling, 32 seqs | 63.43 / 23.54 | 69.5 ms | 0/300 c1 flips |
| `fp8-kv-seqs64` | BF16 weights + FP8 KV, prefix cache, async, 64 seqs | **76.06 / 29.92** | 67.7 ms | DI +0.70 [-1.09,+2.76], P300 cap +2.79 [-0.67,+6.16]; 0/300 c1 flips in two passes; agreement vs stock 98.4% / 96.7% |
| `stock-cache-async` | compiled stock + prefix cache + async, NOT batch-invariant | 73.68 / 27.14 | 50.0 ms | DI +0.24 [-1.30,+1.62], P300 +1.50 [-1.32,+2.92]; 0/300 c1 flips |

Rune's study ended on provider credit, not on a performance ceiling (calibrated FP8 / expert-only NVFP4 were never calibrated or
measured). Production perf study of 7 Oct (different harness): short text c1 21.3 dec/s p50 45 ms, c32 141 dec/s; P300 c1 16.5,
c32 59.7. Rune is HF-gated: accept the gate with your own account and give the container your own `HF_TOKEN`; do not mirror weights.

## Licences

All Apache-2.0 except the router (MIT). Keep each model's LICENSE/NOTICE when redistributing: Quyet's NOTICE starts with
"Quyet by Chinh Nguyen"; d3: `upstream/d3/LICENSE` + `NOTICE` (vLLM Semantic Router Team); pplx-decider: Perplexity AI, derived from
Qwen3.8-27B. Rune derives from Google Gemma 4 (Apache-2.0).

## Not yet validated on a GPU (this repo, this commit)

Everything GPU-side is unvalidated here: the unit tests (`pytest tests`) cover registry, prompts, answer shaping, argv/presets, the
bench harness and `bench/pod_validate.sh` against stand-ins, with no weights. Run `bench/pod_validate.sh <model-id>` on a pod
(see `bench/README.md`) before claiming parity for d3, Quyet or Rune.
