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
| `blink-v0.3-26b-a4b` (s1-fast-2) | **vLLM-backed** | `engine/backends/vllm_readout.py` | `PixilabAI/Blink-v0.3-26B-A4B-NVFP4` | Gemma-4 26B-A4B MoE, NVFP4 (Blackwell), Rune prompt |
| `gutsy-0.8b`, `plumb-4b-q8` | cpu | `engine/backends/gguf_cpu.py` | GGUF Q8_0 | llama.cpp |
| `s1-llm-auto-router` | documented-only | - | `decisionmodels/s1-llm-auto-router` | fixed-head encoder, not a letter readout |

**What "vLLM-backed" means.** Quyet, Rune and Blink are served by vLLM inside the REAX Engine: vLLM prefills the prompt and returns the
logprobs of the option-letter tokens for one generated token (`max_tokens=1`, `allowed_token_ids` = the option letters); the
engine builds the prompt, applies the model's temperature, shapes the answer and provides the server contract, registry and
evaluation harness. There are no native REAX kernels for these three models. The presets below are vLLM flag sets found by the
earlier serving studies.

## Readouts

| model | prompt | option tokens | temperature | max options | input limit |
|---|---|---|---|---|---|
| pplx-decider | system + State/Question/Options, 255 codes | 255 single-token codes | `decision_config.json` | 255 | 8192 tokens |
| d3 | `d3-code-readout-v1` (system + State/Question/Options) | 255 single-token codes (A..Z, AA..) | `decision_config.json` (1.0 at rev 225be8b), fp32 readout | 255 | none upstream; validated to 64k+ |
| Quyet | prompt v2: one user turn, compact JSON state, per-type "KIND" line | letters A.. | choice 1.3007 / score 1.3159 / noul 1.4957 (`quyet_config.json`) | 10 native (`max_options` opt lifts to 255 as the board runs did) | 8000 prompt tokens, state shrunk/truncated to 6000 (min 256) |
| Rune | Surogate decisions v1: system line + `SHARED STATE (JSON string)` user turn | A..Z, then two-letter single-token codes | 2.0, float64 softmax over soft-capped raw logits | 64 | 4096 prompt tokens (reference sidecar limit) |
| Blink | same prompt as Rune (checked token-for-token against the model card's own client code) | A..Z, two-letter codes above 26 | 1.3 for all types (`decision_config.json`), softmax over processed logprobs | 26 native (`max_options` lifts it) | 32000 prompt tokens |

Option order: Quyet `noul` = `[true, false]` (letter A is the true statement), Rune `noul` = `[false, true]`, others as pplx
(`false`, `true`). `Backend.options(question)` returns the keys in the order `run()` returns probabilities; `s1_answer` and the
models' own `answer()` shapes handle the rest.

## d3 and the noncausal attention question

d3's `decision_config.json` (revision 225be8b, 15 Oct 2026) says `"attention_mode": "noncausal_full_attention"`, `"pooling": "last"`,
`"readout_dtype": "float32"`, `"temperature": 1.0`. So **d3's 16 full-attention layers are noncausal** (as pplx-decider's; the
48 Gated-DeltaNet layers stay causal). `PplxEngine` asserts this mode at load. The modelling code in the HF repo was not read for this registry
entry. Practical evidence that the engine implements d3's attention correctly: the earlier private d3 engine (same code path) was
compared with the author runtime on the public sets on 12 Oct 2026 and all score CIs of the paired delta included 0; its live
P300 top-1 agreement with the previous production model was 0.963 (n=297). Per-decision agreement with the author runtime was
measured from this repo on 16 Oct 2026 (table below: P300 top-1 100.0 %, DI 99.42 %).

d3 port notes (from the private engine branch, ported with `engine/d3_contract.py` and additive edits to `pplx_engine.py` /
`fp4_triton.py`, pplx numerics unchanged):
- `engine/d3_contract.py` vendors d3's prompt and answer contract (`d3_format.py`, `normalize_question`, `product_answer`) from
  `vllm-sr/d3`; Apache-2.0, LICENSE/NOTICE in `upstream/d3/`. Revisions 5ce995d and 225be8b render the same prompt bytes and shape
  the same answers (checked by diff); the optional `permutation_average` mode of the author runtime (off by default) is not supported.
- fp32 readout matmul (`PPLX_READOUT_FP32=1`, set automatically for d3), int64 row indices in the fused FP8 MLP and NVFP4 Triton
  kernels (prompts above 64k tokens overflowed int32), opt-in `PPLX_EMBED_CPU=1` (bf16 embedding table in host RAM, 2.4 GiB less VRAM).

Measured with the private branch this port derives from (RTX PRO 6000 Blackwell, DI-1000 / P300, 12 Oct 2026): author bf16 5.40 DI
decisions/s; REAX FP8 15.43 (2.86x), c1 p50 52.6 ms / p95 93 ms, 28.7 GiB, c1 determinism 0 flips in 4 x 199 P300 repeats. Re-measured from this repo on 16 Oct 2026: see "Measured on 16 Oct 2026" below (FP8: 1.86x c1 / 2.54x c32 over the author runtime on the same GPU).

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

**Blink v0.3 26B-A4B** (default `bi1`; the checkpoint is NVFP4, so Blackwell only)

| preset | what | DI / P300 dec/s | c1 p50 | quality / determinism |
|---|---|---|---|---|
| `stock` | the model card's plain vLLM recipe: NVFP4, FP8 KV, prefix caching, 64 seqs, not batch-invariant | 11 Oct study (author-stock arm): 138.2 / 53.5 | 32 ms | 6/300 c1 flips (stock is not deterministic) |
| `bi1` | `VLLM_BATCH_INVARIANT=1`, 256 seqs, 16384-token batches, FP8 KV, prefix caching, util 0.9 | 96.8 / 39.0 (70 % / 73 % of stock) | 40 ms | 0/300 c1 flips over repeated passes; DI -0.09 [-1.88,+1.95], P300 cap +0.15 [-2.99,+4.52] vs stock |
| `bi1-memory30` | `bi1` with `--gpu-memory-utilization 0.30` | throughput not measured | 32 ms | 0/300 c1 flips; 31,045 MiB peak at 32k context |

Blink's model card (Apache-2.0 tag, ungated, no LICENSE file in the repo) publishes the serving recipe and the client code; the
registry entry redistributes only the recipe, not weights. `bi1` is the deterministic choice (production s1-fast-2 runs it); it is
slower than stock vLLM by design. Numbers: 11 Oct 2026, Lium RTX PRO 6000 Blackwell, vLLM 0.30.0.

## Measured on 16 Oct 2026 (this repo, one rented GPU VM)

Run with `bench/pod_validate.sh <model-id>` on one VM with 2x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB each), driver 580.178.04,
Docker with the NVIDIA runtime; one model per GPU, reference arm and REAX arm of the same model one after the other on the **same GPU**
with the same load. Raw files: `results/<model>/2026-10-16-<run>.json` (small JSON, includes the exact environment). These are
**today's numbers**; the 11-12 Oct figures in the sections above come from earlier studies on other hosts and are not re-measured here.

Sets: **P300** = 300 public JevBench v1.6 items (297 answered by pplx/Blink/Rune: 3 items exceed their input cap and both arms reject them),
**DI-1000** = 1000 evenly spaced Decision-Index states = 5,129 decisions (fewer where a model has a lower option cap: Quyet 3,424 answered
of 5,129; Rune 4,981; Blink 5,016). Agreement is per decision on the items both arms answered. "Accuracy delta" is a paired-bootstrap
(1000 replicates) 95 % CI of REAX minus reference argmax accuracy against the gold labels (percentage points) - a proxy, not the official
JevBench capability score. Reference arms: pplx = the author's `DecisionModel` (served one request at a time, as the author does), d3 = the
author's `d3_server.py`, Quyet = the author's `quyet` package (HF transformers bf16), Rune = the reax-subnet `s1-pro` runtime image
(vLLM + reference adapter), Blink = the model card's plain vLLM recipe (preset `stock`) behind the same REAX encoder (no author server exists).

**Agreement with the reference**

| model / preset | P300 top-1 | P300 mean / max abs dp | DI-1000 top-1 | DI mean / max abs dp | accuracy delta P300 | accuracy delta DI |
|---|---|---|---|---|---|---|
| pplx-decider, FP8 (`linear=fp8`) | 98.99 % (n=297) | 0.0044 / 0.256 | 99.08 % (n=5129) | 0.0057 / 0.283 | +0.34 [-0.67,+1.35] | -0.03 [-0.31,+0.26] |
| pplx-decider, bf16 (`linear=bf16`) | 100.00 % | 0.0011 / 0.065 | 99.75 % | 0.0018 / 0.076 | +0.00 [0.00,0.00] | +0.03 [-0.10,+0.15] |
| d3, FP8 (default) | 100.00 % (n=300) | 0.0035 / 0.095 | 99.42 % (n=5129) | 0.0046 / 0.237 | +0.00 [0.00,0.00] | -0.06 [-0.32,+0.17] |
| Quyet, `fp8-kv` | 99.00 % (n=300) | 0.0101 / 0.560 | 98.92 % (n=3424) | 0.0100 / 0.484 | +0.00 [-1.00,+1.00] | -0.21 [-0.52,+0.12] |
| Quyet, `bf16` | 100.00 % (n=300) | 0.0030 / 0.134 | 99.65 % (n=3424) | 0.0032 / 0.157 | +0.00 [0.00,0.00] | -0.06 [-0.24,+0.11] |
| Rune, `deterministic` | 100.00 % (n=297) | 2e-17 / 2e-16 (identical) | 100.00 % (n=4981) | 0 / 0 (identical) | +0.00 | +0.00 |
| Rune, `fp8-kv-seqs64` | 95.96 % (n=297) | 0.0168 / 0.400 | 98.27 % (n=4981) | 0.0167 / 0.430 | -0.67 [-2.69,+1.35] | +0.03 [-0.38,+0.49] |
| Blink, `bi1` (vs stock vLLM) | 93.60 % (n=297) | 0.0282 / 0.729 | 97.43 % (n=5016) | 0.0213 / 0.741 | +0.67 [-1.68,+3.37] | -0.06 [-0.57,+0.45] |

Rune `deterministic` reproduces the reference adapter bit for bit (the 1e-17 is float64 rounding in the softmax of the two code paths). The Blink
row compares the batch-invariant preset with stock vLLM, which flips argmax on its own between passes (6/300 at c1 in the 11 Oct study), so
93.6 % is mostly stock's own noise, not an error of `bi1`; the earlier study found `bi1` itself 0/300 flips. None of the accuracy deltas is
distinguishable from 0.

**Throughput, REAX over reference, same GPU** (decisions/s; c = concurrent clients; 60 s per level after warm-up; P300 one question per
request, DI = the load of that contract: `systemone` models send the 3-6 questions of a state together (200 states), `s1` models one
question per call (1000 states))

| model / preset | P300 c1 | P300 c16 or c32 | c1 p50 latency | DI c16 / c32 | peak GPU memory (device, MiB) ref -> REAX |
|---|---|---|---|---|---|
| pplx-decider FP8 | 8.34 -> 15.6 (**1.87x**) | c16 8.44 -> 20.91 (2.48x); c32 8.44 -> 22.08 (**2.62x**) | 105.8 -> 51.0 ms | c16 1.30 -> 14.86 (11.4x); c32 1.24 -> 14.99 (12.1x) | 69,487 -> 35,293 |
| pplx-decider bf16 | 4.90 -> 10.61 (2.17x, reference stalled in this window; p50 ratio 1.30x) | c16 8.46 -> 12.73 (1.50x) | 105.4 -> 81.1 ms | c16 1.18 -> 9.43 (8.0x) | 72,887 -> 57,295 |
| d3 FP8 (items <= 30k chars) | 8.46 -> 15.77 (**1.86x**) | c16 8.50 -> 21.04 (2.48x); c32 8.51 -> 21.65 (**2.54x**) | 105.1 -> 50.9 ms | c16 1.34 -> 14.82 (11.1x); c32 1.48 -> 14.96 (10.1x) | 76,559 -> 35,293 |
| Quyet `fp8-kv` | 6.00 -> 15.21 (**2.54x**) | c16 6.04 -> 23.02 (3.81x); c32 6.10 -> 20.95 (**3.43x**) | 122.7 -> 51.8 ms | c16 1.65 -> 33.27 (20.2x); c32 1.87 -> 33.51 (17.9x) | 72,837 -> 87,705 (vLLM reserves 0.9 of the card) |
| Quyet `bf16` | 5.74 -> 8.31 (1.45x) | c16 6.02 -> 10.17 (1.69x) | 123.4 -> 97.3 ms | c16 1.69 -> 17.70 (10.5x) | 71,621 -> 87,269 |
| Rune `deterministic` | 15.63 -> 15.84 (1.01x) | c16 46.7 -> 38.84 (**0.83x**) | 62.6 -> 61.9 ms | not measured | 83,719 -> 83,997 |
| Rune `fp8-kv-seqs64` | 15.88 -> 17.39 (1.10x) | c16 45.22 -> 51.34 (1.14x); c32 54.59 -> 55.99 (1.03x) | 61.7 -> 56.2 ms | c1 12.61 -> 19.05 (1.51x); c32 36.0 -> 77.59 (**2.16x**) | 86,151 -> 65,637 |
| Blink `bi1` vs stock vLLM | 70.5 -> 44.0 (0.62x) | c16 238.7 -> 168.8 (0.71x); c32 369.9 -> 260.1 (0.70x) | 14.0 -> 22.5 ms | c1 43.6 -> 33.5 (0.77x); c32 129.1 -> 117.4 (0.91x) | 89,007 -> 90,589 |

How to read these (all of it matters):
- The pplx/d3/Quyet reference servers handle one request at a time, so their c16/c32 rate equals their c1 rate; the ratio at higher
  concurrency is therefore mostly batching. The author's servers also stall now and then on new shapes (one 19 s request in the pplx
  bf16 reference window: 4.9 dec/s at c1 although p50 stayed at 105 ms); the p50 ratios are the robust ones. The DI "11x/18x" figures compare grouped requests (several questions
  per request) against that serial server and are far larger than the P300 ratios; treat them as "what a Decision-Index client sees",
  not as kernel speedups.
- d3 has no input cap (pplx: 8192 tokens). Three P300 items are > 100k characters; uncapped, they dominate a 60 s window of both arms
  (`results/d3/2026-10-16-uncapped.json`: 18x at c1, an artefact). The d3 throughput rows use `--max-chars 30000` (`LOAD_MAX_CHARS`),
  roughly the pplx cap; agreement numbers cover all 300 items.
- The P300 load **replays the same 300 items** for 60 s. Presets with prefix caching (Rune `fp8-kv-seqs64`, Blink both arms, Quyet) see
  cache hits after the first pass (Blink's own 11 Oct notes: 98 % hit ratio in later passes); the cache-free `deterministic` preset and the
  reference adapter do not. The P300 ratio for Rune `fp8-kv-seqs64` and the absolute Blink figures are therefore favourable to caching;
  the DI loads (fewer than one pass per level) are the fairer view of production-like traffic. The 11 Oct studies used a different harness.
- Peak GPU memory is the whole device (sampled every 2 s while loaded and answering). vLLM runs reserve `gpu-memory-utilization` of the card
  whatever the model needs (Blink `bi1-memory30` needed 31 GiB in the 11 Oct study); the native engine used 35 GiB (FP8 weights 28.7 GiB + activations).
- Rune `deterministic` is slower at c16 than the reference adapter (0.83x) although both run the same vLLM flags: the REAX server in front
  of vLLM is a plain threaded HTTP server (`engine/mini_serve.py`), the reference adapter an async server with four workers. Not investigated.
- Rune `fp8-kv-seqs64` on DI: c1 12.6 -> 19.1 dec/s, c32 36.0 -> 77.6 dec/s (2.16x; the 11 Oct study measured 1.79x at c32 with another harness).
- Blink `bi1` is slower than stock vLLM at every level (0.62-0.91x): that is the price of batch invariance, as in the 11 Oct study (70 % / 73 %).

**Not measured today:** Quyet `fp8`/`nvfp4` presets, Rune `previous-f`/`stock-cache-async`, Blink `bi1-memory30` throughput, Plumb-4B and the CPU
models (other slices), the official JevBench capability scorer, c1 determinism (repeat-pass flips) of the REAX arms, long-context (> 8k token) behaviour of d3
beyond the three P300 items, and anything on H100/other GPUs.

## Licences

All Apache-2.0 except the router (MIT); Blink: Apache-2.0 per its model card tag (no LICENSE file in the HF repo). Keep each model's LICENSE/NOTICE when redistributing: Quyet's NOTICE starts with
"Quyet by Chinh Nguyen"; d3: `upstream/d3/LICENSE` + `NOTICE` (vLLM Semantic Router Team); pplx-decider: Perplexity AI, derived from
Qwen3.8-27B. Rune derives from Google Gemma 4 (Apache-2.0).

## What the unit tests cover, and what still needs a GPU

`pytest tests` covers the registry, prompts, answer shaping, argv/presets, the bench harness and `bench/pod_validate.sh` against
stand-ins, with no weights. GPU validation of pplx-decider, d3, Quyet, Rune and Blink is in "Measured on 16 Oct 2026" above; what remains
unvalidated there is listed under "Not measured today". Reproduce a row with `bench/pod_validate.sh <model-id>` (see `bench/README.md`):
two models can run at once on two GPUs (`GPU=0|1`, distinct `PORT_OFF` per concurrent run - two runs with the same `PORT_OFF` collide on
the host ports and give garbage).
