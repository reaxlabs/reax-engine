# GGUF decision models on CPU (llama.cpp)

Backend: `engine/backends/gguf_cpu.py`. Models (pinned file, revision and sha256 in `SPECS`):

| id | model | file | readout | temperature |
|---|---|---|---|---|
| `gutsy-0.8b` | `kouhxp/gutsy` @ f9995fe2 (Qwen3.5-0.8B base, Apache-2.0) | `gutsy-0.8b-v04-q8_0.gguf` | `gutsy` (author's prompt, reject slot Z on choice/score) | choice 1.0628, score 1.0, yes/no 0.7652 |
| `plumb-4b-q8` | `crh225/plumb-4b-GGUF` @ 71c1c57e | `plumb-4b-v5-Q8_0.gguf` | `plumb` (same prompt as the GPU Plumb-4B engine) | 2.07 |

One decision = one prompt evaluation from an empty context, then the option letters' logits at the last position,
divided by the temperature, softmax. Prompts, letters and temperatures are ported unchanged from the REAX subnet GGUF
runtime (`gguf_readouts.py`), so the probabilities are bit-identical to that runtime on the same llama.cpp build
(llama-cpp-python 0.3.36, portable AVX2 build, see `docker/gguf-cpu/Dockerfile`).

Tuning knobs (environment): `GGUF_CONTEXTS` (independent llama contexts, default 1), `GGUF_THREADS` (threads per
context, default physical cores / contexts), `GGUF_N_BATCH` (512), `GGUF_N_UBATCH`.

## Measured (shared AMD Ryzen 5 3600, 8 logical CPUs, no network, 16 Oct 2026)

Every layout tried (1x8, 2x4, 4x2, 8x1, 1x6, 2x3 contexts x threads) returned probabilities **bit-identical** to the
reference runtime (59 of 59 items per layout, plus 298 of 298 and 59 of 59 for the two long runs). Throughput is
bounded by CPU compute (prefill), so several contexts do not beat one wide context: on a busy shared machine the
differences between layouts (0.34 to 0.59 decisions/s on Gutsy) are within run-to-run noise. Default: one context with
one thread per physical core. Reusing the constant prefix (system prompt and template head, 42 tokens on Gutsy) from a
saved state gives identical logits (100 of 100 items) but is not faster, so it is not used.
