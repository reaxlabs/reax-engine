# /opt/jeffan: team ruj changes on pjs's Jeffan engine (bj/jeffan round 1, image bj-g2h, 28 Oct 2026)

The Jeffan engine is **pjs's** (E61 / E62: `jeffan_engine.py`, `jef_pin.py`, `pins.json`, `alt/`; in-process vLLM 0.30.0,
online FP8 weights + FP8 KV cache, `VLLM_BATCH_INVARIANT=1`, pinned Inductor configs). This image runs it pinned to
**pjs's J1 table** (`alt/a1_cakg1.json`, his e62v1hJ1: the board reference's start variant in 5 of 6 of his sessions).
Team ruj added three levers, all **bit-exact** (the same answer bits as pjs's engine on the same pin table: 1024/1024
timed answers, 24/24 c1 answers and the c1 re-run identical in our screens), on by default through `sitecustomize.py`
(`_RUJ_DEFAULTS`; a variable set in the container still wins). Everything outside `/opt/jeffan` is pjs's e62v1h.

| switch | lever | what it changes |
|---|---|---|
| `JEF_GEMM=1` | G: own FP8 GEMMs | vLLM's `_C::cutlass_scaled_mm` (the online-FP8 qkv / o / gate_up / down GEMMs, 86 % of the GPU time of a step) runs on our CUTLASS SM120 kernels (`gemm/`, built at image build time, see `KERNELS.md`) with the CTA tile chosen per (N, K, M) from `gemm_table.json` (14 tile configurations) instead of vLLM's single batch-invariant 64x64x128 configuration. Same mainloop (16x8x32 e4m3 MMA, fp32 accumulation over K in increasing order, no split-K) and the same epilogue (`scale_a * (scale_b * acc)`, round-to-nearest bf16): every configuration gives the same bits as vLLM's kernel (GPU-checked for all model shapes, M = 16..4096). ~700-790 TFLOPS at M >= 512 vs ~470 |
| `JEF_ROWS=1` | readout over the code rows | the logits are projected over the 255 answer-code rows of the tied embedding instead of all 262,144 (same batch-invariant matmul, same soft-cap ops); the other columns are masked to -inf by `allowed_token_ids` anyway |
| `JEF_ASYNC=0`, `JEF_BF_MS=6` | BF: batch formation | vLLM's synchronous scheduler; after each step the engine core waits up to 6 ms (from the end of that step) for as many new requests as the step finished (the re-reads and next questions of the 8 clients), so their reads form one large step instead of a large step followed by one-read steps |

Also in this tree, off unless set: `JEF_HEDGE` (re-read threshold hedge, screens), `JEF_TRACE`, `JEF_STEPLOG`,
`JEF_PROF_DIR`, `JEF_SPEC2`.

Screens (box RTX PRO 6000, runsc, the pinned jeffan reference resident and paused, c8, 4 x 256 items, 28 Oct, session
p3): pjs e62v1c (control) 24.65 dec/s; bj-g1h image (G 10 configs + ROWS) 32.40 (x1.314); G 14 configs + ROWS 32.76
(x1.329); **+ BF 33.66 (x1.365)**; answers identical to the control in every case. Board #341 (bj-g1h): jeffan 1.528x
in session 1 (session 2 failed the item gate on a reference start that did not match the J2 pins; hence J1 here).

Licence: our files Apache-2.0 (as pjs's engine); CUTLASS headers BSD-3-Clause (NVIDIA Corporation), from the base image.

## E62 note (image v1u and later)

This tree is #348's with E62's merge edits; jef_levers.py is team ruj's file from board image #373 (lever SB,
`JEF_STEP_TOK`, and one-group batch formation, `JEF_BF_ALL`; its NV4 hooks stay off, and the native libraries
are built from #348's sources). The switches this image runs with are set in /opt/e62/dispatch.py, not by the
defaults above: `JEF_STEP_TOK=1024`, `JEF_BF_ALL=1`, `JEF_BF_MS=10`, `JEF_GATHER_MS=0`, the re-read threshold
hedge with E62's capped shift (`JEF_HEDGE=0.01`, `JEF_HEDGE_CAP=0.5`), J6 (`JEF_PAR_REREAD=1`) and FULL CUDA
graphs up to 1,024 tokens (`JEF_CG_MODE=FULL`).
