# DeepInfer -> REAX Engine technique map (source review, 20 Oct 2026)

Sources reviewed (read only, nothing executed on Sandy): deepseek-ai/DeepInfer @ 8b16149 (Apache-2.0),
deepseek-ai/deepseek-recipe @ 935817c (MIT), vendored deps deepseek-ai/DeepGEMM (MIT) and deepseek-ai/TileKernels (MIT) at HEAD.

Hard constraints that decide most rows:
- DeepInfer supports **one model (DeepSeek-V4.1-Flash)** and **NVIDIA SM100 (B200/GB200) or Ascend 950**. Our engine GPU is the
  RTX PRO 6000 Blackwell **Server Edition = SM120**. DeepGEMM and TileKernels list "SM90 or SM100" only; DeepGEMM's SM100 kernels use
  tcgen05/TMEM instructions that SM120 does not have. FlashMLA is SM90/SM100 and MLA-specific.
- REAX is prefill-only (one forward pass + option-letter readout). Everything about decoding (MTP/DSpark drafts, confidence-scheduled
  verification, samplers, structured outputs, KV-cache block manager for decode, PD disaggregation) does not apply.

| # | DeepInfer technique | where | REAX setting | applicable? | verdict |
|---|---|---|---|---|---|
| 1 | DeepGEMM FP8/FP4/BF16 GEMMs, m-grouped contiguous MoE GEMM, Mega-MoE (fused dispatch+GEMM+combine) | vendor/DeepGEMM, v4_modules/mlp.py | native Qwen3.5 GEMMs (torch._scaled_mm, cuBLASLt), MoE models via vLLM | **no** on SM120 (SM90/SM100 only, tcgen05) | not portable; idea "M-independent tiles" used in #8 |
| 2 | FlashMLA (dense + sparse MLA decode/prefill) | vendor/FlashMLA, attention_backend.py | none of our models uses MLA | no | skip |
| 3 | DeepEP expert-parallel all-to-all, EPLB | vendor/DeepEP, eplb/ | single GPU | no | skip |
| 4 | TileKernels: fused SwiGLU + per-token FP8 cast, norm + cast, MoE top-k gate | tile_kernels.quant / moe | REAX already has fused add+RMSNorm+FP8 quant, SiLU*mul+quant, gated-norm+quant, NVFP4 static-quant fusions (engine/pplx_engine.py) | already done in REAX | no gain expected |
| 5 | DeepSeek FP8 recipe: 1x128 activation / 128x128 weight block scales; FP8 act x MXFP4 weights (W4A8) | mlp.py, quant.py | REAX FP8 = per-token x per-channel; NVFP4 = W4A4 | partly: SM120 has MXFP8 block-scaled MMA; W4A8 runs at FP8 rate on SM120 (no compute gain in compute-bound prefill) | MXFP8 measured as a GEMM microbench (quality lever, not speed) |
| 6 | SWA bounded replay (YOCO prefill truncation: consumer layers only process a bounded suffix) | v4_modules/yoco.py | needs a YOCO architecture (n_prefill_layer < n_layer). Prefill-only analogue = compute only readout rows where nothing downstream needs the others | the generic analogue is **already in REAX** ("last-layer pruning": final layer computes Q/o_proj/MLP/readout only for readout tokens). Gemma-4 models end in a full-attention layer, so no deeper truncation is possible | no new gain |
| 7 | Asynchronous (zero-overhead) batch scheduler, C++ scheduler | csrc/engine/scheduler.hpp | REAX server idle measured at 1.4 % of wall (5x job R0) | no lever | skip |
| 8 | **Batch-invariant inference** by deterministic kernels with fixed, batch-size-independent tiling (DeepGEMM `use_deterministic_algorithms`, FlashMLA `enable_batch_invariant`) | worker/runtime_config.py | Rune / Blink deterministic presets run vLLM `VLLM_BATCH_INVARIANT=1`, which swaps in slow fallback kernels (fixed 64x64x32 Triton MoE tile; NVFP4 forced to unfused CUTLASS) | **yes, as an idea**: any fixed M-independent tile is batch-invariant; pick a fast one | **prototyped: P1 (Rune) and P2 (Blink)** |
| 9 | Prefill packing with token budget + `prefill_max_sum_seqlen_k` (attention-cost budget) | config.py, scheduler.hpp | REAX packs varlen steps by token budget + splitting of oversized shared requests | already similar | skip |
| 10 | Context parallel (balanced CP), attention DP | parallel_utils | single GPU per model | no | skip |
| 11 | KV/prefix reuse | (KV cache system "not included in this release") | REAX shared-state mode (pplx), vLLM prefix caching (Rune/Blink presets) | nothing released to reuse | skip |
| 12 | mHC / Engram / lightning indexer / sparse attention | v4_modules | V4-specific architecture | no | skip |
| 13 | deepseek-recipe (Rust prompt/API-format conversion, tool-call parsing) | deepseek-recipe | REAX has its own prompt encoders | no speed relevance | skip |
| 14 | CUDA-graph timer / PDL (programmatic dependent launch) | csrc/graph_timer, set_pdl | REAX native path launches large prefill kernels; launch overhead small at 9k tokens/step | marginal | not prototyped |

## Measured on 20 Oct 2026 (RTX PRO 6000 SM120, see RESULTS.md)
| prototype | model | speed | answers | status |
|---|---|---|---|---|
| P1 fixed 128x128x64 batch-invariant MoE tile (row 8) | Rune `deterministic` | full DI-1000 set +12.5-16 %, c1 p50 60 -> 50 ms | bit-identical | shipped (this PR) |
| P1 same | Rune `fp8-kv-seqs64` | +14 % fixed workload, +6 % DI c32 | bit-identical | shipped (this PR) |
| P2 FlashInfer CUTLASS NVFP4 MoE under batch invariance (row 8) | Blink `bi1` | (not measured) | NOT batch-invariant (281/297 differ c1 vs c32) | rejected |
| P2b keep stock dense NVFP4 linear under batch invariance | Blink `bi1` | 102.05 vs 102.33 dec/s DI c32 (no gain) | invariant | rejected |
| P3 MXFP8 1x32 block-scaled GEMM (row 5) | GEMM microbench, pplx/Plumb shapes | 2.0-4.4 % slower than per-token FP8 | same error on random data | not pursued (quality tool only) |
Native models (pplx-decider, d3, Plumb-4B, incl. the s1-fast exact reference): no DeepInfer technique applies beyond what the engine
already does (rows 4, 6, 7, 9), so nothing was changed there.
