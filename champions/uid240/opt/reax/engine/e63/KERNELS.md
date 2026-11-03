# E63 mammoth engine: kernels and their sources

Nothing in /opt/e63 is precompiled: every kernel below is JIT-compiled at engine start from the source named here.

| kernel | source | what it does |
|---|---|---|
| `_ba_kernel` (Triton) | /opt/e63/e63_engine.py `_ba_kernel` / `ba_linear` | GatedDeltaNet in_proj_b\|a GEMM (bf16 in/out, fp32 accumulation, fixed 32x128x128 tile, fixed K order: batch-invariant) |
| FLA fla-core 0.5.2 kernels (conv1d, l2norm, chunk_local_cumsum, kkt_solve, recompute_w_u, fwd_h, fwd_o) | base image /opt/venv/lib/python3.12/site-packages/fla (MIT) | unchanged; called with static chunk indices / offsets; autotune replaced by the configs in /opt/e63/fla_pins.json |
| REAX Engine kernels (`_gnq_kernel`, `_smqs_kernel`, compiled rmsnorm/quant helpers via torch.compile) | base image /opt/reax/engine/pplx_engine.py | unchanged |
| FP8 GEMMs (`torch._scaled_mm`: CUTLASS rowwise, cuBLASLt tensorwise), varlen flash attention (`torch.nn.attention.varlen.varlen_attn`, num_splits=1) | PyTorch 2.14.0+cu130 in the base image | unchanged |

CUDA graphs: /opt/e63/e63_engine.py `E63Graph.capture` records the whole step per (token bucket, sequence bucket) at start.
