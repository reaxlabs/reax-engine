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

## Quick start
```bash
MODEL_DIR=./model bash tools/setup.sh
python engine/server.py --model ./model --backend engine --linear fp8 --compile 1 --gva 1 --port 8741
curl -s localhost:8741/v1/systemone -H 'content-type: application/json' -d '{"model":"pplx","state":"Order #123 arrived broken.",
  "questions":{"team":{"type":"choice","instructions":"Which team handles this?","criteria":{"billing":"refunds","support":"technical issues","sales":"buying"}}}}'
```
See DEPLOY.md for hardware, memory and operational notes. Validated on RTX PRO 6000 Blackwell (sm_120); also measured
on H100, H200, B200 and RTX 5090 (32 GB cards: `CIE_LAZY_LOAD=1`).
