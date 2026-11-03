# Third-party notices

REAX code is MIT (LICENSE); fast_fp8_gemm.py stays Apache-2.0 as a derivative of finegrained-fp8. Third-party code below keeps its own licence; the root licence does not relicense it.

| Component | Source and pin | Licence | Treatment |
|---|---|---|---|
| KnowLine frontend/layout | PelaAI/KnowLine-31B-Gen1, 8f052a8130e599b53e5324c34f0f5ff90696e1c6 | MIT, file headers; adapted from llm2jev0.6.1/OpenJev, AnyJev contributors | Unchanged files in third_party/knowline; native temperatures |
| Finegrained FP8 | kernels-community/finegrained-fp8, 81ed101eb51b0801a4d9469541eda6ef73b0b6f6, kernel5fe9dbb3d45f6f2652958c1f70e5381891fa7aa0, version4 | Apache-2.0, upstream metadata.json | Unchanged Python/Triton in third_party/finegrained-fp8, all11 digests verified; includes i64 addressing |
| Finegrained FP8 GEMM, derived | fast_fp8_gemm.py, derived from w8a8_block_dynamic_fp8_matmul in the finegrained-fp8 row above (copyright 2025 The HuggingFace Inc. team) | Apache-2.0 | Modified in this package: activation quantization split into its own kernel, GEMM rewritten around it, own tile parameters, routing wrapper; arithmetic unchanged, bits compared in test_fast_fp8_gemm.py. Copyright line, licence notice and statement of changes are in the file header |
| d3 native runtime | vllm-sr/d3-ultra, 52ba9e7a00402f9560c7fba8c039608e7ddee7fb | Apache-2.0, repo LICENSE | Loaded from part-q; not vendored or relicensed here; keep full upstream licence and model metadata |
| causal-conv1d | Dao-AILab/causal-conv1d v1.7.0 / PyPI1.7.0 | BSD-3-Clause | Pinned dependency; installed with upstream licence metadata |
| Decision Index kit | apolinario/decision-index, 25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d | MIT, Copyright2026 Apolinário Passos | External kit; installed/supplied separately; no kit source distributed inside engine/ |
| PyTorch / Transformers / FLA / Triton | Exact runtime versions in Dockerfile and run pip-freeze evidence | Respective upstream BSD/Apache/MIT licences | External dependencies, not relicensed |

No UID240/miner projection code is required or bundled by this mode. Model weights have separate upstream and derivative-model obligations; the code licence does not license the weights. Do not remove upstream licences or credit files when assembling part-g and part-q. Stand-in repositories are pinned for reproducibility and do not represent REAX1.3 release weights.
