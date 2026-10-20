# Lead observations not in files (copied from the lead's terminal, 20 Oct 2026)
Before arm rC the lead restored vLLM's original fused_moe.py from /workspace/fused_moe.py.orig (grep -c REAX_BI_MOE -> 0), then ran this repo's module:
  $ python3 -m engine.backends.vllm_patch        -> "vllm 0.30.0: patched /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe.py"
  $ python3 -m engine.backends.vllm_patch --check -> "vllm 0.30.0: patched (...fused_moe.py)"
  preset env printed by V.preset_for("rune","deterministic").env -> {'VLLM_BATCH_INVARIANT': '1', 'REAX_BI_MOE_CONFIG': '{"BLOCK_SIZE_M":128,...,"num_stages":3}'}
During rC's load phase: tr '\0' '\n' < /proc/<EngineCore pid>/environ | grep -E "REAX_BI|VLLM_BATCH" ->
  REAX_BI_MOE_CONFIG={"BLOCK_SIZE_M":128,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":64,"GROUP_SIZE_M":8,"SPLIT_K":1,"num_warps":8,"num_stages":3}
  VLLM_BATCH_INVARIANT=1
The Blink arms used harness/patch_vllm_bi2.py (env-gated; inert unless REAX_BI_ALLOW_MOE / REAX_BI_KEEP_LINEAR set); bX1 = bi1 + REAX_BI_ALLOW_MOE=FlashInferExperts, bX3 = bi1 + REAX_BI_KEEP_LINEAR=1.
The Docker build of docker/vllm-readout was not run.
- bX1 load numbers contaminated by overlapping bX3 start (discarded; invariance fetches were before overlap).
