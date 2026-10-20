"""Experimental, env-gated vLLM 0.30.0 patches (pod only):
REAX_BI_ALLOW_MOE=ClassA,ClassB : these MoE expert classes pass the batch-invariance gate (to test empirically whether they are invariant)
REAX_BI_KEEP_LINEAR=1           : VLLM_BATCH_INVARIANT does not force the NVFP4 linear backend (auto selection stays)"""
import pathlib, vllm.model_executor.layers.fused_moe.modular_kernel as mk, vllm.model_executor.kernels.linear as lin
p = pathlib.Path(mk.__file__); s = p.read_text()
old = "        elif envs.VLLM_BATCH_INVARIANT and not cls._supports_batch_invariance():"
new = ("        elif envs.VLLM_BATCH_INVARIANT and not cls._supports_batch_invariance() and cls.__name__ not in "
       "__import__('os').environ.get('REAX_BI_ALLOW_MOE', '').split(','):")
if "REAX_BI_ALLOW_MOE" not in s:
    assert s.count(old) == 1; p.write_text(s.replace(old, new)); print("patched", p)
p = pathlib.Path(lin.__file__); s = p.read_text()
old = "    if envs.VLLM_BATCH_INVARIANT:\n        bi_supported, reason = CutlassNvFp4LinearKernel.is_supported()"
new = ("    if envs.VLLM_BATCH_INVARIANT and __import__('os').environ.get('REAX_BI_KEEP_LINEAR') != '1':\n"
       "        bi_supported, reason = CutlassNvFp4LinearKernel.is_supported()")
if "REAX_BI_KEEP_LINEAR" not in s:
    assert s.count(old) == 1, "linear anchor"; p.write_text(s.replace(old, new)); print("patched", p)
