"""Patch vLLM 0.30.0 fused_moe.get_default_config: under VLLM_BATCH_INVARIANT, use the fixed (M-independent) tile from
$REAX_BI_MOE_CONFIG (JSON) when set; otherwise unchanged (64/64/32). Idempotent. A fixed tile keeps batch invariance:
every output row is computed by the same K-loop order regardless of how many tokens are in the batch."""
import pathlib, sys, vllm.model_executor.layers.fused_moe.fused_moe as m
p = pathlib.Path(m.__file__); s = p.read_text()
old = '''    if envs.VLLM_BATCH_INVARIANT:
        return {
            "BLOCK_SIZE_M": 64,'''
new = '''    if envs.VLLM_BATCH_INVARIANT:
        import json as _j, os as _o
        if _o.environ.get("REAX_BI_MOE_CONFIG"):
            return dict(_j.loads(_o.environ["REAX_BI_MOE_CONFIG"]))
        return {
            "BLOCK_SIZE_M": 64,'''
if "REAX_BI_MOE_CONFIG" in s: print("already patched", p); sys.exit(0)
assert s.count(old) == 1, "anchor not found"
p.write_text(s.replace(old, new)); print("patched", p)
