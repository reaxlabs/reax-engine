# SPDX-License-Identifier: LicenseRef-REAX-Engine-1.1 AND Apache-2.0
# REAX Engine code (LICENSE). It edits one function of vLLM 0.30.0 (Apache License 2.0, Copyright the vLLM project) inside the
# image at build time; the edited vLLM file stays under Apache-2.0 (modification: the batch-invariant default MoE tile can be set
# from the environment).
"""Faster batch-invariant MoE tile for vLLM-backed MoE models (Rune 26B-A4B).

Under VLLM_BATCH_INVARIANT=1, vLLM 0.30.0 ignores its tuned fused-MoE configs and runs every Triton MoE GEMM with one small fixed
tile (BLOCK 64x64x32, default warps/stages): `get_default_config` in vllm/model_executor/layers/fused_moe/fused_moe.py. A fixed
tile is what makes the kernel batch-invariant, not its size: each output element is the same K-loop over the same fp32
accumulator whatever the number of tokens. The idea follows DeepSeek's DeepInfer, whose batch-invariant mode keeps its fast
kernels with fixed, batch-size-independent schedules instead of swapping in slow fallbacks (no DeepInfer code is used here).

This module rewrites that one branch so that $REAX_BI_MOE_CONFIG (JSON) replaces the 64x64x32 tile when it is set. Unset, vLLM
behaves exactly as shipped. Batch invariance holds by construction (fixed tile, SPLIT_K 1). Identical bits to the 64x64x32 tile
were measured, not proven (BLOCK_K changes too): 20 Oct 2026, RTX PRO 6000 Blackwell (SM120), pinned vLLM 0.30.0, Rune
`deterministic` preset, 3 patched vs 2 unpatched runs: bit-identical answers (P300 297/297, DI-1000 4,981/4,981), the full DI-1000
set +12.5-16 % faster, c1 p50 latency -15 to -17 % (RESULTS.md, results/rune-26b-a4b/2026-10-20-bi-moe-tile/). Re-check the answers on any
other GPU, Triton version or model.

    python3 -m engine.backends.vllm_patch            apply to the installed vLLM (Docker build step; idempotent)
    python3 -m engine.backends.vllm_patch --check    exit 0 if the installed vLLM is patched
"""
from __future__ import annotations

import json
import sys

# The tile measured on SM120 (Rune shapes: hidden 2816, 128 experts, moe_intermediate 704, top-8): BLOCK 128x128x64, 8 warps, 3 stages.
BI_MOE_TILE = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "SPLIT_K": 1,
               "num_warps": 8, "num_stages": 3}
ENV = "REAX_BI_MOE_CONFIG"
VLLM_VERSION = "0.30.0"

_ANCHOR = '''    if envs.VLLM_BATCH_INVARIANT:
        return {
            "BLOCK_SIZE_M": 64,'''
_REPLACEMENT = '''    if envs.VLLM_BATCH_INVARIANT:
        import json as _reax_json, os as _reax_os  # REAX Engine patch (engine/backends/vllm_patch.py)
        if _reax_os.environ.get("REAX_BI_MOE_CONFIG"):
            return dict(_reax_json.loads(_reax_os.environ["REAX_BI_MOE_CONFIG"]))
        return {
            "BLOCK_SIZE_M": 64,'''


def env_value() -> str:
    """The preset environment value: compact JSON of BI_MOE_TILE."""
    return json.dumps(BI_MOE_TILE, separators=(",", ":"))


def patch_text(src: str) -> str:
    """fused_moe.py source -> patched source. Idempotent; raises if the anchor is not found exactly once."""
    if ENV in src:
        return src
    if src.count(_ANCHOR) != 1:
        raise RuntimeError("vllm_patch: batch-invariant MoE anchor not found exactly once; vLLM version changed?")
    return src.replace(_ANCHOR, _REPLACEMENT)


def _target():
    import vllm
    from vllm.model_executor.layers.fused_moe import fused_moe
    return vllm.__version__, fused_moe.__file__


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    version, path = _target()
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if "--check" in argv:
        ok = ENV in src
        print(f"vllm {version}: {'patched' if ok else 'NOT patched'} ({path})")
        return 0 if ok else 1
    if version != VLLM_VERSION:
        print(f"vllm_patch: vLLM {version} != {VLLM_VERSION}; not patching (the measured tile is for {VLLM_VERSION})")
        return 2
    new = patch_text(src)
    if new != src:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)
    print(f"vllm {version}: patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
