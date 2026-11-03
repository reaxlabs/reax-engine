# SPDX-License-Identifier: Apache-2.0
# Blink vision engine: loaded by every Python process of the image (PYTHONPATH=/opt/vision). Both patches act when the
# patched module is imported (the vision tower and linear layers live in the spawned engine core):
#  - BLINK_BI_TILES=1: blink_bi.py, per-M configs for vLLM's batch-invariant dense matmul (Triton tiles / our CUTLASS
#    sm80 GEMM, bit-identical), table bi_tiles.json (Blink text shapes + the Gemma4 vision tower shapes);
#  - VIS_FUSE=1: vfuse.py, exact fused element-wise kernels of the Gemma4 vision tower;
#  - VIS_E5 / VIS_E3 / VIS_E4 = 1 (V3 levers, each default off): whole-encoder graph, fused GEMMs, norm chains (vfuse.py).
import os
import sys

if os.environ.get("BLINK_BI_TILES", "0") == "1":
    try:
        import blink_bi  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[blink-bi] not installed: {exc!r}", file=sys.stderr, flush=True)
if any(os.environ.get(k, "0") == "1" for k in ("VIS_FUSE", "VIS_E5", "VIS_E3", "VIS_E4")):   # V3: E5 / E3 / E4 levers
    try:
        import vfuse  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[vfuse] not installed: {exc!r}", file=sys.stderr, flush=True)
