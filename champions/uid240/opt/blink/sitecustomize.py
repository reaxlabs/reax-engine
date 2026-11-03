# SPDX-License-Identifier: Apache-2.0
# Blink engine: loaded by every Python process of the image (PYTHONPATH=/opt/blink). It installs
#  - the numerics pin (blink_pin.py): fixed Inductor kernel configs, nothing timed at start;
#  - convoy batching in the engine core (blink_core.py, BLINK_CONVOY=1).
# Both act only in the process that imports the patched module (the spawned vLLM engine core).
import os
import sys

if os.environ.get("BLINK_PIN", "1") == "1":
    try:
        import blink_pin  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[blink-pin] not installed: {exc!r}", file=sys.stderr, flush=True)
if os.environ.get("BLINK_RED", "0") == "1":   # K2: exact one-pass red_fused_add_bi_mm_mul_rms_norm_0/_2 (blink_red.py)
    try:
        import blink_red  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[blink-red] not installed: {exc!r}", file=sys.stderr, flush=True)
if (os.environ.get("BLINK_CONVOY", "0") == "1" or os.environ.get("BLINK_PIPE", "0") == "1"
        or os.environ.get("BLINK_TUNE_FILE") or os.environ.get("BLINK_CORE_TRACE")):
    try:
        import blink_core  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[blink-core] not installed: {exc!r}", file=sys.stderr, flush=True)
if any(os.environ.get(v, "0") == "1" for v in ("BLINK_HOST_NOMETA", "BLINK_HOST_LEAN", "BLINK_HOST_DSAMP",
                                              "BLINK_HOST_DFREE", "BLINK_HOST_XDIET", "BLINK_HOST_LATEFREE",
                                              "BLINK_HOST_UNIFY")) \
        or os.environ.get("BLINK_HOST_SWITCH_US", "0") not in ("", "0"):
    try:        # bj/blink4 + bj/blink5 (team ruj): exact host-side levers of the engine core (blink_host.py)
        import blink_host  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[blink-host] not installed: {exc!r}", file=sys.stderr, flush=True)
if any(os.environ.get(v, "0") == "1" for v in ("BLINK_MOE_HEAD", "BLINK_MOE_TAIL", "BLINK_MOE_GEMM")):
    try:
        import blink_moe  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[blink-moe] not installed: {exc!r}", file=sys.stderr, flush=True)
if os.environ.get("BLINK_BI_TILES", "0") == "1" or os.environ.get("BLINK_LABEL_HEAD", "0") == "1":
    try:
        import blink_bi  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[blink-bi] not installed: {exc!r}", file=sys.stderr, flush=True)
if os.environ.get("BLINK_PYPROF_DIR"):      # development only
    try:
        import blink_prof  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[blink-prof] not installed: {exc!r}", file=sys.stderr, flush=True)
if os.environ.get("BLINK_PAR", "0") == "1":
    try:
        import blink_par  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[blink-par] not installed: {exc!r}", file=sys.stderr, flush=True)
if os.environ.get("BLINK_RTR", "0") == "1":
    try:
        import blink_rtr  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[blink-rtr] not installed: {exc!r}", file=sys.stderr, flush=True)
if os.environ.get("BLINK_READ", "0") == "1":        # E64 B3: label-only readout through the sampler
    try:
        import blink_read  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[blink-read] not installed: {exc!r}", file=sys.stderr, flush=True)
if os.environ.get("BLINK_FINE_FILTER", "0") == "1":  # E64 B1 A/B only: pad to the LADDER sizes when fine_pad is false
    try:
        import blink_fine  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[blink-fine] not installed: {exc!r}", file=sys.stderr, flush=True)
# v2d (V1): the picture path. VIS_FUSE=1: our exact Gemma4 vision-tower kernels and encoder graphs (/opt/vision/vfuse.py,
# e61-vision:v5). BLINK_VISION=1: P-mode (blink_pmode.py: picture steps run the eager vision reference's numerics);
# imported LAST so that its model-runner wrapper is the outermost one. A failed P-mode install is fatal for picture
# requests: the frontends refuse them (blink_engine.py checks blink_pmode in the engine core through the census).
if os.environ.get("VIS_FUSE", "0") == "1":
    try:
        if "/opt/vision" not in sys.path:
            sys.path.append("/opt/vision")
        import vfuse  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[vfuse] not installed: {exc!r}", file=sys.stderr, flush=True)
if os.environ.get("BLINK_VISION", "0") == "1":
    try:
        import blink_pmode  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[blink-pm] not installed: {exc!r}", file=sys.stderr, flush=True)
