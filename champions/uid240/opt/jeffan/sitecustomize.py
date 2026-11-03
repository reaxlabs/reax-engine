# SPDX-License-Identifier: Apache-2.0
# Jeffan engine: loaded by every Python process of the image (PYTHONPATH=/opt/jeffan). It installs the numerics pin
# (jef_pin.py: fixed Inductor kernel configs and combo decisions, nothing timed at start). It acts only in the process
# that imports the patched Inductor modules (the spawned vLLM engine core).
import os
import sys

# (ruj) lever defaults of this image; a variable already set in the container wins. Empty in the screening copy.
_RUJ_DEFAULTS: dict = {"JEF_GEMM": "1", "JEF_ROWS": "1", "JEF_ASYNC": "0", "JEF_BF_MS": "6",
                       "JEF_PIN_TABLE": "/opt/jeffan/alt/a1_cakg1.json",
                       # (ruj line J4) GQ (jef_gq.py via jef_levers.py) + FT (jeffan_engine.py, jef_levers.py); round 2:
                       # LF + EM (jeffan_engine.py), line B's host levers (jef_host.py) and OC (vLLM output chunk 1)
                       "JEF_GQ": "1", "JEF_FT": "1", "JEF_LF": "1", "JEF_EM": "1",
                       "JEF_HOST_NOMETA": "1", "JEF_HOST_LEAN": "1", "JEF_HOST_DSAMP": "1", "JEF_HOST_DFREE": "1",
                       "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE": "1",
                       # (ruj line J5) FR + TF (jeffan_engine.py); round 2: SUB (jef_sub.py)
                       "JEF_J5_FR": "1",
                       "JEF_J5_TF": "1",
                       "JEF_J5_SUB": "1"}
for _k, _v in _RUJ_DEFAULTS.items():
    os.environ.setdefault(_k, _v)

if os.environ.get("JEF_PIN", "1") == "1":
    try:
        import jef_pin  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[jef-pin] not installed: {exc!r}", file=sys.stderr, flush=True)
if os.environ.get("JEF_STEPLOG"):   # screens only (ruj): per-step GPU time log, see jef_steplog.py
    try:
        import jef_steplog  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[jef-step] not installed: {exc!r}", file=sys.stderr, flush=True)
if (os.environ.get("JEF_ROWS", "0") == "1" or os.environ.get("JEF_GEMM", "0") == "1"
        or float(os.environ.get("JEF_BF_MS", "0") or 0) > 0
        or os.environ.get("JEF_GQ", "0") == "1"):   # (ruj) levers, see jef_levers.py
    try:
        import jef_levers  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[jef-lever] not installed: {exc!r}", file=sys.stderr, flush=True)
if os.environ.get("JEF_KV16", "0") == "1":   # (E62) lever KV16: 16-token prefix-cache granularity, see jef_kv16.py
    try:
        import jef_kv16  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[jef-kv16] not installed: {exc!r}", file=sys.stderr, flush=True)
if any(os.environ.get(k, "0") == "1" for k in ("JEF_HOST_NOMETA", "JEF_HOST_LEAN", "JEF_HOST_DSAMP", "JEF_HOST_DFREE")):
    try:   # (ruj line J4 round 2) after jef_levers / jef_kv16: jef_host's import finder guards the shared targets
        import jef_host  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[jef-host] not installed: {exc!r}", file=sys.stderr, flush=True)
if os.environ.get("JEF_J5_SUB", "0") == "1":
    try:   # (ruj line J5 round 2) after jef_levers / jef_kv16 / jef_host: sub-block prefix-cache hits, see jef_sub.py
        import jef_sub  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[jef-sub] not installed: {exc!r}", file=sys.stderr, flush=True)
