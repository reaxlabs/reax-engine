# SPDX-License-Identifier: Apache-2.0
"""Development A/B switch for finer CUDA-graph sizes (BLINK_FINE_FILTER=1, needs BLINK_TUNE_FILE).

With BLINK_GRAPH_SIZES = the engine's LADDER plus extra sizes (e.g. every 32 tokens), the tune key "fine_pad": false
makes vLLM's CUDA-graph dispatcher pad each batch to the next LADDER size instead of the next captured size, exactly as
an engine started without the extra sizes would; "fine_pad": true (default) leaves the dispatcher alone. Both behaviours
then run in one engine start. The padded rows are independent of the real ones (every kernel is row-invariant), so the
answers do not depend on the choice. Installed by sitecustomize.py; patches vllm.v1.worker.gpu.cudagraph_utils.
"""
import bisect
import importlib.abc
import json
import os
import sys
import time

_TARGET = "vllm.v1.worker.gpu.cudagraph_utils"
TUNE_FILE = os.environ.get("BLINK_TUNE_FILE", "")
LADDER = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1152,
          1280, 1408, 1536, 1664, 1792, 1920, 2048, 2304, 2560, 2816, 3072, 3584, 4096, 4608, 5120, 6144, 7168, 8192]
_S = {"fine": os.environ.get("BLINK_FINE_PAD", "1") == "1", "t": 0.0}


def _fine():
    if TUNE_FILE:
        now = time.monotonic()
        if now - _S["t"] > 0.5:
            _S["t"] = now
            try:
                with open(TUNE_FILE) as fh:
                    _S["fine"] = bool(json.load(fh).get("fine_pad", _S["fine"]))
            except Exception:  # noqa: BLE001
                pass
    return _S["fine"]


def _patch(m):
    cls = m.CudaGraphManager
    orig = cls.dispatch
    none_mode = m.CUDAGraphMode.NONE

    def dispatch(self, num_reqs, num_tokens, *args, **kwargs):
        if num_tokens > 0 and not _fine():
            i = bisect.bisect_left(LADDER, num_tokens)
            if i < len(LADDER) and LADDER[i] != num_tokens:
                desc = orig(self, num_reqs, LADDER[i], *args, **kwargs)
                if desc.cg_mode != none_mode:
                    return desc
        return orig(self, num_reqs, num_tokens, *args, **kwargs)

    cls.dispatch = dispatch
    print("[blink-fine] CUDA-graph dispatch filter installed (tune key fine_pad)", file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        spec = None
        for f in sys.meta_path:
            if f is self:
                continue
            try:
                spec = f.find_spec(fullname, path, target)
            except Exception:  # noqa: BLE001
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


if _TARGET in sys.modules:
    _patch(sys.modules[_TARGET])
else:
    sys.meta_path.insert(0, _Finder())
