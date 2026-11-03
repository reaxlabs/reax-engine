# SPDX-License-Identifier: Apache-2.0
"""E64 rank 2(b) (BLINK_LEAN_OUT=1): a leaner output update in the engine core, telemetry only.

The scheduler's update_from_output finalises each finished request's prefill statistics with
KVCacheManager.estimate_cached_tokens (a Python walk over the request's blocks in every KV-cache group); the value only
sets PrefillStats.num_cache_creation_tokens, a metrics field. With stats logging off (Blink runs with
disable_log_stats), this module returns 0 for it, so the field reads 0 and the walk leaves the output path. Answers,
scheduling, KV blocks and prefix caching are unchanged. Imported by blink_core.py when BLINK_LEAN_OUT=1."""
import importlib.abc
import os
import sys

ON = os.environ.get("BLINK_LEAN_OUT", "0") == "1"
_TARGET = "vllm.v1.core.kv_cache_manager"


def _patch(mod):
    cls = mod.KVCacheManager
    if getattr(cls.estimate_cached_tokens, "_blink_out", False):
        return

    def estimate_cached_tokens(self, request):
        return 0

    estimate_cached_tokens._blink_out = True
    cls.estimate_cached_tokens = estimate_cached_tokens
    print("[blink-out] prefill-stats cache estimate skipped (telemetry only)", file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    _busy = set()

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET or fullname in _Finder._busy:
            return None
        spec = None
        _Finder._busy.add(fullname)
        try:
            for f in sys.meta_path:
                if f is self:
                    continue
                try:
                    spec = f.find_spec(fullname, path, target)
                except Exception:  # noqa: BLE001
                    spec = None
                if spec is not None:
                    break
        finally:
            _Finder._busy.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


if ON:
    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
    else:
        sys.meta_path.insert(0, _Finder())
