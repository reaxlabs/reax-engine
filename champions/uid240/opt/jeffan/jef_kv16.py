# SPDX-License-Identifier: Apache-2.0
"""E62 lever KV16 (JEF_KV16=1): prefix-cache hits at 16-token granularity for Jeffan's hybrid KV cache.

vLLM gives every KV-cache group the same page bytes. Jeffan's full-attention layers store K = V (one tensor), so their
16-token page is half a sliding-window page, and vLLM doubles their block size to 32 tokens; the scheduler block size
(the lcm of the group block sizes) becomes 32 and prefix-cache hits snap to 32 tokens (a decision's read 1 recomputes
~16-21 template tokens, read 2 ~15.5 of read 1's prefix). With KV16 the full-attention layers keep 16-token blocks and
their pages are padded to the common page instead (vLLM's own page_size_padded mechanism, read through a strided view),
so every group has 16-token blocks and hits land on 16-token boundaries. Costs half of each full-attention page (KV cache
capacity about -8 %). Numerics: the K / V of every token are written to and read from the FP8 paged cache exactly as
before (same values; only the page layout and the hit length change), the Triton attention kernel's tiles do not depend
on the block size (no tensor-descriptor path on CUDA), and a token whose K / V come from the cache instead of being
recomputed gets identical values under batch-invariant kernels. Hooked into vllm.v1.core.kv_cache_utils (engine core)."""
from __future__ import annotations

import importlib.abc
import os
import sys

ON = os.environ.get("JEF_KV16", "0") == "1"


def _log(msg):
    print(f"[jef-kv16] {msg}", file=sys.stderr, flush=True)


def _patch(m):
    if getattr(m, "_jef_kv16", False):
        return
    m._jef_kv16 = True
    orig = m.unify_kv_cache_spec_page_size

    def unify_kv_cache_spec_page_size(kv_cache_spec):
        from dataclasses import replace

        from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec
        pages = {s.page_size_bytes for s in kv_cache_spec.values()}
        if len(pages) > 1:
            mx = max(pages)
            new, n = {}, 0
            for name, s in kv_cache_spec.items():
                if (s.page_size_bytes != mx and isinstance(s, AttentionSpec) and not isinstance(s, MLAAttentionSpec)
                        and mx % s.page_size_bytes == 0):
                    new[name] = replace(s, page_size_padded=mx)
                    n += 1
                else:
                    new[name] = s
            kv_cache_spec = new
            _log(f"{n} attention layers keep their block size (block sizes {sorted({s.block_size for s in new.values()})}"
                 f"); their pages are padded to {mx} bytes")
        return orig(kv_cache_spec)

    m.unify_kv_cache_spec_page_size = unify_kv_cache_spec_page_size
    _log("on: hybrid KV cache groups keep 16-token blocks (prefix-cache hits at 16 tokens)")


_TARGETS = {"vllm.v1.core.kv_cache_utils": _patch}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS:
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

        def exec_module(module, _orig=orig, _fn=_TARGETS[fullname]):
            _orig(module)
            _fn(module)

        spec.loader.exec_module = exec_module
        return spec


if ON:
    for _name, _fn in _TARGETS.items():
        if _name in sys.modules:
            _fn(sys.modules[_name])
    sys.meta_path.insert(0, _Finder())
