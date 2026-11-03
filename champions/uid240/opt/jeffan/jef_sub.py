# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 team ruj. Line J5 round 2 (Jeffan, workflow v2-bj-5): lever SUB for pjs's Jeffan engine (E61 / E62,
# in-process vLLM 0.30.0, hybrid KV cache of six 16-token groups with his KV16). Credit pjs (the engine, KV16).
"""Exact engine-core lever SUB (JEF_J5_SUB=1): prefix-cache hits that end inside a 16-token block.

vLLM's prefix cache hits whole blocks: a read whose prompt shares P tokens with KV that is already in the cache reuses
floor(P / 16) * 16 of them and computes the rest again. Jeffan's read 2 (the read-twice prompt) shares read 1's prompt
minus its 9-token chat suffix, so every read 2 recomputes on average 7.5 tokens that read 1 has just computed; every
read 1 recomputes the 0-15 tokens of the shared template that end inside its first item-specific block. On the screen
items that is 234.4 computed prompt tokens per decision, 228.2 with SUB's tails and 226.5 with tails + heads (token
granularity: 225.2) - r2/scripts/kvsim2.py.

SUB lets a hit end inside the next block. When a read finishes, its last two blocks (one KVCacheBlock per KV-cache group)
are indexed by the chain hash of the block before them together with the token ids they hold: a full block (vLLM has
cached it) without a reference, used only while it still carries the same block hash (i.e. until vLLM evicts it in its
LRU order); the partial last block (vLLM does not cache it and would hand it out first) with one extra reference (at
most JEF_J5_SUB_CAP such entries, released oldest first, and all at once whenever an allocation fails). The block at
which the read's own prefix hit ended (a full, cached block: the "head", e.g. the block where the shared template
ends) is indexed too, without a reference. When the scheduler looks up a new read (Scheduler._get_local_prefix_cache_hit), the
full-block hit H (16 * h tokens) is extended by m = the longest common prefix of the read's tokens [H, H + 16) with an
indexed block under the read's chain hash at block h - 1 (capped at num_tokens - 1, as vLLM caps its own hits). The
extended hit is handed to vLLM's own sub-block machinery of 0.30.0: KVCacheManager.allocate_slots with a hit that ends
inside a block -> SingleTypeKVCacheManager.add_local_computed_blocks records the partial tail, allocate_new_blocks gives
the read a private copy-on-write block (_apply_cow), the scheduler sends the copies with the step
(SchedulerOutput.kv_cache_block_copies) and the V2 model runner copies the whole source page into the new block of every
KV-cache tensor before the step's forward (copy_kv_cache_blocks_inplace, update_requests); the source stays referenced
until the copy has run (vLLM's cow-retained blocks). The read computes from H + m on.

Exactness: the K / V of the m reused positions are the values the earlier read wrote for the same token ids at the same
positions after the same prefix; with batch-invariant kernels a token's K / V do not depend on the batch, the chunk or
the step it was computed in (the same property every prefix-cache hit of this engine and pjs's KV16 32 -> 16 change rely
on: answers bit-identical). The read's remaining tokens attend to the same K / V either way; its answer row is computed
as before. Only which tokens are computed (fewer) and the per-step token counts change.
Scope: only reads of at most the smallest sliding window (Jeffan: 1,024 tokens; JEF_J5_SUB_MAXLEN lowers it) are
extended. Beyond the window the engine's own answers depend on how a read is batched and chunked (line J5 round 2,
session s4b: the control's c8 and c1 answers to 24 long items differ on 20), so moving where a long read's hit ends
would move its bits too; such reads keep vLLM's block hits. Every item of the release's jeffan pool is far shorter
(states <= 340 characters, reads <= ~450 tokens).
Installed by sitecustomize.py (JEF_J5_SUB=1) after jef_levers / jef_kv16 / jef_host; patches
vllm.v1.core.kv_cache_manager and vllm.v1.core.sched.scheduler when the engine core imports them."""
from __future__ import annotations

import importlib.abc
import os
import sys
import time
from collections import OrderedDict

ON = os.environ.get("JEF_J5_SUB", "0") == "1"
HEAD = os.environ.get("JEF_J5_SUB_HEAD", "1") == "1"
CAP = max(1, int(os.environ.get("JEF_J5_SUB_CAP", "64") or 64))          # held tail entries
CAP_UNHELD = max(1, int(os.environ.get("JEF_J5_SUB_CAP_HEAD", "4096") or 4096))
PER_KEY = max(1, int(os.environ.get("JEF_J5_SUB_PER_KEY", "4") or 4))
MIN_M = max(1, int(os.environ.get("JEF_J5_SUB_MIN", "1") or 1))
MAXLEN_ENV = int(os.environ.get("JEF_J5_SUB_MAXLEN", "0") or 0)   # 0 = the smallest sliding window of the KV-cache groups
_S = {"on": ON, "B": None, "pool": None, "checked": False, "ext": 0, "tok": 0, "reg": 0, "rel": 0, "fails": 0,
      "stale": 0, "long": 0, "maxlen": None, "t_rep": 0.0, "err": "", "patched_kvm": False, "patched_sched": False}


def _log(msg):
    print(f"[jef-sub] {msg}", file=sys.stderr, flush=True)


class _Entry:
    __slots__ = ("key", "toks", "blocks", "hashes", "held")

    def __init__(self, key, toks, blocks, hashes, held):
        self.key, self.toks, self.blocks, self.hashes, self.held = key, toks, blocks, hashes, held


_held: "OrderedDict[tuple, _Entry]" = OrderedDict()      # (key, toks) -> held entry, oldest first
_unheld: "OrderedDict[tuple, _Entry]" = OrderedDict()    # (key, toks) -> head entry (no reference), oldest first
_index: dict = {}                                         # key -> [entries], most recent last
_first_hit: dict = {}                                     # request id -> its own prefix hit (tokens)


def _disable(exc):
    if _S["on"]:
        _S["on"] = False
        _S["err"] = repr(exc)
        _log(f"OFF after an error: {exc!r} (held entries released)")
        try:
            _release_all()
        except Exception:  # noqa: BLE001
            pass


def _unindex(e):
    lst = _index.get(e.key)
    if lst is not None:
        for i, x in enumerate(lst):
            if x is e:
                del lst[i]
                break
        if not lst:
            del _index[e.key]


def _drop(e):
    """forget an entry; a held entry gives its references back"""
    _unindex(e)
    if e.held:
        _held.pop((e.key, e.toks), None)
        e.held = False
        _S["pool"].free_blocks(e.blocks)
        _S["rel"] += 1
    else:
        _unheld.pop((e.key, e.toks), None)


def _release_all():
    for e in list(_held.values()):
        _drop(e)


def _add(key, toks, blocks, held, hashes=None):
    k = (key, toks)
    old = _held.get(k) or _unheld.get(k)
    if old is not None:
        _drop(old)
    if held:
        _S["pool"].touch(blocks)          # one reference per block: not evictable, not reusable while indexed
    e = _Entry(key, toks, blocks, hashes, held)
    (_held if held else _unheld)[k] = e
    lst = _index.setdefault(key, [])
    lst.append(e)
    if len(lst) > PER_KEY:
        _drop(lst[0])
    while len(_held) > CAP:
        _drop(next(iter(_held.values())))
    while len(_unheld) > CAP_UNHELD:
        _drop(next(iter(_unheld.values())))
    _S["reg"] += 1


def _check(kvm):
    """once: one block size for every KV-cache group, hashes at that granularity, vLLM's own fine-grained hits off"""
    _S["checked"] = True
    try:
        coord = kvm.coordinator
        sizes = {m.block_size for m in coord.single_type_managers}
        hb = kvm.block_pool.hash_block_size
        why = ""
        if len(sizes) != 1:
            why = f"block sizes {sorted(sizes)}"
        elif hb != next(iter(sizes)):
            why = f"hash block size {hb} != block size {sizes}"
        elif getattr(coord, "enable_partial_hash_hits", False):
            why = "vLLM's own fine-grained hits are on"
        elif not kvm.enable_caching:
            why = "prefix caching off"
        if why:
            _S["on"] = False
            _log(f"OFF: {why}")
            return
        _S["B"] = next(iter(sizes))
        _S["pool"] = kvm.block_pool
        # only reads that fit in every sliding window: for them the engine's numerics are proven independent of where a
        # hit ends (line J5's screens: bit-identical answers across batchings); a longer read is left to vLLM as before
        wins = [m.sliding_window for m in coord.single_type_managers if getattr(m, "sliding_window", None)]
        lim = [x for x in ([min(wins)] if wins else []) + ([MAXLEN_ENV] if MAXLEN_ENV > 0 else [])]
        _S["maxlen"] = min(lim) if lim else None
        _log(f"on: {len(coord.single_type_managers)} KV-cache groups of {_S['B']}-token blocks "
             f"({type(coord).__name__}); tails cap {CAP}, heads {'on' if HEAD else 'off'}, min {MIN_M} token(s), "
             f"reads of at most {_S['maxlen']} tokens")
    except Exception as exc:  # noqa: BLE001
        _disable(exc)


def _reg_block(lists, idx, n, bh, toks_all, may_hold):
    """index block idx of a finishing read: a full block that vLLM has cached is indexed without a reference (used
    only while it still carries its block hash, i.e. until vLLM evicts it in LRU order); the partial last block
    (vLLM does not cache it, it would be reused first) is held by one reference when may_hold"""
    B = _S["B"]
    if idx > 0 and len(bh) < idx:
        return
    blocks = []
    for lst in lists:
        if idx >= len(lst):
            return
        b = lst[idx]
        if b.is_null:
            return
        blocks.append(b)
    end = min(n, idx * B + B)
    if end <= idx * B:
        return
    toks = tuple(toks_all[idx * B:end])
    key = bh[idx - 1] if idx > 0 else None
    hashes = [b.block_hash for b in blocks]
    if end - idx * B == B and all(h is not None for h in hashes):
        _add(key, toks, blocks, False, hashes)
    elif may_hold:
        _add(key, toks, blocks, True)


def _register(kvm, request):
    """before vLLM frees a request's blocks: index its last two blocks (held) and its head block (not held)"""
    rid = request.request_id
    hit0 = _first_hit.pop(rid, None)
    if not _S["on"]:
        return
    if not _S["checked"]:
        _check(kvm)
        if not _S["on"]:
            return
    try:
        if getattr(request, "num_in_flight_tokens", 0):
            return
        n = min(request.num_computed_tokens, request.num_prompt_tokens)
        if n <= 0:
            return
        lists = []
        for mgr in kvm.coordinator.single_type_managers:
            lst = mgr.req_to_blocks.get(rid)          # .get: req_to_blocks is a defaultdict
            if not lst:
                return
            lists.append(lst)
        B = _S["B"]
        bh = request.block_hashes
        toks_all = request._all_token_ids
        last = (n - 1) // B
        for idx in (last - 1, last):
            if idx >= 0:
                _reg_block(lists, idx, n, bh, toks_all, True)
        if HEAD and hit0 is not None:
            hidx = hit0 // B
            if hidx < last - 1:
                _reg_block(lists, hidx, n, bh, toks_all, False)
    except Exception as exc:  # noqa: BLE001
        _disable(exc)


def _extend(sched, request, res):
    """the scheduler's prefix-cache lookup of a new read, extended into the next block (see the module docstring)"""
    blocks, num_local, spb, div = res
    if not _S["on"] or div:
        return res
    kvm = sched.kv_cache_manager
    if not _S["checked"]:
        _check(kvm)
        if not _S["on"]:
            return res
    try:
        _first_hit[request.request_id] = num_local
        if not _index:
            return res
        B = _S["B"]
        H = num_local
        maxlen = request.num_tokens - 1
        if _S["maxlen"] is not None and request.num_tokens > _S["maxlen"]:
            _S["long"] += 1
            return res
        if H % B or H >= maxlen or not kvm.prefix_cache_lookup_enabled(request):
            return res
        h = H // B
        bh = request.block_hashes
        if h > 0 and len(bh) < h:
            return res
        cands = _index.get(bh[h - 1] if h > 0 else None)
        if not cands:
            return res
        nxt = request._all_token_ids[H:H + B]
        best_m, best = 0, None
        for e in reversed(cands):
            t = e.toks
            lim = min(len(t), len(nxt))
            m = 0
            while m < lim and t[m] == nxt[m]:
                m += 1
            if m > best_m:
                if not e.held and any(b.is_null or b.block_hash != x for b, x in zip(e.blocks, e.hashes)):
                    _S["stale"] += 1
                    continue
                best_m, best = m, e
        m = min(best_m, maxlen - H)
        if best is None or m < MIN_M:
            return res
        groups = blocks.blocks
        if len(groups) != len(best.blocks) or any(len(g) != h for g in groups):
            return res
        new = tuple(list(g) + [best.blocks[i]] for i, g in enumerate(groups))
        _S["ext"] += 1
        _S["tok"] += m
        _first_hit[request.request_id] = H + m
        now = time.monotonic()
        if now - _S["t_rep"] > 10.0:
            _S["t_rep"] = now
            _log(f"{_S['ext']} reads extended, {_S['tok']} tokens not recomputed (mean "
                 f"{_S['tok'] / max(1, _S['ext']):.2f}); entries {len(_held)} held / {len(_unheld)} heads, "
                 f"{_S['reg']} registered, {_S['rel']} released, {_S['fails']} allocation-failure releases, "
                 f"{_S['stale']} stale heads skipped, {_S['long']} reads over {_S['maxlen']} tokens left to vLLM")
        return kvm.create_kv_cache_blocks(new), H + m, spb, div
    except Exception as exc:  # noqa: BLE001
        _disable(exc)
        return res


def _patch_kvm(m):
    if _S["patched_kvm"]:
        return
    _S["patched_kvm"] = True
    cls = m.KVCacheManager
    o_free, o_pop, o_alloc, o_reset = cls.free, cls.pop_blocks_for_free, cls.allocate_slots, cls.reset_prefix_cache

    def free(self, request):
        _register(self, request)
        return o_free(self, request)

    def pop_blocks_for_free(self, request):
        _register(self, request)
        return o_pop(self, request)

    def allocate_slots(self, *a, **k):
        res = o_alloc(self, *a, **k)
        if res is None and _held:          # KV pressure: give the held tails back (the next attempt sees them free)
            _S["fails"] += 1
            try:
                _release_all()
            except Exception as exc:  # noqa: BLE001
                _disable(exc)
        return res

    def reset_prefix_cache(self, *a, **k):
        try:
            _release_all()
            _unheld.clear()
            _index.clear()
        except Exception as exc:  # noqa: BLE001
            _disable(exc)
        return o_reset(self, *a, **k)

    cls.free, cls.pop_blocks_for_free, cls.allocate_slots = free, pop_blocks_for_free, allocate_slots
    cls.reset_prefix_cache = reset_prefix_cache
    _log(f"KVCacheManager hooks installed in pid {os.getpid()}")


def _patch_sched(m):
    if _S["patched_sched"]:
        return
    _S["patched_sched"] = True
    cls = m.Scheduler
    if not hasattr(cls, "_get_local_prefix_cache_hit"):
        _S["on"] = False
        _log("OFF: Scheduler._get_local_prefix_cache_hit not found")
        return
    orig = cls._get_local_prefix_cache_hit

    def _get_local_prefix_cache_hit(self, request):
        return _extend(self, request, orig(self, request))

    cls._get_local_prefix_cache_hit = _get_local_prefix_cache_hit
    _log(f"scheduler hook installed in pid {os.getpid()}")


_PATCHES = {"vllm.v1.core.kv_cache_manager": _patch_kvm, "vllm.v1.core.sched.scheduler": _patch_sched}


class _Finder(importlib.abc.MetaPathFinder):
    _busy = set()

    def find_spec(self, fullname, path, target=None):
        if fullname not in _PATCHES or fullname in _Finder._busy:
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

        def exec_module(module, _orig=orig, _fn=_PATCHES[fullname]):
            _orig(module)
            _fn(module)

        spec.loader.exec_module = exec_module
        return spec


if ON:
    _todo = False
    for _name, _fn in _PATCHES.items():
        if _name in sys.modules:
            _fn(sys.modules[_name])
        else:
            _todo = True
    if _todo:
        sys.meta_path.insert(0, _Finder())
