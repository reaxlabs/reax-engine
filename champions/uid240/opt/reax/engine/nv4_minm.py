"""nv4_minm.py (kai): small-step FP8 fallback for NVFP4 projections. KAI_NV4_MINM=<rows>; unset, empty or 0 = off (nothing
is imported, patched or kept).

A step whose rows (the tokens entering the layer stack: one c1 item = one step of ~100-200 rows on the reax1 pool) are fewer
than MINM runs every NVFP4 projection through the FP8 Linear that an all-FP8 engine holds for that slot; a step of MINM rows
or more runs the NVFP4 engine unchanged. The choice is per step, made where the step enters PplxEngine._core_gen, because
the engine picks a projection's input producer (fused add+rms+quant to FP8 or to NVFP4) from the slot's object before the
GEMM is called: a small step therefore runs exactly the FP8 engine (same objects, same producers, same router), not an FP8
GEMM behind an NVFP4 producer. For every converted layer of our plans the step rows are the GEMM's M (the pruned last
layer, whose rows are the read rows, is never converted; mammoth's state-once steps count their shared-layout rows).

Both NVFP4 paths are covered:
  rgh's MAM_NV4 (nv4.py NV4Linear): the FP8 objects nv4.install replaces are kept (`slots` before, `replaced` after).
  our nvfp4= groups (pplx_engine.Linear mode nvfp4): the FP8 Linear is derived from the checkpoint exactly as the FP8 engine
    builds it (same bf16 tensors, same concatenation, Linear(w, "fp8")).
Cost: the FP8 copies stay resident (`kept_bytes`). Refused: KAI_E63 (own layer loop), PPLX_STREAMS > 1 (interleaved step
generators would share the swapped layer list), nvfp4_smooth (folded into the post-norm gain: no FP8 copy matches).
"""
from __future__ import annotations

import json
import os
import time

ENV = "KAI_NV4_MINM"
PROJ_SLOTS = ("gu", "down", "in_qkvz", "out", "qkv", "o")  # = nv4.SLOT values
TYPE_OF = {"gu": "gu", "down": "down", "in_qkvz": "qkvz", "out": "out", "qkv": "qkv", "o": "o"}  # slot -> nv4 type
# kai fix M3 (logging only): steps that ran the FP8 fallback / the NVFP4 engine (patched _core_gen; kai_status.py reports)
COUNTS = {"fallback_steps": 0, "nv4_steps": 0}


def resolve(env) -> int:
    """the MINM rows for this environment (0 = off); ValueError on a bad value or with KAI_E63"""
    raw = (env.get(ENV) or "").strip()
    if not raw:
        return 0
    if not raw.isdigit():
        raise ValueError(f"{ENV}={raw!r}: a row count (integer >= 0; 0 or unset = off)")
    m = int(raw)
    if m and env.get("KAI_E63", "0") == "1":
        raise ValueError(f"{ENV} is not supported under KAI_E63=1 (pjs's graph engine runs its own FP8 layer loop)")
    return m


def is_nvfp4(obj) -> bool:
    """rgh's NV4Linear (fq4, mode stays 'fp8') or our nvfp4 Linear"""
    return bool(getattr(obj, "fq4", False)) or getattr(obj, "mode", None) == "nvfp4"


def slots(eng) -> dict:
    """{(layer, slot): object} of every projection slot (taken before nv4.install)"""
    return {(i, s): L[s] for i, L in enumerate(eng.layers) for s in PROJ_SLOTS if s in L}


def replaced(eng, before) -> dict:
    """{(layer, slot): FP8 object} that nv4.install (or anything else) replaced by an NVFP4 projection since `before`"""
    out = {}
    for (i, s), old in before.items():
        cur = eng.layers[i].get(s)
        if cur is not old and is_nvfp4(cur) and not is_nvfp4(old):
            out[(i, s)] = old
    return out


def _loader(model_dir, device):
    """nv4.install's tensor getter: the checkpoint's bf16 tensors by text-model key, opened on `device`"""
    from safetensors import safe_open
    ip = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(ip):
        idx = json.load(open(ip))["weight_map"]
    else:
        with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt", device="cpu") as sf:
            idx = {k: "model.safetensors" for k in sf.keys()}
    pre = "language_model." if any(k.startswith("language_model.") for k in idx) else "model.language_model."
    handles = {}

    def get(key):
        f = idx[pre + key]
        h = handles.get(f)
        if h is None:
            h = handles[f] = safe_open(os.path.join(model_dir, f), framework="pt", device=device)
        return h.get_tensor(pre + key)
    return get, handles.clear


def _derive_ours(pp, eng, model_dir, todo) -> dict:
    """FP8 Linears for our nvfp4 slots, built as the FP8 engine builds them (fused concatenation, Linear(w, 'fp8'))"""
    import nv4
    get, close = _loader(model_dir, str(eng.device))
    out = {}
    try:
        for i, s in todo:
            w = nv4.weight(get, i, TYPE_OF[s])  # gate|up, qkv|z, q|k|v concatenated in the engine's order
            out[(i, s)] = pp.Linear(w, "fp8")
            del w
    finally:
        close()
    return out


def install(pp, eng, model_dir, minm: int, keep: dict) -> int:
    """register the FP8 fallback of every NVFP4 projection of `eng` (keep = rgh's replaced FP8 objects; our nvfp4 slots are
    derived here), set eng.nv4_minm and patch PplxEngine._core_gen once. Returns the number of NVFP4 projections covered."""
    t0 = time.time()
    if getattr(eng, "n_streams", 1) > 1:
        raise ValueError(f"{ENV} needs PPLX_STREAMS=1 (interleaved step generators would share the swapped layer list)")
    if getattr(eng, "linear_mode", "fp8") != "fp8":
        raise ValueError(f"{ENV} needs linear_mode fp8 (the fallback is the FP8 engine)")
    conv = {k: o for k, o in slots(eng).items() if is_nvfp4(o)}
    other = [(i, s) for i, L in enumerate(eng.layers) for s, o in L.items()
             if s not in PROJ_SLOTS and getattr(o, "mode", None) == "nvfp4"]
    if other:
        raise ValueError(f"{ENV}: NVFP4 in slots without an FP8 fallback {other[:4]}")
    if not conv:
        raise ValueError(f"{ENV}={minm}: the engine has no NVFP4 projection (set MAM_NV4 or nvfp4= groups)")
    ours = sorted(k for k, o in conv.items() if not getattr(o, "fq4", False))
    if ours and getattr(eng, "nv_smooth", None):
        raise ValueError(f"{ENV} with nvfp4_smooth: the smoothing is folded into the post-norm gain, no FP8 copy matches")
    if any(type(conv[k]) is not pp.Linear for k in ours):
        raise ValueError(f"{ENV}: our nvfp4 slots must be fused Linears (engine fuse=True)")
    missing = sorted(k for k, o in conv.items() if getattr(o, "fq4", False) and k not in keep)
    if missing:
        raise ValueError(f"{ENV}: no kept FP8 object for rgh NV4 slots {missing[:4]} (take `slots` before nv4.install)")
    f8 = {k: keep[k] for k in conv if k in keep}
    f8.update(_derive_ours(pp, eng, model_dir, ours))
    reg = {}
    for (i, s), o in sorted(f8.items()):
        reg.setdefault(i, {})[s] = o
    eng._minm_f8 = reg
    eng.nv4_minm = int(minm)
    patch_engine(pp)
    if str(getattr(eng, "device", "")).startswith("cuda"):
        import torch
        torch.cuda.empty_cache()
    print(f"kai nv4_minm: steps < {minm} rows run FP8 in {len(f8)} NVFP4 projections ({len(f8) - len(ours)} rgh NV4 kept, "
          f"{len(ours)} ours derived), FP8 copies {kept_bytes(eng) / 2**20:.0f} MiB, {time.time() - t0:.1f} s", flush=True)
    return len(f8)


def kept_bytes(eng) -> int:
    """bytes of the resident FP8 fallback copies (FP8 weight + fp32 per-channel scales)"""
    n = 0
    for d in getattr(eng, "_minm_f8", {}).values():
        for o in d.values():
            n += o.w8.numel() * o.w8.element_size() + o.ws.numel() * o.ws.element_size()
    return n


def f8_view(eng, rows: int):
    """the layer list a step of `rows` rows runs: None = the engine's own (NVFP4) layers; else a list in which every layer
    holding an NVFP4 projection with a registered fallback is a copy of its dict with the FP8 objects in those slots"""
    m = getattr(eng, "nv4_minm", 0)
    reg = getattr(eng, "_minm_f8", None)
    if not m or not reg or rows >= m:
        return None
    layers = eng.layers
    out = None
    for i, f8s in reg.items():
        L = layers[i]
        sub = {s: o for s, o in f8s.items() if L.get(s) is not o and is_nvfp4(L.get(s))}
        if sub:
            if out is None:
                out = list(layers)
            out[i] = {**L, **sub}
    return out


def patch_engine(pp):
    """PplxEngine._core_gen: a step below the engine's nv4_minm runs on f8_view's layer list (restored when the step ends,
    also on an error); engines without the registry are unchanged"""
    PE = pp.PplxEngine
    if getattr(PE, "_kai_minm_patched", False):
        return
    orig = PE._core_gen

    def _core_gen(self, ids, *a, **k):
        view = f8_view(self, ids.shape[0])
        if view is None:
            COUNTS["nv4_steps"] += 1
            return (yield from orig(self, ids, *a, **k))
        COUNTS["fallback_steps"] += 1
        base = self.layers
        self.layers = view
        try:
            return (yield from orig(self, ids, *a, **k))
        finally:
            self.layers = base
    PE._core_gen = _core_gen
    PE._kai_minm_patched = True
