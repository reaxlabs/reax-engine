"""nv4_plan.py (kai): which projections rgh's q48m NVFP4 path (nv4.py, MAM_NV4) replaces, and a preset for its plan.

MAM_NV4 unset or empty      -> off (nv4.py and libnv4g.so are never imported).
MAM_NV4=<rgh spec>          -> passed to nv4.py verbatim, e.g. "gu+down@48-62:mse,qkvz+out@48-63:mse,qkv+o@48-62:mse"
                               (types gu, down, qkvz, out, qkv, o; layers "a-b" joined by '+'; method rtn | mse).
MAM_NV4=q48                 -> rgh's q48m plan (all48mse) with the layer sets as env lists:
    MAM_NV4_MLP    layers of the MLP gate|up + down projections          (default 48-62)
    MAM_NV4_REST   layers of GatedDeltaNet in_proj_qkv|z + out_proj and   (default: rgh's 48-63 / 48-62)
                   full-attention q|k|v + o_proj; "none" = MLP only
    MAM_NV4_METHOD weight block-scale method rtn | mse                   (default mse)
    MAM_NV4_REST_TYPES  '+'-joined subset of qkvz, out, qkv, o for the REST layers   (default all four)
  Lists take "a-b" ranges and single layers joined by ',' or '+' (e.g. "40-46,48-62"), so the set can follow our sweep's
  order (nv4sweep s2b: least sensitive first 61 60 59 62 57 58 56 55 53 54 52 51 50 49 48 47 46 45 44 43 42 41 63 ...).
With no MAM_NV4_* override the preset is rgh's image ENV string exactly.

rgh mx (dashboard #398, 86.38.238.34/rgh/engine@sha256:03e0a1b6...): method `gptq` (nv4.py lane QX: GPTQ weights, Hessians
from the PUBLIC seed set computed at start) plus rgh's modifiers `+x46` / `+perm` (method = <rtn|mse|gptq>[+x46][+perm]);
rgh's two plans as comma-free presets (they fit KAI_MODEL_ENV_<SLOT>):
MAM_NV4=g46   -> gu+down@46-62:gptq,qkvz+out@46-63:gptq,qkv+o@46-62:gptq   (mammoth, 68 projections)
MAM_NV4=rq44  -> gu+down@44-62:gptq,qkvz@44-63:gptq                       (reax-1, 53 projections)
REAX1_NV4 (rgh's reax-1 stack switch; plan or preset) replaces MAM_NV4 for the reax1 family only.
Calibrated methods (gptq, +perm) are installed by the backend before its warm-up; calibration() is the compliance guard:
MAM_NV4_CAL = seed only (rgh's bodies-file option is refused), MAM_NV4_CAL_N 1..1024 (the validator's public seed set:
engrunner.reference_self_check, items_for_pool(pool, 1024, seed=0x52454158); n takes its first n items).
"""
import re

TYPES = ("gu", "down", "qkvz", "out", "qkv", "o")
METHODS = ("rtn", "mse", "gptq")
MODIFIERS = ("x46", "perm")
REST_TYPES = ("qkvz", "out", "qkv", "o")
RGH_G46 = "gu+down@46-62:gptq,qkvz+out@46-63:gptq,qkv+o@46-62:gptq"  # rgh mx, mammoth MAM_NV4
RGH_RQ44 = "gu+down@44-62:gptq,qkvz@44-63:gptq"                       # rgh mx, reax-1 REAX1_NV4
PRESETS = {"g46": RGH_G46, "rq44": RGH_RQ44}
SEED_SET_N = 1024  # engrunner.reference_self_check: items_for_pool(pool, timed_items=1024, seed=0x52454158) (mammoth, reax1)
SLOT_OF = {"gu": "gu", "down": "down", "qkvz": "in_qkvz", "out": "out", "qkv": "qkv", "o": "o"}  # = nv4.SLOT
RGH_Q48 = "gu+down@48-62:mse,qkvz+out@48-63:mse,qkv+o@48-62:mse"
_RANGE = re.compile(r"^\d+(-\d+)?$")


def _layers(text, what):
    parts = [p for p in re.split(r"[,+]", text.strip()) if p]
    if not parts:
        raise ValueError(f"{what}: empty layer list")
    for p in parts:
        if not _RANGE.match(p):
            raise ValueError(f"{what}: bad layer range {p!r} (use a-b or n)")
        a, _, b = p.partition("-")
        if b and int(b) < int(a):
            raise ValueError(f"{what}: range {p!r} runs backwards")
    return "+".join(parts)


def _check(spec):
    for part in filter(None, spec.split(",")):
        types, _, rest = part.partition("@")
        rng, _, method = rest.partition(":")
        for t in types.split("+"):
            if t not in TYPES:
                raise ValueError(f"MAM_NV4: unknown projection type {t!r} (one of {TYPES})")
        if method:
            base, *mods = method.split("+")
            if base not in METHODS or any(m not in MODIFIERS for m in mods) or len(set(mods)) != len(mods):
                raise ValueError(f"MAM_NV4: unknown method {method!r} (rtn | mse | gptq, then +x46 / +perm)")
        if rng not in ("", "all"):
            _layers(rng, "MAM_NV4")
    return spec


def _rest_types(env):
    raw = (env.get("MAM_NV4_REST_TYPES") or "").strip()
    if not raw:
        return REST_TYPES
    sel = raw.split("+")
    if any(t not in REST_TYPES for t in sel) or len(set(sel)) != len(sel):
        raise ValueError(f"MAM_NV4_REST_TYPES {raw!r}: '+'-joined subset of {REST_TYPES}")
    return tuple(sel)


def resolve(env, family=None) -> str:
    """the nv4.py spec string for this environment ('' = off); family "reax1": REAX1_NV4 (rgh) wins over MAM_NV4"""
    spec = (env.get("MAM_NV4") or "").strip()
    if family == "reax1" and (env.get("REAX1_NV4") or "").strip():
        spec = env["REAX1_NV4"].strip()
    if not spec:
        return ""
    if env.get("KAI_E63", "0") == "1":  # pjs's graph engine reads .w8 of the MLP projections and asserts FP8
        raise ValueError("MAM_NV4 is not supported under KAI_E63=1 (E63 needs FP8 projections)")
    if spec in PRESETS:
        return PRESETS[spec]
    if spec != "q48":
        return _check(spec)
    m = (env.get("MAM_NV4_METHOD") or "mse").strip()
    if m not in METHODS:
        raise ValueError(f"MAM_NV4_METHOD {m!r}: rtn | mse | gptq")
    mlp = _layers(env.get("MAM_NV4_MLP") or "48-62", "MAM_NV4_MLP")
    rest = (env.get("MAM_NV4_REST") or "").strip()
    sel = _rest_types(env)
    gdn, att = "+".join(t for t in ("qkvz", "out") if t in sel), "+".join(t for t in ("qkv", "o") if t in sel)
    parts = [f"gu+down@{mlp}:{m}"]
    if not rest:
        parts += [f"{gdn}@48-63:{m}"] * bool(gdn) + [f"{att}@48-62:{m}"] * bool(att)
    elif rest != "none":
        r = _layers(rest, "MAM_NV4_REST")
        parts += [f"{gdn}@{r}:{m}"] * bool(gdn) + [f"{att}@{r}:{m}"] * bool(att)
    return ",".join(parts)


def needs_h(spec) -> bool:
    """a calibrated method (gptq, +perm) in the plan: the backend installs it before the warm-up (Hessians from the seed set)"""
    for part in filter(None, (spec or "").split(",")):
        method = part.partition("@")[2].partition(":")[2] or "rtn"
        base, *mods = method.split("+")
        if base == "gptq" or "perm" in mods:
            return True
    return False


def has_x_mods(spec) -> bool:
    """a +x46 / +perm method in the plan (activation quantised by nv4q.quant_dyn's x46 / perm kernel, not quant_x)"""
    for part in filter(None, (spec or "").split(",")):
        mods = (part.partition("@")[2].partition(":")[2] or "rtn").split("+")[1:]
        if mods:
            return True
    return False


def calibration(env):
    """(n items, GiB of Hessians per chunk) for nv4.calibrate; refuses anything but the public seed set (rules: conversion
    calibration-free or from activations computed at start from the public seed set 0x52454158)"""
    src = (env.get("MAM_NV4_CAL") or "seed").strip()
    if src != "seed":
        raise ValueError(f"MAM_NV4_CAL {src!r}: only 'seed' (the public seed set rendered at start) is allowed")
    try:
        n = int(env.get("MAM_NV4_CAL_N") or "256")
        gb = float(env.get("MAM_NV4_CAL_GB") or "6")
    except ValueError:
        raise ValueError("MAM_NV4_CAL_N / MAM_NV4_CAL_GB: numbers") from None
    if not 1 <= n <= SEED_SET_N:
        raise ValueError(f"MAM_NV4_CAL_N {n}: 1..{SEED_SET_N} (the validator's public seed set)")
    if not gb > 0:
        raise ValueError(f"MAM_NV4_CAL_GB {gb}: > 0")
    if (env.get("MAM_NV4_GPTQ_GRAPH") or "1") not in ("0", "1"):
        raise ValueError("MAM_NV4_GPTQ_GRAPH: 0 | 1")
    return n, gb


def projections(spec, layer_types):
    """[(layer, type)] that nv4.install replaces for this spec: the slot must exist in the layer (GatedDeltaNet layers have
    in_qkvz/out, full-attention layers qkv/o) and the last layer's qkv/o are skipped (nv4.install's rule)."""
    import nv4
    n = len(layer_types)
    plan = nv4.parse(spec, n)
    has = {"linear_attention": {"gu", "down", "in_qkvz", "out"}, "full_attention": {"gu", "down", "qkv", "o"}}
    out = []
    for (i, t), _m in sorted(plan.items()):
        if SLOT_OF[t] not in has[layer_types[i]] or (i == n - 1 and t in ("qkv", "o")):
            continue
        out.append((i, t))
    return out
