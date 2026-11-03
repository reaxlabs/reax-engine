"""kai_model_env.py (kai): per-model switch sets for ONE image that serves several models (e.g. mammoth + reax1).

KAI_MODEL_ENV_<SLOT>="K=V,K=V" (SLOT: MAMMOTH, REAX1) is applied to os.environ for the model in $REAX_MODEL, at the very top
of serve.py, before any engine module reads its switches (pplx_engine, nv4, f8g ... read MAM_* / KAI_* at import). A
model's set overrides an image-level value of the same key. Unset (or a model without a set) = nothing changes.
Keys: KAI_* / PPLX_* / MAM_* / R1_* / E63_* / MXA_* / REAX1_* (rgh mx) only (never LD_PRELOAD, PATH, proxies ...); values must not contain ','."""
from __future__ import annotations

import re

SLOT_OF = {"mammoth-1": "MAMMOTH", "mammoth": "MAMMOTH", "reax-1": "REAX1", "reax1": "REAX1"}
_KEY = re.compile(r"^(KAI|PPLX|MAM|R1|E63|MXA|REAX1)_[A-Z0-9_]+$")
_VAL = re.compile(r"^[A-Za-z0-9_.:=@+/-]*$")


def parse(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in (x.strip() for x in raw.split(",")):
        if not item:
            continue
        k, sep, v = item.partition("=")
        if not sep or not _KEY.match(k) or not _VAL.match(v) or k.startswith("KAI_MODEL_ENV_"):
            raise ValueError(f"KAI_MODEL_ENV entry {item!r}: need KEY=VALUE with KEY in KAI_/PPLX_/MAM_/R1_/E63_/MXA_/REAX1_*")
        out[k] = v
    return out


def apply(environ) -> dict[str, str]:
    """Apply the served model's set to `environ` (os.environ in serve.py); returns what was set."""
    slot = SLOT_OF.get((environ.get("REAX_MODEL") or "").strip())
    raw = environ.get(f"KAI_MODEL_ENV_{slot}", "") if slot else ""
    if not raw.strip():
        return {}
    vals = parse(raw)
    environ.update(vals)
    return vals
