"""kai_status.py (kai, round-6 fix M3; logging only): what the mammoth / reax1 engine actually runs, and how often.

switches(): every resolved lever of pplx_engine (rgh's MAM_* f8m levers, kai's lever I / D / Q / B / N1 / N2, MINM,
R1_C1GRAPH) after the backend's start-up checks, with the conditions that can silently switch a lever off (KAI_EARLY's,
N1's start-up check) and the path each one acts on. counters(): GDNPS / GDNFUSE calls, steps per DeltaNet path
(pplx_engine.KAI_COUNTS), KAI_EARLY steps, MINM fallback steps (nv4_minm.COUNTS), NVFP4 tile-table hits (nv4t.Router),
N1 fused / fallback / plain gate|up calls (n1.stats), PDL-routed f8g / NVFP4 calls (pdl.STATE's routers); n1 and pdl are
read from sys.modules, never imported. Status: one start-up line ("kai switches: {json}", with the warm-up's counts),
the counts since then in /health (serve.py) and once at shutdown ("kai counts: {json}").

Reads module globals and engine attributes only: nothing here changes what a step computes. Any error while reading is
reported in the line instead of failing the start."""
from __future__ import annotations

import json
import os
import sys

PROJ_SLOTS = ("gu", "down", "in_qkvz", "out", "qkv", "o")


def _g(obj, name, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:  # noqa: BLE001
        return default


def _early(pp, eng) -> dict:
    """KAI_EARLY: on only with MAM_H2D1 and PplxEngine._early_ok (each condition listed); forward_groups steps only"""
    out = {"env": bool(_g(pp, "_EARLY", False)), "h2d1": bool(_g(pp, "_H2D1", False))}
    try:
        emb = _g(eng, "embed")
        out.update(gpu_embed=bool(emb is not None and emb.is_cuda), one_stream=_g(eng, "n_streams", 1) <= 1,
                   no_ovl=not _g(pp, "_OVL", False), layer0_gdn=eng.layers[0]["type"] == "linear_attention",
                   minm_keeps_l0=bool(eng._minm_keeps_l0()), no_trace=_g(eng, "_trace") is None)
        ok = bool(eng._early_ok())
    except Exception as e:  # noqa: BLE001 - a stub engine
        out["error"] = f"{type(e).__name__}: {e}"
        ok = False
    out["on"] = bool(out["env"] and out["h2d1"] and ok)
    out["path"] = "forward_groups (state-once) only"
    return out


def _noxstore(pp, eng) -> dict:
    """kai lever R8-5 (KAI_NOXSTORE): asked / on after the start-up check, and the producer calls per step that skip the x
    store with the layer objects serving now (post = before an FP8 MLP, pre = before an FP8 full-attention layer)"""
    out = {"env": os.environ.get("KAI_NOXSTORE", "0") == "1", "on": bool(_g(pp, "_NOXSTORE", False))}
    try:
        layers = list(eng.layers)
        out["post"] = sum(1 for L in layers if L["gu"].mode == "fp8" and pp.x_unread_post(L))
        out["pre"] = sum(1 for i in range(len(layers) - 1)
                         if eng._first_mode(layers[i + 1]) == "fp8" and pp.x_unread_pre(layers[i + 1]))
    except Exception as e:  # noqa: BLE001 - a stub engine
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def _nv4(eng) -> dict:
    nv4 = sys.modules.get("nv4")
    by = {}
    for L in _g(eng, "layers", None) or []:
        for s in PROJ_SLOTS:
            o = L.get(s) if isinstance(L, dict) else None
            if o is not None and (_g(o, "fq4", False) or _g(o, "mode") == "nvfp4"):
                by[s] = by.get(s, 0) + 1
    return {"spec": (_g(nv4, "SPEC", "") or "") if nv4 is not None else "", "projections": sum(by.values()), "by_slot": by}


def _nv4tab(backend) -> dict:
    r = _g(backend, "_nv4t")
    tab = _g(r, "table")
    if r is None or tab is None:
        return {"on": False}
    return {"on": True, "entries": sum(len(v) for v in tab.values()), "shapes": len(tab)}


def _actq(backend) -> dict:
    a = sys.modules.get("actq")
    asked = _g(backend, "_actq_parts")
    return {"asked": sorted(asked or ()), "active": sorted(_g(a, "PARTS", ()) or ()) if asked else [],
            "silu_eager": _g(a, "SILU_EAGER")}


def _n1(eng, backend) -> dict:
    """kai lever N1 (KAI_N1): asked = installed at start; on = still active after its start-up check (which can switch it
    off); projections = gate|up projections holding the interleaved rows"""
    m = sys.modules.get("n1")
    asked = _g(backend, "_n1_on") is not None
    n = 0
    for L in _g(eng, "layers", None) or []:
        gu = L.get("gu") if isinstance(L, dict) else None
        n += _g(gu, "n1w") is not None
    return {"asked": bool(asked), "on": bool(m is not None and _g(m, "ACTIVE", False)), "projections": n}


def _pdl() -> dict:
    """kai lever N2 (KAI_PDL): pdl.describe() when pdl.py was imported"""
    m = sys.modules.get("pdl")
    if m is None:
        return {"on": False}
    try:
        return dict(m.describe())
    except Exception as e:  # noqa: BLE001
        return {"on": bool(_g(m, "ON", False)), "error": f"{type(e).__name__}: {e}"}


def switches(pp, eng=None, backend=None) -> dict:
    """the resolved levers of this process (see the module doc)"""
    sw = {}
    try:
        rgh = pp.rgh_switches()
    except Exception as e:  # noqa: BLE001
        rgh = {"error": repr(e)}
    sw.update(rgh)
    f8 = _g(pp, "_F8")
    sw["f8tab"] = bool(f8 is not None and _g(f8, "_TABON", False))
    sw["h2d1"] = bool(_g(pp, "_H2D1", False))
    gd = {}
    try:
        gd = dict(pp.kai_gdn_switches())
    except Exception as e:  # noqa: BLE001
        gd = {"error": repr(e)}
    gd["path"] = "chain path only"
    gd["co_a64_staged"] = not (gd.get("gdnps") and gd.get("gdnps_ok") is not False)
    sw["gdnps"] = gd
    hci = bool(_g(pp, "_HCI", False))
    sw["hci"] = {"on": hci, "forward_groups": hci and sw["h2d1"], "forward_marked": hci}
    sw["rdasync"] = bool(_g(pp, "_RDASYNC", False))
    sw["early"] = _early(pp, eng)
    sw["b11"] = {"env": bool(_g(pp, "_B11", False)), "ovl": bool(_g(pp, "_OVL", False)),
                 "lanes": int(_g(eng, "n_streams", 1) or 1), "path": "every DeltaNet layer (not layer 0 under KAI_EARLY)"}
    sw["noxstore"] = _noxstore(pp, eng)
    sw["actq"] = _actq(backend)
    sw["nv4"] = _nv4(eng)
    sw["nv4tab"] = _nv4tab(backend)
    sw["n1"] = _n1(eng, backend)
    sw["pdl"] = _pdl()
    sw["minm"] = int(_g(eng, "nv4_minm", 0) or 0)
    r1g = _g(eng, "_r1g", None)
    sw["c1graph"] = {"env": bool(_g(pp, "_R1G", False)), "on": bool(_g(eng, "_r1g_on", False)),
                     "graphs": len(r1g) if isinstance(r1g, dict) else 0}
    if backend is not None:
        fam = _g(backend, "family")
        sw["family"] = fam
        sw["step_path"] = ("forward_marked" if fam == "reax1" else
                           "forward_groups" if _g(backend, "shared", False) else "forward")
    return sw


def counters(pp, eng=None, backend=None) -> dict:
    """usage counters (cumulative since the process started), flat"""
    c = {"gdnps_calls": int(_g(pp, "_GDNPS_CALLS", 0) or 0)}
    for k, v in (_g(pp, "_GDNFUSE_CALLS", None) or {}).items():  # kai lever #5: fused pass P / fused conv calls
        c[f"gdnfuse_{k}_calls"] = int(v)
    c.update({k: int(v) for k, v in (_g(pp, "KAI_COUNTS", None) or {}).items()})
    m = sys.modules.get("nv4_minm")
    for k, v in (_g(m, "COUNTS", None) or {}).items():
        c["minm_" + k] = int(v)
    r = _g(backend, "_nv4t")
    if r is not None:
        c["nv4t_hits"], c["nv4t_fallbacks"] = int(_g(r, "hits", 0)), int(_g(r, "fallbacks", 0))
    m = sys.modules.get("n1")
    if m is not None:
        for k, v in (_g(m, "stats", None) or {}).items():
            c["n1_" + k] = int(v)
    p = sys.modules.get("pdl")
    st = _g(p, "STATE", None) if p is not None else None
    if st:
        if st.get("f8g") is not None:
            c["pdl_f8g_calls"] = int(_g(st["f8g"], "calls", 0))
        if st.get("nv4"):
            c["pdl_nv4_calls"] = sum(int(_g(x, "calls", 0)) for x in st["nv4"])
    return c


class Status:
    """switches at start + counters since then (the warm-up's own counts are kept apart)"""

    def __init__(self, pp, eng=None, backend=None):
        self.pp, self.eng, self.be = pp, eng, backend
        try:
            self.sw = switches(pp, eng, backend)
        except Exception as e:  # noqa: BLE001 - logging must never fail the start
            self.sw = {"error": f"{type(e).__name__}: {e}"}
        self.warm = self._now()
        self._reported = False

    def _now(self) -> dict:
        try:
            return counters(self.pp, self.eng, self.be)
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}

    def served(self) -> dict:
        now = self._now()
        return {k: (v - self.warm.get(k, 0) if isinstance(v, int) else v) for k, v in now.items()}

    def start_line(self) -> str:
        return "kai switches: " + json.dumps({"switches": self.sw, "warm_counts": self.warm, "pid": os.getpid()},
                                             sort_keys=True, default=repr)

    def health(self) -> dict:
        return json.loads(json.dumps({"switches": self.sw, "counts": self.served(), "warm_counts": self.warm},
                                     default=repr))  # plain JSON types (serve.py's json.dumps has no default)

    def report_line(self):
        """the counts since start, once (shutdown); None afterwards"""
        if self._reported:
            return None
        self._reported = True
        return "kai counts: " + json.dumps(self.served(), sort_keys=True, default=repr)
