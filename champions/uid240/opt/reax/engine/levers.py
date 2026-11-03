"""levers.py (kai, dev only): switch the engine's levers at run time in ONE loaded engine (tools/leverbench.py).

Never imported by the production path (serve.py, server.py, pplx_engine.py, backends/qwen35.py). A LeverSet captures the
start state of a loaded mammoth / reax1 PplxEngine (process switches, engine attributes, the Linear object of every
projection slot) and set(levers) puts the engine into "start state + these changes". set() is absolute: a lever left
out goes back to its start value, so the order of configs never matters and set({}) restores the very start objects.
Everything a switch touches is read by the forward code at call time, so a switched engine runs the code of an engine
started with that lever (tests/test_kai_levers.py checks the objects and switches against started engines).

Keys: the env names with or without MAM_ / PPLX_ (case-insensitive), plus `nvfp4` (our KAI_ENGINE_OPTS option).
Bool values 1/0, on/off, true/false, yes/no. See SWITCHABLE; CONTAINER_ONLY lists what needs one container per variant.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Mapping, Optional

# key -> what it switches (doc; the bench prints it)
SWITCHABLE = {
    "F8": "MAM_F8: f8g FP8 rowwise GEMM for every FP8 Linear (pplx_engine._F8_RW; imports f8g if the engine started without)",
    "F8_GU": "MAM_F8_GU: f8g fused gate/up GEMM + SwiGLU/quant epilogue (pplx_engine._F8_GU)",
    "F8TAB": "MAM_F8TAB: rgh c8a measured tile table (f8g._TABON; the plan cache is dropped)",
    "H2D1": "MAM_H2D1: one pinned H2D copy of forward_groups' metadata (pplx_engine._H2D1)",
    "HCI": "KAI_HCI: kai lever I, fla chunk metadata built on the host and staged with H2D1 (pplx_engine._HCI; no head syncs)",
    "RDASYNC": "KAI_RDASYNC: kai lever I, readout counts copied from pinned memory without a host wait (pplx_engine._RDASYNC)",
    "EARLY": "KAI_EARLY: kai lever I, step head + layer-0 projections launched before forward_groups' metadata (pplx_engine._EARLY; needs H2D1)",
    "OVL": "MAM_OVL: side stream in the state-once DeltaNet layers (pplx_engine._OVL)",
    "B11": "KAI_B11: kai lever B11, bf16 in_ba of every DeltaNet layer on a side stream beside in_qkvz (pplx_engine._B11)",
    "SIDENT": "MAM_SIDENT: unique-state steps skip the state copy (pplx_engine._SIDENT)",
    "CHAIN": "MAM_CHAIN: gdnchain one-launch state recurrence (pplx_engine._CHAIN; needs MAM_FLA_PIN=1 at start)",
    "SCONV": "MAM_SCONV: lag-table conv (engine.sconv)",
    "GDNM": "MAM_GDNM: merged GDN stages under SCONV (engine.gdnm)",
    "SCONV3": "MAM_SCONV3: one-launch q/k/v conv under SCONV (engine.sconv3)",
    "GDNDIRECT": "MAM_GDNDIRECT: direct GDN forward under SCONV (engine.gdndirect)",
    "HOST_CU": "PPLX_HOST_CU: host-side cu_seqlens for fla (engine.host_cu)",
    "TRITON_ARQ": "PPLX_TRITON_ARQ: fused add+rms+FP8 quant (engine.triton_arq; off while our nvfp4 is on, as at start)",
    "NV4": "MAM_NV4: rgh NVFP4 W4A4 projections, spec or q48 (derived from /model on first use, cached; FP8 kept)",
    "NV4_MLP": "MAM_NV4_MLP: q48 preset MLP layers",
    "NV4_REST": "MAM_NV4_REST: q48 preset GDN / attention layers (none = MLP only)",
    "NV4_METHOD": "MAM_NV4_METHOD: q48 preset weight block scales rtn | mse",
    "NV4_GEMM": "MAM_NV4_GEMM: cutlass | cublaslt (nv4.GEMM)",
    "NV4_VAR": "MAM_NV4_VAR: default CUTLASS variant (read per call from the environment)",
    "NV4_ASCALE": "MAM_NV4_ASCALE: kai lever C, NV4 activation block scales rtn | 46 | mse (nv4qa; with NV4 configs)",
    "NV4_AGLOBAL": "MAM_NV4_AGLOBAL: kai lever C, NV4 activation global scale pow2 | pow2h | mse (nv4qa; with NV4 configs)",
    "nvfp4": "KAI_ENGINE_OPTS nvfp4=: our NVFP4 MLP layers (gu@a-b,down@a-b; needs nvfp4_shadow at start)",
    "PDL": "KAI_PDL: kai lever N2, programmatic dependent launch of the Triton / Inductor / f8g / NVFP4 kernels (pdl.py; "
           "registered when the image has pdl.py)",
    "ACTQ": "KAI_ACTQ: kai lever Q parts deq,xs,silu | 1 | 0 (actq.PARTS: NV4 activation quant fed from the FP8 producers; "
            "acts on NV4 projections only, needs LeverSet(..., nv4=True) to be on)",
    "GDNFUSE": "KAI_GDNFUSE: kai lever #5, kkt_solve fused into lever D's pass P + q/k l2norm and g cumsum fused into "
               "_sconv3 (gdnfuse.py; chain path; start-up bit check per part on the first switch-on)",
    "N1": "KAI_N1: kai lever N1, SwiGLU + down-quant amax in the NVFP4 gate|up epilogue (n1.ACTIVE; registered with "
          "LeverSet(..., nv4=True); interleaved rows kept beside the originals)",
    "NOXSTORE": "KAI_NOXSTORE: kai lever R8-5, add_rms_q without the unread bf16 x store before FP8 MLPs and FP8 attention "
                "layers (pplx_engine._NOXSTORE; registered when the image has add_rms_q_nox)",
    "NV4TAB": "MAM_NV4TAB: kai lever B, exact NVFP4 tile table (1 = shipped nv4t_tab.json | a .json path | 0): nv4t.Router "
              "over nv4's library, built and bit-checked (nv4t.check vs the variant nv4.py runs) on its first use; "
              "registered with LeverSet(..., nv4=True)",
}
_C_START = "read once at start"
_C_LOOP = "serving loop (step formation / host overlap): invisible to a fixed-batch bench, use quick.sh"
_C_ENC = "encoding / step planning of the fixed items"
CONTAINER_ONLY = {
    "FLA_PIN": _C_START + ": fla autotuners pinned at import; unpinning re-autotunes under load (E4 numerics lottery)",
    "compile": _C_START + ": torch.compile of the kernels (opt compile)",
    "E63": "pjs whole-step CUDA graphs: another engine (KAI_E63)",
    "linear": _C_START + ": weight format of every projection (opt linear)", "fp8_skip": _C_START + ": weight format",
    "nvfp4_scales": _C_START + ": our NVFP4 copies are built with one option set", "nvfp4_wscale": _C_START + ": as nvfp4_scales",
    "nvfp4_ascale": _C_START + ": as nvfp4_scales", "nvfp4_rot": _C_START + ": as nvfp4_scales",
    "nvfp4_smooth": _C_START + ": folded into the weights", "NV4_TAB": _C_START + ": per-projection variant table",
    "F8RQ2": _C_START + " (and inert: the shipped libf8g.so has no f8_gemm2)",
    "F8_MIN_M": "not wired (f8g constant)", "F8_RASTER_BYTES": "not wired (f8g constant)",
    "STREAMS": "not wired (engine constant)",
    "budget": _C_ENC, "max_input_tokens": _C_ENC, "shared_state": _C_ENC, "shared_group": _C_ENC, "shared_until": _C_ENC,
    "shared_mixq": _C_ENC, "encode_once": _C_ENC, "FASTENC": _C_ENC, "R1_FASTENC": _C_ENC, "warm_steps": _C_START,
    "R1_SCONV": "reax1 forward_marked lag table: read from the environment per call; not wired",
    "R1_C1GRAPH": _C_START + ": rgh u4r2 reax1 single-question CUDA graphs, captured after the warm-up (a switched engine "
                  "replays them only in its start state: r1g_follow)",
    "R1_C1GRAPH_RANGE": _C_START + ": the captured sequence lengths of R1_C1GRAPH",
    "FASTRENDER": _C_ENC + " (rgh m1 MAM_FASTRENDER: same texts, built with the backend)",
    "ADAPT": _C_LOOP, "ADAPT_S": _C_LOOP, "MAX_WAIT_MS": _C_LOOP, "SCHED_CLOSE": _C_LOOP, "ENCTHREAD": _C_LOOP,
    "ENCTHREADS": _C_LOOP, "GC_FREEZE": _C_LOOP, "SWITCH_INTERVAL": _C_LOOP,
}
MODULE_FLAGS = {"F8": "_F8_RW", "F8_GU": "_F8_GU", "H2D1": "_H2D1", "OVL": "_OVL", "SIDENT": "_SIDENT", "CHAIN": "_CHAIN",
                "HCI": "_HCI", "RDASYNC": "_RDASYNC", "EARLY": "_EARLY", "B11": "_B11"}
ENGINE_FLAGS = {"SCONV": "sconv", "GDNM": "gdnm", "SCONV3": "sconv3", "GDNDIRECT": "gdndirect", "HOST_CU": "host_cu"}
BOOLS = set(MODULE_FLAGS) | set(ENGINE_FLAGS) | {"F8TAB", "TRITON_ARQ"}
NV4_KEYS = ("NV4", "NV4_MLP", "NV4_REST", "NV4_METHOD")
PROJ_SLOTS = ("gu", "down", "in_qkvz", "out", "qkv", "o")  # nv4.SLOT values (rgh's projection types)
_LOWER = {"nvfp4", "compile", "linear", "fp8_skip", "nvfp4_scales", "nvfp4_wscale", "nvfp4_ascale", "nvfp4_rot",
          "nvfp4_smooth", "budget", "max_input_tokens", "shared_state", "shared_group", "shared_until", "shared_mixq",
          "encode_once", "warm_steps"}


def parse_bool(v: Any, key: str = "value") -> bool:
    s = str(v).strip().lower()
    if s in ("1", "on", "true", "yes"):
        return True
    if s in ("0", "off", "false", "no"):
        return False
    raise ValueError(f"{key}={v!r}: use 1/0, on/off, true/false")


def normalize_key(k: str) -> str:
    """'MAM_F8TAB' / 'f8tab' -> 'F8TAB'; 'KAI_E63' -> 'E63'; opts keep their lower-case names ('nvfp4', 'compile')"""
    s = str(k).strip()
    if s.lower() in _LOWER:
        return s.lower()
    u = s.upper()
    for p in ("MAM_", "PPLX_", "KAI_"):
        if u.startswith(p) and u != p:
            u = u[len(p):]
            break
    return u


def r1g_follow(eng, state: Mapping[str, Any], start: Mapping[str, Any]) -> None:
    """R1_C1GRAPH: the captured graphs hold the start state's kernels; they replay only while the engine is in it"""
    if getattr(eng, "_r1g", None):
        eng._r1g_on = all(state.get(k) == v for k, v in start.items())
    else:
        eng._r1g_on = False


def nv4_plan_of(t: Mapping[str, str], layer_types) -> list:
    """[(layer, type, method)] that nv4.install would replace for the NV4 keys of `t` (the engine's env names)"""
    import nv4
    import nv4_plan
    if t.get("NV4", "").strip() != "q48" and any(t.get(k, "").strip() for k in NV4_KEYS[1:]):
        raise ValueError("NV4_MLP / NV4_REST / NV4_METHOD only apply to NV4=q48")
    env = {"MAM_" + k: t.get(k, "") for k in NV4_KEYS}
    env["KAI_E63"] = os.environ.get("KAI_E63", "0")
    spec = nv4_plan.resolve(env)
    if not spec:
        return []
    if nv4_plan.needs_h(spec):  # rgh mx gptq / perm: Hessians from the seed set through the backend (_after_build only)
        raise ValueError(f"NV4 {spec!r}: calibrated methods (gptq / +perm) are installed at start-up only, not by LeverSet")
    meth = nv4.parse(spec, len(layer_types))
    return [(i, ty, meth[(i, ty)]) for i, ty in nv4_plan.projections(spec, list(layer_types))]


def _ours_units(spec: str, n_layers: int) -> set:
    """'gu@47-62,down@47-62' (our nvfp4= syntax; a single layer without '-' too) -> {(layer, kind)}"""
    out = set()
    for g in (x.strip() for x in spec.split(",") if x.strip()):
        kind, _, rng = g.lstrip(".").partition("@")
        if kind not in ("gu", "down"):
            raise ValueError(f"nvfp4 group {g!r}: gu@a-b or down@a-b")
        rs = rng.split("+") if rng else [f"0-{n_layers - 1}"]
        for r in rs:
            a, _, b = r.partition("-")
            a, b = int(a), int(b or a)
            if a > b or a < 0 or b >= n_layers:
                raise ValueError(f"nvfp4 group {g!r}: range {r!r} outside 0..{n_layers - 1}")
            out |= {(i, kind) for i in range(a, b + 1)}
    return out


def _actq_text(parts) -> str:
    return ",".join(sorted(parts))


class _Lever:
    __slots__ = ("get", "set", "parse")

    def __init__(self, get, set, parse):
        self.get, self.set, self.parse = get, set, parse


class LeverSet:
    """Run-time lever switching of one loaded engine. nv4=True patches PplxEngine._mlp with rgh's NV4 router once (what an
    image with MAM_NV4 set runs; for FP8 layers it calls the original); leave it False when no config uses NV4."""

    def __init__(self, engine, pp, model_dir: Optional[str] = None, *, nv4: bool = False,
                 log: Callable[[str], None] = lambda s: None):
        for a in ("layers", "n_layers", "layer_types", "linear_mode", "device", *ENGINE_FLAGS.values(), "triton_arq",
                  "nv4_shadow", "set_nvfp4"):
            if not hasattr(engine, a):
                raise RuntimeError(f"levers: the engine has no {a!r} (an image older than levers.py expects)")
        for a in (*MODULE_FLAGS.values(), "_F8", "_FLA_PINNED", "add_rms_q_triton"):
            if not hasattr(pp, a):
                raise RuntimeError(f"levers: pplx_engine has no {a!r} (an image older than levers.py expects)")
        self.eng, self.pp, self.model_dir, self.log = engine, pp, model_dir, log
        self.extra: dict[str, _Lever] = {}
        self.base = {(i, s): L[s] for i, L in enumerate(engine.layers) for s in PROJ_SLOTS if s in L}
        if any(getattr(o, "fq4", False) for o in self.base.values()):
            raise RuntimeError("levers: the engine started with rgh NVFP4 projections (MAM_NV4 set): start it with MAM_NV4= "
                               "so every switch has its FP8 Linear")
        self.arq_avail = (pp.add_rms_q_triton is not None and engine.linear_mode == "fp8"
                          and all(o.mode != "bf16" for o in self.base.values()))
        f8 = self._f8g(required=False)
        start = {k: bool(getattr(pp, a)) for k, a in MODULE_FLAGS.items()}
        start.update({k: bool(getattr(engine, a)) for k, a in ENGINE_FLAGS.items()})
        # the value an engine started without our nvfp4 has (a shadow engine starts with it off)
        start["TRITON_ARQ"] = os.environ.get("PPLX_TRITON_ARQ", "0") == "1" and self.arq_avail
        start["F8TAB"] = bool(f8._TABON) if f8 is not None else os.environ.get("MAM_F8TAB", "0") == "1"
        start.update({k: "" for k in NV4_KEYS})
        start["NV4_GEMM"] = os.environ.get("MAM_NV4_GEMM", "cutlass")
        start["NV4_VAR"] = os.environ.get("MAM_NV4_VAR", "")
        start["nvfp4"] = ""
        import actq
        start["ACTQ"] = _actq_text(actq.resolve(os.environ))
        self.start = start
        self.nv4 = bool(nv4)
        self.cache: dict = {}  # (layer, type, method) -> NV4Linear
        self._tab = None
        if self.nv4 and not getattr(pp, "_kai_nv4_patched", False):
            import nv4 as N
            N.patch_engine(pp)
            pp._kai_nv4_patched = True
        if self.nv4:
            import nv4 as N
            actq.install(N, pp, actq.PARTS)  # once (idempotent); reads actq.PARTS per call
        self.state = dict(start)
        self._apply(self.state, [], set())
        if self.nv4:
            self._register_nv4qa()
            self._register_nv4tab()
            self._register_n1()
        self._register_f8c()
        self._register_pdl()
        self._register_gdnfuse()
        self._register_noxstore()

    # ---------------------------------------------------------------- extension
    def register(self, name: str, get: Callable[[], Any], set: Callable[[Any], None],
                 parse: Callable[[Any], Any] = lambda v: v) -> None:
        """a lever of another module (guard, split-K, ...): get/set a value the forward reads at call time"""
        key = normalize_key(name)
        if key in self.start:
            raise ValueError(f"lever {key} exists")
        self.extra[key] = _Lever(get, set, parse)
        self.start[key] = get()
        self.state[key] = self.start[key]

    def _register_nv4qa(self) -> None:
        """kai lever C (nv4qa: NV4 activation scaling): nv4.quant_x routed through nv4qa (its default modes call rgh's
        nv4q.quant_dyn itself), the two modes registered as levers. An image without nv4qa.py has neither."""
        try:
            import nv4qa
        except ImportError:
            return
        import nv4
        nv4qa.install(nv4)
        self.register("NV4_ASCALE", get=lambda: nv4qa.BLOCK, set=lambda v: nv4qa.set_modes(block=v),
                      parse=nv4qa.parse_block)
        self.register("NV4_AGLOBAL", get=lambda: nv4qa.GLOBAL, set=lambda v: nv4qa.set_modes(glob=v),
                      parse=nv4qa.parse_global)

    def _register_f8c(self) -> None:
        """kai lever R5-1 (f8c, MAM_F8CUT: CUTLASS sm120 FP8 GEMMs with f8g's math): F8CUT = 0 | 1 (f8c._TAB plans) | a
        MAM_F8CUT_TAB spec (those plans for every M >= f8g.MIN_M). Absent when f8g, f8c.py or libf8c.so is missing."""
        f8 = self._f8g(required=False)
        if f8 is None:
            return
        try:
            import f8c
            f8c.load()
        except (ImportError, OSError):
            return
        self.register("F8CUT", get=lambda: f8c.lever_get(f8), set=lambda v: f8c.lever_set(f8, v), parse=f8c.lever_parse)

    def _register_pdl(self) -> None:
        """kai lever N2 (pdl.py, KAI_PDL): programmatic dependent launch of the step's Triton / Inductor / f8g / NVFP4
        kernels, switched per pass (installed on the first PDL=1; off = the original launches). An image without pdl.py
        has no such lever."""
        try:
            import pdl
        except ImportError:
            return
        pp = self.pp
        self.register("PDL", get=lambda: pdl.ON, set=lambda v: pdl.switch(v, pp, log=self.log),
                      parse=lambda v: parse_bool(v, "PDL"))

    def _register_noxstore(self) -> None:
        """kai lever R8-5 (KAI_NOXSTORE): pplx_engine._NOXSTORE, read per producer call by _core_gen. An image without
        add_rms_q_nox has no such lever. Switching it on runs the twin without a start-up check: the bench's bitwise
        comparison is the evidence (the image runs pplx_engine.check_noxstore at start)."""
        pp = self.pp
        if not hasattr(pp, "_NOXSTORE") or not hasattr(pp.K, "add_rms_q_nox"):
            return

        def set_(v):
            pp._NOXSTORE = bool(v)
        self.register("NOXSTORE", get=lambda: bool(pp._NOXSTORE), set=set_, parse=lambda v: parse_bool(v, "NOXSTORE"))

    def _register_nv4tab(self) -> None:
        """kai lever B (nv4t.py, MAM_NV4TAB) as a lever: the start-time install (backends/qwen35.py, after nv4.install)
        needs MAM_NV4 at start, which the bench never has. On its first use per table the Router is built over nv4's
        library and bit-checked (nv4t.check, entries that differ are dropped) against the variant nv4.py runs for each
        (M, N, K) (NV4Linear.variant over rgh's per-bucket table); on = nv4._lib is that Router (its base = the library
        before it, so PDL's routers stay underneath), off = that base. Registered before PDL, so pdl.switch(True) in the
        same set() routes the Router's base and tile library as the start-time hook does. An image without nv4t.py has no
        NV4TAB."""
        try:
            import nv4t
        except ImportError:
            return
        import types

        import nv4
        box: dict = {"path": "", "routers": {}}

        def parse(v):
            s = "" if v is None else str(v).strip()
            if s.lower() in ("", "0", "off", "false", "no"):
                return ""
            if s.lower() in ("on", "true", "yes"):
                s = "1"
            try:
                path = nv4t.resolve({"MAM_NV4TAB": s})
                nv4t.load(path)
            except (ValueError, OSError) as exc:
                raise ValueError(f"NV4TAB={v!r}: {exc}") from None
            return path

        def ref(M, N, K):
            return nv4.NV4Linear.variant(types.SimpleNamespace(tab=self.tab(), out_features=N, in_features=K), M)

        def below():
            cur = nv4.lib()
            return cur.base if isinstance(cur, nv4t.Router) else cur

        def set_(path):
            if not path:
                if isinstance(getattr(nv4, "_lib", None), nv4t.Router):
                    nv4._lib = nv4._lib.base
                box["path"] = ""
                return
            base = below()
            r = box["routers"].get(path)
            if r is None:
                import torch
                r = nv4t.Router(base, nv4t.open_lib(), nv4t.load(path))
                kept, dropped = nv4t.check(r, torch, torch.device(self.eng.device), ref)
                self.log(f"levers: NV4TAB {os.path.basename(path)}: {kept} entries ({dropped} dropped by the bit check)")
                box["routers"][path] = r
            r.base = base
            nv4._lib = r
            box["path"] = path
        self.register("NV4TAB", get=lambda: box["path"], set=set_, parse=parse)

    def _register_gdnfuse(self) -> None:
        """kai lever #5 (gdnfuse.py, KAI_GDNFUSE): the exact GDN fusions on the chain path, switched per pass
        (pplx_engine.gdnfuse_switch; the start-up check of both parts runs at the first switch-on and a failed part
        stays off). An image without gdnfuse_switch has no GDNFUSE."""
        pp = self.pp
        if not hasattr(pp, "gdnfuse_switch"):
            return
        def set_(v):
            on = pp.gdnfuse_switch(bool(v))
            c = getattr(pp, "_GDNFUSE_CALLS", {})  # proof the fused path ran: the fused call counts at every switch
            self.log(f"GDNFUSE {'on' if on else 'off'}: fused calls so far pre {c.get('pre', 0)} conv {c.get('conv', 0)}, "
                     f"checks {dict(getattr(pp, '_GDNFUSE_OK', {}))}")
            return on

        self.register("GDNFUSE", get=lambda: bool(pp._GDNFUSE_ON), set=set_, parse=lambda v: parse_bool(v, "GDNFUSE"))

    def _register_n1(self) -> None:
        """kai lever N1 (n1.py, KAI_N1): the fused SwiGLU + amax gate|up epilogue as a lever. Switching it on interleaves
        the gate|up rows of the NV4 MLP projections in place KEEPING the original rows (lin.n1w beside lin.wq), so the
        off state runs rgh's path at its own speed; n1.ACTIVE is read per call. An image without n1.py has no N1."""
        try:
            import n1
        except ImportError:
            return
        import nv4

        def parse(v):
            b = parse_bool(v, "N1")
            if b:
                try:
                    n1.open_lib()
                except OSError as exc:
                    raise ValueError(f"N1: libnv4s.so cannot be loaded ({exc})") from None
            return b

        def set_(on):
            if on:
                for _, lin in n1.targets(self.eng, nv4):
                    if getattr(lin, "n1w", None) is None:
                        n1.interleave(lin, keep=True)
                n1.install_patches(nv4, self.pp)
            n1.ACTIVE = bool(on)
        self.register("N1", get=lambda: bool(n1.ACTIVE), set=set_, parse=parse)

    # ---------------------------------------------------------------- resolve
    def resolve(self, levers: Mapping[str, Any]) -> dict:
        """start state + the changes, validated (ValueError); nothing is applied"""
        t = dict(self.start)
        for k, v in levers.items():
            key = normalize_key(k)
            if key in CONTAINER_ONLY:
                raise ValueError(f"{k}: not switchable at run time ({CONTAINER_ONLY[key]}); one container per variant")
            if key in BOOLS:
                t[key] = parse_bool(v, key)
            elif key == "ACTQ":
                import actq
                t[key] = _actq_text(actq.parse(v))
            elif key in NV4_KEYS or key in ("NV4_GEMM", "NV4_VAR", "nvfp4"):
                t[key] = "" if v is None else str(v).strip()
            elif key in self.extra:
                t[key] = self.extra[key].parse(v)
            else:
                raise ValueError(f"{k}: unknown lever (switchable: {', '.join(sorted(SWITCHABLE) + sorted(self.extra))})")
        if t["CHAIN"] and not self.start["CHAIN"] and not self.pp._FLA_PINNED:
            raise ValueError("CHAIN=1 needs MAM_FLA_PIN=1 at start (the pins are container-only)")
        if t["TRITON_ARQ"] and not self.arq_avail:
            raise ValueError("TRITON_ARQ=1: add_rms_q_triton unavailable or not an all-FP8 engine")
        if t["ACTQ"] != self.start["ACTQ"] and not self.nv4:
            raise ValueError("ACTQ acts on rgh's NV4 projections: LeverSet(..., nv4=True)")
        if t["NV4_GEMM"] not in ("cutlass", "cublaslt"):
            raise ValueError(f"NV4_GEMM={t['NV4_GEMM']!r}: cutlass | cublaslt")
        if t["NV4_VAR"] and not t["NV4_VAR"].isdigit():
            raise ValueError(f"NV4_VAR={t['NV4_VAR']!r}: a CUTLASS variant number")
        if any(t[k] and not self.start[k] for k in ("F8", "F8_GU", "F8TAB")) and self._f8g(required=False) is None:
            raise ValueError("F8 / F8_GU / F8TAB: f8g (libf8g.so) cannot be loaded")
        return t

    # ---------------------------------------------------------------- apply
    def set(self, levers: Mapping[str, Any]) -> dict:
        """switch to start state + `levers`; returns the levers that differ from the start state. Atomic: on ValueError
        (or a failed NV4 derivation) the engine is left as it was."""
        t = self.resolve(levers)
        plan = nv4_plan_of(t, self.eng.layer_types)
        if plan and not self.nv4:
            raise ValueError("NV4 configs need LeverSet(..., nv4=True) (rgh's _mlp router patched at start)")
        ours = _ours_units(t["nvfp4"], self.eng.n_layers)
        if ours and not self.eng.nv4_shadow:
            raise ValueError("nvfp4=: the engine holds no NVFP4 copies (start it with nvfp4_shadow=gu,down)")
        if ours - set(self.eng.nv4_shadow):
            raise ValueError(f"nvfp4=: no NVFP4 copy of {sorted(ours - set(self.eng.nv4_shadow))[:4]}")
        both = sorted({(i, ty) for i, ty, _ in plan} & ours)
        if both:
            raise ValueError(f"nvfp4= and NV4 both replace {both[:4]}: one NVFP4 path per projection")
        self._derive(plan)
        self._apply(t, plan, ours)
        self.state = t
        return {k: v for k, v in t.items() if v != self.start[k]}

    def _apply(self, t, plan, ours):
        eng, pp = self.eng, self.pp
        r1g_follow(eng, t, self.start)
        for (i, s), obj in self.base.items():
            eng.layers[i][s] = obj
        if eng.nv4_shadow:
            eng.set_nvfp4(ours)
        if plan:
            import nv4
            for i, ty, m in plan:
                eng.layers[i][nv4.SLOT[ty]] = self.cache[(i, ty, m)]
        if (t["F8"] or t["F8_GU"]) and pp._F8 is None:
            pp._F8 = self._f8g(required=True)
        for k, a in MODULE_FLAGS.items():
            setattr(pp, a, bool(t[k]))
        for k, a in ENGINE_FLAGS.items():
            setattr(eng, a, bool(t[k]))
        eng.triton_arq = bool(t["TRITON_ARQ"]) and not ours
        f8 = self._f8g(required=False)
        if f8 is not None:
            f8._TABON = bool(t["F8TAB"])
            f8._plans.clear()
        if self.nv4:
            import nv4
            nv4.GEMM = t["NV4_GEMM"]
            env = {"MAM_" + k: t[k] for k in NV4_KEYS}
            import nv4_plan
            nv4.SPEC = nv4_plan.resolve({**env, "KAI_E63": os.environ.get("KAI_E63", "0")})
        if t["NV4_VAR"]:
            os.environ["MAM_NV4_VAR"] = t["NV4_VAR"]
        else:
            os.environ.pop("MAM_NV4_VAR", None)
        if self.nv4:
            import actq
            actq.PARTS = actq.parse(t["ACTQ"])
        for k, lv in self.extra.items():
            lv.set(t[k])

    # ---------------------------------------------------------------- NV4 derivation (nv4.install's, without the release)
    def _derive(self, plan):
        todo = [(i, ty, m) for i, ty, m in plan if (i, ty, m) not in self.cache]
        if not todo:
            return
        import time

        import nv4
        if self.model_dir is None:
            raise ValueError("NV4: LeverSet needs model_dir to derive the NVFP4 weights")
        t0 = time.time()
        get, close = self._loader()
        try:
            for i, ty, m in todo:
                b = self.base[(i, nv4.SLOT[ty])]
                w = nv4.weight(get, i, ty)
                self.cache[(i, ty, m)] = nv4.NV4Linear(w, m, b.out_features, b.in_features, self.tab())
                del w
        finally:
            close()
        if str(self.eng.device).startswith("cuda"):
            import torch
            torch.cuda.empty_cache()
        self.log(f"levers: {len(todo)} NV4 projections derived in {time.time() - t0:.1f} s ({len(self.cache)} cached)")

    def tab(self):
        if self._tab is None:
            import nv4
            p = os.environ.get("MAM_NV4_TAB", os.path.join(os.path.dirname(os.path.abspath(nv4.__file__)), "nv4_tab.json"))
            self._tab = json.load(open(p)) if os.path.exists(p) else {}
        return self._tab

    def _loader(self):
        """nv4.install's tensor getter: the checkpoint's bf16 tensors, opened on the engine's device"""
        import safetensors
        d = self.model_dir
        ip = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(ip):
            idx = json.load(open(ip))["weight_map"]
        else:
            with safetensors.safe_open(os.path.join(d, "model.safetensors"), framework="pt", device="cpu") as sf:
                idx = {k: "model.safetensors" for k in sf.keys()}
        pre = "language_model." if any(k.startswith("language_model.") for k in idx) else "model.language_model."
        dev = str(self.eng.device)
        handles: dict = {}

        def get(key):
            f = idx[pre + key]
            h = handles.get(f)
            if h is None:
                h = handles[f] = safetensors.safe_open(os.path.join(d, f), framework="pt", device=dev)
            return h.get_tensor(pre + key)
        return get, handles.clear

    def _f8g(self, required: bool):
        if self.pp._F8 is not None:
            return self.pp._F8
        try:
            import f8g
            return f8g
        except (OSError, ImportError) as exc:
            if required:
                raise ValueError(f"f8g unavailable: {exc}") from None
            return None

    def describe(self) -> dict:
        return {"start": dict(self.start), "switchable": sorted(SWITCHABLE) + sorted(self.extra),
                "nv4": self.nv4, "nvfp4_shadow": sorted({k for _, k in self.eng.nv4_shadow}),
                "nv4_cached": len(self.cache)}
