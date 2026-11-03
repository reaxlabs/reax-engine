# SPDX-License-Identifier: Apache-2.0
"""P-mode (BLINK_VISION=1): picture steps of the Blink engine run exactly what an --enforce-eager vision engine runs.

The blink slot and the vision slot start the same image with the same REAX_MODEL. Text steps (T-mode) keep v2b's
compiled path unchanged: Inductor graphs, FULL CUDA graphs and every BLINK_* lever. A step that carries a picture
request (P-mode) runs the numerics of the vision reference (vLLM 0.30 + --enforce-eager, batch-invariant):
  * the plain model forward: need_eager for the CUDA-graph dispatch and skip_compiled in the forward context (vLLM's own
    "run the uncompiled forward" switches, model_runner.py / compilation/decorators.py);
  * every CustomOp instance bound to forward_cuda, which is what an eager engine binds at construction (custom_ops
    "all"); a compiled engine binds forward_native (or a torch.compiled native inside opaque ops);
  * vLLM-IR op priorities and torch-op wrapping as an eager engine sets them (["vllm_c", "native"], wrap off);
  * every v2b lever that touches the forward, the logits or the sampler OFF (module attributes swapped back to the
    stock functions for the step): K1, LAST, LABEL_HEAD, READ, the MoE family (stock run_cutlass_moe_fp4), PAR
    (stock Gemma4DecoderLayer.forward), DSAMP, NOMETA; RED / pins / RTR act on Inductor code only and are counted;
  * kept on in P steps: blink_bi's per-M tile table (the v5 engine's exact BI matmul configs, vision tower shapes
    included) and v5's vision tower kernels (vfuse.py, VIS_FUSE / VIS_ENC_GRAPH), both proven exact in v5.
The P flag covers the step's execute_model AND its sample_tokens (the logits and the sampler run in sample_tokens in
the V2 model runner). This wrapper is the outermost layer over blink_host's execute_model wrapper.
Scheduling (blink_core.py with BLINK_VISION): picture requests never share a step with text requests (no-mix filter
in Scheduler.schedule), are scheduled whole (never chunked), and run in vLLM's own step loop (no PIPE gather, no
pre-scheduling). Picture requests carry their own prefix-cache salt (frontend), so their KV blocks never mix with
compiled-numerics text KV.
Census (BLINK_PM_CENSUS=1, default): every hook that may only act in T steps counts its calls while a P step runs; the
first P warm-up step must leave every counter at 0, else the engine start fails. Start-up asserts: vLLM's own encoder
CUDA graphs / compile are off, and no attention group mixes sliding and full-attention layers.
Installed by sitecustomize.py (imported LAST so that its runner wrapper is the outermost one).
"""
import contextlib
import importlib.abc
import inspect
import os
import sys
import time

ON = os.environ.get("BLINK_VISION", "0") == "1"
CENSUS = os.environ.get("BLINK_PM_CENSUS", "1") == "1"
CAP = int(os.environ.get("BLINK_PM_TOKENS", "8192"))         # P steps: whole requests up to this many tokens
TRACE = os.environ.get("BLINK_PM_TRACE", "")                  # development only: one JSON line per P step
TCOUNT = os.environ.get("BLINK_PM_TCOUNT", "1") == "1"        # log Triton compiles after the ready marker
READY_PREFIX = "pmready-"      # request id of the frontends' last warm-up decision
N_FRONTENDS = max(1, int(os.environ.get("BLINK_FRONTENDS", "1") or 1))   # ready once every frontend sent its marker
GROUP_HOLD_S = 3.0             # a warm-up group (pmwarm-g<n>-<tag>-) is held at most this long for its last member
# L1: v2b levers that replicate kernels the eager reference runs too, allowed in P steps one group at a time after
# their own gate (default off): K1 (dense GEMM chain), MOE (the NVFP4 MoE family = stock run_cutlass_moe_fp4's
# arithmetic), PAR (MoE on the aux stream, same kernels), LAST (last layer on the last rows, batch invariance),
# HEAD (letter-only LM head: every other logit is masked to -inf by allowed_token_ids before any reduction)
_ENV_L1 = {k: os.environ.get("BLINK_PM_" + k, "0") == "1" for k in ("K1", "MOE", "PAR", "LAST", "HEAD")}
TUNE = os.environ.get("BLINK_PM_TUNE_FILE", "")   # development only: live L1 switches (JSON {"K1": 0|1, ...})
_PMG_ON = os.environ.get("BLINK_PM_GRAPHS", "0") == "1"       # development only: G1 P-mode FULL graphs (blink_pmg.py)
ATTNLOG = os.environ.get("BLINK_PM_ATTNLOG", "0") == "1"     # development: log every new unified_attention constexpr set
SYNCDBG = int(os.environ.get("BLINK_PM_SYNCDEBUG", "0") or 0)   # development only: list host syncs of the first N P steps
_pp = os.environ.get("BLINK_PM_PROF", "")       # development only: "dir:first_step:n_steps" torch.profiler over P steps
PROF = (_pp.split(":")[0], int(_pp.split(":")[1]), int(_pp.split(":")[2])) if _pp else None


class _PM:
    on = False                 # True while a P step's execute_model / sample_tokens runs
    graph = False              # G1 (dev): this P step replays a P-mode FULL graph (blink_pmg.py)
    cur = None                 # TRACE: the running P step's record
    pend = []                  # TRACE: records waiting for their events
    ids = set()                # request ids of picture requests the runner holds
    steps = 0
    mixed = 0
    t_on = 0.0
    census = {}                # hook -> calls during P steps
    census_checked = False
    compiles = []              # Triton compiles (time, name, in_p) after the engine reported ready
    ready = False


_ST = {"insts": [], "swaps": None, "ir": None, "n_ops": 0, "runner": None, "errors": []}


def _log(msg):
    print(f"[blink-pm] {msg}", file=sys.stderr, flush=True)


def is_p_request(req):
    """a scheduler Request (or NewRequestData) that carries a picture"""
    for f in getattr(req, "mm_features", None) or ():
        if getattr(f, "modality", None) in ("image", "video"):
            return True
    return False


# ------------------------------------------------------------------------------------------------ census
def _count(name):
    if _PM.on:
        _PM.census[name] = _PM.census.get(name, 0) + 1


def _census_wrap(owner, attr, name):
    """wrap owner.attr so that calls during P steps are counted (T steps: one flag check)"""
    fn = getattr(owner, attr, None)
    if fn is None or getattr(fn, "_pm_census", False):
        return
    import functools

    @functools.wraps(fn)
    def wrapped(*a, **k):
        if _PM.on:
            _PM.census[name] = _PM.census.get(name, 0) + 1
        return fn(*a, **k)

    wrapped._pm_census = True
    setattr(owner, attr, wrapped)


# ------------------------------------------------------------------------------------------------ CustomOp tracking
def _patch_custom_op(m):
    cls = m.CustomOp
    if getattr(cls.__init__, "_pm", False):
        return
    orig = cls.__init__

    def __init__(self, *a, **k):
        orig(self, *a, **k)
        _ST["insts"].append(self)

    __init__._pm = True
    cls.__init__ = __init__


def _p_method(inst):
    """the method an --enforce-eager engine binds (CustomOp.dispatch_forward with the op enabled, CUDA platform)"""
    from vllm.platforms import current_platform
    if current_platform.is_rocm():
        return inst.forward_hip
    return inst.forward_cuda


def _closure_fn(fn, var):
    try:
        return inspect.getclosurevars(fn).nonlocals.get(var)
    except Exception:  # noqa: BLE001
        return None


def _build(runner):
    """the per-step swap list (built once; rebuilt if CustomOps were created later)"""
    swaps = []
    ops = 0
    seen = set()
    for inst in _ST["insts"]:
        if id(inst) in seen:
            continue
        seen.add(id(inst))
        t = inst.__dict__.get("_forward_method")
        if t is None:
            continue
        p = _p_method(inst)
        if getattr(t, "__func__", None) is getattr(p, "__func__", object()) and getattr(t, "__self__", None) is inst:
            continue
        swaps.append((inst, "_forward_method", t, p))
        ops += 1
    names = {}
    for inst, _, t, _ in swaps:
        k = type(inst).__name__
        names[k] = names.get(k, 0) + 1
    # v2b levers, OFF in P steps unless their L1 gate passed (_l1)
    l1 = _l1()
    _ST["l1"] = dict(l1)
    bb = sys.modules.get("blink_bi")
    if bb is not None:
        if getattr(bb, "LABEL_HEAD", False) and not l1["HEAD"]:
            swaps.append((bb, "LABEL_HEAD", True, False))
        if getattr(bb, "_K1", None) is not None and not l1["K1"]:
            swaps.append((bb, "_K1", bb._K1, None))
        if getattr(bb, "_LAST", None) is not None and not l1["LAST"]:
            swaps.append((bb, "_LAST", bb._LAST, None))
    k1m = sys.modules.get("blink_k1")
    if k1m is not None and l1["K1"] and _k1p_table():
        swaps.append((k1m, "cfg", k1m.cfg, _k1p_cfg(k1m.cfg)))
    if k1m is not None and l1["K1"] and callable(getattr(k1m, "_cta", None)):
        # P steps run the MoE on the main stream (no PAR), so a K1 row's CTA cap "moe" (leave the w13 GEMM's SMs free)
        # only slows the dense GEMM: all SMs instead (scheduling only, the same fp32 chain per element), and no
        # device-property query per call
        def _cta_p(c, m):
            v = c.get("CTA", 0)
            return 0 if v == "moe" else int(v)
        swaps.append((k1m, "_cta", k1m._cta, _cta_p))
    cm = sys.modules.get("vllm.model_executor.layers.fused_moe.experts.cutlass_moe")
    if cm is not None and getattr(cm, "_blink_orig_run_cutlass_moe_fp4", None) is not None and not l1["MOE"]:
        swaps.append((cm, "run_cutlass_moe_fp4", cm.run_cutlass_moe_fp4, cm._blink_orig_run_cutlass_moe_fp4))
    g4 = sys.modules.get("vllm.model_executor.models.gemma4")
    if g4 is not None and not l1["PAR"]:
        cur = g4.Gemma4DecoderLayer.__dict__.get("forward")
        stock = getattr(g4.Gemma4DecoderLayer, "_pm_stock_forward", None)
        if stock is not None and cur is not stock:
            swaps.append((g4.Gemma4DecoderLayer, "forward", cur, stock))
    mr = sys.modules.get("vllm.v1.worker.gpu.model_runner")
    if mr is not None:
        cur = mr.GPUModelRunner.__dict__.get("sample")
        stock = getattr(mr.GPUModelRunner, "_pm_stock_sample", None)
        if stock is not None and cur is not stock:
            swaps.append((mr.GPUModelRunner, "sample", cur, stock))
    bh = sys.modules.get("blink_host")
    if bh is not None and isinstance(getattr(bh, "_LIVE", None), dict):
        _ST["host_live"] = bh._LIVE
    _ST["swaps"] = swaps
    _ST["n_ops"] = ops
    _ST["n_insts"] = len(_ST["insts"])
    # IR priorities / torch wrap of an eager engine (platforms/cuda.py get_default_ir_op_priority, mode NONE)
    from vllm.ir.op import IrOp
    from vllm import envs
    default = ["vllm_c", "native"]
    rms = (["oink"] + default) if envs.VLLM_USE_OINK_OPS else default
    want = {"rms_norm": rms, "fused_add_rms_norm": rms, "gelu_and_mul_sparse": ["triton", "native"]}
    ir = []
    for name, prio in want.items():
        op = IrOp.registry.get(name)
        if op is not None:
            ir.append((op, prio))
    _ST["ir"] = ir
    _log(f"swap list: {ops} CustomOp instances to forward_cuda ({names}); {len(swaps) - ops} lever swaps "
         f"{[(type(o).__name__ if not isinstance(o, type(sys)) else o.__name__) + '.' + a for o, a, _, _ in swaps[ops:]]}; "
         f"IR {[(o.name, p) for o, p in ir]}")


K1P_PATH = os.environ.get("BLINK_PM_K1P", os.path.join(os.path.dirname(os.path.abspath(__file__)), "k1_tiles_p.json"))


def _k1p_table():
    """P-only K1 rows for the row counts of picture steps (above the T table's last row), {(N, K): [(max_m, cfg)]};
    every tile width is the same fp32 chain (k1p_bench.py bit-checks each one against today's path)"""
    if "k1p" not in _ST:
        tab = {}
        try:
            import json
            with open(K1P_PATH) as fh:
                raw = json.load(fh)
            for key, rows in raw.get("gemm", {}).items():
                tab[tuple(int(x) for x in key.split(","))] = sorted(((int(m), c) for m, c in rows),
                                                                   key=lambda x: x[0])
            _log(f"P-only K1 rows {K1P_PATH}: {len(tab)} shapes")
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001
            _log(f"P-only K1 rows not loaded: {exc!r}")
        _ST["k1p"] = tab
    return _ST["k1p"]


def _k1p_cfg(orig):
    tab = _k1p_table()

    def cfg(part, n, k, m):
        c = orig(part, n, k, m)
        if c is not None or part != "gemm":
            return c
        for max_m, cc in tab.get((n, k), ()):
            if m <= max_m:
                return cc
        return None

    return cfg


def _l1():
    """the L1 levers allowed in P steps: env defaults, overridden live by BLINK_PM_TUNE_FILE (development A/B)"""
    cur = dict(_ENV_L1)
    if TUNE:
        now = time.monotonic()
        if now - _ST.get("tune_t", 0.0) > 0.5:
            _ST["tune_t"] = now
            try:
                import json
                with open(TUNE) as fh:
                    j = json.load(fh)
                _ST["tune"] = {k: bool(j[k]) for k in list(cur) + ["PG"] if k in j}
            except Exception:  # noqa: BLE001
                pass
        cur.update({k: v for k, v in _ST.get("tune", {}).items() if k in cur})
    return cur


@contextlib.contextmanager
def pctx():
    """the P-mode context: entered for a picture step's execute_model and for its sample_tokens"""
    if (_ST["swaps"] is None or len(_ST["insts"]) != _ST.get("n_insts")
            or (TUNE and _l1() != _ST.get("l1"))):
        _build(_ST["runner"])
    import vllm.ir as vir
    swaps = _ST["swaps"]
    live = _ST.get("host_live")
    old_live = dict(live) if live is not None else None
    stack = contextlib.ExitStack()
    n_ops = _ST["n_ops"]
    for obj, attr, _t, p in swaps[:n_ops]:
        obj.__dict__[attr] = p
    for obj, attr, _t, p in swaps[n_ops:]:
        setattr(obj, attr, p)
    if live is not None:      # blink_host's live switches: DSAMP, NOMETA and XDIET (FUSEW, NODRAFT) off for the step
        for key in ("dsamp", "nometa", "fusew", "nodraft"):
            if key in live:
                live[key] = False
    _PM.on = True
    dbg = SYNCDBG and _PM.steps <= SYNCDBG
    if dbg:
        _sync_debug(True)
    try:
        for op, prio in _ST["ir"]:
            stack.enter_context(op.set_priority(prio))
        stack.enter_context(vir.enable_torch_wrap(False))
        yield
    finally:
        try:
            stack.close()
            if dbg:
                _sync_debug(False)
        finally:
            _PM.on = False
            for obj, attr, t, _p in reversed(swaps[n_ops:]):
                setattr(obj, attr, t)
            for obj, attr, t, _p in swaps[:n_ops]:
                obj.__dict__[attr] = t
            if live is not None:
                live.update(old_live)


# ------------------------------------------------------------------------------------------------ runner
def _p_step(runner, so):
    """True when this scheduler output schedules a picture request"""
    ids = _PM.ids
    for rid in getattr(so, "finished_req_ids", ()) or ():
        ids.discard(rid)
    p = t = 0
    for r in so.scheduled_new_reqs:
        if not _PM.ready and str(r.req_id).startswith(READY_PREFIX):
            _PM.n_ready = getattr(_PM, "n_ready", 0) + 1
            if _PM.n_ready >= N_FRONTENDS:
                _PM.ready = True
                _log(f"engine ready markers from all {N_FRONTENDS} frontend(s); Triton compiles from now on are logged")
        if is_p_request(r):
            ids.add(r.req_id)
            p += 1
        else:
            t += 1
    for rid in so.scheduled_cached_reqs.req_ids:
        if rid in ids:
            p += 1
        else:
            t += 1
    if p and t:
        _PM.mixed += 1
        _log(f"WARNING: a step mixes {p} picture and {t} text requests (runs as P)")
    return p > 0


def _patch_runner(m):
    cls = m.GPUModelRunner
    if getattr(cls.execute_model, "_pm", False):
        return
    # stock functions that v2b levers replaced (the P swaps put them back for picture steps)
    rs = cls.__dict__.get("sample")
    if getattr(rs, "_blink_read", False):
        stock = _closure_fn(rs, "orig")
        if stock is not None and getattr(stock, "__qualname__", "") == "GPUModelRunner.sample":
            cls._pm_stock_sample = stock
        else:
            _ST["errors"].append("blink_read's stock GPUModelRunner.sample not found")

    orig_d = m.dispatch_cg_and_sync_dp

    def dispatch_cg_and_sync_dp(*a, **k):
        if _PM.on:
            if _PM.graph:
                _count("pm.dispatch_pgraph")
            else:
                k["need_eager"] = True
                _count("pm.dispatch_eager")
        return orig_d(*a, **k)

    m.dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp
    orig_f = m.set_forward_context

    def set_forward_context(*a, **k):
        if _PM.on:
            k["skip_compiled"] = True
        return orig_f(*a, **k)

    m.set_forward_context = set_forward_context
    orig_exec = cls.execute_model
    orig_samp = cls.sample_tokens
    orig_load = cls.load_model
    orig_kv = cls.initialize_kv_cache

    def execute_model(self, scheduler_output, *a, **k):
        dummy = k.get("dummy_run", False) or (len(a) >= 2 and bool(a[1]))
        if dummy:
            self._pm_step = False
            return orig_exec(self, scheduler_output, *a, **k)
        p = _p_step(self, scheduler_output)
        self._pm_step = p
        if not p:
            return orig_exec(self, scheduler_output, *a, **k)
        _PM.steps += 1
        t0 = time.perf_counter()
        if PROF:
            _prof_step(_PM.steps)
        if TRACE:
            _ev_poll()
            _PM.cur = {"ev": "pstep", "n": len(scheduler_output.num_scheduled_tokens),
                       "tok": scheduler_output.total_num_scheduled_tokens,
                       "enc": sum(len(v) for v in (scheduler_output.scheduled_encoder_inputs or {}).values()),
                       "t0": time.time(), "evs": [_ev()]}
        pmg = sys.modules.get("blink_pmg") if _PMG_ON else None
        use_g = pmg is not None and _ST.get("tune", {}).get("PG", True) and pmg.qualifies(scheduler_output)
        if use_g:       # G1 (dev): replay a P-mode FULL graph for this step's LM forward
            saved = self.cudagraph_manager
            self.cudagraph_manager = pmg._G["mgr"]
            pmg._G["active"] = True
            _PM.graph = True
        try:
            with pctx():
                r = orig_exec(self, scheduler_output, *a, **k)
        finally:
            if use_g:
                self.cudagraph_manager = saved
                pmg._G["active"] = False
                _PM.graph = False
        if TRACE:
            _PM.cur["g"] = int(use_g)
            _PM.cur["exec_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
            _PM.cur["evs"].append(_ev())
        return r

    def sample_tokens(self, *a, **k):
        if not getattr(self, "_pm_step", False):
            return orig_samp(self, *a, **k)
        self._pm_step = False
        t0 = time.perf_counter()
        with pctx():
            r = orig_samp(self, *a, **k)
        if TRACE and getattr(_PM, "cur", None) is not None:
            _PM.cur["samp_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
            _PM.cur["evs"].append(_ev())
            _PM.pend.append(_PM.cur)
            _PM.cur = None
        if CENSUS and not _PM.census_checked:
            _check_census()
        return r

    def load_model(self, *a, **k):
        r = orig_load(self, *a, **k)
        _ST["runner"] = self
        cc = self.vllm_config.compilation_config
        bad = [f for f in ("cudagraph_mm_encoder", "compile_mm_encoder") if getattr(cc, f, False)]
        if bad:
            raise RuntimeError(f"[blink-pm] vLLM encoder graphs / compile must stay off for P-mode: {bad}")
        if TRACE:
            msc = type(getattr(self, "model_state", None))
            orig_pie = getattr(msc, "prepare_inputs_embeds", None)
            if orig_pie is not None and not getattr(orig_pie, "_pm", False):
                def prepare_inputs_embeds(ms, *pa, _o=orig_pie, **pk):
                    out = _o(ms, *pa, **pk)
                    if _PM.on and getattr(_PM, "cur", None) is not None:
                        _PM.cur["evs"].append(_ev())
                        _PM.cur["enc_ms"] = round((time.time() - _PM.cur["t0"]) * 1e3, 3)
                    return out
                prepare_inputs_embeds._pm = True
                msc.prepare_inputs_embeds = prepare_inputs_embeds
        _log(f"load_model: cudagraph_mm_encoder={getattr(cc, 'cudagraph_mm_encoder', None)} "
             f"compile_mm_encoder={getattr(cc, 'compile_mm_encoder', None)}; {len(_ST['insts'])} CustomOp instances")
        return r

    def initialize_kv_cache(self, *a, **k):
        r = orig_kv(self, *a, **k)
        _check_groups(self)
        return r

    if _PMG_ON:
        orig_cap = cls.capture_model

        def capture_model(self, *ca, **ck):
            r = orig_cap(self, *ca, **ck)
            if not ck.get("profile_only", False):
                import blink_pmg
                try:
                    blink_pmg.capture(self, pctx)
                except Exception as exc:  # noqa: BLE001  (dev: G1 off, the engine keeps eager P steps)
                    import traceback
                    blink_pmg._G["mgr"] = None
                    blink_pmg._G["active"] = False
                    blink_pmg._G["why"] = f"capture failed: {exc!r}"
                    _log(f"G1 P-graph capture FAILED, P steps stay eager: {exc!r}\n{traceback.format_exc()[-1500:]}")
            return r

        cls.capture_model = capture_model
    execute_model._pm = True
    cls.execute_model = execute_model
    cls.sample_tokens = sample_tokens
    cls.load_model = load_model
    cls.initialize_kv_cache = initialize_kv_cache
    cgm = sys.modules.get("vllm.v1.worker.gpu.cudagraph_utils")
    owner = getattr(cgm, "CudaGraphManager", None) if cgm is not None else None
    if owner is not None:
        _census_wrap(owner, "run_pw_graph", "cudagraph.run_pw_graph")
        orig_rf = owner.run_fullgraph
        if not getattr(orig_rf, "_pm_census", False):
            def run_fullgraph(mgr, *ra, **rk):
                if _PM.on:
                    pmg = sys.modules.get("blink_pmg")
                    key = "pm.pgraph" if pmg is not None and mgr is pmg._G.get("mgr") else "cudagraph.run_fullgraph"
                    _PM.census[key] = _PM.census.get(key, 0) + 1
                return orig_rf(mgr, *ra, **rk)
            run_fullgraph._pm_census = True
            owner.run_fullgraph = run_fullgraph
        if _PMG_ON:
            import blink_pmg
            blink_pmg.patch_cg_context(cgm)
    _log(f"installed in pid {os.getpid()} (P steps: eager forward, forward_cuda CustomOps, eager IR priorities; "
         f"levers off; census {CENSUS}; whole requests <= {CAP} tokens)")


def _check_groups(runner):
    """no attention group may hold both sliding and full-attention layers (gemma4_mm clears the image ranges of the
    full-attention layers in place on their group's metadata object)"""
    try:
        cfg = runner.model_config.hf_config
        tcfg = getattr(cfg, "text_config", cfg)
        types = list(getattr(tcfg, "layer_types", []) or [])
        bad = []
        n = 0
        for groups in runner.attn_groups:
            for g in groups:
                kinds = set()
                for name in g.layer_names:
                    if ".layers." not in name:
                        continue
                    i = int(name.split(".layers.")[1].split(".")[0])
                    if i < len(types):
                        kinds.add(types[i])
                n += 1
                if len(kinds) > 1:
                    bad.append(sorted(kinds))
        if bad:
            raise RuntimeError(f"[blink-pm] an attention group mixes layer types {bad}")
        kc = getattr(runner, "kv_cache_config", None)
        ng = len(kc.kv_cache_groups) if kc is not None else "?"
        _log(f"attention groups checked: {n} groups in {ng} KV-cache group(s), none mixes sliding and full layers "
             f"({[sorted(set(types[int(x.split('.layers.')[1].split('.')[0])] for x in g.layer_names if '.layers.' in x)) for gs in runner.attn_groups for g in gs]})")
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"[blink-pm] attention group check failed: {exc!r}") from None


def _check_census():
    """after the first P step: every T-only hook must have stayed silent"""
    _PM.census_checked = True
    l1 = _ST.get("l1") or {}
    allowed = {"blink_moe.run_cutlass_moe_fp4": l1.get("MOE"), "blink_k1.mm": l1.get("K1"),
               "blink_last.mm": l1.get("LAST"), "blink_last.moe_rows": l1.get("LAST") or l1.get("PAR")}
    t_only = {k: v for k, v in _PM.census.items() if not k.startswith("pm.") and not allowed.get(k)}
    if _ST["errors"]:
        raise RuntimeError(f"[blink-pm] P-mode set-up errors: {_ST['errors']}")
    if t_only:
        raise RuntimeError(f"[blink-pm] T-only hooks fired in a P step: {t_only}")
    if not (_PM.census.get("pm.dispatch_eager") or _PM.census.get("pm.dispatch_pgraph")):
        raise RuntimeError("[blink-pm] the P step did not take the P dispatch")
    _log(f"census of the first P step: T-only hooks silent; {_PM.census}")
    try:        # the frontends serve pictures only once this file exists (blink_engine.py _start_vision)
        with open(os.environ.get("BLINK_PM_OK_FILE", "/tmp/blink-pm-ok"), "w") as fh:
            fh.write(f"{os.getpid()} {time.time():.3f} {_PM.census}\n")
    except OSError as exc:
        raise RuntimeError(f"[blink-pm] cannot write the P-mode ok file: {exc!r}") from None


def _ev():
    import torch
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def _ev_poll(block=False):
    """write the trace records of finished P steps: host ms of execute_model / sample_tokens and the GPU timeline ms
    between the step's events (start, encoder done, execute end, sample end)"""
    keep = []
    for rec in _PM.pend:
        evs = rec["evs"]
        if not block and not evs[-1].query():
            keep.append(rec)
            continue
        evs[-1].synchronize()
        rec["gpu_ms"] = [round(evs[i].elapsed_time(evs[i + 1]), 3) for i in range(len(evs) - 1)]
        del rec["evs"]
        _trace(rec)
    _PM.pend = keep


_PROF = {}


def _prof_step(n):
    """development: torch.profiler (CPU + CUDA) over P steps [first, first + n_steps); the step after the window
    writes key_averages tables (by CPU self time and by CUDA time) and a chrome trace to the directory"""
    d, first, cnt = PROF
    import torch
    if n == first and "p" not in _PROF:
        acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        _PROF["p"] = torch.profiler.profile(activities=acts, record_shapes=False, with_stack=False)
        _PROF["p"].__enter__()
        _PROF["t0"] = time.time()
        _log(f"profiler on at P step {n}")
    elif n == first + cnt and "p" in _PROF and "done" not in _PROF:
        _PROF["done"] = True
        torch.cuda.synchronize()
        p = _PROF["p"]
        p.__exit__(None, None, None)
        try:
            os.makedirs(d, exist_ok=True)
            ka = p.key_averages()
            with open(os.path.join(d, f"prof_cpu_{os.getpid()}.txt"), "w") as fh:
                fh.write(ka.table(sort_by="self_cpu_time_total", row_limit=60, max_name_column_width=90))
            with open(os.path.join(d, f"prof_cuda_{os.getpid()}.txt"), "w") as fh:
                fh.write(ka.table(sort_by="cuda_time_total", row_limit=80, max_name_column_width=90))
            p.export_chrome_trace(os.path.join(d, f"prof_trace_{os.getpid()}.json"))
            _log(f"profiler: {cnt} P steps written to {d} ({time.time() - _PROF['t0']:.1f} s window)")
        except Exception as exc:  # noqa: BLE001
            _log(f"profiler write failed: {exc!r}")


_SYNC = {"n": {}, "hooked": False, "reported": False}


def _sync_debug(on):
    """development: torch's sync debug mode during P steps; every synchronizing call is counted by its Python stack"""
    import torch
    import warnings
    if not _SYNC["hooked"]:
        _SYNC["hooked"] = True
        orig_show = warnings.showwarning

        def show(message, category, filename, lineno, file=None, line=None):
            if "synchroniz" in str(message) and _PM.on:
                import traceback
                fr = [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in traceback.extract_stack()[:-2]]
                key = str(message).split(".")[0][:80] + " @ " + " < ".join(reversed(fr[-7:]))
                _SYNC["n"][key] = _SYNC["n"].get(key, 0) + 1
                return
            return orig_show(message, category, filename, lineno, file, line)

        warnings.showwarning = show
        warnings.simplefilter("always")
    torch.cuda.set_sync_debug_mode(1 if on else 0)
    if not on and _PM.steps >= SYNCDBG and not _SYNC["reported"]:
        _SYNC["reported"] = True
        tot = sum(_SYNC["n"].values())
        _log(f"sync census over {SYNCDBG} P steps: {tot} synchronizing calls, {len(_SYNC['n'])} sites")
        for k, v in sorted(_SYNC["n"].items(), key=lambda kv: -kv[1]):
            _log(f"  sync x{v}: {k}")


def _trace(rec):
    import json
    try:
        rec["t"] = round(time.time(), 4)
        with open(TRACE, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------------------------------------ other hooks
def _patch_gemma4(m):
    """remember the stock Gemma4DecoderLayer.forward before blink_par replaces it (blink_par imports first in
    sitecustomize, so the stock function is taken from its closure)"""
    cls = m.Gemma4DecoderLayer
    cur = cls.__dict__.get("forward")
    if cur is None:
        return
    stock = cur
    if getattr(cur, "__module__", "") != m.__name__:
        stock = _closure_fn(cur, "orig_forward")
        if stock is None or getattr(stock, "__qualname__", "") != "Gemma4DecoderLayer.forward":
            _ST["errors"].append("stock Gemma4DecoderLayer.forward not found")
            return
    cls._pm_stock_forward = stock


def _patch_census_inductor(m):
    """T-only: every Inductor-generated Triton kernel launch (RED and the pins act there)"""
    cls = getattr(m, "CachingAutotuner", None)
    if cls is not None:
        _census_wrap(cls, "run", "inductor.CachingAutotuner.run")


def _patch_census_extern(m):
    from torch._inductor.select_algorithm import extern_kernels
    _census_wrap(extern_kernels, "mm_dtype", "inductor.extern_kernels.mm_dtype")


def _patch_cutlass_moe(m):
    if getattr(m, "_blink_orig_run_cutlass_moe_fp4", None) is not None:
        _census_wrap(m, "run_cutlass_moe_fp4", "blink_moe.run_cutlass_moe_fp4")


def _patch_scheduler(m):
    """no-mix filter + whole picture requests (P steps take whole requests up to CAP tokens)"""
    cls = m.Scheduler
    if getattr(cls.schedule, "_pm", False):
        return
    orig = cls.schedule
    from vllm.v1.core.sched.request_queue import create_request_queue

    import re
    grp_re = re.compile(r"^pmwarm-g(\d+)-([0-9a-z]+)-")

    def schedule(self, *a, **k):
        reqs = list(self.waiting) + list(self.skipped_waiting)
        if not reqs:
            return orig(self, *a, **k)
        # P warm-up groups (frontend 0's start-up, blink_engine._pm_warmup): the n requests of a group run as ONE step
        # of exactly those n requests, so the warm-up covers each step-shape class whatever the encoder speed
        groups = {}
        for r in reqs:
            m = grp_re.match(str(r.request_id))
            if m:
                groups.setdefault(m.group(2), (int(m.group(1)), []))[1].append(r)
        if groups:
            now = time.time()
            ready = [t for t, (n, rs) in groups.items()
                     if len(rs) >= n or now - min(x.arrival_time for x in rs) > GROUP_HOLD_S]
            if ready:
                tag = min(ready, key=lambda t: min(x.arrival_time for x in groups[t][1]))
                keep = set(id(x) for x in groups[tag][1])
                held_g = [x for x in reqs if id(x) not in keep]
            else:
                held_g = [x for _, rs in groups.values() for x in rs]
            return _held_schedule(self, orig, held_g, a, k)
        kinds = [is_p_request(r) for r in reqs]
        pend = [r for r in self.running if r.num_computed_tokens < r.num_prompt_tokens]
        if not any(kinds) and not any(is_p_request(r) for r in pend):
            return orig(self, *a, **k)
        if pend:      # a prefill continuation is scheduled first: its kind decides the step
            want = is_p_request(min(pend, key=lambda r: r.arrival_time))
        else:
            first = min(range(len(reqs)), key=lambda i: reqs[i].arrival_time)
            want = kinds[first]
        held = [r for r, kd in zip(reqs, kinds) if kd != want]
        if want:      # picture step: whole requests, oldest first, up to CAP tokens
            tok = 0
            for r in sorted((r for r, kd in zip(reqs, kinds) if kd), key=lambda r: r.arrival_time):
                n = r.num_tokens - r.num_computed_tokens
                if tok and tok + n > CAP:
                    held.append(r)
                else:
                    tok += n
        if not held:
            return orig(self, *a, **k)
        return _held_schedule(self, orig, held, a, k)

    def _held_schedule(self, orig, held, a, k):
        """schedule() with the given waiting requests held back (re-queued in front, original order)"""
        if not held:
            return orig(self, *a, **k)
        hs = set(id(r) for r in held)
        hw = [r for r in self.waiting if id(r) in hs]
        hk = [r for r in self.skipped_waiting if id(r) in hs]
        if hw:
            self.waiting.remove_requests(hw)
        if hk:
            self.skipped_waiting.remove_requests(hk)
        try:
            return orig(self, *a, **k)
        finally:
            for q, rs in ((self.waiting, hw), (self.skipped_waiting, hk)):
                if rs:
                    tmp = create_request_queue(self.policy)
                    for r in rs:
                        tmp.add_request(r)
                    q.prepend_requests(tmp)

    schedule._pm = True
    cls.schedule = schedule


def _patch_encoder_runner(m):
    """every vision-encoder execution runs with P bindings: in P steps it already does; outside them (vLLM's encoder
    memory profile at start) it is wrapped in the P-mode context too, so anything an encoder kernel module captures
    lazily at its first call (vfuse's encoder graphs, a whole-encoder graph with the multimodal embedder's vLLM RMSNorm)
    holds the eager (forward_cuda) numerics, never the compiled engine's forward_native binding"""
    cls = getattr(m, "EncoderRunner", None)
    if cls is None or getattr(cls.execute_mm_encoder, "_pm", False):
        return
    orig = cls.execute_mm_encoder

    def execute_mm_encoder(self, *a, **k):
        if _PM.on or _ST.get("runner") is None:
            return orig(self, *a, **k)
        _count("pm.encoder_outside_p")
        with pctx():
            return orig(self, *a, **k)

    execute_mm_encoder._pm = True
    cls.execute_mm_encoder = execute_mm_encoder


_ATTN = {"pe": set(), "pg": set(), "t": set()}


def _patch_attnlog(m):
    """development (BLINK_PM_ATTNLOG=1): every distinct constexpr / launch-option set of kernel_unified_attention, by
    phase: pe = eager P step, pg = P-mode graph capture (G1), t = text; each new set is logged once (the G1 gate
    requires set(pg) == set(pe))"""
    fn = getattr(m, "kernel_unified_attention", None)
    if fn is None or getattr(fn, "_pm_attnlog", False):
        return

    class _Logged:
        _pm_attnlog = True

        def __init__(self, f):
            self._f = f

        def __getitem__(self, grid):
            launch = self._f[grid]

            def run(*a, **k):
                pmg = sys.modules.get("blink_pmg")
                ph = ("pg" if (pmg is not None and pmg._G.get("active")) else "pe") if _PM.on else "t"
                key = tuple(sorted((kk, vv) for kk, vv in k.items()
                                   if (kk.isupper() or kk in ("num_warps", "num_stages")) and not hasattr(vv, "shape")))
                if key not in _ATTN[ph]:
                    _ATTN[ph].add(key)
                    _log(f"attn constexpr NEW ({ph}, #{len(_ATTN[ph])}): {dict(key)}")
                return launch(*a, **k)

            return run

        def __getattr__(self, n):
            return getattr(self._f, n)

    m.kernel_unified_attention = _Logged(fn)


def _patch_triton(m):
    """count Triton JIT compiles once the engine is ready (none may happen during timed runs)"""
    if getattr(m.compile, "_pm", False):
        return
    orig = m.compile

    def compile(src, *a, **k):
        if _PM.ready:
            name = getattr(getattr(src, "fn", None), "__name__", None) or getattr(src, "name", "?")
            _PM.compiles.append((round(time.time(), 3), str(name), _PM.on))
            _log(f"Triton compile after ready: {name} (P step {_PM.on}); total {len(_PM.compiles)}")
        return orig(src, *a, **k)

    compile._pm = True
    m.compile = compile


_PATCHES = {
    "vllm.model_executor.custom_op": _patch_custom_op,
    "vllm.v1.worker.gpu.mm.encoder_runner": _patch_encoder_runner,
    "vllm.v1.worker.gpu.model_runner": _patch_runner,
    "vllm.model_executor.models.gemma4": _patch_gemma4,
    "vllm.v1.core.sched.scheduler": _patch_scheduler,
}
if CENSUS:
    _PATCHES.update({
        "torch._inductor.runtime.triton_heuristics": _patch_census_inductor,
        "torch._inductor.kernel.mm": _patch_census_extern,
        "vllm.model_executor.layers.fused_moe.experts.cutlass_moe": _patch_cutlass_moe,
    })
if TCOUNT:
    _PATCHES["triton.compiler.compiler"] = _patch_triton
if ATTNLOG:
    _PATCHES["vllm.v1.attention.ops.triton_unified_attention"] = _patch_attnlog
if _PMG_ON:
    def _patch_tattn(m):
        import blink_pmg
        blink_pmg.patch_builder(m)
    _PATCHES["vllm.v1.attention.backends.triton_attn"] = _patch_tattn


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


def _install_late():
    """census hooks on v2b's own modules (imported before this one by sitecustomize); vfuse's E5 capture predicate"""
    vf = sys.modules.get("vfuse")
    e5 = getattr(vf, "_E5", None) if vf is not None else None
    if isinstance(e5, dict) and "capture_ok" in e5:
        e5["capture_ok"] = lambda: _PM.on      # E5 captures its whole-encoder graphs only inside the P-mode context
    if not CENSUS:
        return
    k1 = sys.modules.get("blink_k1")
    if k1 is not None:
        for f in ("mm",):
            _census_wrap(k1, f, f"blink_k1.{f}")
    last = sys.modules.get("blink_last")
    if last is not None:
        for f in ("mm", "moe_rows"):
            _census_wrap(last, f, f"blink_last.{f}")


if ON:
    for _name, _fn in _PATCHES.items():
        if _name in sys.modules:
            _fn(sys.modules[_name])
    sys.meta_path.insert(0, _Finder())
    _install_late()
