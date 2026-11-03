# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 team ruj (levers for pjs's Jeffan engine, E61/E62; each one off unless its switch is set)
"""Engine-core levers for the Jeffan engine, installed by sitecustomize.py in every process; they act only in the
spawned vLLM engine core, where the patched modules are imported.

JEF_ROWS=1  readout over the answer-code rows only. The engine reads one next-token distribution per read, restricted
            to the option codes (allowed_token_ids), so only the code columns of the logits matter. vLLM projects the
            last hidden state of every read onto all 262,144 rows of the tied embedding (2.8 GB read per engine
            step). Here the same projection (the same lm_head method, i.e. the batch-invariant matmul under
            VLLM_BATCH_INVARIANT=1, whose per-element K order does not depend on N) runs over the 255 code rows
            (JEF_CODE_IDS, set by the engine from the checkpoint's renderer), followed by the same soft-cap ops; the
            other columns are filled with the cap's floor (-soft_cap) and are masked to -inf by allowed_token_ids
            before any probability is formed. The code columns are the same values as before.
JEF_GEMM=1  lever G: the FP8 dense GEMMs (qkv / o / gate_up / down of all 60 layers, 86 % of the GPU time of a step)
            run on our CUTLASS SM120 kernels with a larger CTA tile per (N, K, M) instead of vLLM's single 64x64x128
            configuration. Same mainloop (16x8x32 e4m3 MMA, fp32 accumulation over K in increasing order, no split-K)
            and the same epilogue (scale_a * (scale_b * acc), round-to-nearest bf16): the same bits, verified per shape.
JEF_BF_MS=W lever BF (batch formation, with JEF_ASYNC=0 = vLLM's synchronous scheduler): after an engine step whose
            outputs finished F requests, the engine core waits up to W ms (from the end of that step) until F new
            requests are waiting (their successors: the re-reads and the next questions) before it schedules the next
            step, so the closed loop of the validator's 8 clients forms one large step per cycle instead of a large
            step followed by small ones. Order and contents of every read are unchanged.
JEF_BF_ALL=1 (with JEF_BF_MS) one group: the post-step wait also counts the requests that were already waiting when
            the step ended, i.e. it waits (up to W ms) until the successors of every request in flight are in, so all
            of the validator's reads gather into one step instead of two alternating halves.
JEF_STEP_TOK=T lever SB (step budget): the scheduler schedules at most T new tokens per engine step (vLLM's own
            per-step token budget, max_num_scheduled_tokens; default = max_num_batched_tokens 8192); the read that
            does not fit is split by vLLM's chunked prefill and its rest runs first in the next step. The FP8 GEMMs'
            cost per token is a saw-tooth in the step's token count M (CTA-tile waves: 60 layers cost 42.0 ms at
            M = 512 but 53.6 ms at M = 544, 81.8 ms at 1024 and 93.4 ms at 1040), so with T on a tile boundary (512,
            768, 1024) the steps of a busy engine sit at the cheap end of a tooth. The compiled graph, its compile
            range and the buffers are unchanged (only the scheduler's budget is lowered); every token's arithmetic
            is the same (batch-invariant kernels, prefix-cached K/V of the earlier chunk read back as written).
"""
from __future__ import annotations

import importlib.abc
import os
import sys

ROWS = os.environ.get("JEF_ROWS", "0") == "1"
GEMM = os.environ.get("JEF_GEMM", "0") == "1"
BF_MS = float(os.environ.get("JEF_BF_MS", "0") or 0)
BF_ALL = os.environ.get("JEF_BF_ALL", "0") == "1"
STEP_TOK = int(os.environ.get("JEF_STEP_TOK", "0") or 0)
FT_MEMO = os.environ.get("JEF_FT", "0") == "1"   # (ruj, bj-g6) lever FT's engine-core part: BF wait lookups memoised
GQ = os.environ.get("JEF_GQ", "0") == "1"        # (ruj, bj-g6) lever GQ: one-pass GELU-mul + FP8 quant (jef_gq.py)
# (E73 SA) step alignment: candidate step sizes (the FULL CUDA-graph sizes)
STEP_ALIGN = sorted({int(x) for x in os.environ.get("JEF_STEP_ALIGN", "").split(",") if x.strip()})
_HERE = os.path.dirname(os.path.abspath(__file__))
GEMM_DIR = os.environ.get("JEF_GEMM_DIR", os.path.join(_HERE, "gemm"))
GEMM_TABLE = os.environ.get("JEF_GEMM_TABLE", os.path.join(_HERE, "gemm_table.json"))
NV4 = os.environ.get("JEF_NV4", "").strip()
NV4_TABLE = os.environ.get("JEF_NV4_TABLE", os.path.join(_HERE, "nv4_table.json"))


def _log(msg):
    print(f"[jef-lever] {msg}", file=sys.stderr, flush=True)


def _patch_gemma4(m):
    if not ROWS:
        return
    ids_env = os.environ.get("JEF_CODE_IDS", "")
    ids = [int(x) for x in ids_env.split(",") if x.strip()]
    if not ids:
        _log("JEF_ROWS=1 but no JEF_CODE_IDS: readout unchanged")
        return
    cls = m.Gemma4ForCausalLM
    orig = cls.compute_logits

    def compute_logits(self, hidden_states):
        import torch
        import jef_e73lev   # (E73) the model, at its first logits call (profile run, weights final)
        jef_e73lev.on_model(self)
        st = getattr(self, "_jef_rows", None)
        if st is None:
            import types
            w = self.lm_head.weight
            idx = torch.tensor(ids, device=w.device, dtype=torch.long)
            rows = w.index_select(0, idx).contiguous()
            head = types.SimpleNamespace(weight=rows, quant_method=self.lm_head.quant_method,
                                         tp_size=getattr(self.lm_head, "tp_size", 1))
            st = self._jef_rows = (idx, head)
            _log(f"readout over {len(ids)} code rows (of {w.shape[0]})")
        idx, head = st
        lp = self.logits_processor
        if lp.logits_as_input or getattr(head, "tp_size", 1) != 1:
            return orig(self, hidden_states)
        codes = lp._apply_head(head, hidden_states, None)
        if lp.soft_cap is not None:
            codes = codes / lp.soft_cap
            codes = torch.tanh(codes)
            codes = codes * lp.soft_cap
        if lp.scale != 1.0:
            codes *= lp.scale
        fill = -float(lp.soft_cap) if lp.soft_cap is not None else -1e4
        out = codes.new_full((codes.shape[0], lp.org_vocab_size), fill)
        out.index_copy_(1, idx, codes)
        return out

    cls.compute_logits = compute_logits
    _log("JEF_ROWS on: Gemma4ForCausalLM.compute_logits over the code rows")


def _install_gemm(m=None):
    """JEF_GEMM=1 (lever G): vLLM's _C::cutlass_scaled_mm runs on our CUTLASS FP8 kernels (jef_fp8gemm.so, routed by
    jef_fp8override.so), with the CTA tile chosen per (N, K, M) from gemm_table.json. Same arithmetic, same bits
    (jef_fp8_common.cuh); the compiled graph and every other kernel are unchanged."""
    if not GEMM or _STATE.get("gemm"):
        return
    import json

    import torch
    try:
        torch.ops.load_library(os.path.join(GEMM_DIR, "jef_fp8gemm.so"))
        rows = json.load(open(GEMM_TABLE))["table"]
        flat = [int(v) for r in rows for v in r]
        torch.ops.jef.fp8_set_table(flat)
        torch.ops.load_library(os.path.join(GEMM_DIR, "jef_fp8override.so"))
        _STATE["gemm"] = True
        import jef_e73lev   # (E73) JEF_LLP: last-layer row pruning in front of jef::fp8_mm
        jef_e73lev.install_gemm_override()
        if NV4 and os.path.exists(NV4_TABLE):
            nrows = json.load(open(NV4_TABLE))["table"]
            torch.ops.jef.nv4_set_table([int(v) for r in nrows for v in r])
            _log(f"JEF_NV4 tile table: {len(nrows)} rows")
        _log(f"JEF_GEMM on: _C::cutlass_scaled_mm -> jef::fp8_mm ({len(rows)} table rows; cfgs "
             f"{torch.ops.jef.fp8_cfg_names()})")
    except Exception as exc:  # noqa: BLE001
        _log(f"JEF_GEMM could not be installed ({exc!r}); vLLM's kernel stays")


_GQ = {"calls": 0}


def _install_gq(m=None):
    """JEF_GQ=1 (lever GQ, team ruj bj-g6): the compiled graph's GELU(tanh)-mul + per-token FP8 quant kernel of the
    MLP (one generated source file, the same in every start state; jef_gq.GQ_FILE / GQ_NAME) is launched as
    jef_gq.jef_gq_kernel instead: the same Triton statements in one pass over the [M, 43008] gate_up row instead of two
    (the generated kernel reads the row once for the amax and again for the quantisation). Same q and scale bits
    (scripts/gq_test.py: torch.equal against the generated kernel). Hooked at CachingAutotuner.run (jef_pin.py patches
    benchmark_all_configs of the same class); every other kernel and the graph are unchanged."""
    if not GQ or _STATE.get("gq"):
        return
    try:
        import torch._inductor.runtime.triton_heuristics as th
        import jef_gq
    except Exception as exc:  # noqa: BLE001
        _log(f"JEF_GQ could not be installed ({exc!r}); Inductor's kernel stays")
        return
    cls = th.CachingAutotuner
    orig = cls.run

    def run(self, *args, **kwargs):
        mine = getattr(self, "_jef_gq", None)
        if mine is None:
            mine = (os.path.basename(getattr(self, "filename", None) or "") == jef_gq.GQ_FILE
                    and getattr(getattr(self, "fn", None), "__name__", "") == jef_gq.GQ_NAME
                    and len(args) == 5 and int(args[4]) == jef_gq.N_HALF)
            self._jef_gq = mine
            if mine:
                _log(f"JEF_GQ: {jef_gq.GQ_NAME} ({jef_gq.GQ_FILE}) runs as jef_gq.jef_gq_kernel (one pass)")
        if mine:
            _GQ["calls"] += 1
            jef_gq.gelu_quant(args[0], args[1], args[2], int(args[3]))
            return None
        return orig(self, *args, **kwargs)

    cls.run = run
    _STATE["gq"] = True
    _log("JEF_GQ=1: one-pass GELU-mul + FP8 quant hook on (CachingAutotuner.run)")


_NV4_MOD = {"self_attn.qkv_proj": "qkv", "self_attn.o_proj": "o", "mlp.gate_up_proj": "gu", "mlp.down_proj": "down"}


def _nv4_plan(spec, n_layers=60):
    out = {}
    for part in filter(None, spec.split(",")):
        types, _, rest = part.partition("@")
        rng, _, method = rest.partition(":")
        layers = []
        for r in (rng or f"0-{n_layers - 1}").split("+"):
            a, _, b = r.partition("-")
            layers += list(range(int(a), int(b or a) + 1))
        for t in types.split("+"):
            if t not in _NV4_MOD.values():
                raise ValueError(f"JEF_NV4: unknown projection type {t!r}")
            for i in layers:
                out[(i, t)] = 0 if method == "rtn" else 1
    return out


def _nv4_convert(model):
    """lever NV4: re-quantise the planned FP8 weights to NVFP4 and register them with jef::fp8_mm (see JEF_NV4)."""
    import re
    import time

    import torch
    plan = _nv4_plan(NV4)
    if not plan:
        return
    _install_gemm()
    if not _STATE.get("gemm"):
        _log("JEF_NV4 needs JEF_GEMM=1 (the NVFP4 path lives in jef::fp8_mm): NV4 off")
        return
    t0 = time.time()
    pat = re.compile(r"layers\.(\d+)\.(self_attn\.qkv_proj|self_attn\.o_proj|mlp\.gate_up_proj|mlp\.down_proj)$")
    keep, nbytes, seen = [], 0, set()
    for name, mod in model.named_modules():
        mt = pat.search(name)
        if not mt:
            continue
        key = (int(mt.group(1)), _NV4_MOD[mt.group(2)])
        if key not in plan:
            continue
        w = getattr(mod, "weight", None)
        if w is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2 or w.stride(0) != 1:
            _log(f"JEF_NV4: {name}: weight is not a column-major e4m3 [K, N] "
                 f"({None if w is None else (w.dtype, tuple(w.shape), w.stride())}); kept FP8")
            continue
        qw, sfw = torch.ops.jef.nv4_quant_w(w.t(), plan[key])
        torch.ops.jef.nv4_register(w, qw, sfw)
        keep.append((qw, sfw))
        nbytes += qw.numel() + sfw.numel()
        seen.add(key)
    torch.cuda.synchronize()
    _STATE["nv4_keep"] = keep
    miss = sorted(set(plan) - seen)
    _log(f"JEF_NV4 on: {len(keep)} projections NVFP4 ({NV4}; registered {torch.ops.jef.nv4_count()}), "
         f"{nbytes / 2**30:.2f} GiB, {time.time() - t0:.1f} s" + (f"; NOT found: {miss[:8]}" if miss else ""))


def _nv4_hits():
    try:
        import torch
        return int(torch.ops.jef.nv4_hits())
    except Exception:  # noqa: BLE001
        return -1


def _patch_runner(m):
    """NV4: convert right after vLLM's model load (online FP8 weights final), before profiling / compile / capture.
    vLLM 0.30 has two runners (v1/worker/gpu_model_runner.py and the V2 v1/worker/gpu/model_runner.py, the one this
    engine uses); both are patched when imported."""
    if not NV4:
        return
    cls = getattr(m, "GPUModelRunner", None)
    if cls is None or getattr(cls, "_jef_nv4_hooked", False):
        return
    cls._jef_nv4_hooked = True
    orig = cls.load_model

    def load_model(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            _nv4_convert(self.model)
        except Exception as exc:  # noqa: BLE001
            _log(f"JEF_NV4 FAILED ({exc!r}); the FP8 path stays")
        return out

    cls.load_model = load_model
    _log(f"JEF_NV4={NV4}: model-load hook on ({m.__name__}.GPUModelRunner)")


_BF = {"finished": 0, "t_out": 0.0, "waits": 0, "waited_ms": 0.0, "timeouts": 0, "t_rep": 0.0, "fills": 0, "steps": 0,
       "tok": 0, "full": 0, "hist": {}}


def _patch_sched(m):
    """BF: count the requests each step finishes and when its outputs were processed; SB: the step token budget."""
    if BF_MS <= 0 and STEP_TOK <= 0:
        return
    import time
    cls = m.Scheduler
    orig = cls.update_from_output

    def update_from_output(self, scheduler_output, model_runner_output):
        out = orig(self, scheduler_output, model_runner_output)
        try:
            n = sum(1 for eco in out.values() for o in eco.outputs if o.finish_reason is not None)
        except Exception:  # noqa: BLE001
            n = 0
        if n:
            _BF["finished"] += n
            _BF["t_out"] = time.monotonic()
        return out

    cls.update_from_output = update_from_output
    _log(f"JEF_BF_MS={BF_MS:g}: scheduler output counter on")
    if STEP_TOK > 0:
        oinit = cls.__init__

        def __init__(self, *args, **kwargs):
            oinit(self, *args, **kwargs)
            old = self.max_num_scheduled_tokens
            self.max_num_scheduled_tokens = min(STEP_TOK, old)
            _log(f"JEF_STEP_TOK={STEP_TOK}: per-step token budget {old} -> {self.max_num_scheduled_tokens} "
                 f"(chunked prefill {self.scheduler_config.enable_chunked_prefill})")

        cls.__init__ = __init__
    osched = cls.schedule

    def _pending(sched, stop):
        """(E73 SA) pending new tokens: running remainders + waiting reads, prefix hits excluded (up to stop)"""
        n = 0
        try:
            for r in sched.running:
                n += max(0, r.num_tokens - r.num_computed_tokens)
            kv = sched.kv_cache_manager
            for r in sched.waiting:
                try:
                    hit = kv.get_computed_blocks(r)[1]
                except Exception:  # noqa: BLE001
                    hit = 0
                n += max(1, r.num_tokens - hit)
                if n >= stop:
                    return n
        except Exception:  # noqa: BLE001
            return 0
        return n

    def schedule(self, *args, **kwargs):   # step-size counters for the BF report (no GPU work, no sync)
        cap = None
        if STEP_ALIGN and len(self.waiting) + len(self.running) >= 3:   # (E73 SA) not for a lone decision's reads
            budget = self.max_num_scheduled_tokens
            pend = _pending(self, budget + 1)
            b = max((x for x in STEP_ALIGN if x <= pend), default=0)
            if 0 < b < budget and pend <= budget:
                cap = budget
                self.max_num_scheduled_tokens = b
                _BF["aligned"] = _BF.get("aligned", 0) + 1
        try:
            so = osched(self, *args, **kwargs)
        finally:
            if cap is not None:
                self.max_num_scheduled_tokens = cap
        try:
            t = int(so.total_num_scheduled_tokens)
            if t > 0:
                _BF["steps"] += 1
                _BF["tok"] += t
                if t >= self.max_num_scheduled_tokens:
                    _BF["full"] += 1
                b = next(x for x in (128, 256, 384, 512, 640, 768, 896, 1024, 1536, 2048, 1 << 30) if t <= x)
                _BF["hist"][b] = _BF["hist"].get(b, 0) + 1
        except Exception:  # noqa: BLE001
            pass
        return so

    cls.schedule = schedule


def _patch_core(m):
    """BF: the post-step gather window of the engine core's busy loop (synchronous scheduler only)."""
    if BF_MS <= 0:
        return
    import json
    import queue
    import time
    cls = m.EngineCoreProc
    orig = cls._process_input_queue
    W = BF_MS / 1000.0

    def _budget_full(sched, memo=None):
        """JEF_STEP_TOK: True when the work already queued fills the next step's token budget (the rest of a split
        read plus the new tokens of the waiting reads, prefix-cache hits excluded). (ruj, bj-g6 lever FT) memo: the
        prefix-cache hit of each waiting request, looked up once per post-step wait (no step runs during the wait, so
        the cache's contents and the hits cannot change; get_computed_blocks is a read-only lookup)."""
        budget = sched.max_num_scheduled_tokens
        n = 0
        try:
            for r in sched.running:
                n += max(0, r.num_tokens - r.num_computed_tokens)
            kv = sched.kv_cache_manager
            for r in sched.waiting:
                hit = None if memo is None else memo.get(r.request_id)
                if hit is None:
                    try:
                        hit = kv.get_computed_blocks(r)[1]
                    except Exception:  # noqa: BLE001
                        hit = 0
                    if memo is not None:
                        memo[r.request_id] = hit
                n += max(1, r.num_tokens - hit)
                if n >= budget:
                    return True
        except Exception:  # noqa: BLE001
            return False
        return n >= budget

    def _process_input_queue(self):
        orig(self)
        want = _BF["finished"]
        if want <= 0:
            return
        _BF["finished"] = 0
        if getattr(self, "batch_queue", None):
            return
        sched = self.scheduler
        if BF_ALL:   # one group: also wait for the successors of the requests that were already waiting
            want += len(sched.waiting)
        t_end = _BF["t_out"] + W
        now = time.monotonic()
        memo = {} if FT_MEMO else None   # (ruj, bj-g6 lever FT) one prefix-hit lookup per request and wait
        if len(sched.waiting) >= want or now >= t_end or (STEP_TOK and _budget_full(sched, memo)):
            return
        t0 = now
        while len(sched.waiting) < want:
            left = t_end - time.monotonic()
            if left <= 0:
                _BF["timeouts"] += 1
                break
            try:
                req = self.input_queue.get(timeout=left)
            except queue.Empty:
                _BF["timeouts"] += 1
                break
            self._handle_client_request(*req)
            while not self.input_queue.empty():
                self._handle_client_request(*self.input_queue.get_nowait())
            if STEP_TOK and _budget_full(sched, memo):
                _BF["fills"] += 1
                break
        _BF["waits"] += 1
        _BF["waited_ms"] += 1000.0 * (time.monotonic() - t0)
        if time.monotonic() - _BF["t_rep"] > 10.0:
            _BF["t_rep"] = time.monotonic()
            st = max(1, _BF["steps"])
            _log(f"BF: {_BF['waits']} waits, mean {(_BF['waited_ms'] / max(1, _BF['waits'])):.2f} ms, "
                 f"{_BF['timeouts']} timeouts, {_BF['fills']} budget fills; steps {_BF['steps']}, "
                 f"{_BF['tok'] / st:.1f} tok/step, {_BF['full']} at the budget, {_BF.get('aligned', 0)} aligned, hist "
                 f"{json.dumps({str(k): v for k, v in sorted(_BF['hist'].items())})}"
                 + (f"; NV4 calls {_nv4_hits()}" if NV4 else ""))

    cls._process_input_queue = _process_input_queue
    _log(f"JEF_BF_MS={BF_MS:g}: post-step gather window on" + (" (one group: JEF_BF_ALL=1)" if BF_ALL else ""))


_STATE = {}
_TARGETS = {"vllm.model_executor.models.gemma4": _patch_gemma4, "vllm._custom_ops": _install_gemm,
            "vllm.v1.core.sched.scheduler": _patch_sched, "vllm.v1.engine.core": _patch_core}
if NV4:   # the runner hook only with NV4 (jef_steplog.py hooks the same module; two exec_module wrappers recurse)
    _TARGETS.update({"vllm.v1.worker.gpu_model_runner": _patch_runner, "vllm.v1.worker.gpu.model_runner": _patch_runner})
if GQ:    # (ruj, bj-g6) installed when vLLM's compile backend is imported (engine core, before the graph is compiled)
    _TARGETS["vllm.compilation.backends"] = _install_gq


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


if ROWS or GEMM or BF_MS > 0 or STEP_TOK > 0 or GQ:
    for _name, _fn in _TARGETS.items():
        if _name in sys.modules:
            _fn(sys.modules[_name])
    sys.meta_path.insert(0, _Finder())
