#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Jeffan engine (REAX engine track v2, slot `jeffan`): our decision server around an in-process vLLM 0.30.0.

Contract: POST /internal/v1/decide on $RUNTIME_UDS (header X-S1M-Internal-Token, token from $S1M_INTERNAL_TOKEN or the
file $S1M_INTERNAL_TOKEN_FILE), one question per call; GET /health (200 once warm, 503 before).
Model protocol: the checkpoint's own dm_decision_engine renderer (tattrongvu/Jeffan-31b-1710, inference/, Apache-2.0),
imported at start from the read-only weights mount (nothing of it is shipped in this image): layout state_first,
single-token answer codes, the next-token distribution at the answer position restricted to the codes (processed
logprobs), temperature 1; a question whose top-1 probability is below 0.9 is read again with the read-twice prompt and
that distribution replaces the first.
Runtime: vLLM (Apache-2.0) in-process (AsyncLLM, engine core in a SPAWNED process), the checkpoint's serving flags
(online FP8 weights, FP8 KV cache, batch-invariant kernels, prefix caching), CUDA graphs over a ladder (piecewise by default; this image sets FULL graphs up to 1,024 tokens through JEF_CG_MODE / JEF_GRAPH_SIZES, /opt/e62/dispatch.py),
a short gather window, and the re-read issued as soon as the first read returns.

Environment (optional): JEF_GRAPH_SIZES (comma list | "default"), JEF_GATHER_MS (4), JEF_QUIET_MS (1.0), JEF_UTIL (0.44)
"""
from __future__ import annotations

import asyncio
import hmac
import json
import math
import os
import sys
import time
import uuid

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

MODEL_DIR = os.environ.get("REAX_MODEL_DIR", "/model")
MODEL_NAME = "jeffan-31b-1710"
TEMPERATURE = 1.0
REREAD_BELOW = 0.9
MAX_PROMPT = 8192           # first read
MAX_REREAD = 16440          # read-twice prompt (vLLM max_model_len 16448 - 8)
QTYPES = ("choice", "score", "noul")
SPEC2 = os.environ.get("JEF_SPEC2", "0") == "1"   # (ruj lever S2) issue read 2 together with read 1
PAR1 = os.environ.get("JEF_PAR_REREAD", "0") == "1"   # (E62 J6) the same, only for a decision that is alone
PAR_QUIET = float(os.environ.get("JEF_PAR_QUIET_S", "1.0"))
# (ruj lever H) re-read threshold hedge: JEF_HEDGE=<delta> (0 = off), JEF_HEDGE_DMAX (default 2.0); JEF_TRACE=1 adds the
# read probabilities to each answer (screens only)
HEDGE = float(os.environ.get("JEF_HEDGE", "0") or 0)
HEDGE_DMAX = float(os.environ.get("JEF_HEDGE_DMAX", "2.0") or 2.0)
HEDGE_CAP = float(os.environ.get("JEF_HEDGE_CAP", "0") or 0)   # (E62) > 0: capped shift instead of the midpoint
TRACE = os.environ.get("JEF_TRACE", "0") == "1"
# (ruj lever FE, bj-g5 round 2) front-end without thread hops: the question is encoded in the event loop instead of the
# default executor, and each read enters vLLM as the EngineInput that vLLM's raw-prompt path would build in its single
# renderer thread (InputProcessor.process_inputs -> parse_model_prompt + Renderer.render_cmpl with the default
# completion tokenize params), so AsyncLLM.add_request takes its synchronous path. Same EngineCoreRequest (token ids,
# sampling params), so the same numbers; only when the reads reach the engine core changes. JEF_UVLOOP=1: uvloop loop.
FE = os.environ.get("JEF_FE", "0") == "1"
# (ruj lever FT, bj-g6 line J4) front-end turnaround (with FE): (a) a decision that must read again first lets the
# other coroutines woken by the same engine step run (one event-loop turn), so the HTTP answers of that step leave
# before its re-reads are issued (the clients' round trip is the longer path to the next step); (b) the read-twice
# prompt (read 2's token ids) is built after read 1 has entered the engine instead of before (a decision that is alone,
# J6's joint path, still builds both first). Same token ids, same requests: only their order in time changes.
FT = FE and os.environ.get("JEF_FT", "0") == "1"
# (ruj lever LF, line J4 round 2) lean front-end: the reads' outputs are not detokenized (SamplingParams detokenize=False:
# the answer uses only the code tokens' logprob values, never their text; vLLM then skips the per-read incremental
# detokenizer and the per-logprob token decoding in the output processor) and vLLM's statistics are off
# (disable_log_stats: no per-step scheduler stats in the engine core, no per-request stats / metric loggers in the
# front-end; nothing in this engine reads them). Same token ids, same sampling, same logprob values.
LF = os.environ.get("JEF_LF", "0") == "1"
# (ruj lever EM, line J4 round 2) encode memo: the checkpoint renderer tokenises every text piece of a question (state,
# question, each option, closing line) with tok.encode, once for read 1 and again for the read-twice prompt (a second
# CodeRenderer); EM memoises tok.encode per piece string (shared by both renderers, cleared when it holds 4,096 pieces),
# so read 2's pieces and repeated question / option texts are not tokenised again. tok.encode is a pure function of
# the string: the same token ids. The renderer's own code is unchanged (its _enc is wrapped per instance).
EM = os.environ.get("JEF_EM", "0") == "1"
# (ruj lever FR, line J5) fast read path (with FE): each read's EngineCoreRequest is a copy of the request vLLM's own
# InputProcessor.process_inputs built for the first read with the same number of answer codes (the same sampling
# parameters), with this read's token ids, a fresh request id and arrival time; it enters AsyncLLM.generate as an
# EngineCoreRequest (deprecated but supported in the pinned vLLM 0.30.0, which logs one warning; prove FR again on any
# vLLM bump), so the per-read render / validation / clone work (~0.1 ms of event-loop time per read) runs once per code
# count. The request the engine core receives is the same message (all fields equal but the id and the arrival time;
# checked on CPU for every read of the screen items and for synthetic reads of every option count up to 16,289 tokens).
FR = FE and os.environ.get("JEF_J5_FR", "0") == "1"
# (ruj lever TF, line J5) fast tokenizer call (with EM): the renderer's tok.encode(s, add_special_tokens=False) goes to
# the tokenizer's Rust backend directly (tok.backend_tokenizer.encode(s, add_special_tokens=False).ids), without
# transformers' batch-encoding wrapper; the same ids (checked on CPU for every piece of the screen items)
TF = EM and os.environ.get("JEF_J5_TF", "0") == "1"
LOG_FLOOR = 1e-6   # the release's s1scoring.LOG_FLOOR
_LATER = object()  # (ruj lever FT) read 2's ids not built yet


def _flogs(p):
    return [math.log(max(float(x), LOG_FLOOR)) for x in p]


def _dist(p, q):
    """The release's logit distance at T = 1 (s1scoring.logit_distance): max |centred log p - centred log q|."""
    a, b = _flogs(p), _flogs(q)
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    return max(abs((x - ma) - (y - mb)) for x, y in zip(a, b))


def _toward(a, b, s):
    """(E62 H cap) The distribution whose floored centred logs lie on the segment from a's toward b's at distance s
    (max norm) from a's: d(result, a) = min(s, d(a, b)), d(result, b) = d(a, b) - that."""
    la, lb = _flogs(a), _flogs(b)
    ma, mb = sum(la) / len(la), sum(lb) / len(lb)
    ca, cb = [x - ma for x in la], [y - mb for y in lb]
    d = max(abs(x - y) for x, y in zip(ca, cb))
    t = 0.0 if d == 0 else min(1.0, s / d)
    l = [x + t * (y - x) for x, y in zip(ca, cb)]
    top = max(l)
    e = [math.exp(x - top) for x in l]
    z = sum(e)
    return [x / z for x in e]


def _midpoint(p, q):
    """The distribution whose floored centred logs are the midpoint of p's and q's: distance d(p, q) / 2 to both."""
    l = [(x + y) / 2.0 for x, y in zip(_flogs(p), _flogs(q))]
    top = max(l)
    e = [math.exp(x - top) for x in l]
    s = sum(e)
    return [x / s for x in e]
LADDER = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1152,
          1280, 1408, 1536, 1664, 1792, 1920, 2048, 2304, 2560, 2816, 3072, 3584, 4096, 4608, 5120, 6144, 7168, 8192]


def log(*a):
    print(f"[jeffan {time.strftime('%H:%M:%S')}]", *a, flush=True)


def _parse_model_prompt(model_config, prompt):
    """(ruj lever FE) vLLM 0.30's parse_model_prompt (vllm/renderers/inputs/preprocess.py)."""
    from vllm.renderers.inputs.preprocess import parse_model_prompt
    return parse_model_prompt(model_config, prompt)


class BadRequest(ValueError):
    def __init__(self, msg, status=400):
        super().__init__(msg)
        self.status = status


def load_renderer(tok):
    """The checkpoint's renderer, from the weights mount (author code, Apache-2.0). Only render.py is loaded (the
    package __init__ imports the author's offline engine, which needs packages this image does not have)."""
    import importlib.util
    path = os.path.join(MODEL_DIR, "inference", "dm_decision_engine", "render.py")
    spec = importlib.util.spec_from_file_location("jeffan_render", path)
    R = importlib.util.module_from_spec(spec)
    sys.modules["jeffan_render"] = R      # dataclasses resolve their module by name
    spec.loader.exec_module(R)
    r1, r2 = R.CodeRenderer(tok, MAX_PROMPT, reads=1), R.CodeRenderer(tok, MAX_REREAD, reads=2)
    if EM:                                            # (ruj lever EM) one tok.encode per distinct piece string
        memo = {}
        if TF:                                        # (ruj lever TF) the Rust backend's encode, same ids
            bk = tok.backend_tokenizer

            def _tf(s, _bk=bk):
                return _bk.encode(s, add_special_tokens=False).ids
            enc1 = enc2 = _tf
        else:
            enc1, enc2 = r1._enc, r2._enc

        def wrap(enc):
            def _enc(s):
                v = memo.get(s)
                if v is None:
                    v = enc(s)
                    if len(memo) >= 4096:
                        memo.clear()
                    memo[s] = v
                return v
            return _enc
        r1._enc, r2._enc = wrap(enc1), wrap(enc2)
    return R, r1, r2


def softmax(values):
    z = [float(v) / TEMPERATURE for v in values]
    top = max(z)
    e = [math.exp(x - top) for x in z]
    s = sum(e)
    return [x / s for x in e]


def answer(qtype, keys, probs):
    if any(not math.isfinite(p) or p < 0 for p in probs) or sum(probs) <= 0:
        raise RuntimeError("non-finite option probabilities")
    total = sum(probs)
    probs = [p / total for p in probs]
    dist = dict(zip(keys, probs))
    out = {"type": qtype, "probabilities": dist, "confidence": max(probs)}
    if qtype == "noul":
        out["noul"] = dist["true"]
    elif qtype == "choice":
        out["choice"] = keys[max(range(len(probs)), key=probs.__getitem__)]
    else:
        out["score"] = sum(i * p for i, p in enumerate(probs))
    return out


def engine_args():
    from vllm.engine.arg_utils import AsyncEngineArgs
    kw = dict(model=MODEL_DIR, served_model_name=["reax"], dtype="bfloat16", max_model_len=16448,
              logprobs_mode="processed_logprobs", max_logprobs=256, seed=0, max_num_seqs=64, enable_prefix_caching=True,
              quantization="fp8", kv_cache_dtype="fp8", gpu_memory_utilization=float(os.environ.get("JEF_UTIL", "0.44")),
              limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0})
    if LF:                                            # (ruj lever LF) no vLLM statistics
        kw["disable_log_stats"] = True
    if os.environ.get("JEF_ASYNC", "") in ("0", "1"):   # (ruj) vLLM's async scheduling off / on (default: vLLM's choice)
        kw["async_scheduling"] = os.environ["JEF_ASYNC"] == "1"
    prof = os.environ.get("JEF_PROF_DIR", "")      # screens only (ruj): torch profiler, started by GET /prof/start
    if prof:
        from vllm.config.profiler import ProfilerConfig
        kw["profiler_config"] = ProfilerConfig(profiler="torch", torch_profiler_dir=prof, torch_profiler_with_stack=False)
    cc = {}
    sizes = os.environ.get("JEF_GRAPH_SIZES", "")
    if sizes != "default":
        ladder = [int(x) for x in sizes.split(",") if x.strip()] or LADDER
        cc["cudagraph_capture_sizes"] = sorted(set(ladder))
    if os.environ.get("JEF_CG_MODE"):                 # (E62 J3) vLLM cudagraph_mode, e.g. FULL; unset = default
        cc["cudagraph_mode"] = os.environ["JEF_CG_MODE"]
    if cc:
        kw["compilation_config"] = cc
    return AsyncEngineArgs(**kw)


class Engine:
    def __init__(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.R, self.r1, self.r2 = load_renderer(tok)
        os.environ.setdefault("JEF_CODE_IDS", ",".join(str(int(c)) for c in self.r1.code_ids))  # (ruj) for JEF_ROWS
        self.llm, self.defaults, self.ready = None, {}, False
        self.decisions = self.rereads = self.steps = self.hedged = 0
        self.inflight, self.busy_until, self.pars = 0, 0.0, 0   # (E62 J6)
        self.gather = float(os.environ.get("JEF_GATHER_MS", "4")) / 1e3
        self.quiet = float(os.environ.get("JEF_QUIET_MS", "1.0")) / 1e3
        self.pending, self.timer, self.win_end, self.last = [], None, 0.0, 0.0
        self.fr_tmpl, self.tasks = {}, None   # (ruj lever FR) EngineCoreRequest templates per code count
        tokv = os.environ.get("S1M_INTERNAL_TOKEN", "")
        path = os.environ.get("S1M_INTERNAL_TOKEN_FILE", "")
        if not tokv and path and os.path.exists(path):
            tokv = open(path).read().strip()
        if not tokv and os.environ.get("RUNTIME_ALLOW_NO_AUTH") != "1":
            sys.exit("jeffan: no token (S1M_INTERNAL_TOKEN or S1M_INTERNAL_TOKEN_FILE)")
        self.token = tokv.encode()

    def encode(self, state, q):
        try:
            row = self.R.from_wire(state, q)
        except (KeyError, TypeError, ValueError) as exc:
            raise BadRequest(f"bad question: {type(exc).__name__}") from None
        p1 = self.r1.pieces(row)
        if p1 is None:
            if not 1 <= row.n <= len(self.r1.codes):
                raise BadRequest("a question needs 1 to 255 options")
            raise BadRequest(f"maximum context length exceeded: input exceeds {MAX_PROMPT} model tokens", 422)
        ids1 = self.r1.assemble(p1)
        p2 = self.r2.pieces(row)
        ids2 = self.r2.assemble(p2) if p2 is not None else None
        return row.keys, ids1, ids2, self.r1.code_ids[: row.n]

    def encode1(self, state, q):
        """(ruj lever FT) encode()'s read-1 part (same checks, same ids); the row is kept for encode2."""
        try:
            row = self.R.from_wire(state, q)
        except (KeyError, TypeError, ValueError) as exc:
            raise BadRequest(f"bad question: {type(exc).__name__}") from None
        p1 = self.r1.pieces(row)
        if p1 is None:
            if not 1 <= row.n <= len(self.r1.codes):
                raise BadRequest("a question needs 1 to 255 options")
            raise BadRequest(f"maximum context length exceeded: input exceeds {MAX_PROMPT} model tokens", 422)
        return row.keys, self.r1.assemble(p1), row, self.r1.code_ids[: row.n]

    def encode2(self, row):
        """(ruj lever FT) encode()'s read-2 part: the read-twice prompt's ids, None when it does not fit."""
        p2 = self.r2.pieces(row)
        return self.r2.assemble(p2) if p2 is not None else None

    async def start(self, llm=None):
        from vllm.usage.usage_lib import UsageContext
        from vllm.v1.engine.async_llm import AsyncLLM
        t0 = time.time()
        self.llm = llm or AsyncLLM.from_engine_args(engine_args(), usage_context=UsageContext.OPENAI_API_SERVER)
        self.defaults = dict(self.llm.model_config.get_diff_sampling_param() or {})
        if FR:
            self.tasks = await self.llm.get_supported_tasks()
        log("engine up in", round(time.time() - t0), "s; sampling defaults", self.defaults)
        warm = {"type": "choice", "instructions": "Pick one.", "criteria": {"a": "A", "b": "B", "c": "C"}}
        await asyncio.gather(*(self.decide({"warm": i}, warm, gather=False) for i in range(8)))
        self.ready = True
        log("ready after", round(time.time() - t0), "s")

    def sampling(self, code_ids):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        d = self.defaults
        return SamplingParams.from_optional(
            n=1, repetition_penalty=d.get("repetition_penalty", 1.0), temperature=0.0, top_p=d.get("top_p", 1.0),
            top_k=d.get("top_k", 0), min_p=d.get("min_p", 0.0), max_tokens=1, logprobs=len(code_ids),
            allowed_token_ids=list(code_ids), output_kind=RequestOutputKind.FINAL_ONLY, skip_clone=True,
            detokenize=not LF)                               # (ruj lever LF) no text for the code tokens

    def _engine_input(self, prompt):
        """(ruj lever FE) what InputProcessor.process_inputs builds for a raw TokensPrompt (vLLM 0.30, the deprecated
        raw-prompt branch, run by process_inputs_async in the renderer's one-thread executor), built in the loop."""
        r = self.llm.renderer
        tok_params = r.default_cmpl_tok_params.with_kwargs()
        (ei,) = r.render_cmpl([_parse_model_prompt(self.llm.model_config, prompt)], tok_params)
        return ei

    def _fast_request(self, ids, code_ids):
        """(ruj lever FR) this read's EngineCoreRequest: the per-code-count template from vLLM's process_inputs, with
        this read's token ids, a fresh id and arrival time, and its own (shallow) copy of the sampling parameters"""
        import copy
        import msgspec
        n = len(code_ids)
        t = self.fr_tmpl.get(n)
        if t is None:
            from vllm.inputs import TokensPrompt
            t = self.llm.input_processor.process_inputs(
                "jef-fr", self._engine_input(TokensPrompt(prompt_token_ids=list(ids))), self.sampling(code_ids),
                supported_tasks=self.tasks)
            self.fr_tmpl[n] = t
        return msgspec.structs.replace(t, request_id=uuid.uuid4().hex, prompt_token_ids=ids, arrival_time=time.time(),
                                       sampling_params=copy.copy(t.sampling_params))

    async def _read(self, ids, code_ids):
        final = None
        if FR:                                               # (ruj lever FR) pre-built request
            req = self._fast_request(ids, code_ids)
            async for out in self.llm.generate(req, req.sampling_params, req.request_id):
                final = out
            lp = final.outputs[0].logprobs[0]
            return softmax([lp[c].logprob for c in code_ids])
        from vllm.inputs import TokensPrompt
        prompt = TokensPrompt(prompt_token_ids=ids)
        if FE:
            prompt = self._engine_input(prompt)
        async for out in self.llm.generate(prompt, self.sampling(code_ids), uuid.uuid4().hex):
            final = out
        lp = final.outputs[0].logprobs[0]
        return softmax([lp[c].logprob for c in code_ids])

    async def decide(self, state, q, gather=True):
        self.inflight += 1                                   # (E62 J6) decisions in flight
        try:
            return await self._decide(state, q, gather)
        finally:
            self.inflight -= 1

    async def _decide(self, state, q, gather=True):
        loop = asyncio.get_running_loop()
        row = None
        if FT:                                               # (ruj lever FT) read 2's ids after read 1 is issued
            keys, ids1, row, code_ids = self.encode1(state, q)
            ids2 = _LATER
        elif FE:                                             # (ruj lever FE) no executor hop (~0.2 ms of CPU)
            keys, ids1, ids2, code_ids = self.encode(state, q)
        else:
            keys, ids1, ids2, code_ids = await loop.run_in_executor(None, self.encode, state, q)
        if gather:
            await self._gate(loop)
        n = len(ids1)
        now = loop.time()                                     # (E62 J6) alone = nothing else in flight, now or
        if self.inflight > 1:                                 # within the last PAR_QUIET seconds
            self.busy_until = now + PAR_QUIET
        alone = PAR1 and self.inflight == 1 and now >= self.busy_until
        if ids2 is _LATER and (SPEC2 or alone):              # (ruj lever FT) the joint path needs both ids now
            ids2 = self.encode2(row)
        if (SPEC2 or alone) and ids2 is not None:
            self.pars += alone
            # (ruj lever S2) both reads enter the engine together: read 2's shared prefix (read 1 minus its chat
            # suffix) is a prefix-cache hit on read 1's blocks, cached when read 1 is scheduled (also within the same
            # step), so the extra work is read 2's own continuation; the decision needs one engine step, not two
            t1 = asyncio.ensure_future(self._read(ids1, code_ids))
            t2 = asyncio.ensure_future(self._read(ids2, code_ids))
            try:
                probs = await t1
            except BaseException:
                t2.cancel()
                raise
            p1, p2 = probs, None
            if HEDGE > 0 and abs(max(p1) - REREAD_BELOW) < HEDGE:   # (E62) lever H in the joint path, as below
                self.rereads += 1
                p2 = await t2
                n += len(ids2)
                d = _dist(p1, p2)
                if HEDGE_CAP > 0:                              # (E62 H cap)
                    self.hedged += 1
                    mine, other = (p2, p1) if max(p1) < REREAD_BELOW else (p1, p2)
                    probs = _toward(mine, other, min(d / 2.0, HEDGE_CAP))
                elif d <= HEDGE_DMAX:
                    self.hedged += 1
                    probs = _midpoint(p1, p2)
                else:
                    probs = p2 if max(p1) < REREAD_BELOW else p1
            elif max(p1) < REREAD_BELOW:
                self.rereads += 1
                probs = p2 = await t2
                n += len(ids2)
            else:
                t2.cancel()
            self.decisions += 1
            out = answer(q["type"], keys, probs)
            if TRACE:
                out["dm_trace"] = {"p1": p1, "p2": p2}
            return out, n
        if ids2 is _LATER:                                    # (ruj lever FT) read 1 enters the engine first
            t1 = asyncio.ensure_future(self._read(ids1, code_ids))
            await asyncio.sleep(0)
            try:
                ids2 = self.encode2(row)
            except BaseException:
                t1.cancel()
                raise
            probs = await t1
        else:
            probs = await self._read(ids1, code_ids)
        p1 = probs
        p2 = None
        if ids2 is not None and HEDGE > 0 and abs(max(p1) - REREAD_BELOW) < HEDGE:
            # (ruj lever H) the first read sits within HEDGE of the re-read threshold: the reference, whose numerics move
            # a little from start to start, may decide either way. Read 2 is computed, and when the two reads are
            # within HEDGE_DMAX of each other the answer is their midpoint in the gates' (floored, centred) log space,
            # which is d/2 from both, so the item stays within the item gate whichever read the reference returns.
            self.rereads += 1
            if FT:                                            # (ruj lever FT) this step's HTTP answers leave first
                await asyncio.sleep(0)
            p2 = await self._read(ids2, code_ids)
            n += len(ids2)
            d = _dist(p1, p2)
            if HEDGE_CAP > 0:                                  # (E62 H cap)
                self.hedged += 1
                mine, other = (p2, p1) if max(p1) < REREAD_BELOW else (p1, p2)
                probs = _toward(mine, other, min(d / 2.0, HEDGE_CAP))
            elif d <= HEDGE_DMAX:
                self.hedged += 1
                probs = _midpoint(p1, p2)
            else:
                probs = p2 if max(p1) < REREAD_BELOW else p1
        elif ids2 is not None and max(probs) < REREAD_BELOW:    # read twice; issued at once, not after a whole batch
            self.rereads += 1
            if FT:                                            # (ruj lever FT) this step's HTTP answers leave first
                await asyncio.sleep(0)
            probs = p2 = await self._read(ids2, code_ids)
            n += len(ids2)
        self.decisions += 1
        out = answer(q["type"], keys, probs)
        if TRACE:
            out["dm_trace"] = {"p1": p1, "p2": p2}
        return out, n

    # gather window (as the Blink engine): the first request after an idle gap opens a window of at most JEF_GATHER_MS
    # that closes early after JEF_QUIET_MS without a new request; the burst is released together
    async def _gate(self, loop):
        if not self.gather:
            return
        fut = loop.create_future()
        self.pending.append(fut)
        self.last = loop.time()
        if self.timer is None:
            self.win_end = self.last + self.gather
            self.timer = loop.call_later(min(self.gather, self.quiet), self._tick)
        await fut

    def _tick(self):
        loop = asyncio.get_running_loop()
        now = loop.time()
        if now >= self.win_end or now - self.last >= self.quiet:
            self.timer = None
            live, self.pending = [f for f in self.pending if not f.done()], []
            self.steps += 1
            for f in live:
                f.set_result(None)
            return
        self.timer = loop.call_later(min(self.win_end - now, self.quiet - (now - self.last)), self._tick)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                m = await receive()
                if m["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif m["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        path, method = scope["path"], scope["method"]
        if path == "/health" and method == "GET":
            if not self.ready:
                return await self._json(send, 503, {"ok": False, "status": "warming up"})
            h = {"ok": True, "model": MODEL_NAME, "kind": "vllm", "decisions": self.decisions, "rereads": self.rereads,
             "par": self.pars, "hedged": self.hedged}
            if HEDGE > 0:
                h["hedged"] = self.hedged
            return await self._json(send, 200, h)
        if path in ("/prof/start", "/prof/stop") and os.environ.get("JEF_PROF_DIR"):   # screens only (ruj)
            got = b""
            for k, v in scope["headers"]:
                if k == b"x-s1m-internal-token":
                    got = v
                    break
            if self.token and not hmac.compare_digest(got, self.token):
                return await self._json(send, 401, {"error": {"message": "unauthorized"}})
            await (self.llm.start_profile() if path == "/prof/start" else self.llm.stop_profile())
            return await self._json(send, 200, {"ok": True, "profile": path})
        if method != "POST" or path != "/internal/v1/decide":
            return await self._json(send, 404, {"error": {"message": "not found"}})
        if self.token:
            got = b""
            for k, v in scope["headers"]:
                if k == b"x-s1m-internal-token":
                    got = v
                    break
            if not hmac.compare_digest(got, self.token):
                return await self._json(send, 401, {"error": {"message": "unauthorized"}})
        if not self.ready:
            return await self._json(send, 503, {"error": {"message": "warming up"}})
        chunks = []
        while True:
            m = await receive()
            chunks.append(m.get("body", b""))
            if not m.get("more_body"):
                break
        try:
            body = json.loads(b"".join(chunks))
            questions, state = body["questions"], body["state"]
            if not isinstance(questions, dict) or len(questions) != 1:
                raise ValueError("exactly one question per call")
            (qid, q), = questions.items()
            if not isinstance(q, dict) or q.get("type") not in QTYPES:
                raise ValueError("bad question type")
            if body.get("images"):
                return await self._json(send, 422, {"error": {"message": "this model takes text only"}, "code": "unsupported_images"})
        except Exception as exc:  # noqa: BLE001
            return await self._json(send, 400, {"error": {"message": f"bad request: {type(exc).__name__}"}})
        try:
            ans, n = await self.decide(state, q)
        except BadRequest as exc:
            return await self._json(send, exc.status, {"error": {"message": str(exc)}})
        except Exception as exc:  # noqa: BLE001
            return await self._json(send, 500, {"error": {"message": type(exc).__name__}})
        await self._json(send, 200, {"model": MODEL_NAME, "answers": {qid: ans}, "usage": {"input_tokens": n, "output_tokens": 0}})

    @staticmethod
    async def _json(send, code, obj):
        raw = json.dumps(obj).encode()
        await send({"type": "http.response.start", "status": code,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode())]})
        await send({"type": "http.response.body", "body": raw})


async def amain():
    import uvicorn
    from concurrent.futures import ThreadPoolExecutor
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(8))
    eng = Engine()
    await eng.start()
    cfg = uvicorn.Config(eng, uds=os.environ["RUNTIME_UDS"], log_level="warning", access_log=False, lifespan="on",
                         timeout_keep_alive=75, interface="asgi3")
    await uvicorn.Server(cfg).serve()


# ------------------------------------------------------------------------------------------------ several frontends
# (ruj lever MF, line J5) JEF_FRONTENDS=N > 1: N frontend processes (HTTP + question encoding + vLLM AsyncLLM client
# each) on one socket and one engine core, with vLLM's multi-API-server plumbing (client_index / client_count): pjs's
# BLINK_FRONTENDS code from his Blink engine (E59 / E62 blink_engine.py, credit pjs) with team ruj's #376 connection
# balancing (JEF_FE_BALANCE=1: the parent accepts every connection and hands it to the frontend with the fewest open
# connections), here for the Jeffan engine. The socket is bound under a temporary name and renamed to $RUNTIME_UDS once
# every frontend has answered its warm-up decisions. Each read is the same request as with one frontend; only which
# process encodes the question and talks to the engine core changes (the decisions of one connection stay in one
# frontend, so every decision's reads, re-read and hedge are those of the one-frontend engine).
async def _fe_main(sock, client_config, ready, fe_sock=None, fe_closed=None):
    import uvicorn
    from concurrent.futures import ThreadPoolExecutor
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(8))
    cfg = dict(client_config)
    count, index = cfg.pop("client_count", 1), cfg.pop("client_index", 0)
    ea = engine_args()
    ea._api_process_count, ea._api_process_rank = count, index
    vllm_config = ea.create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)
    llm = AsyncLLM.from_vllm_config(vllm_config, usage_context=UsageContext.OPENAI_API_SERVER,
                                    disable_log_stats=ea.disable_log_stats, client_addresses=cfg, client_count=count,
                                    client_index=index)
    eng = Engine()
    await eng.start(llm)
    ready.put(index)
    if fe_sock is not None:
        server = _balanced_server(uvicorn, eng, fe_sock, fe_closed, index)
        await server.serve(sockets=[])
        return
    server = uvicorn.Server(uvicorn.Config(eng, fd=sock.fileno(), log_level="warning", access_log=False,
                                           lifespan="on", timeout_keep_alive=75, interface="asgi3"))
    await server.serve()


def _balanced_server(uvicorn, eng, fe_sock, fe_closed, index):
    """(team ruj #376) a uvicorn server without a listener of its own; connections arrive as file descriptors on
    fe_sock (SCM_RIGHTS, one per datagram) and are served with uvicorn's own HTTP protocol; each closed connection is
    counted in fe_closed[index] (written only by this process), so the parent sees the open count"""
    import socket

    class _Server(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=[])
            loop = asyncio.get_running_loop()
            config = self.config

            def proto():
                p = config.http_protocol_class(config=config, server_state=self.server_state,
                                               app_state=self.lifespan.state, _loop=loop)
                orig_lost = p.connection_lost

                def connection_lost(exc, _orig=orig_lost):
                    fe_closed[index] += 1
                    return _orig(exc)
                p.connection_lost = connection_lost
                return p

            async def attach(c):
                try:
                    await loop.connect_accepted_socket(proto, sock=c)
                except Exception as exc:  # noqa: BLE001
                    fe_closed[index] += 1
                    log(f"frontend {index}: could not attach a connection: {exc!r}")
                    c.close()

            def on_fd():
                while True:
                    try:
                        _msg, fds, _flags, _addr = socket.recv_fds(fe_sock, 16, 16)
                    except (BlockingIOError, InterruptedError):
                        return
                    except OSError as exc:
                        log(f"frontend {index}: fd channel error {exc!r}")
                        return
                    if not fds:
                        return
                    for fd in fds:
                        c = socket.socket(fileno=fd)
                        c.setblocking(False)
                        loop.create_task(attach(c))

            fe_sock.setblocking(False)
            loop.add_reader(fe_sock.fileno(), on_fd)
            log(f"frontend {index}: balanced mode (connections handed over by the parent)")

    return _Server(uvicorn.Config(eng, log_level="warning", access_log=False, lifespan="on", timeout_keep_alive=75,
                                  interface="asgi3"))


def _fe_worker(listen_address, sock, args, client_config):
    socks = getattr(args, "fe_socks", None)
    index = client_config.get("client_index", 0)
    fe_sock = socks[index] if socks else None
    fe_closed = getattr(args, "fe_closed", None)
    if os.environ.get("JEF_UVLOOP", "0") == "1":
        import uvloop
        uvloop.run(_fe_main(sock, client_config, args.ready, fe_sock, fe_closed))
    else:
        asyncio.run(_fe_main(sock, client_config, args.ready, fe_sock, fe_closed))


def serve_multi(n):
    import multiprocessing
    import socket
    import types
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.utils import get_engine_zmq_addresses, launch_core_engines
    from vllm.v1.executor.abstract import Executor
    from vllm.v1.utils import APIServerProcessManager
    t0 = time.time()
    uds = os.environ["RUNTIME_UDS"]
    tmp = uds + ".starting"
    for path in (tmp, uds):
        if os.path.exists(path):
            os.unlink(path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(tmp)
    os.chmod(tmp, 0o666)                  # as uvicorn does for a unix socket
    sock.listen(2048)
    ea = engine_args()
    ea._api_process_count, ea._api_process_rank = n, -1
    vllm_config = ea.create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)
    executor_class = Executor.get_class(vllm_config)
    addresses = get_engine_zmq_addresses(vllm_config, n, defer_api_server_ports=True)
    ready = multiprocessing.get_context("spawn").Queue()
    args = types.SimpleNamespace(ready=ready)
    balance = os.environ.get("JEF_FE_BALANCE", "1") == "1"
    if balance:      # team ruj #376: see _balanced_server
        pairs = [socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM) for _ in range(n)]
        closed = multiprocessing.get_context("spawn").RawArray("i", n)
        args = types.SimpleNamespace(ready=ready, fe_socks=[p[1] for p in pairs], fe_closed=closed)
    with launch_core_engines(vllm_config, executor_class, not ea.disable_log_stats, addresses) as launch:
        addresses = launch.addresses
        mgr = APIServerProcessManager(listen_address=tmp, sock=sock, args=args, num_servers=n,
                                      input_addresses=addresses.inputs, output_addresses=addresses.outputs,
                                      target_server_fn=_fe_worker, tensor_queue=launch.tensor_queue)
        addresses.inputs, addresses.outputs = mgr.gather_actual_addresses()
        launch.watched_frontend_processes = mgr.processes   # a frontend that dies aborts the engine start-up
    cores = list(getattr(launch.engine_manager, "processes", []) or [])
    got = set()
    while len(got) < n:
        try:
            got.add(ready.get(timeout=5))
        except Exception:  # noqa: BLE001  (queue.Empty)
            dead = [p.name for p in mgr.processes if not p.is_alive()]
            if dead:
                sys.exit(f"jeffan: frontend(s) {dead} exited during start-up")
    if balance:
        import threading

        def acceptor():
            assigned = [0] * n
            while True:
                try:
                    conn, _ = sock.accept()
                except OSError as exc:
                    log(f"acceptor: accept failed {exc!r}")
                    time.sleep(0.01)
                    continue
                try:
                    i = min(range(n), key=lambda j: (assigned[j] - closed[j], assigned[j], j))
                    socket.send_fds(pairs[i][0], [b"c"], [conn.fileno()])
                    assigned[i] += 1
                except OSError as exc:
                    log(f"acceptor: hand-over failed {exc!r}")
                finally:
                    conn.close()

        threading.Thread(target=acceptor, name="jeffan-acceptor", daemon=True).start()
        log(f"connections balanced over {n} frontends (least open connections)")
    os.rename(tmp, uds)
    log(f"{n} frontends ready after {round(time.time() - t0)} s; socket {uds}")
    while True:
        for p in list(mgr.processes) + cores:
            p.join(timeout=0.5)
            if not p.is_alive():
                sys.exit(f"jeffan: process {p.name} exited ({p.exitcode})")


if __name__ == "__main__":
    _nfe = int(os.environ.get("JEF_FRONTENDS", "1") or 1)
    if _nfe > 1:                                     # (ruj lever MF, line J5) several frontend processes
        serve_multi(_nfe)
    elif os.environ.get("JEF_UVLOOP", "0") == "1":   # (ruj lever FE) uvloop event loop (vLLM's API server uses it)
        import uvloop
        uvloop.run(amain())
    else:
        asyncio.run(amain())
