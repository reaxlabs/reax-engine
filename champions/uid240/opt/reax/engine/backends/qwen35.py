"""Backend for the Qwen3.5-hybrid decision models served by the native PplxEngine: pplx-decider-v1.1-27b, d3, d3-turbo, reax-1.

All checkpoints share the architecture (48 Gated-DeltaNet + 16 full-attention layers, hidden 5120, 255-way
readout) and the engine; they differ in prompt/answer contract and attention mode:

  family "pplx"  perplexity-ai/pplx-decider-v1.1-27b   prompt in pplx_engine.py (adapted from the author's code)
  family "d3"    vllm-sr/d3, vllm-sr/d3-turbo          prompt/answer contract in d3_contract.py (vendored, upstream/d3)
  family "reax1" decisionmodels/reax-1                 pplx-decider v1.2 joint format (protocol v3), tokenised by the author's
                                                       own Encoder from the checkpoint's source/src/pplx_decider (foreign code,
                                                       read from the read-only weights mount), readout at the question marker

pplx and d3 have NONCAUSAL full attention (decision_config.json: "attention_mode": "noncausal_full_attention"). d3-turbo
says "prefix_bidir": the state block attends only to itself, the question block reads the whole row. With `shared_state`
every single decision is then run state-once through PplxEngine.forward_groups (prefix = state block, one suffix), and
queued calls with a token-identical state block are merged into one group (`shared_group`): the production recipe of
the s1-pro d3-turbo node (private engine branch d3-turbo-shared2-20261022 @75d0659, `--shared-state 1 --shared-until 64
--shared-mixq 0 --shared-group 1`), so an answer does not depend on which sibling calls arrive with it.

The torch/transformers imports are lazy so the module (and the registry) import on a CPU-only machine.
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import sys
from pathlib import Path

from .. import d3_contract, fastenc1, kai_status, qev_contract, startup
from .base import Encoded

_ENGINE_DIR = str(Path(__file__).resolve().parent.parent)
FAMILIES = ("pplx", "d3", "reax1", "mammoth")
_DEFAULT_NAMES = {"pplx": "pplx-decider-v1.1-27b", "d3": "d3", "reax1": "reax-1", "mammoth": "mammoth-1"}
NV4_FAMILIES = ("mammoth", "reax1")  # rgh's MAM_NV4 hook (+ KAI_NV4_MINM): the Qwen3.5-27B families our screens run


# rgh f8m (MAM_FASTENC; rgh: default on, here off): qev_contract.encode_request for ONE question with both texts tokenised in
# one call of the Rust tokenizer (tokenizers.Tokenizer.encode_batch: the call transformers' tok.encode itself makes for a
# single text, here with the full and the prefix text together; it runs without the GIL, so concurrent encode threads
# overlap). Same rendered texts, same token ids (rgh: 4000/4000 items of both pools checked), same prefix-length rule and
# errors. Copied verbatim from rgh's u4 image (/opt/reax-m/engine/backends/qwen35.py).
_FASTENC = os.environ.get("MAM_FASTENC", "0") == "1"
# rgh u4r (R1_FASTENC; rgh: default on, here off): REAX-1 encode = the author's validate + rendered text, tokenised by one Rust
# Tokenizer.encode_batch call (what the processor's tokenizer runs for one text, GIL released) instead of the processor
# __call__ (BatchEncoding, torch tensors, image-token bookkeeping for no images); same ids, marker, length rule, errors.
# Copied from rgh's u4r image (/opt/reax-r/engine/backends/qwen35.py, recipe /opt/rgh/reax1/patch_r1x.py).
_R1_FASTENC = os.environ.get("R1_FASTENC", "0") == "1"


# rgh m1 / mam-c8d (MAM_FASTRENDER; rgh: default on, here off): FASTENC's two chat-template renders held the GIL for ~2/3 of
# an encode (rgh lane mam session mS2: the 8 encodes of a closed-loop c8 step ran nearly back to back). For qev's two
# messages (system = SYSTEM_PROMPT, user = state block + suffix, add_generation_prompt, enable_thinking False) the mammoth
# template is plain concatenation with the user content trimmed: render(...) == head + (state_block + suffix).strip() + tail.
# head / tail are cut out of the template's own output (sentinel content) when the backend is built, then checked against
# apply_chat_template on rgh's probe contents; any difference (another template) keeps the reference renders. Same texts ->
# same token ids (tests/test_rgh_m1.py: >= 1,000 real items). Copied from rgh's m1 image
# (/opt/reax-m/engine/backends/qwen35.py, mam-c8d), switch default off.
_FASTRENDER = os.environ.get("MAM_FASTRENDER", "0") == "1"
_RENDER_PROBES = ("x", "  lead", "trail \n", "<state>\nQEV_PREFIX_MARK\n</state>\n\n", "<|im_start|>user\n\u65e5 \U0001f642\u3000",
                  "\t\x1c a \x1f", "", "<tool_response>x", "<think>a</think>b", "{\"k\": [1, null]}")


def _fast_renderer(tokenizer):
    """(head, tail) with qev_contract.render(tokenizer, state, suffix) == head + (state_block(state) + suffix).strip() + tail,
    or None (fast path off)"""
    if not _FASTRENDER:
        return None

    def tmpl(content):
        msgs = [{"role": "system", "content": qev_contract.SYSTEM_PROMPT}, {"role": "user", "content": content}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    try:
        sent = "\x00RGH_RENDER_SENTINEL\x00"
        t = tmpl(sent)
        if t.count(sent) != 1:
            return None
        head, tail = t.split(sent)
        if any(tmpl(c) != head + c.strip() + tail for c in _RENDER_PROBES):
            return None
        return head, tail
    except Exception:  # noqa: BLE001 - any template error: keep the reference renders
        return None


def _encode_request_fast(tokenizer, state, question, rend=None):
    q = question
    if rend is not None:  # MAM_FASTRENDER
        head, tail = rend
        sb = qev_contract.state_block(state)
        full_text = head + (sb + q.suffix).strip() + tail
        text = head + (sb + qev_contract.PREFIX_MARK).strip() + tail
    else:
        full_text = qev_contract.render(tokenizer, state, q.suffix)
        text = qev_contract.render(tokenizer, state, qev_contract.PREFIX_MARK)
    enc = tokenizer._tokenizer.encode_batch([full_text, text[: text.index(qev_contract.PREFIX_MARK)]],
                                            add_special_tokens=False)
    full, prefix = [list(enc[0].ids)], list(enc[1].ids)
    n = len(prefix)
    while n > 0 and not all(ids[:n] == prefix[:n] for ids in full):
        n -= 1
    if n <= 0 or any(len(ids) <= n for ids in full):
        raise ValueError("a branch has an empty suffix; state block boundary mismatch")
    return full, n


def _pplx_engine_module():
    """pplx_engine.py and its kernels import each other as top-level modules (legacy flat layout)."""
    if _ENGINE_DIR not in sys.path:
        sys.path.insert(0, _ENGINE_DIR)
    import pplx_engine
    return pplx_engine


def nv_acc_kw(rot="", smooth="", wscale="amax", ascale="amax", shadow="") -> dict:
    """NVFP4 accuracy levers (engine/nvfp4_acc.py) and the offline sweep hook nvfp4_shadow (tools/nv4sweep.py) for PplxEngine,
    passed only when set: the default call is unchanged."""
    kw = {"nvfp4_rot": str(rot or ""), "nvfp4_smooth": str(smooth or ""), "nvfp4_wscale": str(wscale or "amax"),
          "nvfp4_ascale": str(ascale or "amax"), "nvfp4_shadow": str(shadow or "")}
    return {k: v for k, v in kw.items() if v not in ("", "amax")}


class Qwen35Backend:
    kind = "native-gpu"
    serve_warmup = False  # serve.py leaves warming to the `warm_steps` option (runs in the constructor)
    # d3-turbo (prefix_bidir) state-once routing, set in __init__; class defaults keep stubbed instances valid
    prefix_bidir = False
    shared = False
    shared_group = False
    shared_min = 64
    encode_once = False  # lever B5 (mammoth): one render + one tokenise per call, checked against encode_request at first
    encode_once_check = 256
    _once_seen = 0
    _r1_tok = None  # R1_FASTENC: the Rust tokenizer (None = the author's Encoder.plan path)
    _enc1 = None  # kai lever I KAI_ENC1 (with MAM_FASTENC): fastenc1.Enc1, one render + one tokenisation per question
    _render = None  # rgh m1 MAM_FASTRENDER: mammoth fast-render (head, tail) or None
    _nv4t = None  # kai lever B (MAM_NV4TAB): the installed nv4t.Router (kept table, hit counters) or None
    _kai = None  # kai fix M3: kai_status.Status (start line, /health, shutdown counts)
    _nv4_defer = None  # rgh mx: model_dir of a calibrated NV4 plan (gptq) waiting for _after_build
    _nv4_be = None  # the backend nv4.install calibrates through (set only during that deferred install)

    def __init__(self, model_dir, *, family="pplx", name=None, linear="fp8", compile="1", gva="1", fp8_skip="", nvfp4="",
                 nvfp4_scales="", budget=16384, max_input_tokens=None, warm_steps=0, device="cuda", shared_state=False,
                 shared_group=False, shared_min=64, shared_until=None, shared_mixq=None, encode_once=False, encode_once_check=256,
                 nvfp4_rot="", nvfp4_smooth="", nvfp4_wscale="amax", nvfp4_ascale="amax", nvfp4_shadow=""):
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        self.family = family
        if encode_once and family != "mammoth":
            raise ValueError("encode_once is for the mammoth family (qev contract) only")
        self.encode_once, self.encode_once_check = bool(encode_once), int(encode_once_check)
        self.name = name or _DEFAULT_NAMES[family]
        # pplx / reax-1: the 8192-token limit; d3: the author's server has none (long prompts were validated to 64k+).
        self.max_input_tokens = max_input_tokens if max_input_tokens is not None else (262144 if family == "d3" else 8192)
        self.max_batch_tokens = budget
        with startup.phase("engine_import"):
            self._pp = _pplx_engine_module()
        model_dir = str(model_dir)
        with startup.phase("tokenizer"):
            if family == "mammoth":
                from transformers import AutoTokenizer
                self._tok = AutoTokenizer.from_pretrained(model_dir)
                self._init_render()
                if _FASTENC and fastenc1.enabled():  # kai lever I (KAI_ENC1; unset = off)
                    self._enc1 = fastenc1.Enc1(self._tok, self._fastenc_call)
                    print("KAI_ENC1: " + ("on (self-check of the first 256 calls)" if self._enc1.on else
                                          f"off ({self._enc1.why})"), file=sys.stderr, flush=True)
            elif family == "d3":
                from transformers import AutoTokenizer
                self._d3 = d3_contract
                cfg = json.loads((Path(model_dir) / "decision_config.json").read_text())
                self._tok, self._codes = AutoTokenizer.from_pretrained(model_dir), cfg["codes"]
                if cfg.get("readout_dtype", "float32") == "float32":
                    os.environ.setdefault("PPLX_READOUT_FP32", "1")  # author runtime: hidden.float() @ readout.T in fp32
            elif family == "reax1":
                self._init_reax1(model_dir)
            else:
                self._prompter = self._pp.Prompter(model_dir)
        self.engine = self._pp.PplxEngine(model_dir, device=device, linear_mode=linear, compile=compile,
                                          fp8_skip=tuple(x for x in fp8_skip.split(",") if x), gva_native=gva,
                                          nvfp4_groups=tuple(g for g in nvfp4.split(",") if g), nvfp4_scales=nvfp4_scales or None,
                                          **nv_acc_kw(nvfp4_rot, nvfp4_smooth, nvfp4_wscale, nvfp4_ascale, nvfp4_shadow))
        # rgh q48m (MAM_NV4; unset = off) and kai's small-step FP8 fallback (KAI_NV4_MINM; unset = off): _install_nv4.
        # mammoth and (kai) reax1: the same Qwen3.5-27B PplxEngine; forward_marked runs the same _core_gen / _mlp.
        if family in NV4_FAMILIES:
            self._install_nv4(model_dir)
        self._device = device
        self.steps = 0
        self.prefix_bidir = bool(getattr(self.engine, "prefix_bidir", False))
        if shared_until is not None:
            self.engine.shared_until = shared_until
        if shared_mixq is not None:
            self.engine.shared_mixq = bool(shared_mixq)
        if shared_state or shared_group:
            if not self.prefix_bidir:
                raise ValueError("shared_state / shared_group need a prefix_bidir checkpoint (vllm-sr/d3-turbo): the sharing "
                                 "is exact only when the state block attends to itself")
            self.shared, self.shared_group, self.shared_min = bool(shared_state), bool(shared_group), shared_min
            if self.shared:
                print(f"state-once mode: until layer {self.engine.shared_until} of {self.engine.n_layers}, "
                      f"state rows see {'all questions' if self.engine.shared_mixq else 'the state only'}", flush=True)
        self._after_build(warm_steps)

    def _pdl_apply(self):
        """kai lever N2 (KAI_PDL; unset = off, pdl.py never imported): programmatic dependent launch for the Triton, Inductor,
        f8g and NVFP4 kernels of a step; run by _after_build after _install_nv4, also after a deferred (rgh mx gptq) one
        (it routes the libraries installed there), and before the warm-up (which then compiles the PDL twins)"""
        if os.environ.get("KAI_PDL", "") in ("", "0"):
            return
        import pdl
        with startup.phase("pdl"):
            pdl.apply(os.environ, self._pp, log=lambda s: print(s, flush=True))

    def _after_build(self, warm_steps):
        """warm-up, KAI_ACTQ's check, then (reax1, rgh u4r2 R1_C1GRAPH) the single-question CUDA graphs: captured last, so
        they hold the kernels the engine serves with (the check may switch actq parts off); never without a warm-up (fla's
        caches and every compile / autotune must exist before a capture)"""
        defer = getattr(self, "_nv4_defer", None)
        if defer is not None:  # rgh mx (gptq): Hessians from the public seed set through this backend, before the warm-up
            self._nv4_defer = None
            self._nv4_be = self
            steps = self.steps
            # kai R8-5 review: the Hessian forwards run before _noxstore_check, so they use the full producer (an unchecked
            # K.add_rms_q_nox must not reach the derived weights); restored for the warm-up and the check
            pp = getattr(self, "_pp", None)
            nox = getattr(pp, "_NOXSTORE", None)
            if nox:
                pp._NOXSTORE = False
            try:
                self._install_nv4(defer)
            finally:
                self._nv4_be = None
                self.steps = steps  # calibration steps are not serving steps
                if nox:
                    pp._NOXSTORE = nox
        self._pdl_apply()  # kai N2: after every NV4 install (a deferred gptq plan included), before the warm-up
        self._gdnfuse_pdl_check()  # kai ck7 review: GDNFUSE's PDL twins checked before they serve (and before the warm-up)
        if warm_steps:
            self.warmup(warm_steps)
        self._actq_check()
        self._n1_check()
        self._noxstore_check()
        if self.family == "reax1" and warm_steps and hasattr(self.engine, "r1_capture"):
            on = os.environ.get("R1_C1GRAPH", "0") == "1"
            with startup.phase("r1_capture") if on else contextlib.nullcontext():
                self.engine.r1_capture()
        self._kai_start()

    def _kai_start(self):
        """kai fix M3: ONE line with every resolved switch (after every start-up check), counts of the warm-up apart"""
        self._kai = kai_status.Status(getattr(self, "_pp", None), getattr(self, "engine", None), self)
        print(self._kai.start_line(), flush=True)

    def kai_status(self) -> dict:
        """/health: the start-up switches and the usage counts since then ({} before the start line)"""
        return self._kai.health() if self._kai is not None else {}

    def kai_report(self):
        """shutdown: the usage counts since start, once (None afterwards or without a start line)"""
        return self._kai.report_line() if self._kai is not None else None

    def _init_render(self):
        """MAM_FASTRENDER (mammoth): the (head, tail) of the chat template, or None (reference renders)"""
        self._render = _fast_renderer(self._tok) if self.family == "mammoth" else None
        if _FASTRENDER:
            print(f"rgh c8: mammoth fast render {'on' if self._render else 'off'}", flush=True)

    def _fastenc_call(self, tok, state, n):
        """FASTENC with this backend's fast render (KAI_ENC1's fallback / self-check calls it)"""
        if self._render is None:  # the reference renders (default)
            return _encode_request_fast(tok, state, n)
        return _encode_request_fast(tok, state, n, self._render)

    def _actq_check(self):
        """kai lever Q (KAI_ACTQ): after the warm-up (the compiled producers are built and autotuned: the check must not be
        their first caller), each installed part is checked bit-for-bit on this GPU; a failing part goes back to nv4's path"""
        parts = getattr(self, "_actq_parts", None)
        if not parts:
            return
        import actq
        with startup.phase("actq_check"):
            good = actq.check(self._pp, parts, self.engine.device)
            actq.PARTS = good
        print(f"kai actq: parts {sorted(good)} active (asked {sorted(parts)}; silu eager={actq.SILU_EAGER})", flush=True)

    def _n1_check(self):
        """kai lever N1 (KAI_N1): after the warm-up (and actq's check), the fused epilogue is checked bit-for-bit on this GPU
        (every bf16 silu input vs the compiled K.silu_mul, the whole chain on real weights); a difference restores the
        original gate|up rows and the base chain"""
        if not getattr(self, "_n1_on", False):
            return
        import n1
        import nv4
        with startup.phase("n1_check"):
            self._n1_on = n1.check(nv4, self._pp, self.engine, self.engine.device)

    def _gdnfuse_pdl_check(self):
        """kai ck7 review (KAI_GDNFUSE x KAI_PDL): PplxEngine.__init__ checked the fused GDN kernels before PDL existed; with
        PDL on they serve as PDL twins, so every part that passed is checked again now (pplx_engine.gdnfuse_recheck); a
        difference keeps the unfused kernels. Without PDL (pdl.py not imported or off) the __init__ check covers serving."""
        pp = getattr(self, "_pp", None)
        if pp is None or not getattr(pp, "_GDNFUSE_ON", False):
            return
        pdl = sys.modules.get("pdl")
        if pdl is None or not getattr(pdl, "ON", False):
            return
        with startup.phase("gdnfuse_pdl_check"):
            pp.gdnfuse_recheck(log=lambda s: print(s, flush=True))

    def _noxstore_check(self):
        """kai lever R8-5 (KAI_NOXSTORE): after the warm-up (the compiled producers and their PDL twins exist and serve) and
        before reax1's graph capture, K.add_rms_q_nox is checked byte-for-byte against K.add_rms_q on this GPU; a difference
        (or no check possible) switches the lever off"""
        pp = getattr(self, "_pp", None)
        if pp is None or not getattr(pp, "_NOXSTORE", False):
            return
        eng = self.engine
        with startup.phase("noxstore_check"):
            ok = pp.check_noxstore(eng.device, int(eng.layers[0]["post_norm"].shape[0]), float(eng.eps),
                                   w=eng.layers[0]["post_norm"], log=lambda s: print(s, flush=True))
        if not ok:
            pp._NOXSTORE = False

    def _install_nv4(self, model_dir):
        """rgh q48m (MAM_NV4; unset = off): selective NVFP4 W4A4 projections derived here from the bf16 checkpoint in
        model_dir (nv4.py + nv4q.py + libnv4g.so, rgh's files verbatim; team image 86.38.238.34/rgh/engine@sha256:859001e1...;
        nv4.py / nv4q.py of rgh mx 03e0a1b6...: + method gptq). rgh mx REAX1_NV4 is the reax1 family's plan when set.
        A calibrated plan (gptq) is deferred to _after_build (before the warm-up), where it runs with be=self.
        nv4_plan.resolve adds the kai preset MAM_NV4=q48 (+ MAM_NV4_MLP / MAM_NV4_REST / MAM_NV4_METHOD layer lists).
        kai KAI_NV4_MINM=<rows> (unset = off; nv4_minm.py): steps below that many rows run the FP8 objects of every NVFP4
        projection (rgh's NV4 or our nvfp4= groups), kept resident; rgh's are the very objects nv4.install replaces."""
        import nv4_minm
        import nv4_plan
        import nv4qa
        spec = nv4_plan.resolve(os.environ, getattr(self, "family", None))
        be = getattr(self, "_nv4_be", None)
        if spec and nv4_plan.has_x_mods(spec) and (os.environ.get("KAI_ACTQ", "").strip() or not nv4qa.is_default(
                nv4qa.resolve(os.environ))):
            raise ValueError("MAM_NV4 methods with +x46 / perm quantise x in nv4q (perm gathers it): not with KAI_ACTQ or "
                             "MAM_NV4_ASCALE / MAM_NV4_AGLOBAL, which feed the GEMM their own quantisation")
        if spec and nv4_plan.needs_h(spec) and be is None:
            # rgh mx: a calibrated plan (gptq) runs this backend over the public seed set, so it is installed (with
            # KAI_NV4_MINM and the NVFP4 levers below) by _after_build, before the warm-up; the guard refuses at once
            nv4_plan.calibration(os.environ)
            self._nv4_defer = model_dir
            return
        minm = nv4_minm.resolve(os.environ)
        acc = nv4qa.resolve(os.environ)  # kai lever C (MAM_NV4_ASCALE / MAM_NV4_AGLOBAL; unset = rgh's quantiser)
        keep = {}
        if not spec and not nv4qa.is_default(acc):
            print(f"kai nv4qa: ignored (MAM_NV4_ASCALE / MAM_NV4_AGLOBAL = {acc[0]} / {acc[1]} without an NV4 plan)", flush=True)
        if spec:
            with startup.phase("nvfp4_derive"):
                import nv4
                nv4.SPEC = spec
                nv4.patch_engine(self._pp)
                before = nv4_minm.slots(self.engine) if minm else {}
                if be is not None:  # rgh mx: calibrated plan, H from the public seed set (nv4seed.py) through `be`
                    nv4_plan.calibration(os.environ)
                    nv4.install(self.engine, model_dir, be=be)
                else:
                    nv4.install(self.engine, model_dir)
                keep = nv4_minm.replaced(self.engine, before)
                # kai lever B (MAM_NV4TAB, default unset): exact NVFP4 tile table (nv4t.py, libnv4t.so), start-up
                # bit-equality check vs the nv4g variant nv4 runs there, per entry; installed before lever K so nv4k's shapes win
                if os.environ.get("MAM_NV4TAB", "").strip():
                    import nv4t
                    tab = nv4t.resolve(os.environ)
                    if tab:
                        self._nv4t = nv4t.install(nv4, tab, self.engine)
                # kai lever K (MAM_NV4K, default unset): CUTLASS stream-K NVFP4 variants for listed shapes (nv4k.py)
                if os.environ.get("MAM_NV4K", "").strip():
                    import nv4k
                    nv4k.install(nv4, os.environ["MAM_NV4K"])
                # kai lever C: NVFP4 activation block / global scale choice (nv4qa.py), checked against its torch reference
                if not nv4qa.is_default(acc):
                    nv4qa.set_modes(*acc)
                    chk = nv4qa.selfcheck(self.engine.device, pairs=[acc])
                    nv4qa.install(nv4)
                    print(f"kai nv4qa: activation scales {acc[0]} / global {acc[1]} (self-check {chk})", flush=True)
                # kai lever Q (KAI_ACTQ, default unset): NV4 activation quant fed from the FP8 producers (actq.py);
                # checked on the GPU after the warm-up (_actq_check)
                if os.environ.get("KAI_ACTQ", "").strip():
                    import actq
                    parts = actq.resolve(os.environ)
                    if parts:
                        actq.install(nv4, self._pp, parts)
                        self._actq_parts = parts
                        if not nv4qa.is_default(acc):
                            print("kai actq: lever C is on (MAM_NV4_ASCALE / AGLOBAL), actq yields to it (no Q gain)", flush=True)
                # kai lever N1 (KAI_N1, default unset): SwiGLU + the down projection's NVFP4 amax in the gate|up GEMM's
                # epilogue (n1.py, libnv4s.so); gate|up rows interleaved here, checked on the GPU after the warm-up (_n1_check)
                if os.environ.get("KAI_N1", "").strip():
                    import n1
                    tab = n1.resolve(os.environ)
                    if tab is not None:
                        self._n1_on = n1.install(nv4, self._pp, self.engine, tab) > 0
        if minm:
            with startup.phase("nv4_minm"):
                nv4_minm.install(self._pp, self.engine, str(model_dir), minm, keep)

    def _init_reax1(self, model_dir):
        """decisionmodels/reax-1: pplx-decider v1.2 joint protocol v3. Tokenisation is the author's own Encoder (identical
        prompts and validation); the production node runs it with `--v12-split single` (one sequence per question, the
        gateway fans multi-question requests out), which is what the one-question-per-call contract does as well."""
        mdir = Path(model_dir)
        sys.path.insert(0, str(mdir / "source" / "src"))
        from pplx_decider.protocol import PROTOCOL, Encoder, LengthError, RequestError, options as r1_options
        from transformers import AutoProcessor
        joint = json.loads((mdir / "joint_config.json").read_text())
        assert joint == PROTOCOL, f"joint_config.json does not match the author's PROTOCOL: {joint} != {PROTOCOL}"
        codes = json.loads((mdir / "decision_config.json").read_text())["codes"]
        self._encoder = Encoder(AutoProcessor.from_pretrained(model_dir), codes)
        self._v12_errors = (LengthError, RequestError)
        self._r1_options = r1_options
        self._r1_tok = getattr(self._encoder.processor.tokenizer, "_tokenizer", None) if _R1_FASTENC else None
        assert self.max_input_tokens <= self.max_batch_tokens, "max_input_tokens must fit one step (budget)"

    # ---------------------------------------------------------------- Backend API
    def options(self, question):
        if self.family == "mammoth":
            return list(qev_contract.normalize_question(question).keys)
        if self.family == "d3":
            return list(self._d3.normalize_question(question).keys)
        return list(self._pp.options(question)[0])

    def encode(self, state, question):
        aux = None
        if self.family == "mammoth":
            n = qev_contract.normalize_question(question)
            once = self._encode_once(state, n) if self.encode_once else None
            if once is not None:
                ids, aux = once
            else:
                if _FASTENC and self._enc1 is not None:  # kai lever I KAI_ENC1: FASTENC's result, one render + one tokenise
                    full, aux = self._enc1.encode(state, n)
                elif _FASTENC:  # rgh f8m (B5 encode_once wins when both are on); m1: + MAM_FASTRENDER
                    full, aux = self._fastenc_call(self._tok, state, n)
                else:
                    full, aux = qev_contract.encode_request(self._tok, state, [n])
                ids = full[0]
            n_opt = len(n.keys)
        elif self.family == "d3":
            n = self._d3.normalize_question(question)  # ValueError with the reason
            text = self._d3.render(self._tok, state, n.rendered, self._codes)
            if self.prefix_bidir:
                ids, aux = self._d3.encode_split(self._tok, text, state)
            else:
                ids = self._tok(text, add_special_tokens=False)["input_ids"]
            n_opt = len(n.keys)
        elif self.family == "reax1":
            return self._encode_reax1(state, question)
        else:
            validate_pplx_question(question)
            ids, n_opt = self._prompter.ids(state, question)
        if len(ids) > self.max_input_tokens:
            raise ValueError(f"maximum context length exceeded: {len(ids)} > {self.max_input_tokens} tokens")
        return Encoded(ids, len(ids), n_opt, aux)

    def _encode_once(self, state, n):
        """encode_once with a self-check: the first `encode_once_check` calls are compared with encode_request; one
        difference turns encode_once off for good (and that call is served from encode_request)."""
        got = qev_contract.encode_once(self._tok, state, n)
        if got is None or self._once_seen >= self.encode_once_check:
            self._once_seen += got is not None
            return got
        self._once_seen += 1
        full, p = qev_contract.encode_request(self._tok, state, [n])
        if got != (full[0], p):
            self.encode_once = False
            print(f"encode_once: differs from encode_request (P {got[1]} vs {p}, {len(got[0])} vs {len(full[0])} ids); "
                  "turned off", file=sys.stderr, flush=True)
            return full[0], p
        return got

    def _encode_reax1(self, state, question):
        if not isinstance(question, dict) or question.get("type") not in ("choice", "score", "noul"):
            raise ValueError("each question needs type choice|score|noul")
        if getattr(self, "_r1_tok", None) is not None:
            return self._encode_reax1_fast(state, question)
        try:
            chunks = self._encoder.plan({"state": state, "questions": {"q": question}}, self.max_input_tokens)
        except self._v12_errors[0] as exc:  # over-long single question: the engine's 422 context_length
            raise ValueError(f"maximum context length exceeded: {exc}") from None
        except self._v12_errors[1] as exc:
            raise ValueError(str(exc)) from None
        if len(chunks) != 1:  # one question cannot be split
            raise ValueError("maximum context length exceeded: the question needs more than one sequence")
        c = chunks[0]
        return Encoded(c.input_ids.tolist(), int(c.length), int(c.counts[0]), int(c.markers[0]))

    def _encode_reax1_fast(self, state, question):
        """R1_FASTENC: Encoder.plan for one question without the processor call (see _R1_FASTENC)."""
        e = self._encoder
        request = {"state": state, "questions": {"q": question}}
        try:
            e.validate(request)
        except self._v12_errors[1] as exc:
            raise ValueError(str(exc)) from None
        ids = self._r1_tok.encode_batch([e.text(request)], add_special_tokens=True)[0].ids
        marks = [i for i, t in enumerate(ids) if t == e.marker_id]
        if len(marks) != 1:
            raise ValueError("Readout markers do not match questions after tokenization")
        n, cap = len(ids), self.max_input_tokens
        if cap is not None and n > cap:
            raise ValueError(f"maximum context length exceeded: Question 'q' needs {n} tokens with the state and images, "
                             f"over the {cap}-token limit; shorten the state, lower the image resolution or raise max_length")
        return Encoded(list(ids), n, len(self._r1_options(question)[0]), marks[0])

    def run(self, batch):
        import torch
        if self.family == "reax1":
            ids = torch.tensor([t for e in batch for t in e.payload], device=self._device)
            logits = self.engine.forward_marked(ids, [len(e.payload) for e in batch], [[e.aux] for e in batch])
            self.steps += 1
            return self.engine.probs(logits, [e.n_options for e in batch])
        if self.shared:
            return self._run_shared(batch)
        ids = torch.tensor([t for e in batch for t in e.payload], device=self._device)
        states = [e.aux for e in batch] if self.prefix_bidir else None
        logits = self.engine.forward(ids, [len(e.payload) for e in batch], states)
        self.steps += 1
        return self.engine.probs(logits, [e.n_options for e in batch])

    # ---------------------------------------------------------------- d3-turbo: state-once routing
    def plan_shared(self, batch):
        """Split a step into (plain, groups) the way the production d3-turbo node does (engine/server.py of 75d0659).

        groups: [(prefix_ids, [(suffix_ids, batch_index), ...]), ...] for PplxEngine.forward_groups; plain: batch indices that
        cannot use the shared path (state block shorter than the 3-token conv window) and run through forward().
        With shared_group, queued calls whose first `shared_min` tokens AND state-block length agree are merged into one
        group when their common prefix covers exactly the state block (>= shared_min tokens); every other single decision
        becomes a group with one suffix, so the answer does not depend on sibling co-arrival."""
        items = [(e.payload, e.aux, i) for i, e in enumerate(batch)]
        singles, merged = [], []
        if self.shared_group:
            buckets = {}
            for it in items:
                if len(it[0]) <= self.shared_min:
                    singles.append(it)
                else:
                    buckets.setdefault((tuple(it[0][:self.shared_min]), it[1]), []).append(it)
            for its in buckets.values():
                if len(its) < 2:
                    singles += its
                    continue
                seqs = [x[0] for x in its]
                m = min(map(len, seqs)) - 1
                k = 0
                while k < m and all(x[k] == seqs[0][k] for x in seqs):
                    k += 1
                sb = its[0][1]
                k = sb if k >= sb else 0  # the shared prefix is exactly the (token-identical) state block
                if k < self.shared_min:
                    singles += its
                else:
                    merged.append((seqs[0][:k], [(x[0][k:], x[2]) for x in its]))
        else:
            singles = items
        conv, plain = [], []
        for ids, sb, i in singles:
            if sb is not None and 3 <= sb <= len(ids) - 1:  # forward_groups needs a prefix >= the conv window (3)
                conv.append((ids[:sb], [(ids[sb:], i)]))
            else:
                plain.append(i)
        return plain, conv + merged

    def _run_shared(self, batch):
        import torch
        out = [None] * len(batch)
        plain, groups = self.plan_shared(batch)
        if plain:
            ids = torch.tensor([t for i in plain for t in batch[i].payload], device=self._device)
            logits = self.engine.forward(ids, [len(batch[i].payload) for i in plain], [batch[i].aux for i in plain])
            for i, p in zip(plain, self.engine.probs(logits, [batch[i].n_options for i in plain])):
                out[i] = p
        if groups:
            order = [i for _, ss in groups for _, i in ss]
            logits = self.engine.forward_groups([(p, [s for s, _ in ss]) for p, ss in groups])
            for i, p in zip(order, self.engine.probs(logits, [batch[i].n_options for i in order])):
                out[i] = p
        self.steps += 1
        return out

    # ---------------------------------------------------------------- family answer shapes (/v1/systemone)
    def answer(self, question, probs):
        if self.family == "mammoth":
            return qev_contract.product_answer(qev_contract.normalize_question(question), probs)
        if self.family == "d3":
            return self._d3.product_answer(self._d3.normalize_question(question), probs)
        a = self._pp.answer(question, probs)
        if question["type"] == "noul":
            a["probabilities"] = {"false": 1.0 - a["noul"], "true": a["noul"]}
        return a

    # ---------------------------------------------------------------- warm-up
    def _enc(self, n, n_opt=2):
        """Synthetic step item of n tokens (ids discarded); d3-turbo needs a state block, the marker is the last token."""
        if self.family == "reax1":
            return Encoded([1] * n, n, n_opt, n - 1)
        return Encoded([1] * n, n, n_opt, max(1, n // 2) if self.prefix_bidir else None)

    def warmup(self, steps=200):
        """Trigger every torch.compile specialisation / Triton autotune before serving (first requests of each new batch
        shape otherwise stall the GPU loop for seconds). Synthetic token ids; results discarded."""
        cap = min(self.max_input_tokens, 8192)

        def go(lens, n_opt=2):
            lens = [min(n, cap) for n in lens]
            while sum(lens) > self.max_batch_tokens and len(lens) > 1:
                lens.pop()
            self.run([self._enc(n, n_opt) for n in lens])

        shapes = [[n] for n in (17, 64, 300, 700, 1500, 3000, 6000, 8192)]
        shapes += [[300] * k for k in (2, 3, 4, 5, 7, 8, 12, 16, 24, 32)]
        shapes += [[700] * 11 + [64], [4000, 4000, 4000, 4000], [2000] * 8, [8192, 8000], [500] * 32, [16000]]
        # start-up lines (startup.py): run() returns host lists, so each phase ends with its GPU work done
        sweep = [lens for _ in range(2) for lens in shapes]
        with startup.phase("first_forward"):
            go(sweep[0])
        with startup.phase("warmup_shapes"):
            for lens in sweep[1:]:
                go(lens)
        rng = random.Random(7)
        with startup.phase("warmup_steps"):
            for _ in range(steps):
                lens, n = [], rng.choice([1, 1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 40])
                for _ in range(n):
                    length = int(min(cap, max(16, rng.lognormvariate(6.3, 0.8))))
                    if sum(lens) + length > self.max_batch_tokens:
                        break
                    lens.append(length)
                go(lens or [64], rng.randint(2, 12))
        if self.shared:
            with startup.phase("warmup_shared"):
                self._warmup_shared(steps, rng, cap)

    def _warmup_shared(self, steps, rng, cap):
        """The shared-state kernels as the production node warms them: a deterministic sweep over every step-size bucket
        (fla's conv kernel specialises and re-autotunes per 1024-token bucket of the step; a size first seen while serving
        stalls the GPU loop for tens of seconds), then random steps of shared groups (long prefix + short suffixes)."""
        eng, budget = self.engine, self.max_batch_tokens
        full = eng.shared_until < eng.n_layers
        for T in range(512, budget + 1, 512):
            for n_seq in (1, 3, 9):
                lens = [max(16, T // n_seq)] * n_seq
                if max(lens) > cap:
                    continue
                self.run([self._enc(n) for n in lens])
                if n_seq > 1:  # one shared group of this full size: prefix + n_seq suffixes
                    P = T // n_seq - 40 if full else T - 40 * n_seq
                    if P < 64:
                        continue
                    self._warm_groups([([1] * P, [([1] * 40, 2)] * n_seq)])
        for _ in range(max(40, steps // 2)):
            groups, tot = [], 0
            for _ in range(rng.choice([1, 2, 3, 4, 6, 8, 12])):
                P = int(min(cap - 600, max(64, rng.lognormvariate(6.8, 0.8))))
                subs = [([1] * int(max(8, rng.lognormvariate(4.5, 0.7))), rng.randint(2, 12))
                        for _ in range(rng.choice([1, 2, 3, 5, 8, 12, 32, 64]))]
                n = (P * len(subs) if full else P) + sum(len(s) for s, _ in subs)
                if tot + n > budget:
                    break
                groups.append(([1] * P, subs))
                tot += n
            if groups:
                self._warm_groups(groups)

    def _warm_groups(self, groups):
        logits = self.engine.forward_groups([(p, [s for s, _ in ss]) for p, ss in groups])
        self.engine.probs(logits, [n for _, ss in groups for _, n in ss])
        self.steps += 1


def validate_pplx_question(question):
    """Same checks as the legacy server (engine/server.py validate) for one question; raises ValueError."""
    if not isinstance(question, dict) or question.get("type") not in ("choice", "score", "noul"):
        raise ValueError("each question needs type choice|score|noul")
    crit = question.get("criteria")
    if question["type"] == "choice" and (not isinstance(crit, dict) or not 1 <= len(crit) <= 255):
        raise ValueError("choice questions take 1 to 255 options per choice")
    if question["type"] == "score" and (not isinstance(crit, list) or not 2 <= len(crit) <= 10):
        raise ValueError("a score takes 2 to 10 levels")
