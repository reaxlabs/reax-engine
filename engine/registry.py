"""Model registry of the REAX Engine: id -> ModelSpec (weights, licence, backend factory, hardware notes, status).

`python -m engine.registry` prints the table; `python -m engine.registry --json` the full records.
Factories import their backend lazily, so listing the registry needs neither torch nor GPUs.

status
  native           REAX kernels (packed varlen prefill, fused/quantised GEMMs) in this repo
  vllm-backed      decision readout through a vLLM server (prefill + 1 token); see engine/backends/vllm_readout.py
  cpu              llama.cpp / ONNX on CPU
  classifier       model kind `classifier_heads`: encoder + N fixed heads on onnxruntime (engine/backends/onnx_heads.py)
  documented-only  described in docs/, no backend
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Callable

STATUSES = ("native", "vllm-backed", "cpu", "classifier", "documented-only")
MODEL_KINDS = ("decision", "classifier_heads")
BACKEND_KINDS = ("native-gpu", "vllm", "gguf-cpu", "onnx-cpu", "none")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    hf_repo: str
    revision: str                       # full commit sha the numbers in docs/ were measured on (or "" if none)
    licence: str                        # SPDX
    licence_notice: str                 # what must be kept / done on redistribution
    params: str
    arch: str
    readout: str                        # readout summary: prompt family, option tokens, temperature
    backend_kind: str                   # one of BACKEND_KINDS
    status: str                         # one of STATUSES
    hardware: str                       # hardware / memory notes
    factory: Callable | None = field(default=None, compare=False, repr=False)   # factory(model_dir, **opts) -> Backend
    aliases: tuple = ()
    gated: bool = False                 # HF gate: each operator accepts the licence with their own account
    weights_gb: float = 0.0             # download size (0 = not applicable)
    notes: str = ""
    model_kind: str = "decision"        # "decision" (readout over answer options) | "classifier_heads" (encoder + fixed heads)
    production_opts: dict = field(default_factory=dict, compare=False)   # backend options of the production recipe, applied by
    #                                                                       serve.py when REAX_PROFILE=reference (the engine track v2 images)

    def __post_init__(self):
        assert self.status in STATUSES, self.status
        assert self.backend_kind in BACKEND_KINDS, self.backend_kind
        assert self.model_kind in MODEL_KINDS, self.model_kind

    def record(self) -> dict:
        d = asdict(self)
        d.pop("factory")
        d["has_factory"] = self.factory is not None
        return d


# ------------------------------------------------------------------------------------------------ factories (lazy)
def _plumb(model_dir, **opts):
    import sys
    from pathlib import Path
    here = str(Path(__file__).resolve().parent)  # plumb_engine imports pplx_engine as a flat top-level module
    if here not in sys.path:
        sys.path.insert(0, here)
    from plumb_engine import PlumbBackend
    return PlumbBackend(model_dir, **opts)


def _onnx_heads(model_id):
    def make(model_dir, **opts):
        from engine.backends import onnx_heads
        assert model_id == onnx_heads.S1_LLM_AUTO_ROUTER.model_id
        return onnx_heads.OnnxHeadsBackend(model_dir, spec=onnx_heads.S1_LLM_AUTO_ROUTER, **opts)
    return make


def _pplx(model_dir, **opts):
    from engine.backends.qwen35 import Qwen35Backend
    return Qwen35Backend(model_dir, family="pplx", name="pplx-decider-v1.1-27b", **opts)


def _d3(model_dir, **opts):
    from engine.backends.qwen35 import Qwen35Backend
    return Qwen35Backend(model_dir, family="d3", name="d3", **opts)


def _d3_turbo(model_dir, **opts):
    from engine.backends.qwen35 import Qwen35Backend
    return Qwen35Backend(model_dir, family="d3", name="d3-turbo", **opts)


def _mammoth(model_dir, **opts):
    from engine.backends.qwen35 import Qwen35Backend
    return Qwen35Backend(model_dir, family="mammoth", name="mammoth-1", **opts)


def _reax1(model_dir, **opts):
    from engine.backends.qwen35 import Qwen35Backend
    return Qwen35Backend(model_dir, family="reax1", name="reax-1", **opts)


def _quyet(model_dir, **opts):
    from engine.backends.vllm_readout import VllmReadoutBackend
    return VllmReadoutBackend(model_dir, family="quyet", name="quyet-1.0-large", **opts)


def _jeffan(model_dir, **opts):
    from engine.backends.vllm_readout import VllmReadoutBackend
    return VllmReadoutBackend(model_dir, family="jeffan", name="jeffan-31b-1710", **opts)


def _rune(model_dir, **opts):
    from engine.backends.vllm_readout import VllmReadoutBackend
    return VllmReadoutBackend(model_dir, family="rune", name="rune-26b-a4b", **opts)


def _blink(model_dir, **opts):
    from engine.backends.vllm_readout import VllmReadoutBackend
    return VllmReadoutBackend(model_dir, family="blink", name="blink-v0.3-26b-a4b", **opts)


def _gguf(name):
    def make(model_dir, **opts):
        from engine.backends import gguf_cpu
        return gguf_cpu.load(name, model_dir, **opts)
    return make


# ------------------------------------------------------------------------------------------------ entries
_SPECS = (
    ModelSpec(
        id="plumb-4b", aliases=("s1-fast", "plumb"),
        hf_repo="crh225/plumb-4b", revision="55de037801a8a9b9de3db5c0e16cef86210c2186",
        licence="Apache-2.0", licence_notice="Apache-2.0; base Qwen3.5-4B (JevK5 LoRA merged). Keep the model's NOTICE.",
        params="4.2B", arch="Qwen3.5 hybrid: 24 Gated-DeltaNet + 8 full-attention layers (causal), hidden 2560, tied embeddings",
        readout="jevk5 prompt (system + JSON user turn), option letters A..P, last-position letter logits / T=2.07, one token each",
        backend_kind="native-gpu", status="native", factory=_plumb, weights_gb=8.4,
        hardware="one CUDA GPU, ~9 GB used (manifest minimum 24 GB); max 4096 input tokens, 16 options"),
    ModelSpec(
        id="pplx-decider-v1.1-27b", aliases=("pplx", "pplx-decider"),
        hf_repo="perplexity-ai/pplx-decider-v1.1-27b", revision="6195aa55a72ae21e503c8a60198f7b390d42adb5",
        licence="Apache-2.0", licence_notice="Apache-2.0; derived from Qwen/Qwen3.8-27B (Apache-2.0). Keep the repo NOTICE.",
        params="26.1B", arch="Qwen3.8 hybrid: 48 Gated-DeltaNet + 16 NONCAUSAL full-attention layers, hidden 5120, custom code",
        readout="255-way code readout at the last position, T from decision_config.json; prompt adapted from the author's code",
        backend_kind="native-gpu", status="native", factory=_pplx, weights_gb=52,
        hardware="80+ GB card for bf16; FP8 28.7 GiB; 32 GB cards with CIE_LAZY_LOAD=1; measured on RTX PRO 6000 (sm_120), H100/H200/B200, RTX 5090",
        notes="stock vLLM/SGLang/TensorRT-LLM cannot serve it (noncausal attention)",
        production_opts={"budget": 4096, "warm_steps": 200}),
    ModelSpec(
        id="d3", aliases=("vllm-sr-d3",),
        hf_repo="vllm-sr/d3", revision="225be8b1e2270becab69c5fe05b26f99cbbdd66b",
        licence="Apache-2.0", licence_notice="Apache-2.0; (c) vLLM Semantic Router Team; keep upstream/d3/LICENSE and NOTICE. Built on Qwen/Qwen3.8-27B.",
        params="26.1B", arch="Qwen3.5 hybrid (same text stack as pplx-decider): 48 Gated-DeltaNet + 16 NONCAUSAL full-attention layers; vision tower unused",
        readout="d3-code-readout-v1: system + State/Question/Options user turn, 255 single-token codes (A..Z, AA..), fp32 readout, T from decision_config.json",
        backend_kind="native-gpu", status="native", factory=_d3, weights_gb=52,
        hardware="same as pplx-decider; long prompts (64k+ tokens) need the int64 index fixes in this repo",
        notes="decision_config.json says attention_mode=noncausal_full_attention. Contract verified against revisions 5ce995d and 225be8b."),
    ModelSpec(
        id="d3-turbo", aliases=("d3turbo", "vllm-sr-d3-turbo"),
        hf_repo="vllm-sr/d3-turbo", revision="cfa0d41d253467e4742d5aedb46d4f0ef3b84b41",
        licence="Apache-2.0", licence_notice="Apache-2.0; (c) vLLM Semantic Router Team; keep upstream/d3/LICENSE and NOTICE. Built on Qwen/Qwen3.8-27B.",
        params="26.1B", arch="Qwen3.5 hybrid (same text stack as d3): 48 Gated-DeltaNet + 16 full-attention layers with PREFIX_BIDIR attention (the state block attends to itself only, the question block reads the whole row)",
        readout="d3-code-readout-v1 as d3 (system + State/Question/Options user turn, 255 single-token codes, fp32 readout); decision_config.json says attention_mode=prefix_bidir",
        backend_kind="native-gpu", status="native", factory=_d3_turbo, weights_gb=52,
        hardware="same as d3; FP8 ~31 GiB with CIE_LAZY_LOAD=1, step budget 16384 tokens ~35 GiB",
        notes="production recipe of the s1-pro d3-turbo node: state-once through forward_groups for every call (REAX_PROFILE=reference applies it: "
              "shared_state, shared_group, shared_until 64, budget 16384, 8192-token cap per question); without the profile the engine runs plain prefix_bidir.",
        production_opts={"budget": 16384, "max_input_tokens": 8192, "warm_steps": 200, "shared_state": True, "shared_group": True,
                         "shared_until": 64, "shared_mixq": False}),
    ModelSpec(
        id="mammoth-1", aliases=("mammoth",),
        hf_repo="baselabs/mammoth-1", revision="61cec44c281c693992e3d6ebeb44dd3402dfa186",
        licence="MIT", licence_notice="MIT; keep upstream/mammoth/LICENSE. qev contract port: Apache-2.0, OneJev authors (NOTICE).",
        params="27B", arch="Qwen3.5 hybrid: Gated-DeltaNet causal, full attention prefix_bidir",
        readout="qev-labels-v2; A..Z then two-letter slots; slot_lm fp32, T=1.0",
        backend_kind="native-gpu", status="native", factory=_mammoth, weights_gb=54.7,
        hardware="FP8 state-once ~35.2 GB peak; two-engine fit requires GPU check on RTX PRO 6000 96 GB",
        notes="Production private engine 5117ec3; cap 16384, state-once prefix_bidir. CIE_LAZY_LOAD=1 in reference image.",
        production_opts={"linear": "fp8", "compile": "1", "gva": "1", "budget": 16384, "max_input_tokens": 16384,
                         "warm_steps": 200, "shared_state": True, "shared_group": True, "shared_until": 64, "shared_mixq": False}),
    ModelSpec(
        id="reax-1", aliases=("reax1", "decisionmodels-reax-1"),
        hf_repo="decisionmodels/reax-1", revision="ca2a0cc54a537c1f5cf24e5b09387cfb357b6518",
        licence="Apache-2.0", licence_notice="Apache-2.0 (decisionmodels.io REAX 1 release). Keep the repo LICENSE/NOTICE.",
        params="26.1B", arch="Qwen3.5 hybrid as pplx-decider-v1.1-27b (48 Gated-DeltaNet + 16 NONCAUSAL full-attention layers), joint request format (protocol v3)",
        readout="pplx-decider v1.2 joint format: one sequence per request, 255-way code readout at the question marker token, T from decision_config.json; "
                "tokenised by the author's Encoder from the checkpoint's source/src/pplx_decider (run from the read-only weights mount)",
        backend_kind="native-gpu", status="native", factory=_reax1, weights_gb=48.6,
        hardware="same as pplx-decider; FP8 ~31-34.5 GiB on RTX PRO 6000 (production recipe: CIE_LAZY_LOAD=1, budget 8192)",
        notes="served as one question per sequence (the production node's --v12-split single); a question over 8192 tokens gets 422",
        production_opts={"budget": 8192, "max_input_tokens": 8192, "warm_steps": 200}),
    ModelSpec(
        id="quyet-1.0-large", aliases=("quyet", "quyet-1-large"),
        hf_repo="chinhnc/Quyet-1.0-Large", revision="d997d58951e3c0da4486cf0884ca9d4d916e95dc",
        licence="Apache-2.0", licence_notice='Apache-2.0; keep NOTICE including its first line "Quyet by Chinh Nguyen" on redistribution. Base google/gemma-4-31B-it.',
        params="31.3B", arch="Gemma-4 dense: 60 layers = 50 sliding-window (1024) + 10 full attention (causal), hidden 5376, merged LoRA r16",
        readout="prompt v2 (single user turn, compact JSON state), option letters A..J (native max 10 options), per-type temperatures from quyet_config.json (choice 1.3007 / score 1.3159 / noul 1.4957)",
        backend_kind="vllm", status="vllm-backed", factory=_quyet, weights_gb=62.5,
        hardware="80+ GB card (bf16 62.5 GB); measured RTX PRO 6000 96 GB",
        notes="Reference profile uses production FP8 + FP8 KV recipe, utilization 0.45 for two-engine fit (GPU gate pending).",
        production_opts={"preset": "ref-prod", "max_input_tokens": 8000, "max_options": 10, "no_truncation": True}),
    ModelSpec(
        id="jeffan-31b-1710", aliases=("jeffan",),
        hf_repo="tattrongvu/Jeffan-31b-1710", revision="cebfa86c2bed2b5c83b12e44de4374b20c87fd79",
        licence="Apache-2.0", licence_notice="Apache-2.0 (LICENSE in the repo since 22 Oct 2026); base google/gemma-4-31B-it (Apache-2.0).",
        params="31.3B", arch="Gemma-4 dense (Gemma-4-31B base): 60 layers = 50 sliding-window (1024) + 10 full attention (causal), hidden 5376",
        readout="dm_decision_engine answer-code readout (layout state_first): system line + State/Question/Options user turn, single-token codes A..Z, AA.. (255), restricted softmax T=1, conditional read-twice re-read below top-1 0.9",
        backend_kind="vllm", status="vllm-backed", factory=_jeffan, weights_gb=62.6,
        hardware="FP8 weights + FP8 KV: one 96 GB card shared with another model (util 0.44); bf16 needs 80+ GB",
        notes="Served caps: 8,192 tokens first read (gateway limit), read-twice prompt up to 16,440 (else the first read stands, as in the author path).",
        production_opts={"preset": "fp8-kv-colo-bi", "max_input_tokens": 8192, "max_options": 255}),
    ModelSpec(
        id="rune-26b-a4b", aliases=("rune", "s1-pro", "rune-26b-a4b-v3"),
        hf_repo="surogate/rune-26b-a4b-GGUF", revision="c6b360d47895bb77bdf3805a13ee5a5557ab1921",
        licence="Apache-2.0", licence_notice="Apache-2.0 (base google/gemma-4-26B-A4B-it). HF-gated: each operator accepts the gate with their own account; do not mirror the weights.",
        params="~26B (4B active)", arch="Gemma-4 MoE (from the base config: 30 layers, 25 sliding + 5 full attention, 128 experts top-8); the repo name says GGUF but the files are bf16 safetensors",
        readout="Surogate decisions v1 (system line + SHARED STATE JSON user turn), letters A..Z then single-token two-letter codes, raw soft-capped logits / T=2.0",
        backend_kind="vllm", status="vllm-backed", factory=_rune, gated=True, weights_gb=52,
        hardware="80+ GB card (60 GB minimum); measured RTX PRO 6000 97.9 GB",
        notes="Rune's own config.json is not readable without the gate; architecture is taken from the base model's config. "
              "REAX_PROFILE=reference uses the ref-pool preset = the flags of the REAX s1-pro pool runtime (text only; the weights are "
              "verified against manifests/s1-pro-v1.json by the image entrypoint).",
        production_opts={"preset": "ref-pool", "max_input_tokens": 4096, "max_options": 64}),
    ModelSpec(
        id="blink-v0.3-26b-a4b", aliases=("blink", "s1-fast-2"),
        hf_repo="PixilabAI/Blink-v0.3-26B-A4B-NVFP4", revision="08caad20c1574fb76341ca809ff38fdb081f7d4a",
        licence="Apache-2.0", licence_notice="Apache-2.0 per the model card tag (base google/gemma-4-26B-A4B-it, Apache-2.0); the repo ships no LICENSE file. Weights are not mirrored here.",
        params="~26B (4B active)", arch="Gemma-4 MoE, NVFP4 (ModelOpt) checkpoint, 17.5 GB; needs a Blackwell GPU (NVFP4)",
        readout="Surogate decisions v1 (same prompt as Rune), letters A..Z (two-letter codes above 26 options), softmax over processed logprobs / T=1.3",
        backend_kind="vllm", status="vllm-backed", factory=_blink, weights_gb=17.5,
        hardware="Blackwell GPU (NVFP4); 31 GiB peak measured at utilization 0.30, 32k context; measured RTX PRO 6000",
        notes="serving recipe = vLLM with the bi1 preset (batch-invariant, c1 zero flips); the author's own readme serves it with plain vLLM. "
              "REAX_PROFILE=reference uses the ref-prod preset (the s1-fast production flags at gpu_memory_utilization 0.45).",
        production_opts={"preset": "ref-prod", "max_input_tokens": 8000}),
    ModelSpec(
        id="gutsy-0.8b", aliases=("gutsy", "s1-cpu"),
        hf_repo="kouhxp/gutsy", revision="f9995fe26a337f1a4ea3f1a5ed38c9799e5a8f63",
        licence="Apache-2.0", licence_notice="Apache-2.0; base Qwen3.5-0.8B.",
        params="0.8B", arch="Qwen3.5-0.8B GGUF Q8_0 (llama.cpp)",
        readout="author prompt (State/Question/Options), letters A..P, one pass, per-type temperatures (choice 1.0628 / score 1.0 / yes-no 0.7652)",
        backend_kind="gguf-cpu", status="cpu", factory=_gguf("gutsy-0.8b"), weights_gb=0.8,
        hardware="CPU only (AVX2 or NEON), >= 4 cores, 4 GB RAM; ~1 s/decision on 4 cores",
        notes="backend: engine/backends/gguf_cpu.py (llama.cpp CPU)"),
    ModelSpec(
        id="plumb-4b-q8", aliases=("s1-lite", "plumb-q8"),
        hf_repo="crh225/plumb-4b-GGUF", revision="71c1c57e523219124f7be9c486d475d19891185f",
        licence="Apache-2.0", licence_notice="see model card",
        params="4.2B", arch="Plumb-4B, GGUF Q8_0 (llama.cpp)",
        readout="same jevk5 prompt and T=2.07 as plumb-4b; Q8-vs-bf16 mean probability distance 0.115",
        backend_kind="gguf-cpu", status="cpu", factory=_gguf("plumb-4b-q8"), weights_gb=4.5,
        hardware="CPU works but is slow; manifest asks one NVIDIA GPU >= 8 GB",
        notes="backend: engine/backends/gguf_cpu.py (llama.cpp CPU)"),
    ModelSpec(
        id="s1-llm-auto-router", aliases=("llm-auto-router",),
        hf_repo="decisionmodels/s1-llm-auto-router", revision="572c5b6cdb53b77fbb26ed33568ebbc5b71c69c0",
        licence="MIT", licence_notice="MIT; base jhu-clsp/mmBERT-small.",
        params="~0.1B (ONNX, int8 embeddings, 183 MB)", arch="mmBERT-small student encoder (11 layers) with 7 fixed classification heads",
        readout="NOT a letter-logit readout: encoder + 7 fixed heads (category 9-way, difficulty 5, stakes 4, 4 binary flags), softmax / sigmoid with per-head temperatures from meta.json; no prompt, fixed questions only",
        backend_kind="onnx-cpu", status="classifier", model_kind="classifier_heads", factory=_onnx_heads("s1-llm-auto-router"),
        weights_gb=0.2, hardware="CPU (onnxruntime), p50 13.9 ms on 4 pinned cores (model card); see docs/models/s1-llm-auto-router.md",
        notes="classifier_heads kind: served at POST /v1/classify (and the production-compatible /internal/v1/decide, /v1/systemone) by engine/serve.py; execution is identical to the reference path"),
)

REGISTRY: dict[str, ModelSpec] = {s.id: s for s in _SPECS}
_ALIASES = {a: s.id for s in _SPECS for a in s.aliases}


def get(model_id: str) -> ModelSpec:
    mid = _ALIASES.get(model_id, model_id)
    if mid not in REGISTRY:
        raise KeyError(f"unknown model {model_id!r}; known: {', '.join(REGISTRY)}")
    return REGISTRY[mid]


def build(model_id: str, model_dir, **opts):
    """Construct the Backend of a model: build('quyet-1.0-large', '/model', preset='fp8-kv')."""
    spec = get(model_id)
    if spec.factory is None:
        raise NotImplementedError(f"{spec.id} has status {spec.status}: no backend ({spec.notes})")
    return spec.factory(model_dir, **opts)


def table() -> str:
    cols = ("id", "status", "backend", "params", "licence", "weights", "hf repo @ revision")
    rows = [(s.id, s.status, s.backend_kind, s.params, s.licence, f"{s.weights_gb:g} GB" + (" gated" if s.gated else ""),
             f"{s.hf_repo}@{s.revision[:8]}") for s in REGISTRY.values()]
    w = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    line = lambda r: "  ".join(v.ljust(w[i]) for i, v in enumerate(r)).rstrip()  # noqa: E731
    return "\n".join([line(cols), line(["-" * x for x in w]), *map(line, rows)])


def main(argv=None):
    ap = argparse.ArgumentParser(description="REAX Engine model registry")
    ap.add_argument("--json", action="store_true", help="print full records")
    ap.add_argument("model", nargs="?", help="show one model")
    a = ap.parse_args(argv)
    if a.model:
        print(json.dumps(get(a.model).record(), indent=1))
    elif a.json:
        print(json.dumps([s.record() for s in REGISTRY.values()], indent=1))
    else:
        print(table())


if __name__ == "__main__":
    main()
