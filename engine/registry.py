"""Model registry of the REAX Engine: id -> ModelSpec (weights, licence, backend factory, hardware notes, status).

`python -m engine.registry` prints the table; `python -m engine.registry --json` the full records.
Factories import their backend lazily, so listing the registry needs neither torch nor GPUs.

status
  native           REAX kernels (packed varlen prefill, fused/quantised GEMMs) in this repo
  vllm-backed      decision readout through a vLLM server (prefill + 1 token); see engine/backends/vllm_readout.py
  cpu              llama.cpp / ONNX on CPU
  documented-only  described in docs/, no backend
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Callable

STATUSES = ("native", "vllm-backed", "cpu", "documented-only")
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

    def __post_init__(self):
        assert self.status in STATUSES, self.status
        assert self.backend_kind in BACKEND_KINDS, self.backend_kind

    def record(self) -> dict:
        d = asdict(self)
        d.pop("factory")
        d["has_factory"] = self.factory is not None
        return d


# ------------------------------------------------------------------------------------------------ factories (lazy)
def _plumb(model_dir, **opts):
    from engine import plumb_engine  # module comes from the plumb slice: expected hook load(model_dir, **opts) -> Backend
    return plumb_engine.load(model_dir, **opts)


def _pplx(model_dir, **opts):
    from engine.backends.qwen35 import Qwen35Backend
    return Qwen35Backend(model_dir, family="pplx", name="pplx-decider-v1.1-27b", **opts)


def _d3(model_dir, **opts):
    from engine.backends.qwen35 import Qwen35Backend
    return Qwen35Backend(model_dir, family="d3", name="d3", **opts)


def _quyet(model_dir, **opts):
    from engine.backends.vllm_readout import VllmReadoutBackend
    return VllmReadoutBackend(model_dir, family="quyet", name="quyet-1.0-large", **opts)


def _rune(model_dir, **opts):
    from engine.backends.vllm_readout import VllmReadoutBackend
    return VllmReadoutBackend(model_dir, family="rune", name="rune-26b-a4b", **opts)


def _blink(model_dir, **opts):
    from engine.backends.vllm_readout import VllmReadoutBackend
    return VllmReadoutBackend(model_dir, family="blink", name="blink-v0.3-26b-a4b", **opts)


def _gguf(name):
    def make(model_dir, **opts):
        from engine.backends import gguf_cpu  # module comes from the cpu slice
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
        notes="stock vLLM/SGLang/TensorRT-LLM cannot serve it (noncausal attention)"),
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
        id="quyet-1.0-large", aliases=("quyet",),
        hf_repo="chinhnc/Quyet-1.0-Large", revision="9a0f051112ba4301f6e4c8b87fc088ce00b11aae",
        licence="Apache-2.0", licence_notice='Apache-2.0; keep NOTICE including its first line "Quyet by Chinh Nguyen" on redistribution. Base google/gemma-4-31B-it.',
        params="31.3B", arch="Gemma-4 dense: 60 layers = 50 sliding-window (1024) + 10 full attention (causal), hidden 5376, merged LoRA r16",
        readout="prompt v2 (single user turn, compact JSON state), option letters A..J (native max 10 options), per-type temperatures from quyet_config.json (choice 1.3007 / score 1.3159 / noul 1.4957)",
        backend_kind="vllm", status="vllm-backed", factory=_quyet, weights_gb=62.5,
        hardware="80+ GB card (bf16 62.5 GB); measured RTX PRO 6000 96 GB",
        notes="Weights, config, tokenizer and quyet_config.json are byte-identical between the measured revision d997d58951e3 and the pinned one (only README/MANIFEST differ)."),
    ModelSpec(
        id="rune-26b-a4b", aliases=("rune", "s1-pro"),
        hf_repo="surogate/rune-26b-a4b-GGUF", revision="c6b360d47895bb77bdf3805a13ee5a5557ab1921",
        licence="Apache-2.0", licence_notice="Apache-2.0 (base google/gemma-4-26B-A4B-it). HF-gated: each operator accepts the gate with their own account; do not mirror the weights.",
        params="~26B (4B active)", arch="Gemma-4 MoE (from the base config: 30 layers, 25 sliding + 5 full attention, 128 experts top-8); the repo name says GGUF but the files are bf16 safetensors",
        readout="Surogate decisions v1 (system line + SHARED STATE JSON user turn), letters A..Z then single-token two-letter codes, raw soft-capped logits / T=2.0",
        backend_kind="vllm", status="vllm-backed", factory=_rune, gated=True, weights_gb=52,
        hardware="80+ GB card (60 GB minimum); measured RTX PRO 6000 97.9 GB",
        notes="Rune's own config.json is not readable without the gate; architecture is taken from the base model's config."),
    ModelSpec(
        id="blink-v0.3-26b-a4b", aliases=("blink", "s1-fast-2"),
        hf_repo="PixilabAI/Blink-v0.3-26B-A4B-NVFP4", revision="08caad20c1574fb76341ca809ff38fdb081f7d4a",
        licence="Apache-2.0", licence_notice="Apache-2.0 per the model card tag (base google/gemma-4-26B-A4B-it, Apache-2.0); the repo ships no LICENSE file. Weights are not mirrored here.",
        params="~26B (4B active)", arch="Gemma-4 MoE, NVFP4 (ModelOpt) checkpoint, 17.5 GB; needs a Blackwell GPU (NVFP4)",
        readout="Surogate decisions v1 (same prompt as Rune), letters A..Z (two-letter codes above 26 options), softmax over processed logprobs / T=1.3",
        backend_kind="vllm", status="vllm-backed", factory=_blink, weights_gb=17.5,
        hardware="Blackwell GPU (NVFP4); 31 GiB peak measured at utilization 0.30, 32k context; measured RTX PRO 6000",
        notes="serving recipe = vLLM with the bi1 preset (batch-invariant, c1 zero flips); the author's own readme serves it with plain vLLM."),
    ModelSpec(
        id="gutsy-0.8b", aliases=("gutsy", "s1-cpu"),
        hf_repo="kouhxp/gutsy", revision="f9995fe26a337f1a4ea3f1a5ed38c9799e5a8f63",
        licence="Apache-2.0", licence_notice="Apache-2.0; base Qwen3.5-0.8B.",
        params="0.8B", arch="Qwen3.5-0.8B GGUF Q8_0 (llama.cpp)",
        readout="author prompt (State/Question/Options), letters A..P, one pass, per-type temperatures (choice 1.0628 / score 1.0 / yes-no 0.7652)",
        backend_kind="gguf-cpu", status="cpu", factory=_gguf("gutsy-0.8b"), weights_gb=0.8,
        hardware="CPU only (AVX2 or NEON), >= 4 cores, 4 GB RAM; ~1 s/decision on 4 cores",
        notes="backend module engine/backends/gguf_cpu.py comes from the cpu slice"),
    ModelSpec(
        id="plumb-4b-q8", aliases=("s1-lite", "plumb-q8"),
        hf_repo="crh225/plumb-4b-GGUF", revision="71c1c57e523219124f7be9c486d475d19891185f",
        licence="Apache-2.0", licence_notice="see model card",
        params="4.2B", arch="Plumb-4B, GGUF Q8_0 (llama.cpp)",
        readout="same jevk5 prompt and T=2.07 as plumb-4b; Q8-vs-bf16 mean probability distance 0.115",
        backend_kind="gguf-cpu", status="cpu", factory=_gguf("plumb-4b-q8"), weights_gb=4.5,
        hardware="CPU works but is slow; manifest asks one NVIDIA GPU >= 8 GB",
        notes="backend module engine/backends/gguf_cpu.py comes from the cpu slice"),
    ModelSpec(
        id="s1-llm-auto-router", aliases=("llm-auto-router",),
        hf_repo="decisionmodels/s1-llm-auto-router", revision="572c5b6cdb53b77fbb26ed33568ebbc5b71c69c0",
        licence="MIT", licence_notice="MIT; base jhu-clsp/mmBERT-small.",
        params="~0.1B (ONNX, int8 embeddings, 183 MB)", arch="mmBERT-small student encoder (11 layers) with 7 fixed classification heads",
        readout="NOT a letter-logit readout: fixed heads (category 9-way, difficulty 5, stakes 4, 4 binary flags), per-head temperatures; no prompt, fixed questions only",
        backend_kind="none", status="documented-only", factory=None, weights_gb=0.2,
        hardware="CPU (onnxruntime), p50 13.9 ms on 4 pinned cores",
        notes="Does not fit the arbitrary-question decision contract; a separate fixed-head classifier. Update this entry if the cpu slice ships engine/backends/onnx_router.py."),
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
