"""Start-up phase timestamps: one stderr line per phase, always on, `startup: <phase> <seconds>` (seconds of that phase,
one decimal). Timing lines only: nothing here changes what the engine builds or computes.

Phases (disjoint; their sum plus interpreter start and container creation is the start-up time quick.py measures):
  python_imports  serve.py module imports and argument parsing (from this module's first import)
  engine_import   torch / triton / FLA / pplx_engine imports (qwen35 backends)
  tokenizer       AutoTokenizer.from_pretrained (mammoth, d3) or the reax-1 encoder set-up
  weight_read     safetensors reads (with CIE_LAZY_LOAD=1 each tensor is read when the layer loop consumes it; a read waits
                  for the queued GPU work of the tensor before, since the copy is stream-ordered)
  fp8_convert     the layer loop minus its reads: FP8 (and in-engine NVFP4 / shadow) quantisation, concatenation
  model_build     the rest of PplxEngine.__init__ (config, kernel table, attention backend pick)
  nvfp4_derive    rgh q48m NVFP4 W4A4 projections (nv4.install; only when MAM_NV4 is set)
  nv4_minm        kai small-step FP8 fallback set-up (nv4_minm.install; only when KAI_NV4_MINM is set)
  first_forward   the first warm-up step: torch.compile tracing + Inductor/Triton compiles + first autotunes
  warmup_shapes   the rest of the fixed warm-up shape sweep
  warmup_steps    the random warm-up steps
  warmup_shared   the shared-state warm-up (d3-turbo / mammoth state-once)
  actq_check      kai lever Q start-up bit-equality check (actq.check; only when KAI_ACTQ and MAM_NV4 are set)
  n1_check        kai lever N1 start-up bit-equality check (n1.check; only when KAI_N1 and MAM_NV4 are set)
  r1_capture      rgh u4r2 reax1 single-question CUDA graphs + kai's replay check (PplxEngine.r1_capture; only R1_C1GRAPH=1)
  ready           total since this module's first import, printed when /health turns 200

The module registers itself as both `startup` (flat imports of pplx_engine and its kernels) and `engine.startup`, so every
importer shares one clock.
"""
from __future__ import annotations

import re
import sys
import time

T0 = time.perf_counter()
_NAME = re.compile(r"^[a-z0-9_]+$")

sys.modules.setdefault("startup", sys.modules[__name__])
sys.modules.setdefault("engine.startup", sys.modules[__name__])


def mark(phase: str, seconds: float) -> None:
    if not _NAME.match(phase):
        raise ValueError(f"start-up phase names are [a-z0-9_]+, not {phase!r}")
    print(f"startup: {phase} {float(seconds):.1f}", file=sys.stderr, flush=True)


def since_start() -> float:
    return time.perf_counter() - T0


def ready() -> None:
    mark("ready", since_start())


class phase:
    """with startup.phase("name"): ...  -> one line with the block's wall time (also when the block raises)"""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        mark(self.name, time.perf_counter() - self.t0)
        return False
