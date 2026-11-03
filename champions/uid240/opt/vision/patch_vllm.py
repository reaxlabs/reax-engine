# SPDX-License-Identifier: Apache-2.0
"""Build-time patch of the pinned vLLM 0.30.0 Gemma4 vision encoder chunk budget (gemma4_mm.py _encoder_chunk):
with VISION_ENCODER_BUDGET / REAX_VISION_ENCODER_BUDGET = total10 the encoder chunk budget is 10 % of the card's total
memory instead of min(free / 2, total / 10), so the chunking does not depend on another engine's live allocations.
Same behaviour as the v2 vision reference image (its engine/vision_budget.py, REAX Engine; credited in NOTICE).
The build fails if the pinned source line is not found exactly once."""
from importlib.util import find_spec
from pathlib import Path

OLD = '        budget = min(free_bytes // 2, total_bytes // 10)'
NEW = ('        import os\n'
       '        budget = (total_bytes // 10 if os.environ.get("REAX_VISION_ENCODER_BUDGET") == "total10"\n'
       '                  else min(free_bytes // 2, total_bytes // 10))')

path = Path(next(iter(find_spec("vllm").submodule_search_locations))) / "model_executor/models/gemma4_mm.py"
src = path.read_text()
if src.count(OLD) != 1 or "    def _encoder_chunk(" not in src:
    raise SystemExit("pinned Gemma4 vision chunk source changed; review before patching")
path.write_text(src.replace(OLD, NEW))
print("patched", path)
