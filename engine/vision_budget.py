"""Build-time patch of pinned vLLM 0.30.0 Gemma4 vision chunk budgeting.

Only Gemma4ForConditionalGeneration._encoder_chunk is changed, and only the
opt-in REAX_VISION_ENCODER_BUDGET=total10 branch changes its result. Text does
not call this function. total10 agrees with upstream whenever >=20% is free;
it removes dependence on the other engine's live allocations. GPU parity and
peak-memory checks remain required. The build fails if the pinned source drifts.
"""
from pathlib import Path

ORIGINAL = '        budget = min(free_bytes // 2, total_bytes // 10)'
PINNED = '''        import os
        budget = (total_bytes // 10 if os.environ.get("REAX_VISION_ENCODER_BUDGET") == "total10"
                  else min(free_bytes // 2, total_bytes // 10))'''


def patch_source(source):
    if source.count(ORIGINAL) != 1 or '    def _encoder_chunk(' not in source:
        raise ValueError('pinned Gemma4 vision chunk source changed; review before patching')
    return source.replace(ORIGINAL, PINNED)


def main():
    from importlib.util import find_spec
    spec = find_spec('vllm')
    path = Path(next(iter(spec.submodule_search_locations))) / 'model_executor/models/gemma4_mm.py'
    path.write_text(patch_source(path.read_text()))
    print('Applied opt-in Gemma4 total10 vision budget:', path)


if __name__ == '__main__':
    main()
