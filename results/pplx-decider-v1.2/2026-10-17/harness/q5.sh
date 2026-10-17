#!/bin/bash
# Phase 1 paired runs, interleaved with the author fast path (same pod, same session, fresh caches each run)
cd /workspace; T=/workspace/tools
until grep -q Q4PRE_DONE /workspace/exact.out 2>/dev/null; do sleep 20; done
R="MODEL=/workspace/m12 CODE=/workspace/r12"
env MODEL=/workspace/m12 CODE=/workspace/r12 bash $T/official.sh r12-fp8-joint --linear fp8 --compile 1 --gva 1 --warm-steps 120
bash $T/official12.sh a12L-fast-joint-r2 /workspace/m12 --path fast --split joint --recompile-limit 256
env MODEL=/workspace/m12 CODE=/workspace/r12 bash $T/official.sh r12-bf16-joint --linear bf16 --compile 1 --gva 1 --warm-steps 120
env MODEL=/workspace/m12 CODE=/workspace/r12 bash $T/official.sh r12-fp8-single --linear fp8 --compile 1 --gva 1 --warm-steps 120 --v12-split single
env MODEL=/workspace/m12 CODE=/workspace/r12 bash $T/official.sh r12-nvmid-joint --linear fp8 --compile 1 --gva 1 --nvfp4 gu@4-59,down@4-59,in_qkvz@4-59,out@4-59 --nvfp4-scales engine/scales/nvfp4-act-amax-max512.json --warm-steps 120
bash $T/official12.sh a12L-fp8-joint-r2 /workspace/m12 --path fastfp8 --split joint --recompile-limit 256
env MODEL=/workspace/m12 CODE=/workspace/r12 bash $T/official.sh r12-fp8-joint-r2 --linear fp8 --compile 1 --gva 1 --warm-steps 120
env MODEL=/workspace/m12f CODE=/workspace/r12 bash $T/official.sh rf12-fp8-joint --linear fp8 --compile 1 --gva 1 --warm-steps 120
env MODEL=/workspace/m12f CODE=/workspace/r12 bash $T/official.sh rf12-fp8-single --linear fp8 --compile 1 --gva 1 --warm-steps 120 --v12-split single
echo Q5_DONE
