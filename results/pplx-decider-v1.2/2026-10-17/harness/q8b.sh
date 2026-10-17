#!/bin/bash
# pod B: setup, then the DI-5k pair (same pod): REAX fp8 joint, author fast joint
bash /workspace/tools/pod_setup_b.sh > /workspace/setup.log 2>&1
cd /workspace; T=/workspace/tools
DI_ONLY=1 DI_LIMIT=0 env MODEL=/workspace/m12 CODE=/workspace/r12 bash $T/official.sh r12-fp8-joint-di5k --linear fp8 --compile 1 --gva 1 --warm-steps 120
DI_ONLY=1 DI_LIMIT=0 bash $T/official12.sh a12L-fast-joint-di5k /workspace/m12 --path fast --split joint --recompile-limit 256
echo Q8_DONE
