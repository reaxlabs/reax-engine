#!/bin/bash
cd /workspace; T=/workspace/tools
until [ -f /workspace/owo/runs/r12-fp8-det/done ]; do sleep 20; done
DI_ONLY=1 DI_LIMIT=0 env MODEL=/workspace/m12 CODE=/workspace/r12 bash $T/official.sh r12-fp8-joint-di5k --linear fp8 --compile 1 --gva 1 --warm-steps 120
DI_ONLY=1 DI_LIMIT=0 bash $T/official12.sh a12L-fast-joint-di5k /workspace/m12 --path fast --split joint --recompile-limit 256
echo Q8_DONE
