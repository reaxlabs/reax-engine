#!/bin/bash
cd /workspace; T=/workspace/tools
until [ -f /workspace/jobs/q5.rc ]; do sleep 20; done
bash $T/official12.sh f12L-fast-joint /workspace/m12f --path fast --split joint --recompile-limit 256
bash $T/official12.sh f12L-fast-single /workspace/m12f --path fast --split single --recompile-limit 256
bash $T/official12.sh f12L-fp8-joint /workspace/m12f --path fastfp8 --split joint --recompile-limit 256
echo Q6_DONE
