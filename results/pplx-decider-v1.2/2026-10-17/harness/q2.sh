#!/bin/bash
cd /workspace; T=/workspace/tools
until grep -q Q1_DONE q1.log; do sleep 20; done
bash $T/official12.sh a12L-fast-joint /workspace/m12 --path fast --split joint --recompile-limit 256
bash $T/official12.sh a12L-fast-single /workspace/m12 --path fast --split single --recompile-limit 256
bash $T/official12.sh a12L-fp8-joint /workspace/m12 --path fastfp8 --split joint --recompile-limit 256
echo Q2_DONE
