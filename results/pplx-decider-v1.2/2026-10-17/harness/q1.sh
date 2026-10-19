#!/bin/bash
cd /workspace
T=/workspace/tools
bash $T/official12.sh a12-fast-joint /workspace/m12 --path fast --split joint
bash $T/official12.sh a12-fast-single /workspace/m12 --path fast --split single
bash $T/official12.sh a12-fp8-joint /workspace/m12 --path fastfp8 --split joint
CODE=/workspace/cie bash $T/official.sh reax-C --linear fp8 --compile 1 --gva 1 --shared-state 1 --shared-until 64 --shared-mixq 1 --warm-steps 120
CODE=/workspace/cie bash $T/official.sh reax-fp8 --linear fp8 --compile 1 --gva 1
bash $T/official12.sh f12-fast-joint /workspace/m12f --path fast --split joint
bash $T/official12.sh f12-fast-single /workspace/m12f --path fast --split single
echo Q1_DONE
