#!/bin/bash
cd /workspace; T=/workspace/tools
until [ -f /workspace/jobs/q2.rc ]; do sleep 20; done
DI_ONLY=1 CLIENTS=32 bash $T/official12.sh a12L-fast-joint-c32 /workspace/m12 --path fast --split joint --recompile-limit 256
DI_ONLY=1 CLIENTS=32 bash $T/official12.sh a12L-fp8-joint-c32 /workspace/m12 --path fastfp8 --split joint --recompile-limit 256
echo Q3_DONE
