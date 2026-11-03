#!/bin/bash
# f8p_build.sh (kai lever N2, KAI_PDL): libf8p.so from f8p.cu (which includes rgh's f8g.cu + f8common.cuh) with rgh's
# exact f8g build line (engine/KERNELS.md). Compile only (no GPU needed): nice -n 19.
set -e
cd "$(dirname "$0")"
NVCC=${NVCC:-/usr/local/cuda-13.4/bin/nvcc}
date -u +%T
nice -n 19 $NVCC -O3 -arch=sm_120a -shared -Xcompiler -fPIC -cudart static -Xptxas -v -o libf8p.so f8p.cu > f8p_build.log 2>&1 \
  || { grep -m10 error f8p_build.log; exit 1; }
md5sum libf8p.so f8p.cu f8g.cu f8common.cuh
date -u +%T
