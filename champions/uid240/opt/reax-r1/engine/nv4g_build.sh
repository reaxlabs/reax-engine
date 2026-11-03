#!/bin/bash
# build.sh: libnv4g.so from nv4g.cu (6 variants, one TU each, parallel) + nv4_dispatch.cu. Run on the GPU box host
# under the GPU lock (HOST-QUIET): /srv/r3/boxtools/gpurun.sh 0 0 ./build.sh
set -e
cd "$(dirname "$0")"
NVCC=/usr/local/cuda-13.4/bin/nvcc
CUT=${CUT:-/srv/r3/lanes/quant/pjs/cutlass}
FL="-O3 -std=c++17 -arch=sm_120a --expt-relaxed-constexpr -Xcompiler -fPIC -DNDEBUG -I$CUT/include -I$CUT/tools/util/include"
date -u +%T
for v in 0 1 2 3 4 5; do
  $NVCC $FL -DNV4_VARIANT=$v -c nv4g.cu -o nv4g_$v.o > build_$v.log 2>&1 &
done
$NVCC $FL -c nv4_dispatch.cu -o nv4_dispatch.o
fail=0
for j in $(jobs -p); do wait $j || fail=1; done
for v in 0 1 2 3 4 5; do [ -s nv4g_$v.o ] || { echo "variant $v failed:"; grep -m5 error build_$v.log; fail=1; }; done
[ $fail = 0 ] || exit 1
$NVCC -shared -arch=sm_120a -cudart static -o libnv4g.so nv4g_0.o nv4g_1.o nv4g_2.o nv4g_3.o nv4g_4.o nv4g_5.o nv4_dispatch.o
md5sum libnv4g.so nv4g.cu nv4_dispatch.cu
date -u +%T
