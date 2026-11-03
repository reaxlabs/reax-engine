#!/bin/bash
# nv4k_build.sh (kai lever K): libnv4k.so from nv4k.cu (variants 6 7 8, one TU each, parallel) + nv4k_dispatch.cu.
# CUT = the CUTLASS 4.5.0 tree bundled with flashinfer 0.6.18.post1 (flashinfer/data/cutlass), as for rgh's nv4g_build.sh.
# Compile only (no GPU needed): nice -n 19, 4 jobs.
set -e
cd "$(dirname "$0")"
NVCC=${NVCC:-/usr/local/cuda-13.4/bin/nvcc}
CUT=${CUT:?set CUT to flashinfer 0.6.18.post1 flashinfer/data/cutlass}
FL="-O3 -std=c++17 -arch=sm_120a --expt-relaxed-constexpr -Xcompiler -fPIC -DNDEBUG -I$CUT/include -I$CUT/tools/util/include"
date -u +%T
for v in 6 7 8; do
  nice -n 19 $NVCC $FL -DNV4K_VARIANT=$v -c nv4k.cu -o nv4k_$v.o > nv4k_build_$v.log 2>&1 &
done
nice -n 19 $NVCC $FL -c nv4k_dispatch.cu -o nv4k_dispatch.o
fail=0
for j in $(jobs -p); do wait $j || fail=1; done
for v in 6 7 8; do [ -s nv4k_$v.o ] || { echo "variant $v failed:"; grep -m5 error nv4k_build_$v.log; fail=1; }; done
[ $fail = 0 ] || exit 1
nice -n 19 $NVCC -shared -arch=sm_120a -cudart static -o libnv4k.so nv4k_6.o nv4k_7.o nv4k_8.o nv4k_dispatch.o
rm -f nv4k_6.o nv4k_7.o nv4k_8.o nv4k_dispatch.o nv4k_build_6.log nv4k_build_7.log nv4k_build_8.log
md5sum libnv4k.so nv4k.cu nv4k_dispatch.cu
date -u +%T
