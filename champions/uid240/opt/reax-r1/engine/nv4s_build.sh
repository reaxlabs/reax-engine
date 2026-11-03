#!/bin/bash
# nv4s_build.sh (kai lever N1): libnv4s.so from nv4s.cu (tiles 1 2 3 4 8, one TU each, parallel) + nv4s_dispatch.cu.
# CUT = the CUTLASS 4.5.0 tree bundled with flashinfer 0.6.18.post1 (flashinfer/data/cutlass), as for rgh's nv4g_build.sh.
# N1_SILU = the exp sequence of the epilogue (nv4s_silu.cuh; the shipped value is the one the start-up probe accepts).
# Compile only (no GPU needed): nice -n 19.
set -e
cd "$(dirname "$0")"
NVCC=${NVCC:-/usr/local/cuda-13.4/bin/nvcc}
CUT=${CUT:?set CUT to flashinfer 0.6.18.post1 flashinfer/data/cutlass}
N1_SILU=${N1_SILU:-1}
FL="-O3 -std=c++17 -arch=sm_120a --expt-relaxed-constexpr -Xcompiler -fPIC -DNDEBUG -DN1_SILU=$N1_SILU -I$CUT/include -I$CUT/tools/util/include"
TILES="1 2 3 4 8"
date -u +%T
for t in $TILES; do
  nice -n 19 $NVCC $FL -DNV4S_TILE=$t -c nv4s.cu -o nv4s_$t.o > nv4s_build_$t.log 2>&1 &
done
nice -n 19 $NVCC $FL -c nv4s_dispatch.cu -o nv4s_dispatch.o
fail=0
for j in $(jobs -p); do wait $j || fail=1; done
for t in $TILES; do [ -s nv4s_$t.o ] || { echo "tile $t failed:"; grep -m8 -i error nv4s_build_$t.log; fail=1; }; done
[ $fail = 0 ] || exit 1
OBJS=""; for t in $TILES; do OBJS="$OBJS nv4s_$t.o"; done
nice -n 19 $NVCC -shared -arch=sm_120a -cudart static -o libnv4s.so $OBJS nv4s_dispatch.o
rm -f $OBJS nv4s_dispatch.o; for t in $TILES; do rm -f nv4s_build_$t.log; done
md5sum libnv4s.so nv4s.cu nv4s_silu.cuh nv4s_dispatch.cu
date -u +%T
