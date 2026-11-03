#!/bin/bash
# nv4t_build.sh (kai lever B): libnv4t.so from nv4t.cu (tiles 0-10, one TU each, parallel) + nv4t_dispatch.cu.
# CUT = the CUTLASS 4.5.0 tree bundled with flashinfer 0.6.18.post1 (flashinfer/data/cutlass), as for rgh's nv4g_build.sh.
# Compile only (no GPU needed): nice -n 19.
set -e
cd "$(dirname "$0")"
NVCC=${NVCC:-/usr/local/cuda-13.4/bin/nvcc}
CUT=${CUT:?set CUT to flashinfer 0.6.18.post1 flashinfer/data/cutlass}
FL="-O3 -std=c++17 -arch=sm_120a --expt-relaxed-constexpr -Xcompiler -fPIC -DNDEBUG -I$CUT/include -I$CUT/tools/util/include"
TILES="0 1 2 3 4 5 6 7 8 9 10"
date -u +%T
for t in $TILES; do
  nice -n 19 $NVCC $FL -DNV4T_TILE=$t -c nv4t.cu -o nv4t_$t.o > nv4t_build_$t.log 2>&1 &
done
nice -n 19 $NVCC $FL -c nv4t_dispatch.cu -o nv4t_dispatch.o
fail=0
for j in $(jobs -p); do wait $j || fail=1; done
for t in $TILES; do [ -s nv4t_$t.o ] || { echo "tile $t failed:"; grep -m5 error nv4t_build_$t.log; fail=1; }; done
[ $fail = 0 ] || exit 1
OBJS=""; for t in $TILES; do OBJS="$OBJS nv4t_$t.o"; done
nice -n 19 $NVCC -shared -arch=sm_120a -cudart static -o libnv4t.so $OBJS nv4t_dispatch.o
rm -f $OBJS nv4t_dispatch.o; for t in $TILES; do rm -f nv4t_build_$t.log; done
md5sum libnv4t.so nv4t.cu nv4t_dispatch.cu
date -u +%T
