#!/bin/bash
# f8c_build.sh (kai lever R5-1): libf8c.so from f8c.cu (variants 0-7 x kinds 0 1, one TU each, parallel) + f8c_dispatch.cu
# (which includes rgh's f8g.cu + f8common.cuh). CUT = the CUTLASS 4.5.0 tree bundled with flashinfer 0.6.18.post1
# (flashinfer/data/cutlass), as for rgh's nv4g_build.sh. Compile only (no GPU needed): nice -n 19.
set -e
cd "$(dirname "$0")"
NVCC=${NVCC:-/usr/local/cuda-13.4/bin/nvcc}
CUT=${CUT:?set CUT to flashinfer 0.6.18.post1 flashinfer/data/cutlass}
FL="-O3 -std=c++17 -arch=sm_120a --expt-relaxed-constexpr -Xcompiler -fPIC -DNDEBUG -I$CUT/include -I$CUT/tools/util/include"
VARIANTS="0 1 2 3 4 5 6 7"
KINDS="0 1"
date -u +%T
for v in $VARIANTS; do for k in $KINDS; do
  nice -n 19 $NVCC $FL -DF8C_VARIANT=$v -DF8C_KIND=$k -c f8c.cu -o f8c_${v}_${k}.o > f8c_build_${v}_${k}.log 2>&1 &
done; done
nice -n 19 $NVCC $FL -Xptxas -v -c f8c_dispatch.cu -o f8c_dispatch.o > f8c_build_dispatch.log 2>&1
fail=0
for j in $(jobs -p); do wait $j || fail=1; done
OBJS=""
for v in $VARIANTS; do for k in $KINDS; do
  [ -s f8c_${v}_${k}.o ] || { echo "variant $v kind $k failed:"; grep -m5 error f8c_build_${v}_${k}.log; fail=1; }
  OBJS="$OBJS f8c_${v}_${k}.o"
done; done
[ $fail = 0 ] || exit 1
nice -n 19 $NVCC -shared -arch=sm_120a -cudart static -o libf8c.so $OBJS f8c_dispatch.o
rm -f $OBJS f8c_dispatch.o f8c_build_?_?.log
md5sum libf8c.so f8c.cu f8c_dispatch.cu f8c_build.sh
date -u +%T
