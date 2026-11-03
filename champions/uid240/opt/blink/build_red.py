# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 pjs team (K2)
"""Build blink_red.cu (exact one-pass replicas of Inductor's red_fused_add_bi_mm_mul_rms_norm_0/_2, see blink_red.py)
into $BLINK_GEMM_DIR (default ./gemm) at image build time (no GPU needed). No --use_fast_math (the arithmetic is
spelled out with __fmaf_rn / __fmul_rn / __fadd_rn); -frandom-seed for a reproducible binary, then strip
--strip-unneeded (as the other Blink extensions)."""
import os
import subprocess
import sys
import time

from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("BLINK_GEMM_DIR", os.path.join(HERE, "gemm"))
build = os.path.join(OUT, "blink_red")
os.makedirs(build, exist_ok=True)
t0 = time.time()
load(name="blink_red", sources=[os.path.join(HERE, "blink_red.cu")],
     extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_120a,code=sm_120a", "--expt-relaxed-constexpr",
                        "-DNDEBUG", "-frandom-seed=blink_red"],
     build_directory=build, verbose=False)
so = os.path.join(OUT, "blink_red.so")
os.replace(os.path.join(build, "blink_red.so"), so)
subprocess.run(["strip", "--strip-unneeded", so], check=True)
print(f"blink_red built in {time.time() - t0:.0f} s into {OUT}", file=sys.stderr)
