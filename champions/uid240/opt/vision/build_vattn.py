# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 pjs team (vision engine)
"""Build vattn.cu (exact Gemma4 vision-tower attention forward = PyTorch mem-efficient SDPA arithmetic, see vattn.cu)
into $VATTN_DIR (default ./gemm) at image build time (no GPU needed). No --use_fast_math (the arithmetic is spelled
out with __fmul_rn / __fadd_rn / exp2f / __frcp_rn); -frandom-seed for a reproducible binary, then
strip --strip-unneeded (as the Blink extensions); the source is copied next to the binary (gemm/vattn.cu) and the build directory is removed."""
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
from torch.utils.cpp_extension import load  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("VATTN_DIR", os.path.join(HERE, "gemm"))
build = os.path.join(OUT, "vattn_build")
os.makedirs(build, exist_ok=True)
t0 = time.time()
load(name="vattn", sources=[os.path.join(HERE, "vattn.cu")],
     extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_120a,code=sm_120a", "--expt-relaxed-constexpr",
                        "-DNDEBUG", "-frandom-seed=vattn"] + (["-Xptxas=-v"] if os.environ.get("VATTN_VERBOSE") else []),
     build_directory=build, verbose=bool(os.environ.get("VATTN_VERBOSE")))
so = os.path.join(OUT, "vattn.so")
os.replace(os.path.join(build, "vattn.so"), so)
subprocess.run(["strip", "--strip-unneeded", so], check=True)
shutil.copyfile(os.path.join(HERE, "vattn.cu"), os.path.join(OUT, "vattn.cu"))   # same-stem source next to the .so
shutil.rmtree(build, ignore_errors=True)       # no build intermediates (.o, build.ninja) in the image
print(f"vattn built in {time.time() - t0:.0f} s into {OUT}", file=sys.stderr)
