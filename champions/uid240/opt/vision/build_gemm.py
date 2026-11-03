# SPDX-License-Identifier: Apache-2.0
"""Build the blink_gemm extension (blink_gemm.cu: CUTLASS sm80-class bf16 tensor-op GEMM, see blink_bi.py) into
/opt/blink/gemm at image build time (no GPU needed). Run once from the Dockerfile; the engine loads the .so."""
import os
import sys
import time

from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("BLINK_GEMM_DIR", os.path.join(HERE, "gemm"))
CUT = os.environ.get("CUTLASS_DIR", "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
load(name="blink_gemm", sources=[os.path.join(HERE, "blink_gemm.cu")],
     extra_include_paths=[f"{CUT}/include", f"{CUT}/tools/util/include"],
     extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_120a,code=sm_120a", "--expt-relaxed-constexpr",
                        "-DNDEBUG"],
     build_directory=OUT, verbose=False)
print(f"blink_gemm built in {time.time() - t0:.0f} s into {OUT}", file=sys.stderr)
