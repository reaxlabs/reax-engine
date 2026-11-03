# SPDX-License-Identifier: Apache-2.0
"""Build blink_rgemm.cu (CUTLASS sm80 bf16 x bf16 -> fp32 router GEMM, see blink_rtr.py) into /opt/blink/gemm at image
build time (no GPU needed); a separate step so the other extensions' image layer stays cached."""
import os
import sys
import time

from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("BLINK_GEMM_DIR", os.path.join(HERE, "gemm"))
CUT = os.environ.get("CUTLASS_DIR", "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass")
build = os.path.join(OUT, "blink_rgemm")
os.makedirs(build, exist_ok=True)
t0 = time.time()
load(name="blink_rgemm", sources=[os.path.join(HERE, "blink_rgemm.cu")],
     extra_include_paths=[f"{CUT}/include", f"{CUT}/tools/util/include"],
     extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_120a,code=sm_120a", "--expt-relaxed-constexpr",
                        "-DNDEBUG"],
     build_directory=build, verbose=False)
os.replace(os.path.join(build, "blink_rgemm.so"), os.path.join(OUT, "blink_rgemm.so"))
print(f"blink_rgemm built in {time.time() - t0:.0f} s into {OUT}", file=sys.stderr)
