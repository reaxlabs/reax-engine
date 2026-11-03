# SPDX-License-Identifier: Apache-2.0
"""Build our CUDA extensions into /opt/blink/gemm at image build time (no GPU needed); the engine loads the .so files.
  blink_gemm.cu     CUTLASS sm80-class bf16 tensor-op GEMM for the batch-invariant dense layers (see blink_bi.py)
  blink_moegemm.cu  CUTLASS SM120 NVFP4 grouped GEMM for the MoE experts (see blink_moe.py), if present"""
import os
import sys
import time

from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("BLINK_GEMM_DIR", os.path.join(HERE, "gemm"))
CUT = os.environ.get("CUTLASS_DIR", "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass")
FLAGS = ["-O3", "-std=c++17", "-gencode=arch=compute_120a,code=sm_120a", "--expt-relaxed-constexpr", "-DNDEBUG"]
for name, extra in (("blink_gemm", []), ("blink_moegemm", ["-DCUTLASS_ENABLE_GDC_FOR_SM100=0"])):
    src = os.path.join(HERE, name + ".cu")
    if not os.path.exists(src):
        continue
    out = os.path.join(OUT, name)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    load(name=name, sources=[src], extra_include_paths=[f"{CUT}/include", f"{CUT}/tools/util/include"],
         extra_cuda_cflags=FLAGS + extra, build_directory=out, verbose=False)
    os.replace(os.path.join(out, name + ".so"), os.path.join(OUT, name + ".so"))
    print(f"{name} built in {time.time() - t0:.0f} s into {OUT}", file=sys.stderr)
