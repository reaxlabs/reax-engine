# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 team ruj
"""Build the Jeffan GEMM lever's libraries from the sources next to this file with the image's own toolchain
(torch.utils.cpp_extension + ninja, nvcc, NVIDIA CUTLASS 4.5.0 headers bundled with flashinfer); no GPU needed.
  jef_fp8gemm.so     torch ops jef::fp8_mm, jef::fp8_mm_cfg, jef::fp8_set_table, jef::fp8_cfg_names (jef_fp8gemm.cu +
                     jef_fp8_cfg*.cu, kernels in jef_fp8_common.cuh)
  jef_fp8override.so routes vLLM's _C::cutlass_scaled_mm to jef::fp8_mm (jef_fp8override.cu)
Usage: python3 build_fp8gemm.py [out_dir] [which: all|gemm|override]
E62: + -frandom-seed=<library name> (deterministic symbol names); then strip --strip-unneeded each .so."""
import glob
import os
import shutil
import sys
import time

from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "build")
WHICH = sys.argv[2] if len(sys.argv) > 2 else "all"
CUT = os.environ.get("CUTLASS_DIR", "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass")
FLAGS = ["-O3", "-std=c++17", "-gencode=arch=compute_120a,code=sm_120a", "--expt-relaxed-constexpr", "-DNDEBUG"]
LIBS = {
    "jef_fp8gemm": [os.path.join(HERE, "jef_fp8gemm.cu")] + sorted(glob.glob(os.path.join(HERE, "jef_fp8_cfg*.cu"))),
    "jef_fp8override": [os.path.join(HERE, "jef_fp8override.cu")],
}
for name, srcs in LIBS.items():
    if WHICH not in ("all", {"jef_fp8gemm": "gemm", "jef_fp8override": "override"}[name]):
        continue
    bdir = os.path.join(OUT, name)
    os.makedirs(bdir, exist_ok=True)
    t0 = time.time()
    load(name=name, sources=srcs, extra_include_paths=[f"{CUT}/include", f"{CUT}/tools/util/include"],
         extra_cuda_cflags=FLAGS + [f"-frandom-seed={name}"], build_directory=bdir, verbose=False, is_python_module=False)
    shutil.copyfile(os.path.join(bdir, name + ".so"), os.path.join(OUT, name + ".so"))
    print(f"{name} built in {time.time() - t0:.0f} s into {OUT}", file=sys.stderr)
