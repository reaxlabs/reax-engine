# SPDX-License-Identifier: Apache-2.0
"""Build vis_ops.cu (V3 levers E3 / E4: exact fused GEMM epilogue, norm chains, head norms + RoPE of the Gemma4 vision
tower; see vis_ops.cu) into $VIS_OPS_DIR (default ./gemm) at image build time (no GPU needed). CUTLASS headers: the
base image's (flashinfer's NVIDIA CUTLASS 4.5.0) + the dual-GEMM headers of CUTLASS examples/45_dual_gemm in
./vis_dual (BSD-3-Clause). No --use_fast_math. -frandom-seed for a reproducible binary, then strip --strip-unneeded;
the source is copied next to the binary (gemm/vis_ops.cu) and the build directory is removed. Build at /opt/vision
(the source path is embedded)."""
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
from torch.utils.cpp_extension import load  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("VIS_OPS_DIR", os.path.join(HERE, "gemm"))
CUT = os.environ.get("CUTLASS_DIR", "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass")
build = os.path.join(OUT, "vis_ops_build")
os.makedirs(build, exist_ok=True)
t0 = time.time()
load(name="vis_ops", sources=[os.path.join(HERE, "vis_ops.cu")],
     extra_include_paths=[HERE, f"{CUT}/include", f"{CUT}/tools/util/include"],
     extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_120a,code=sm_120a", "--expt-relaxed-constexpr",
                        "-DNDEBUG", "-frandom-seed=vis_ops"] + (["-Xptxas=-v"] if os.environ.get("VIS_OPS_VERBOSE") else []),
     build_directory=build, verbose=bool(os.environ.get("VIS_OPS_VERBOSE")))
so = os.path.join(OUT, "vis_ops.so")
os.replace(os.path.join(build, "vis_ops.so"), so)
subprocess.run(["strip", "--strip-unneeded", so], check=True)
shutil.copyfile(os.path.join(HERE, "vis_ops.cu"), os.path.join(OUT, "vis_ops.cu"))   # same-stem source next to the .so
shutil.rmtree(build, ignore_errors=True)       # no build intermediates (.o, build.ninja) in the image
print(f"vis_ops built in {time.time() - t0:.0f} s into {OUT}", file=sys.stderr)
sys.stderr.flush()
os._exit(0)      # skip interpreter teardown: the extension load() imported can crash at exit on a GPU-less build host
