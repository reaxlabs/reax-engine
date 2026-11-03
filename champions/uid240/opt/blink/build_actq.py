# SPDX-License-Identifier: Apache-2.0
"""Build blink_actq.cu (fused GELU(tanh)*up + NVFP4 experts quantisation, see blink_moe.py BLINK_MOE_ACTQ) into
/opt/blink/gemm at image build time (no GPU needed); a separate step so the other extensions' image layers stay cached.
No --use_fast_math: the kernel replays vLLM's activation and quantisation arithmetic, which vLLM compiles without it."""
import os
import sys
import time

from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("BLINK_GEMM_DIR", os.path.join(HERE, "gemm"))
build = os.path.join(OUT, "blink_actq")
os.makedirs(build, exist_ok=True)
t0 = time.time()
load(name="blink_actq", sources=[os.path.join(HERE, "blink_actq.cu")],
     extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_120a,code=sm_120a", "--expt-relaxed-constexpr",
                        "-DNDEBUG"],
     build_directory=build, verbose=False)
os.replace(os.path.join(build, "blink_actq.so"), os.path.join(OUT, "blink_actq.so"))
print(f"blink_actq built in {time.time() - t0:.0f} s into {OUT}", file=sys.stderr)
