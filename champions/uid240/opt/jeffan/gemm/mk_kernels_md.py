# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 team ruj
"""Write /opt/jeffan/KERNELS.md at image build time: every native binary of the Jeffan lever G, its sources and the
exact compile / link lines taken from the build.ninja files of the build that produced it, plus sha256 sums."""
import hashlib
import os
import re
import subprocess
import sys

GEMM = sys.argv[1]          # /opt/jeffan/gemm
BUILD = sys.argv[2]         # the build dir (with jef_fp8gemm/ and jef_fp8override/ subdirs)
OUT = sys.argv[3]           # /opt/jeffan/KERNELS.md


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ninja_lines(d):
    txt = open(os.path.join(d, "build.ninja")).read()
    get = lambda k: (re.search(rf"^{k} = (.*)$", txt, re.M) or [None, ""])[1]  # noqa: E731
    return {k: get(k) for k in ("nvcc", "cuda_cflags", "cuda_post_cflags", "ldflags", "cxx")}, \
        re.findall(r"^build (\S+): (\S+) (\S+)", txt, re.M)


nv = subprocess.run(["/usr/local/cuda/bin/nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-2:]
cxx = subprocess.run(["c++", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
import torch  # noqa: E402
lines = ["# KERNELS.md (Jeffan engine, /opt/jeffan): compiled parts added by team ruj (lever G)", "",
         "pjs's E62 image ships no compiled Jeffan parts; this layer adds two shared libraries, built at image build",
         "time from the sources below (ours, Apache-2.0) against NVIDIA CUTLASS 4.5.0 headers (BSD-3-Clause, NVIDIA",
         "Corporation) as bundled with flashinfer in the vLLM 0.30.0 base image",
         "(`/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass`). The engine loads them with",
         "`torch.ops.load_library` (jef_levers.py, JEF_GEMM); nothing is compiled at start.", "",
         f"Toolchain (from the base image): torch {torch.__version__}, {' / '.join(x.strip() for x in nv)}, {cxx}.",
         "Target sm_120a. Rebuild: `cd /opt/jeffan/gemm && python3 build_fp8gemm.py <dir> all` "
         "(torch.utils.cpp_extension.load + ninja).", "",
         "| binary | source | built by | sha256 |", "|---|---|---|---|"]
for name in ("jef_fp8gemm", "jef_fp8override"):
    so = os.path.join(GEMM, name + ".so")
    vars_, builds = ninja_lines(os.path.join(BUILD, name))
    srcs = [os.path.basename(s) for _, rule, s in builds if rule == "cuda_compile"]
    src_txt = ", ".join(f"/opt/jeffan/gemm/{x}" for x in srcs)
    if name == "jef_fp8gemm":
        src_txt += ", /opt/jeffan/gemm/jef_fp8_common.cuh"
    lines.append(f"| /opt/jeffan/gemm/{name}.so | {src_txt} | `cd /opt/jeffan/gemm && python3 build_fp8gemm.py <dir> all` "
                 f"(torch.utils.cpp_extension.load, ninja; exact lines below) | {sha(so)} |")
lines += [""]
for name in ("jef_fp8gemm", "jef_fp8override"):
    vars_, builds = ninja_lines(os.path.join(BUILD, name))
    lines += [f"## {name}.so", "", "Compile (each source, from build.ninja):", "", "```",
              f"{vars_['nvcc']} --generate-dependencies-with-compile --dependency-output <obj>.d {vars_['cuda_cflags']} "
              f"-c <source.cu> -o <obj> {vars_['cuda_post_cflags']}", "```", "", "Link:", "", "```",
              f"{vars_['cxx']} <objs> {vars_['ldflags']} -o {name}.so", "```", ""]
lines += ["## sources (sha256)", ""] + [f"- {f}: {sha(os.path.join(GEMM, f))}" for f in sorted(os.listdir(GEMM))
                                         if f.endswith((".cu", ".cuh", ".py"))]
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"wrote {OUT}")
