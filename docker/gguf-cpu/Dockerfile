# REAX Engine, GGUF CPU image: Gutsy 0.8B (default) or Plumb-4B Q8_0 on llama.cpp. Linux amd64 and arm64 from one file.
# Build from the repository root:
#   docker build -f docker/gguf-cpu/Dockerfile -t reax-engine-gguf-cpu .
# Run (weights mounted read-only; the pinned file name and sha256 are in engine/backends/gguf_cpu.py):
#   docker run --rm --network none --cpus 8 -v /path/to/weights:/model:ro -e RUNTIME_ALLOW_NO_AUTH=1 \
#       reax-engine-gguf-cpu --model gutsy-0.8b
# The entrypoint is engine/serve.py, the same server as every other REAX image (--model plumb-4b-q8 for the Q8_0 Plumb).
# Base and llama-cpp-python are pinned like the REAX subnet GGUF runtime, so the numbers are the same on every
# x86-64-v3 (AVX2) or ARMv8.2 machine (portable build, no -march=native).
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update -qq && apt-get install -y -qq build-essential cmake git >/dev/null && rm -rf /var/lib/apt/lists/*
ARG CMAKE_ARGS_AMD64="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF"
ARG CMAKE_ARGS_ARM64="-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod"
RUN set -e; case "$(uname -m)" in x86_64) A="$CMAKE_ARGS_AMD64";; aarch64) A="$CMAKE_ARGS_ARM64";; *) exit 1;; esac; \
    CMAKE_ARGS="$A" CMAKE_BUILD_PARALLEL_LEVEL=8 pip wheel --no-deps -w /wheels "llama-cpp-python==0.3.36"
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update -qq && apt-get install -y -qq curl libgomp1 >/dev/null && rm -rf /var/lib/apt/lists/*
COPY --from=build /wheels /wheels
RUN pip install -q /wheels/*.whl "numpy==2.3.3" "uvicorn==0.29.0" && rm -rf /wheels
RUN mkdir -p /opt/reax/engine/backends /licenses
COPY engine/__init__.py engine/serve.py engine/registry.py /opt/reax/engine/
COPY engine/backends/__init__.py engine/backends/base.py engine/backends/common.py engine/backends/gguf_cpu.py \
     engine/backends/gguf_readouts.py /opt/reax/engine/backends/
COPY LICENSE NOTICE /licenses/
COPY LICENSES /licenses/LICENSES/
COPY upstream/d3/LICENSE upstream/d3/NOTICE /licenses/d3/
RUN useradd --uid 10002 --create-home s1m
USER s1m
ENV GGUF_MODEL_DIR=/model GGUF_CONTEXTS=1 RUNTIME_HOST=127.0.0.1 RUNTIME_PORT=8103 HOME=/tmp
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 CMD curl -sf "http://127.0.0.1:${RUNTIME_PORT}/health" || exit 1
ENTRYPOINT ["python", "/opt/reax/engine/serve.py"]
CMD ["--model", "gutsy-0.8b"]
