# REAX Engine for Plumb-4B (s1-fast): competition-ready image. Serves the s1 runtime sidecar contract
# (POST /internal/v1/decide with X-S1M-Internal-Token, GET /health) on $RUNTIME_UDS (or RUNTIME_HOST:RUNTIME_PORT).
# Build from the repository root:  docker build -f docker/s1-fast/Dockerfile -t reax-engine-s1-fast .
# Run (as the engine competition does): --network none --read-only --cap-drop ALL --tmpfs /tmp:exec
#   -v <plumb-4b dir>:/model:ro -v <socket dir>:/run/reax -e RUNTIME_UDS=/run/reax/engine.sock
#   -e S1M_INTERNAL_TOKEN_FILE=/run/reax/token
# Same base image digest and the same date-bounded dependency resolution as the reference runtime.
FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null && rm -rf /var/lib/apt/lists/*
RUN python -m pip install -q --target /tmp/pip26 "pip==26.2.1" \
 && PYTHONPATH=/tmp/pip26 python -m pip install -q --uploaded-prior-to=2026-09-25T04:21:31Z \
      "transformers==5.17.0" "accelerate==1.15.0" "huggingface-hub==1.33.0" \
      "flash-linear-attention==0.5.2" "einops==0.8.2" \
      "fastapi==0.115.14" "uvicorn==0.29.0" "uvloop==0.22.1" "httptools==0.8.0" \
 && rm -rf /tmp/pip26
RUN mkdir -p /opt/reax /licenses
COPY manifests/s1-fast-v1.json /opt/reax/s1-fast-v1.json
COPY engine /opt/reax/engine
COPY LICENSE NOTICE /licenses/
COPY LICENSES /licenses/LICENSES/
COPY upstream/d3/LICENSE upstream/d3/NOTICE /licenses/d3/
# Byte-compile now: the root filesystem is read-only at run time.
RUN python -m compileall -q /opt/reax/engine
RUN useradd --uid 10002 --create-home s1m
USER s1m
# Writable caches for any uid (the container runs with only /tmp writable): torch.compile and Triton kernels land in /tmp.
ENV PLUMB_MODEL_DIR=/model REAX_MANIFEST=/opt/reax/s1-fast-v1.json RUNTIME_HOST=127.0.0.1 RUNTIME_PORT=8101 \
    HOME=/tmp TRITON_CACHE_DIR=/tmp/triton TORCHINDUCTOR_CACHE_DIR=/tmp/inductor XDG_CACHE_HOME=/tmp/.cache \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 CMD curl -sf "http://127.0.0.1:${RUNTIME_PORT}/health" || exit 1
ENTRYPOINT ["python", "/opt/reax/engine/serve.py", "--model", "plumb-4b"]
