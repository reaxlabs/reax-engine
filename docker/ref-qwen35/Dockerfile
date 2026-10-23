# Engine track v2 reference image for the Qwen3.5-hybrid slots: d3-turbo (s1-pro), pplx (s1-pro-2), reax1 (REAX 1).
# One image for the three slots (the model is chosen by $REAX_MODEL: d3-turbo | pplx-decider-v1.1-27b | reax-1); it is
# published under the three slot names with one digest. Serves the REAX engine contract of docker/s1-fast: POST
# /internal/v1/decide with X-S1M-Internal-Token and GET /health on the unix socket $RUNTIME_UDS, token from
# $S1M_INTERNAL_TOKEN_FILE, weights read-only at /model, nothing listens on TCP when RUNTIME_UDS is set.
# REAX_PROFILE=reference starts each model with the options of its production recipe (engine/registry.py production_opts).
#
# Stack = the production node images (system1models services/node-d3turbo, node-reax1): nvidia/cuda 13.0.2 runtime
# (Ubuntu 24.04) pinned by digest, torch 2.14.0+cu130, transformers 5.17.0, flash-linear-attention 0.5.2.
# Build from the repository root:  docker build -f docker/ref-qwen35/Dockerfile --build-arg GIT_SHA=$(git rev-parse HEAD) -t reax-engine-ref-qwen35 .
# Run (as the engine competition does): --network none --read-only --cap-drop ALL --tmpfs /tmp:exec --gpus device=N --user 10002:10002
#   -v <weights>:/model:ro -v <socket dir>:/run/reax -e REAX_MODEL=d3-turbo -e RUNTIME_UDS=/run/reax/engine.sock
#   -e S1M_INTERNAL_TOKEN_FILE=/run/reax/token
ARG BASE=nvidia/cuda:13.0.2-runtime-ubuntu24.04@sha256:6a0e31b59e70890446f2c17356d6efc0d54260090a8c63d6ca7c2ad049db95d2
FROM ${BASE}
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="reax-engine-ref-qwen35" \
      org.opencontainers.image.description="REAX engine track v2 reference engine for d3-turbo, pplx and REAX 1 (no weights inside)" \
      org.opencontainers.image.source="https://github.com/reaxlabs/reax-engine" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.licenses="LicenseRef-REAX-Engine-1.1 AND Apache-2.0" \
      ai.reax.engine.models="d3,pplx,reax1" \
      ai.reax.engine.commit="${GIT_SHA}"
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# gcc + Python headers: Triton / torch.compile build their launcher stubs at run time (into /tmp).
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends python3 python3-venv python3-dev gcc g++ curl ca-certificates >/dev/null \
 && rm -rf /var/lib/apt/lists/*
# torch + torchvision from the cu130 index as the node images do; everything else from PyPI as resolved on 21 Oct 2026
# (the day the stack was validated on RTX PRO 6000: pip 26.2.1 --uploaded-prior-to bounds the transitive pins).
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install -q "pip==26.2.1" \
 && PIP_CONFIG_FILE=/dev/null /opt/venv/bin/pip install -q torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130 \
 && /opt/venv/bin/pip install -q --uploaded-prior-to=2026-10-21T00:00:00Z \
      "transformers==5.17.0" "accelerate==1.15.0" "safetensors==0.8.0" "flash-linear-attention==0.5.2" "kernels==0.16.1" \
      "huggingface-hub==1.31.0" "einops==0.8.2" "hf_xet" "pillow" "numpy" \
      "uvicorn==0.29.0" "uvloop==0.22.1" "httptools==0.8.0" \
 && /opt/venv/bin/pip freeze > /opt/pip-freeze.txt \
 && /opt/venv/bin/python -c "import torch, fla, transformers; print('versions', torch.__version__, fla.__version__, transformers.__version__)"
RUN mkdir -p /opt/reax /licenses
COPY engine /opt/reax/engine
COPY LICENSE NOTICE /licenses/
COPY LICENSES /licenses/LICENSES/
COPY upstream/d3/LICENSE upstream/d3/NOTICE /licenses/d3/
# Byte-compile now: the root filesystem is read-only at run time.
RUN /opt/venv/bin/python -m compileall -q /opt/reax/engine
RUN useradd --uid 10002 --create-home --home-dir /home/s1m s1m
USER s1m
WORKDIR /opt/reax
# Writable caches for any uid (the container runs with only /tmp writable): torch.compile and Triton kernels land in /tmp.
# CIE_LAZY_LOAD=1: stream+quantise weights per tensor (the production recipes), so the load peak stays near the steady size.
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/opt/reax REAX_PROFILE=reference CIE_LAZY_LOAD=1 REAX_MAX_WAIT_MS=4 PLUMB_MODEL_DIR=/model \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 OPENAI_API_KEY= \
    HOME=/tmp TRITON_CACHE_DIR=/tmp/triton TORCHINDUCTOR_CACHE_DIR=/tmp/inductor XDG_CACHE_HOME=/tmp/.cache \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s --retries=3 CMD \
  if [ -n "$RUNTIME_UDS" ]; then curl -sf --unix-socket "$RUNTIME_UDS" http://localhost/health; else curl -sf "http://127.0.0.1:${RUNTIME_PORT:-8101}/health"; fi || exit 1
ENTRYPOINT ["python", "-m", "engine.serve"]
