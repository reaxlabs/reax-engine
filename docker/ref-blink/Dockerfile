# Engine track v2 reference image for the Blink slot (s1-fast / s1-vision): vLLM 0.30.0 on container-internal loopback + the REAX
# server in front, started with the s1-fast production recipe at gpu_memory_utilization 0.45 (preset ref-prod; two engines
# must fit on one card). Serves the REAX engine contract: POST /internal/v1/decide + GET /health on the unix socket
# $RUNTIME_UDS, token from $S1M_INTERNAL_TOKEN_FILE, weights read-only at /model. Only vLLM listens on TCP, on 127.0.0.1
# inside the container's own network namespace (the contract runs with --network none); the REAX server does not listen on TCP.
# Same pinned vLLM base as the production node-letters image and docker/vllm-readout. No weights inside.
# Build from the repository root:  docker build -f docker/ref-blink/Dockerfile --build-arg GIT_SHA=$(git rev-parse HEAD) -t reax-engine-ref-blink .
# Run: --network none --read-only --cap-drop ALL --tmpfs /tmp:exec --gpus device=N --user 10002:10002 -v <weights>:/model:ro
#   -v <socket dir>:/run/reax -e REAX_MODEL=blink-v0.3-26b-a4b -e RUNTIME_UDS=/run/reax/engine.sock -e S1M_INTERNAL_TOKEN_FILE=/run/reax/token
ARG BASE=vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
FROM ${BASE}
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="reax-engine-ref-blink" \
      org.opencontainers.image.description="REAX engine track v2 reference engine for Blink v0.3 26B-A4B NVFP4 (vLLM 0.30.0, no weights inside)" \
      org.opencontainers.image.source="https://github.com/reaxlabs/reax-engine" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.licenses="LicenseRef-REAX-Engine-1.1 AND Apache-2.0" \
      ai.reax.engine.models="blink" \
      ai.reax.engine.commit="${GIT_SHA}"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONPATH=/opt/reax PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENAI_API_KEY= \
    REAX_PROFILE=reference REAX_MODEL=blink-v0.3-26b-a4b REAX_MODEL_DIR=/model PLUMB_MODEL_DIR=/model REAX_VLLM_PORT=8000 \
    HOME=/tmp VLLM_CACHE_ROOT=/tmp/vllm-cache TRITON_CACHE_DIR=/tmp/triton TORCHINDUCTOR_CACHE_DIR=/tmp/inductor XDG_CACHE_HOME=/tmp/.cache \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN mkdir -p /opt/reax /licenses
COPY engine /opt/reax/engine
COPY docker/vllm-readout/entrypoint.sh /opt/reax/entrypoint.sh
COPY LICENSE NOTICE /licenses/
COPY LICENSES /licenses/LICENSES/
COPY upstream/d3/LICENSE upstream/d3/NOTICE /licenses/d3/
RUN chmod +x /opt/reax/entrypoint.sh && python3 -m compileall -q /opt/reax/engine
HEALTHCHECK --interval=30s --timeout=5s --start-period=1500s --retries=3 CMD \
  if [ -n "$RUNTIME_UDS" ]; then curl -sf --unix-socket "$RUNTIME_UDS" http://localhost/health; else curl -sf "http://127.0.0.1:${REAX_PORT:-8741}/health"; fi || exit 1
ENTRYPOINT ["/opt/reax/entrypoint.sh"]
