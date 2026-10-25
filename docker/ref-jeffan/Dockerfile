# Jeffan production FP8 + FP8 KV reference at utilization 0.44; no weights included.
ARG BASE=vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
FROM ${BASE}
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="reax-engine-ref-jeffan" \
      org.opencontainers.image.description="REAX engine track v2 reference engine for Jeffan-31b-1710 FP8 (vLLM 0.30.0, no weights inside)" \
      org.opencontainers.image.source="https://github.com/reaxlabs/reax-engine" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.licenses="LicenseRef-REAX-Engine-1.1 AND Apache-2.0" \
      ai.reax.engine.models="jeffan" \
      ai.reax.engine.commit="${GIT_SHA}"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONPATH=/opt/reax PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENAI_API_KEY= \
    REAX_PROFILE=reference REAX_MODEL=jeffan-31b-1710 REAX_MODEL_DIR=/model PLUMB_MODEL_DIR=/model REAX_VLLM_PORT=8000 \
    HOME=/tmp VLLM_CACHE_ROOT=/tmp/vllm-cache TRITON_CACHE_DIR=/tmp/triton TORCHINDUCTOR_CACHE_DIR=/tmp/inductor XDG_CACHE_HOME=/tmp/.cache NUMBA_CACHE_DIR=/tmp/numba \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN mkdir -p /opt/reax /licenses
COPY engine /opt/reax/engine
COPY docker/vllm-readout/entrypoint.sh /opt/reax/entrypoint.sh
COPY LICENSE NOTICE /licenses/
COPY LICENSES /licenses/LICENSES/
COPY upstream/jeffan/LICENSE upstream/jeffan/NOTICE /licenses/jeffan/
COPY upstream/d3/LICENSE upstream/d3/NOTICE /licenses/d3/
RUN chmod +x /opt/reax/entrypoint.sh && python3 -m compileall -q /opt/reax/engine
HEALTHCHECK --interval=30s --timeout=5s --start-period=1500s --retries=3 CMD \
  if [ -n "$RUNTIME_UDS" ]; then curl -sf --unix-socket "$RUNTIME_UDS" http://localhost/health; else curl -sf "http://127.0.0.1:${REAX_PORT:-8741}/health"; fi || exit 1
ENTRYPOINT ["/opt/reax/entrypoint.sh"]
