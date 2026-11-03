# Engine track reference image for the s1-pro POOL model: Surogate Rune 26B-A4B v3 (REAX_MODEL=rune-26b-a4b-v3, alias s1-pro).
# vLLM 0.30.0 (same digest as the s1-pro pool runtime and the other vLLM reference images) started with the pool runtime's
# flags (preset ref-pool = runtimes/s1-pro/start.sh with RUNE_DETERMINISTIC=1 defaults: VLLM_BATCH_INVARIANT=1, no chunked
# prefill, max-num-batched-tokens = max-model-len = 32768, -O0, processed_logits, prefix caching off, 64 seqs, util 0.85),
# and the REAX server in front (the engine contract: POST /internal/v1/decide + GET /health on $RUNTIME_UDS). Text only.
# Before vLLM starts the 18 weight files are checked against the pool's release manifest (/opt/reax/s1-pro-v1.json, manifest 5b11440bf57b5434).
# No weights inside; HF-gated model: mount your own copy read-only at /model.
# Build from the repository root:  docker build -f docker/ref-rune/Dockerfile --build-arg GIT_SHA=$(git rev-parse HEAD) -t reax-engine-ref-rune .
# Run: --network none --read-only --cap-drop ALL --tmpfs /tmp:exec --gpus device=N --user 10002:10002 -v <weights>:/model:ro
#   -v <socket dir>:/run/reax -e RUNTIME_UDS=/run/reax/engine.sock -e S1M_INTERNAL_TOKEN_FILE=/run/reax/token
ARG BASE=vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
FROM ${BASE}
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="reax-engine-ref-rune" \
      org.opencontainers.image.description="REAX engine track reference engine for Surogate Rune 26B-A4B v3 / s1-pro pool (vLLM 0.30.0, deterministic pool flags, no weights inside)" \
      org.opencontainers.image.source="https://github.com/reaxlabs/reax-engine" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.licenses="LicenseRef-REAX-Engine-1.1 AND Apache-2.0" \
      ai.reax.engine.models="rune,s1-pro" \
      ai.reax.engine.commit="${GIT_SHA}"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONPATH=/opt/reax PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENAI_API_KEY= \
    REAX_PROFILE=reference REAX_MODEL=rune-26b-a4b-v3 REAX_MODEL_DIR=/model PLUMB_MODEL_DIR=/model REAX_VLLM_PORT=8000 \
    REAX_S1_MANIFEST=/opt/reax/s1-pro-v1.json \
    HOME=/tmp VLLM_CACHE_ROOT=/tmp/vllm-cache TRITON_CACHE_DIR=/tmp/triton TORCHINDUCTOR_CACHE_DIR=/tmp/inductor XDG_CACHE_HOME=/tmp/.cache NUMBA_CACHE_DIR=/tmp/numba \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN mkdir -p /opt/reax /licenses
COPY engine /opt/reax/engine
COPY docker/vllm-readout/entrypoint.sh /opt/reax/entrypoint.sh
COPY manifests/s1-pro-v1.json /opt/reax/s1-pro-v1.json
COPY upstream/rune/chat_template_nospace.jinja /opt/reax/chat_template_nospace.jinja
COPY LICENSE NOTICE /licenses/
COPY LICENSES /licenses/LICENSES/
COPY upstream/rune/LICENSE upstream/rune/NOTICE /licenses/rune/
RUN chmod +x /opt/reax/entrypoint.sh && python3 -m compileall -q /opt/reax/engine
HEALTHCHECK --interval=30s --timeout=5s --start-period=1500s --retries=3 CMD \
  if [ -n "$RUNTIME_UDS" ]; then curl -sf --unix-socket "$RUNTIME_UDS" http://localhost/health; else curl -sf "http://127.0.0.1:${REAX_PORT:-8741}/health"; fi || exit 1
ENTRYPOINT ["/opt/reax/entrypoint.sh"]
