# REAX Engine image for vLLM-backed decision models (Quyet-1.0-Large, Rune 26B-A4B): vLLM on loopback + the REAX server
# in front. Same pinned vLLM base as the reax-subnet s1-pro runtime (vLLM 0.30.0). NOT built in the authoring environment:
# build it on the GPU host.
#   docker build -f docker/vllm-readout/Dockerfile -t reax-vllm-readout .      (from the repository root)
#   docker run --rm --gpus all -v /path/to/weights:/model:ro -e REAX_MODEL=quyet-1.0-large -e REAX_PRESET=fp8-kv -e S1M_INTERNAL_TOKEN=change-me \
#       -p 127.0.0.1:8741:8741 reax-vllm-readout
# Weights are mounted read-only at /model (Rune: HF-gated, download with your own account). No API keys are needed or passed.
ARG BASE=vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
FROM ${BASE}
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONPATH=/opt/reax PYTHONDONTWRITEBYTECODE=1 OPENAI_API_KEY= \
    REAX_MODEL_DIR=/model REAX_HOST=0.0.0.0 REAX_PORT=8741 REAX_VLLM_PORT=8000
RUN mkdir -p /opt/reax /licenses
COPY engine /opt/reax/engine
COPY docker/vllm-readout/entrypoint.sh /opt/reax/entrypoint.sh
COPY LICENSE NOTICE /licenses/
RUN chmod +x /opt/reax/entrypoint.sh
HEALTHCHECK --interval=30s --timeout=5s --start-period=1200s --retries=3 CMD curl -sf "http://127.0.0.1:${REAX_PORT}/health" || exit 1
ENTRYPOINT ["/opt/reax/entrypoint.sh"]
