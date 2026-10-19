# REAX native engine for the Qwen3.5-hybrid decision models (pplx-decider-v1.1-27b, d3) and the pins their authors' reference
# code needs (transformers 5.17.0, flash-linear-attention 0.5.2), so reference and REAX server run from the same image.
# Reference code of the models (autojev for pplx, d3_server.py for d3, the `quyet` package for Quyet) is foreign code:
# run it only in this container, weights mounted read-only. NOT built in the authoring environment: build on the GPU host.
#   docker build -f docker/qwen35/Dockerfile -t reax-qwen35 .      (from the repository root)
FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 OPENAI_API_KEY= PYTHONDONTWRITEBYTECODE=1
# gcc + headers: torch.compile / Triton build helpers at warm-up
RUN apt-get update -qq && apt-get install -y -qq curl ca-certificates build-essential python3-dev >/dev/null && rm -rf /var/lib/apt/lists/*
# the validated stack of tools/setup.sh (torch 2.14.0+cu130), plus what the reference servers import
RUN python -m pip install -q torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130 \
 && python -m pip install -q transformers==5.17.0 accelerate==1.15.0 safetensors==0.8.0 flash-linear-attention==0.5.2 \
      huggingface-hub==1.31.0 hf_xet pillow numpy einops opencv-python-headless \
      uvicorn==0.29.0 uvloop==0.22.1 httptools==0.8.0 \
 && python -m pip install -q --no-deps quyet==1.0.2
RUN mkdir -p /opt/reax /licenses
COPY engine /opt/reax/engine
COPY bench /opt/reax/bench
COPY upstream /opt/reax/upstream
COPY LICENSE NOTICE /licenses/
COPY LICENSES /licenses/LICENSES/
COPY upstream/d3/LICENSE upstream/d3/NOTICE /licenses/d3/
ENV PYTHONPATH=/opt/reax HOME=/tmp TRITON_CACHE_DIR=/tmp/triton XDG_CACHE_HOME=/tmp/.cache TORCHINDUCTOR_CACHE_DIR=/tmp/inductor
WORKDIR /opt/reax
# default: the REAX server (engine/serve.py; needs S1M_INTERNAL_TOKEN or RUNTIME_ALLOW_NO_AUTH=1); override the command for a reference server
ENTRYPOINT ["python", "-m", "engine.serve"]
