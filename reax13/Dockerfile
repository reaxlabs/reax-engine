FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git libnuma1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir 'torch==2.14.0' 'torchvision==0.29.0' 'transformers==5.17.0' 'safetensors==0.8.0' 'accelerate==1.15.0' 'huggingface-hub==1.33.0' 'peft==0.21.0' 'pillow==12.3.0' 'einops==0.8.2' 'flash-linear-attention==0.5.2' 'numpy==1.26.4' 'bitsandbytes==0.49.2' 'opencv-python-headless==4.11.0.86' 'cuda-toolkit[nvcc]==13.0.3'
# No author code at build time. Save image ID + pip freeze before any forward pass.
# CUDA_HOME shim (lead patch prep11): pip CUDA wheels live in nvidia/cu13 with lib/ and versioned .so only; SGLang/flashinfer JIT link with
# -L$CUDA_HOME/lib64 -lcudart (and lib64/stubs for -lcuda). Build a standard /usr/local/cuda layout of symlinks; no new binaries.
RUN set -eux; C=/usr/local/lib/python3.12/site-packages/nvidia/cu13; test -x $C/bin/nvcc; mkdir -p /usr/local/cuda/lib64/stubs; \
    for e in $C/*; do [ "$(basename $e)" = lib ] || ln -s $e /usr/local/cuda/$(basename $e); done; \
    for f in $C/lib/*.so*; do b=$(basename $f); ln -sf $f /usr/local/cuda/lib64/$b; ln -sf $f /usr/local/cuda/lib64/${b%%.so*}.so; done; \
    ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/local/cuda/lib64/stubs/libcuda.so; ln -s lib64 /usr/local/cuda/lib; \
    ls -la /usr/local/cuda /usr/local/cuda/lib64 | head -60
RUN pip install --no-cache-dir 'requests==2.32.5'

RUN pip install --no-cache-dir 'kernels==0.16.0'
ENV KERNELS_CACHE=/opt/kernels HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 LOCAL_KERNELS=kernels-community/finegrained-fp8=/opt/fp8-kernel
COPY third_party/finegrained-fp8 /opt/fp8-kernel
RUN python -c 'from kernels import get_kernel; k=get_kernel("kernels-community/finegrained-fp8",version=4); print(k.__file__)'
