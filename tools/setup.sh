#!/bin/bash
# Install the validated software stack and fetch the model weights.
# Usage: MODEL_DIR=/path/to/model bash tools/setup.sh   (needs Python 3.12, NVIDIA driver with CUDA 13 support)
set -e
MODEL_DIR=${MODEL_DIR:-./model}
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
pip install transformers==5.17.0 accelerate==1.15.0 safetensors==0.8.0 flash-linear-attention==0.5.2 \
    huggingface-hub==1.31.0 hf_xet pillow numpy
python -c "from huggingface_hub import snapshot_download as s; s('perplexity-ai/pplx-decider-v1.1-27b', revision='6195aa55a72ae21e503c8a60198f7b390d42adb5', local_dir='$MODEL_DIR')"
