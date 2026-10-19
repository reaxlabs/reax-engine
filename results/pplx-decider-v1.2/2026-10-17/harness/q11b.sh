#!/bin/bash
cd /workspace; T=/workspace/tools
until grep -q Q8_DONE /workspace/jobs/q8b.log 2>/dev/null; do sleep 20; done
bash $T/official12.sh a12L-static32k-joint-b /workspace/m12 --path fast --split joint --recompile-limit 256 --buckets 32768
env MODEL=/workspace/m12 CODE=/workspace/r12 bash $T/official.sh r12-fp8-joint-b --linear fp8 --compile 1 --gva 1 --warm-steps 120
bash $T/official12.sh a12L-fast-joint-b /workspace/m12 --path fast --split joint --recompile-limit 256
echo Q11_DONE
HF_HUB_DISABLE_TELEMETRY=1 python -c "from huggingface_hub import snapshot_download as s; s('perplexity-ai/pplx-decider-flash-v1.2', revision='6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d', local_dir='/workspace/m12f', max_workers=16)" > dlf.log 2>&1
env MODEL=/workspace/m12f CODE=/workspace/r12 bash $T/official.sh rf12-bf16-joint-b --linear bf16 --compile 1 --gva 1 --warm-steps 120
bash $T/official12.sh f12L-fast-joint-b /workspace/m12f --path fast --split joint --recompile-limit 256
echo Q11F_DONE
