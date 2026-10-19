#!/bin/bash
# usage: official.sh TAG [server args...] -> /workspace/owo/runs/TAG/{p300.jsonl,latency.json,di-native/,server.log}
tag=$1; shift
export TORCH_LOGS=recompiles HF_HUB_OFFLINE=1 OWO_EVAL=/workspace/owo/eval OWO_KIT=/workspace/owo/kit OWO_P300_INPUT=/workspace/owo/eval/p300-input.jsonl OWO_P300_GOLD=/workspace/owo/eval/p300-gold.jsonl OWO_DI5K_ROWS=/workspace/owo/eval/di5k-rows.jsonl
OUT=/workspace/owo/runs/$tag; mkdir -p $OUT
# fresh compile caches for every run unless WARM=1 (no warm-cache effects from earlier runs on the same pod)
if [ -z "$WARM" ]; then export TRITON_CACHE_DIR=/workspace/cold/$tag/triton TORCHINDUCTOR_CACHE_DIR=/workspace/cold/$tag/inductor; rm -rf /workspace/cold/$tag; mkdir -p $TRITON_CACHE_DIR; fi
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-default} TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-default}" > $OUT/caches.txt
cd ${CODE:-/workspace/cie}
nvidia-smi --query-gpu=name,clocks.max.sm,power.limit,temperature.gpu --format=csv > $OUT/gpu.txt
git config --global --add safe.directory '*'; git rev-parse HEAD > $OUT/commit.txt 2>/dev/null; git diff --stat >> $OUT/commit.txt 2>/dev/null; env | grep '^PPLX_' > $OUT/env.txt
echo "$@" > $OUT/args.txt
python engine/server.py --model ${MODEL:-/workspace/model} --port 8741 "$@" > $OUT/server.log 2>&1 &
SP=$!
for i in $(seq 1 240); do curl -sf localhost:8741/health >/dev/null && break; sleep 5; done
curl -s localhost:8741/health > $OUT/health_start.json
cd /workspace/owo/owo
[ -n "$DI_ONLY" ] || python run_p300.py --base-url http://127.0.0.1:8741 --out $OUT/p300.jsonl --latency-out $OUT/latency.json --workers 16 > $OUT/p300.log 2>&1
python run_di.py --base-url http://127.0.0.1:8741 --out $OUT/di-native --workers 8 --limit ${DI_LIMIT:-1000} > $OUT/di.log 2>&1
curl -s localhost:8741/health > $OUT/health_end.json
kill $SP; wait $SP 2>/dev/null
echo done $tag
