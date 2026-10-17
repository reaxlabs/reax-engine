#!/bin/bash
# usage: official12.sh TAG MODELDIR [v12_server args...]   (author v1.2 code)  -> /workspace/owo/runs/TAG
tag=$1; model=$2; shift 2
export HF_HUB_OFFLINE=1 OWO_EVAL=/workspace/owo/eval OWO_KIT=/workspace/owo/kit OWO_P300_INPUT=/workspace/owo/eval/p300-input.jsonl OWO_P300_GOLD=/workspace/owo/eval/p300-gold.jsonl OWO_DI5K_ROWS=/workspace/owo/eval/di5k-rows.jsonl
OUT=/workspace/owo/runs/$tag; mkdir -p $OUT
export TRITON_CACHE_DIR=/workspace/cold/$tag/triton TORCHINDUCTOR_CACHE_DIR=/workspace/cold/$tag/inductor; rm -rf /workspace/cold/$tag; mkdir -p $TRITON_CACHE_DIR
nvidia-smi --query-gpu=name,clocks.max.sm,power.limit,temperature.gpu --format=csv > $OUT/gpu.txt
echo "$model $@" > $OUT/args.txt
T0=$(date +%s)
python /workspace/tools/v12_server.py --model $model --port 8741 "$@" > $OUT/server.log 2>&1 &
SP=$!
for i in $(seq 1 360); do curl -sf localhost:8741/health >/dev/null && break; sleep 5; done
echo "startup_s $(( $(date +%s) - T0 ))" > $OUT/startup.txt
cd /workspace/owo/owo
[ -n "$DI_ONLY" ] || python run_p300.py --base-url http://127.0.0.1:8741 --out $OUT/p300.jsonl --latency-out $OUT/latency.json --workers 16 > $OUT/p300.log 2>&1
python run_di.py --base-url http://127.0.0.1:8741 --out $OUT/di-native --workers ${CLIENTS:-8} --limit ${DI_LIMIT:-1000} > $OUT/di.log 2>&1
curl -s localhost:8741/health > $OUT/health_end.json
kill $SP; wait $SP 2>/dev/null
echo done $tag
