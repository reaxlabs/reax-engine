#!/bin/bash
# arm.sh LABEL MODEL MODEL_DIR PRESET [extra env assignments...]  -> /workspace/out/LABEL/{p300,di}.jsonl, load-*.json, server logs
set -uo pipefail
L=$1; M=$2; MD=$3; P=$4; shift 4
O=/workspace/out/$L; mkdir -p $O; cd /workspace/reax-engine
export S1M_INTERNAL_TOKEN=localbench OPENAI_API_KEY= VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1
for kv in "$@"; do export "$kv"; done
rm -rf /root/.cache/vllm/torch_compile_cache /root/.triton/cache 2>/dev/null
env | grep -E "REAX_|VLLM_" > $O/env.txt
python3 -m engine.serve --model $M --model-dir $MD --opt preset=$P --opt log_path=$O/vllm.log --host 127.0.0.1 --port 8101 --no-verify > $O/serve.log 2>&1 &
SP=$!
for i in $(seq 1 720); do curl -sf http://127.0.0.1:8101/health >/dev/null && break; sleep 5; done
curl -sf http://127.0.0.1:8101/health >/dev/null || { echo "not healthy"; tail -30 $O/serve.log; kill $SP; exit 3; }
echo "healthy after $((i*5)) s" | tee $O/startup.txt
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done > $O/mem.csv ) & MS=$!
U=http://127.0.0.1:8101
python3 -m bench.agreement --ref $U --contract s1 --set p300 --data-dir /workspace/eval --fetch-only --workers 8 --ref-cache $O/p300.jsonl > $O/ag-p300.log 2>&1
python3 -m bench.agreement --ref $U --contract s1 --set di --limit 1000 --data-dir /workspace/eval --fetch-only --workers 8 --ref-cache $O/di.jsonl > $O/ag-di.log 2>&1
python3 -m bench.loadgen --url $U --contract s1 --set p300 --data-dir /workspace/eval --concurrency 1,32 --seconds ${LOAD_SECONDS:-90} --out $O/load-p300.json > $O/load-p300.log 2>&1
python3 -m bench.loadgen --url $U --contract s1 --set di --limit 1000 --data-dir /workspace/eval --concurrency 1,32 --seconds ${LOAD_SECONDS:-90} --out $O/load-di.json > $O/load-di.log 2>&1
kill $MS; kill $SP; sleep 5; pkill -f "vllm serve" ; sleep 10; pkill -9 -f "vllm serve"; sleep 3
echo ARM_DONE $L
