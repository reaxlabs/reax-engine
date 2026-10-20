#!/bin/bash
# bicheck.sh LABEL MODEL MODEL_DIR PRESET [env...]: empirical batch invariance: P300 answers at 1 worker vs 32 workers vs 32 again (bitwise)
set -uo pipefail
L=$1; M=$2; MD=$3; P=$4; shift 4
O=/workspace/out/$L; mkdir -p $O; cd /workspace/reax-engine
export S1M_INTERNAL_TOKEN=localbench OPENAI_API_KEY= VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1
for kv in "$@"; do export "$kv"; done
env | grep -E "REAX_|VLLM_" > $O/env.txt
python3 -m engine.serve --model $M --model-dir $MD --opt preset=$P --opt log_path=$O/vllm.log --host 127.0.0.1 --port 8101 --no-verify > $O/serve.log 2>&1 &
SP=$!
for i in $(seq 1 720); do curl -sf http://127.0.0.1:8101/health >/dev/null && break; sleep 5; done
U=http://127.0.0.1:8101
for w in 32 1 32; do python3 -m bench.agreement --ref $U --contract s1 --set p300 --data-dir /workspace/eval --fetch-only --workers $w --ref-cache $O/p300-w$w-$(date +%s).jsonl > /dev/null 2>&1; done
python3 -m bench.loadgen --url $U --contract s1 --set p300 --data-dir /workspace/eval --concurrency 1,32 --seconds ${LOAD_SECONDS:-90} --out $O/load-p300.json > $O/load-p300.log 2>&1
python3 -m bench.loadgen --url $U --contract s1 --set di --limit 1000 --data-dir /workspace/eval --concurrency 1,32 --seconds ${LOAD_SECONDS:-90} --out $O/load-di.json > $O/load-di.log 2>&1
kill $SP; sleep 5; pkill -f "vllm[ ]serve"; sleep 10; pkill -9 -f "vllm[ ]serve"; pkill -9 -f EngineCore; sleep 3
echo ARM_DONE $L
