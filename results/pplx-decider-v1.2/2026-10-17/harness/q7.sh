#!/bin/bash
cd /workspace; until [ -f /workspace/jobs/q6.rc ]; do sleep 20; done
export HF_HUB_OFFLINE=1; OUT=/workspace/owo/runs/r12-fp8-det; mkdir -p $OUT
export TRITON_CACHE_DIR=/workspace/cold/det/triton TORCHINDUCTOR_CACHE_DIR=/workspace/cold/det/inductor; mkdir -p $TRITON_CACHE_DIR
cd /workspace/r12
python engine/server.py --model /workspace/m12 --port 8741 --linear fp8 --compile 1 --gva 1 --warm-steps 120 > $OUT/server.log 2>&1 &
SP=$!
for i in $(seq 1 240); do curl -sf localhost:8741/health >/dev/null && break; sleep 5; done
python3 tools/determinism.py http://127.0.0.1:8741 /workspace/owo/eval/di5k-rows.jsonl /workspace/owo/eval/di5k-ids.json 60 3 > $OUT/determinism.json 2>&1
kill $SP; wait $SP 2>/dev/null
echo Q7_DONE > $OUT/done
