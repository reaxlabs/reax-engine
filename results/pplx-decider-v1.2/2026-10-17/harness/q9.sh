#!/bin/bash
cd /workspace/r12; until [ -f /workspace/owo/runs/r12-fp8-det/done ]; do sleep 20; done
export HF_HUB_OFFLINE=1 EXACT_MULTI=1
for m in m12f m12; do
python tools/exact_v12.py author /workspace/$m /workspace/owo/eval/di5k-rows.jsonl 40 /workspace/exactM-$m.json > /workspace/exactM-$m.out 2>&1
python tools/exact_v12.py engine /workspace/$m /workspace/owo/eval/di5k-rows.jsonl 40 /workspace/exactM-$m.json --linear bf16 >> /workspace/exactM-$m.out 2>&1
python tools/exact_v12.py engine /workspace/$m /workspace/owo/eval/di5k-rows.jsonl 40 /workspace/exactM-$m.json --linear fp8 >> /workspace/exactM-$m.out 2>&1
done
echo Q9_DONE > /workspace/q9.done
