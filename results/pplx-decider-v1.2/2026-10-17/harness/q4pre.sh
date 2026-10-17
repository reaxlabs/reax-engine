#!/bin/bash
cd /workspace; until [ -f /workspace/jobs/q3.rc ]; do sleep 20; done
export HF_HUB_OFFLINE=1
bash tools/smoke_r12.sh > smoke.out 2>&1
cd /workspace/r12
python tools/exact_v12.py author /workspace/m12 /workspace/owo/eval/di5k-rows.jsonl 40 /workspace/exact-author.json > /workspace/exact.out 2>&1
python tools/exact_v12.py engine /workspace/m12 /workspace/owo/eval/di5k-rows.jsonl 40 /workspace/exact-author.json --linear bf16 >> /workspace/exact.out 2>&1
python tools/exact_v12.py engine /workspace/m12 /workspace/owo/eval/di5k-rows.jsonl 40 /workspace/exact-author.json --linear fp8 >> /workspace/exact.out 2>&1
echo Q4PRE_DONE >> /workspace/exact.out
