#!/bin/bash
# usage: tools/score.sh CAND [REF]  -> fetch run dirs into runs/, paired compare -> runs/CAND/compare-vs-REF.json
J=/home/flori/jobs/pplx-v12-engine-20261017; B=/home/flori/jobs/custom-inference-engine-20261010/state/owo-bundle
R=${RUNS:-$J/runs}; c=$1; r=${2:-a12L-fast-joint}
for t in $r $c; do [ -f $R/$t/.fetched ] || { timeout 300 python3 $J/tools/pod.py get /workspace/owo/runs/$t/ $R/$t/ >/dev/null && touch $R/$t/.fetched; }; done
cd $B/owo && OWO_EVAL=$B/eval OWO_KIT=$B/kit OWO_HARNESS=/home/flori/jobs/unsloth-decision-experiments-20261008/public-harness \
  OWO_P300_INPUT=$B/eval/p300-input.jsonl OWO_P300_GOLD=$B/eval/p300-gold.jsonl OWO_DI5K_ROWS=$B/eval/di5k-rows.jsonl \
  python3 compare.py $R/$r $R/$c --out $R/$c/compare-vs-$r.json 2>&1 | tail -40
