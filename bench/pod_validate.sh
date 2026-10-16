#!/usr/bin/env bash
# One command per model on a fresh GPU pod (Ubuntu + NVIDIA driver with CUDA 13 + docker with the NVIDIA runtime):
#
#   HF_TOKEN=hf_... REAX_BENCH_DATA=/path/to/eval-data bash bench/pod_validate.sh <model-id>
#
# model-id: pplx-decider-v1.1-27b | d3 | quyet-1.0-large | rune-26b-a4b   (aliases work; see `python3 -m engine.registry`)
# What it does, in order (one model on the GPU at a time):
#   1. downloads the pinned weights (registry revision; HF_TOKEN required for the gated Rune repo)
#   2. builds the images (docker/qwen35 or docker/vllm-readout; the Rune reference image comes from reax-subnet)
#   3. REFERENCE server (the author's / reference code, in a container, weights read-only): agreement responses on P300
#      (+ DI-<DI_LIMIT>) are saved, then c1 and c$CONC throughput is measured, then the server is stopped
#   4. REAX Engine server for the model: the same agreement run against the saved reference responses, the same load levels
#   5. writes results/<model>/<date>.json (bench/summarize.py)
# Environment: WORK (default /workspace/reax-validate)  PRESET (vLLM-backed models; default: faithful preset)  CONC (16)
#   LOAD_SECONDS (60)  DI_LIMIT (1000; 0 = skip DI)  VERIFY (1: check MANIFEST.sha256 when the repo ships one)
#   GPU (docker --gpus value: "all" or a device index; default all)  PORT_OFF (added to the server ports, so two models can run on two GPUs at once)
#   MIN_FREE_GB (default weights*1.15+60)  REAX_OPTS (backend options key=value, space separated)  REF_IMAGE / REAX_SUBNET_DIR (Rune reference)  RO (1: --read-only)
# Stops nothing on the pod and rents nothing; the lead tears the pod down.
set -euo pipefail

MODEL_ARG=${1:?usage: bench/pod_validate.sh <model-id>}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
WORK=${WORK:-/workspace/reax-validate}
DATA=${REAX_BENCH_DATA:?set REAX_BENCH_DATA to the directory with the evaluation sets (see bench/sets.py)}
CONC=${CONC:-16}; LOAD_SECONDS=${LOAD_SECONDS:-60}; DI_LIMIT=${DI_LIMIT:-1000}; VERIFY=${VERIFY:-1}; RO=${RO:-1}
TODAY=$(date -u +%F)
log() { echo "[$(date -u +%T)] $*"; }

# ---- 0. registry facts (stdlib only)
SPEC=$(python3 -m engine.registry "$MODEL_ARG")
field() { python3 -c "import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])" "$SPEC" "$1"; }
ID=$(field id); REPO=$(field hf_repo); REV=$(field revision); GB=$(field weights_gb); GATED=$(field gated); STATUS=$(field status)
case "$STATUS" in native|vllm-backed) ;; *) echo "pod_validate: $ID has status $STATUS (GPU validation covers native and vllm-backed models)"; exit 2 ;; esac
[ -d "$DATA" ] || { echo "no data dir $DATA"; exit 2; }
command -v docker >/dev/null || { echo "docker missing"; exit 2; }
nvidia-smi -L >/dev/null || { echo "no NVIDIA GPU visible"; exit 2; }
mkdir -p "$WORK/raw/$ID" "$WORK/cache/$ID" "$ROOT/results/$ID"
mkdir -p "$WORK/raw/$ID${PRESET:+-$PRESET}${TAG:+-$TAG}"
RAW=$WORK/raw/$ID${PRESET:+-$PRESET}${TAG:+-$TAG}; MDIR=$WORK/weights/$ID
AVAIL=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc 0-9)
NEED=${MIN_FREE_GB:-$(python3 -c "print(int($GB*1.15+60))")}
[ "$AVAIL" -ge "$NEED" ] || { echo "need ~${NEED} GB free in $WORK (weights ${GB} GB + images), have ${AVAIL} GB"; exit 2; }

# per-model wiring: container image, REAX command env, reference command, contract
GPU=${GPU:-all}; PORT_OFF=${PORT_OFF:-0}
GPU_ARG="all"; [ "$GPU" != all ] && GPU_ARG="device=$GPU"
PORT_REF=$((8001+PORT_OFF)); PORT_REAX=$((8741+PORT_OFF)); TOKEN=pod-validate; REF_ENV=()
NAME_REF=reax-ref-$ID; NAME_CAND=reax-cand-$ID
case "$ID" in
  pplx-decider-v1.1-27b|d3)
    ENGINE_IMAGE=reax-qwen35; DOCKERFILE=docker/qwen35/Dockerfile; CONTRACT=systemone
    REAX_OPTS=${REAX_OPTS:-"linear=fp8 compile=1 gva=1 warm_steps=200"}
    REAX_CMD=(--model "$ID" --model-dir /model --host 127.0.0.1 --port $PORT_REAX)
    for o in $REAX_OPTS; do REAX_CMD+=(--opt "$o"); done
    REF_IMAGE=${REF_IMAGE:-reax-qwen35}; REF_ENTRY=python
    if [ "$ID" = d3 ]; then REF_CMD=(/model/d3_server.py --model /model --host 127.0.0.1 --port $PORT_REF)
    else REF_CMD=(engine/server.py --model /model --backend ref --port $PORT_REF); fi ;;
  quyet-1.0-large)
    ENGINE_IMAGE=reax-vllm-readout; DOCKERFILE=docker/vllm-readout/Dockerfile; CONTRACT=systemone
    REAX_CMD=(); REF_IMAGE=${REF_IMAGE:-reax-qwen35}; REF_ENTRY=python
    REF_CMD=(bench/ref/quyet_ref_server.py --model /model --host 127.0.0.1 --port $PORT_REF) ;;
  rune-26b-a4b)
    ENGINE_IMAGE=reax-vllm-readout; DOCKERFILE=docker/vllm-readout/Dockerfile; CONTRACT=s1
    REAX_CMD=(); REF_IMAGE=${REF_IMAGE:-reax-s1-pro-runtime}; REF_ENTRY=""   # the image's own entrypoint (vLLM + Rune adapter)
    REF_CMD=(); REF_ENV=(-e RUNE_PORT=$PORT_REF -e RUNE_SKIP_VERIFY=1) ;;
  *) echo "no pod wiring for $ID"; exit 2 ;;
esac
export S1M_INTERNAL_TOKEN=$TOKEN

# ---- 1. weights
if [ ! -f "$MDIR/.complete-$REV" ]; then
  if [ "$GATED" = True ] && [ -z "${HF_TOKEN:-}" ]; then echo "$REPO is HF-gated: accept its terms with your account and set HF_TOKEN"; exit 2; fi
  log "downloading $REPO @ $REV (${GB} GB) to $MDIR"
  [ -x "$WORK/venv/bin/python" ] || { python3 -m venv "$WORK/venv" && "$WORK/venv/bin/pip" install -q "huggingface_hub>=1.0" hf_xet; }
  "$WORK/venv/bin/python" -c "import sys; from huggingface_hub import snapshot_download as s; s(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])" "$REPO" "$REV" "$MDIR"
  if [ "$VERIFY" = 1 ] && [ -f "$MDIR/MANIFEST.sha256" ]; then log "verifying MANIFEST.sha256"; (cd "$MDIR" && sha256sum --check --strict --quiet MANIFEST.sha256); fi
  touch "$MDIR/.complete-$REV"
fi

# ---- 2. images
log "building $ENGINE_IMAGE"; docker build -q -f "$DOCKERFILE" -t "$ENGINE_IMAGE" . >/dev/null
if [ "$REF_IMAGE" = reax-qwen35 ] && [ "$ENGINE_IMAGE" != reax-qwen35 ]; then log "building reax-qwen35 (reference runtime)"; docker build -q -f docker/qwen35/Dockerfile -t reax-qwen35 . >/dev/null; fi
if [ "$ID" = rune-26b-a4b ] && ! docker image inspect "$REF_IMAGE" >/dev/null 2>&1; then
  [ -n "${REAX_SUBNET_DIR:-}" ] || { echo "Rune reference image $REF_IMAGE missing: build it from reax-subnet (docker build -f runtimes/s1-pro/Dockerfile -t reax-s1-pro-runtime <reax-subnet checkout>) or set REAX_SUBNET_DIR"; exit 2; }
  docker build -q -f "$REAX_SUBNET_DIR/runtimes/s1-pro/Dockerfile" -t "$REF_IMAGE" "$REAX_SUBNET_DIR" >/dev/null
fi

# ---- container helpers (no API keys; weights read-only; loopback only because of --network host + loopback binds)
RO_FLAGS=(); [ "$RO" = 1 ] && RO_FLAGS=(--read-only)
cleanup() { docker rm -f "$NAME_REF" "$NAME_CAND" >/dev/null 2>&1 || true; }
trap cleanup EXIT
start() {  # name image entrypoint(or "") port-for-health -- args...
  local name=$1 image=$2 entry=$3 port=$4; shift 5
  docker rm -f "$name" >/dev/null 2>&1 || true
  local ep=(); [ -n "$entry" ] && ep=(--entrypoint "$entry")
  docker run -d --name "$name" --gpus "$GPU_ARG" --ipc host --network host "${RO_FLAGS[@]}" --tmpfs /tmp:rw,exec,size=32g \
    --cap-drop ALL --security-opt no-new-privileges -e HOME=/tmp -e OPENAI_API_KEY= -e HF_HUB_OFFLINE=1 \
    -e VLLM_CACHE_ROOT=/tmp/vllm -e S1M_INTERNAL_TOKEN="$TOKEN" "${ENV_ARGS[@]}" -v "$MDIR":/model:ro "${ep[@]}" "$image" "$@" >/dev/null
  log "waiting for $name on :$port"
  for _ in $(seq 1 1800); do
    curl -sf -m 3 "http://127.0.0.1:$port/health" >/dev/null && return 0
    docker ps -q -f name="$name" | grep -q . || { docker logs --tail 40 "$name" || true; echo "$name exited"; exit 3; }
    sleep 2
  done
  echo "$name not healthy after 60 min"; docker logs --tail 40 "$name" || true; exit 3
}

GPU_IDX=${GPU/all/0}
phase() {  # label(ref|cand) port -> agreement responses/report + load levels under $RAW; the reference only saves its responses
  local label=$1 port=$2
  # peak GPU memory (whole device, MiB) sampled every 2 s while the server is loaded and answering
  ( while :; do nvidia-smi -i "$GPU_IDX" --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done > "$RAW/mem-$label.csv" 2>/dev/null ) &
  local SAMPLER=$!
  local url=http://127.0.0.1:$port
  local C=$WORK/cache/$ID/$REV; mkdir -p "$C"
  if [ "$label" = ref ]; then
    python3 -m bench.agreement --ref "$url" --contract "$CONTRACT" --set p300 --data-dir "$DATA" --fetch-only --workers 8 --ref-cache "$C/ref-p300.jsonl"
    if [ "$DI_LIMIT" -gt 0 ]; then
      python3 -m bench.agreement --ref "$url" --contract "$CONTRACT" --set di --limit "$DI_LIMIT" --data-dir "$DATA" --fetch-only --workers 8 --ref-cache "$C/ref-di.jsonl"
    fi
  else
    python3 -m bench.agreement --ref "$C/ref-p300.jsonl" --cand "$url" --contract "$CONTRACT" --set p300 --data-dir "$DATA" --workers 8 --out "$RAW/agreement-p300.json" >/dev/null
    if [ "$DI_LIMIT" -gt 0 ]; then
      python3 -m bench.agreement --ref "$C/ref-di.jsonl" --cand "$url" --contract "$CONTRACT" --set di --limit "$DI_LIMIT" --data-dir "$DATA" --workers 8 --out "$RAW/agreement-di.json" >/dev/null
    fi
  fi
  python3 -m bench.loadgen --url "$url" --contract "$CONTRACT" --set p300 --data-dir "$DATA" --concurrency "1,$CONC" --seconds "$LOAD_SECONDS" --out "$RAW/load-$label-p300.json"
  if [ "$DI_LIMIT" -gt 0 ] && [ "$CONTRACT" = systemone ]; then
    python3 -m bench.loadgen --url "$url" --contract systemone --set di --limit 200 --grouped --concurrency "$CONC" --seconds "$LOAD_SECONDS" --out "$RAW/load-$label-di.json"
  fi
  kill "$SAMPLER" 2>/dev/null || true; wait "$SAMPLER" 2>/dev/null || true
  sort -n "$RAW/mem-$label.csv" | tail -1 > "$RAW/peak-mem-$label.txt"
}

# ---- 3. reference
ENV_ARGS=("${REF_ENV[@]}")
log "reference server for $ID"
start $NAME_REF "$REF_IMAGE" "$REF_ENTRY" $PORT_REF -- "${REF_CMD[@]}"
phase ref $PORT_REF
cleanup

# ---- 4. REAX Engine
ENV_ARGS=()
if [ "$ENGINE_IMAGE" = reax-vllm-readout ]; then
  ENV_ARGS=(-e REAX_VLLM_PORT=$((8000+PORT_OFF)) -e REAX_MODEL="$ID" -e REAX_HOST=127.0.0.1 -e REAX_PORT=$PORT_REAX)
  [ -n "${PRESET:-}" ] && ENV_ARGS+=(-e REAX_PRESET="$PRESET")
  [ -n "${REAX_OPTS:-}" ] && ENV_ARGS+=(-e REAX_OPTS="$REAX_OPTS")
  log "REAX Engine (vLLM-backed, preset ${PRESET:-default}) for $ID"
  start $NAME_CAND "$ENGINE_IMAGE" "" $PORT_REAX --
else
  log "REAX Engine (native) for $ID"
  start $NAME_CAND "$ENGINE_IMAGE" "" $PORT_REAX -- "${REAX_CMD[@]}"
fi
phase cand $PORT_REAX

# ---- 5. results
PEAK_REF=$(tr -dc 0-9 < "$RAW/peak-mem-ref.txt"); PEAK_CAND=$(tr -dc 0-9 < "$RAW/peak-mem-cand.txt")
{
  echo "{\"gpu\": \"$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)\","
  echo " \"engine_image_id\": \"$(docker image inspect -f '{{.Id}}' "$ENGINE_IMAGE")\", \"reference_image_id\": \"$(docker image inspect -f '{{.Id}}' "$REF_IMAGE")\","
  echo " \"peak_gpu_mem_mib\": {\"reference\": ${PEAK_REF:-null}, \"reax\": ${PEAK_CAND:-null}}, \"gpu_arg\": \"$GPU_ARG\", \"preset\": \"${PRESET:-default}\", \"reax_opts\": \"${REAX_OPTS:-}\", \"concurrency\": $CONC, \"load_seconds\": $LOAD_SECONDS, \"di_limit\": $DI_LIMIT,"
  echo " \"repo_head\": \"$(git rev-parse HEAD 2>/dev/null || echo unknown)\"}"
} > "$RAW/env.json"
echo "$SPEC" > "$RAW/spec.json"
OUT=$ROOT/results/$ID/$TODAY${PRESET:+-$PRESET}${TAG:+-$TAG}.json
python3 -m bench.summarize --model "$ID" --spec "$RAW/spec.json" --env "$RAW/env.json" --agreement "$RAW/agreement-p300.json" \
  --agreement-di "$RAW/agreement-di.json" --ref-load "$RAW/load-ref-p300.json" --cand-load "$RAW/load-cand-p300.json" \
  --ref-load-di "$RAW/load-ref-di.json" --cand-load-di "$RAW/load-cand-di.json" --out "$OUT"
log "wrote $OUT"
