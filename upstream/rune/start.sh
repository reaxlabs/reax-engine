#!/bin/bash
# REAX s1-pro runtime sidecar = reference Rune entrypoint (vLLM loopback :8000 + decide adapter), with
# changes that do not touch numerics: the weights are checked against the release manifest before start, the adapter
# binds to RUNE_HOST (default 127.0.0.1), and it rejects prompts above 4,096 tokens as the reference engine does. Weights: mount the Rune snapshot read-only at /model.
set -u
MODEL_DIR=${RUNE_MODEL_DIR:-/model}
[ -f "$MODEL_DIR/config.json" ] || { echo "rune: no weights at $MODEL_DIR (mount the snapshot read-only there)"; exit 2; }
if [ "${RUNE_SKIP_VERIFY:-0}" != 1 ]; then
  echo "rune: verifying weights against the release manifest (takes a few minutes)"
  python3 -c "import json;m=json.load(open('/opt/reax/s1-pro-v1.json'));[print(h+'  '+n) for n,h in sorted(m['weights']['files'].items())]" \
    | (cd "$MODEL_DIR" && sha256sum --check --strict --quiet) || { echo "rune: weights do not match the manifest"; exit 2; }
fi
export OPENAI_API_KEY= HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
MAXLEN=${RUNE_MAX_MODEL_LEN:-32768}
# Determinism (measured bit-exact across c1/c16/c32, restarts and separate processes), three parts:
#  1. VLLM_BATCH_INVARIANT=1 (batch-invariant kernels),
#  2. no chunked prefill with a token budget >= max-model-len (a prompt is never split depending on co-scheduled requests),
#  3. -O0 = no torch.compile: separately (cold-)compiled inductor builds give different numerics, an eager process does not.
# RUNE_DETERMINISTIC=0 gives stock vLLM (compiled, ~4-7 % faster, not reproducible across load or processes).
DET_ARGS=()
if [ "${RUNE_DETERMINISTIC:-1}" = 1 ]; then
  export VLLM_BATCH_INVARIANT=1
  DET_ARGS=(--no-enable-chunked-prefill --max-num-batched-tokens "$MAXLEN" -O0)
fi
VLLM_ARGS=(
  "$MODEL_DIR" --served-model-name rune --host 127.0.0.1 --port 8000
  --max-model-len "$MAXLEN"
  --chat-template /opt/rune/chat_template_nospace.jinja
  --limit-mm-per-prompt '{"image":1}'
  --logprobs-mode processed_logits --max-logprobs 300
  --mm-processor-cache-gb 0
  --max-num-seqs "${RUNE_MAX_NUM_SEQS:-64}" --gpu-memory-utilization "${RUNE_GPU_MEM_UTIL:-0.85}"
)
VLLM_ARGS+=("${DET_ARGS[@]}")
# prefix caching: bit-identical to off (measured), only pays off when requests share a leading STATE (the state precedes the question)
if [ "${RUNE_PREFIX_CACHING:-0}" = 1 ]; then VLLM_ARGS+=(--enable-prefix-caching); else VLLM_ARGS+=(--no-enable-prefix-caching); fi
[ -n "${RUNE_QUANTIZATION:-}" ] && VLLM_ARGS+=(--quantization "$RUNE_QUANTIZATION")
# shellcheck disable=SC2206
[ -n "${RUNE_VLLM_EXTRA_ARGS:-}" ] && VLLM_ARGS+=(${RUNE_VLLM_EXTRA_ARGS})
vllm serve "${VLLM_ARGS[@]}" &
VLLM_PID=$!
for i in $(seq 1 360); do
  curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "rune: vLLM died during startup"; exit 3; }
  sleep 2
done
curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null || { echo "rune: vLLM not healthy after 12 min"; kill $VLLM_PID; exit 4; }
python3 /opt/rune/warmup.py http://127.0.0.1:8000 || true
python3 /opt/rune/rune_adapter.py --host "${RUNE_HOST:-127.0.0.1}" --port "${RUNE_PORT:-8102}" --workers "${RUNE_ADAPTER_WORKERS:-4}" \
  --upstream http://127.0.0.1:8000 --tokenizer "$MODEL_DIR" --max-input-tokens 4096 &
AD_PID=$!
trap 'kill $VLLM_PID $AD_PID 2>/dev/null' TERM INT
wait -n $VLLM_PID $AD_PID
echo "rune: a child process exited; shutting down"
kill $VLLM_PID $AD_PID 2>/dev/null
exit 1
