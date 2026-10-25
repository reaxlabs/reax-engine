#!/bin/bash
# vLLM on loopback (flags from the named preset, engine/backends/vllm_readout.py) + the REAX server in front.
#   REAX_MODEL        registry id (quyet-1.0-large | rune-26b-a4b | blink-v0.3-26b-a4b)           required
#   REAX_PRESET       serving preset (see docs/MODELS.md); default per model (REAX_PROFILE=reference: blink -> ref-prod)
#   REAX_PROFILE      reference = the engine track v2 reference image: production recipe of the model (registry production_opts)
#   RUNTIME_UDS       serve the REAX contract on this unix socket instead of TCP (vLLM stays on container-internal loopback)
#   REAX_MODEL_DIR    weights directory (default /model, mount read-only)
#   REAX_VERIFY=1     check Quyet's MANIFEST.sha256 before starting (a few minutes)
#   REAX_OPTS         extra backend options for the server, e.g. 'max_options=255'
#   REAX_VLLM_EXTRA   extra flags appended to `vllm serve`
set -u
MODEL=${REAX_MODEL:?set REAX_MODEL to a registry id}
DIR=${REAX_MODEL_DIR:-/model}
case "$MODEL" in
  quyet|quyet-1-large|quyet-1.0-large) FAMILY=quyet; MODEL=quyet-1.0-large ;;
  jeffan|jeffan-31b-1710) FAMILY=jeffan; MODEL=jeffan-31b-1710 ;;
  rune|s1-pro|rune-26b-a4b) FAMILY=rune; MODEL=rune-26b-a4b ;;
  blink|s1-fast-2|blink-v0.3-26b-a4b) FAMILY=blink; MODEL=blink-v0.3-26b-a4b ;;
  *) echo "reax: $MODEL is not a vLLM-backed model"; exit 2 ;;
esac
[ -f "$DIR/config.json" ] || { echo "reax: no weights at $DIR (mount them read-only there)"; exit 2; }
if [ "${REAX_VERIFY:-0}" = 1 ] && [ -f "$DIR/MANIFEST.sha256" ]; then
  echo "reax: verifying weights against MANIFEST.sha256"
  (cd "$DIR" && sha256sum --check --strict --quiet MANIFEST.sha256) || { echo "reax: weights do not match the manifest"; exit 2; }
fi
VPORT=${REAX_VLLM_PORT:-8000}
if [ -z "${REAX_PRESET:-}" ] && [ "${REAX_PROFILE:-}" = reference ] && { [ "$FAMILY" = blink ] || [ "$FAMILY" = quyet ]; }; then REAX_PRESET=ref-prod; fi
if [ -z "${REAX_PRESET:-}" ] && [ "${REAX_PROFILE:-}" = reference ] && [ "$FAMILY" = jeffan ]; then REAX_PRESET=fp8-kv-colo-bi; fi
if [ "${REAX_PROFILE:-}" = reference ] && [ "$FAMILY" = blink ]; then
  export REAX_VISION_ENCODER_BUDGET=${REAX_VISION_ENCODER_BUDGET:-total10}
fi
case "${REAX_VISION_ENCODER_BUDGET:-}" in
  ""|total10|dynamic) ;;
  *) echo "reax: unsupported REAX_VISION_ENCODER_BUDGET"; exit 2 ;;
esac
# shellcheck disable=SC2086
python3 -m engine.backends.vllm_readout "$FAMILY" ${REAX_PRESET:-} "$DIR" --port "$VPORT" --exec ${REAX_VLLM_EXTRA:-} &
VLLM_PID=$!
for _ in $(seq 1 720); do
  curl -sf -m 3 "http://127.0.0.1:$VPORT/health" >/dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "reax: vLLM died during startup"; exit 3; }
  sleep 2
done
curl -sf -m 3 "http://127.0.0.1:$VPORT/health" >/dev/null || { echo "reax: vLLM not healthy after 24 min"; kill $VLLM_PID; exit 4; }
export REAX_VLLM_BASE_URL="http://127.0.0.1:$VPORT"
OPTS=()
for o in ${REAX_OPTS:-}; do OPTS+=(--opt "$o"); done
[ -n "${REAX_PRESET:-}" ] && OPTS+=(--opt "preset=${REAX_PRESET}")
python3 -m engine.serve --model "$MODEL" --model-dir "$DIR" --host "${REAX_HOST:-0.0.0.0}" --port "${REAX_PORT:-8741}" "${OPTS[@]}" &
SRV_PID=$!
trap 'kill $VLLM_PID $SRV_PID 2>/dev/null' TERM INT
wait -n $VLLM_PID $SRV_PID
echo "reax: a child process exited; shutting down"
kill $VLLM_PID $SRV_PID 2>/dev/null
exit 1
