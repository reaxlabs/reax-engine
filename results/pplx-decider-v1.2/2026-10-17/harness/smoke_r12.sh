#!/bin/bash
# smoke: REAX engine on v1.2 (bf16, no warm-up) -> example request, compare to author fast path (support .99946, urgent .9518)
cd /workspace/r12; export HF_HUB_OFFLINE=1
python engine/server.py --model ${MODEL:-/workspace/m12} --port 8799 --linear ${LIN:-bf16} --compile 1 --gva 1 --warm-steps 0 "$@" > /workspace/smoke-r12.log 2>&1 &
SP=$!
for i in $(seq 1 240); do curl -sf localhost:8799/health >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
tail -5 /workspace/smoke-r12.log
curl -s localhost:8799/v1/systemone -d '{"state":"My integration keeps failing with a 401 since we rotated our API key.","questions":{"team":{"type":"choice","criteria":{"billing":"Charges","support":"Technical errors","sales":"Buying"}},"urgent":{"type":"noul","instructions":"Is the customer blocked?"}}}'; echo
curl -s localhost:8799/v1/systemone -d '{"state":"My integration keeps failing with a 401 since we rotated our API key.","questions":{"urgent":{"type":"noul","instructions":"Is the customer blocked?"}}}'; echo
kill $SP; wait $SP 2>/dev/null
