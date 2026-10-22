# Engine track v2: reference images

The v2 engine competition measures one candidate image on each production model against a pinned reference engine for that
model. The reference of a slot is numerically the engine that serves the model in production. Four images, built by
`.github/workflows/images.yml` (job `ref`) and published to GHCR by digest:

| slot | `REAX_MODEL` | image | weights (read-only at `/model`) |
|---|---|---|---|
| `d3turbo` (s1-pro) | `d3-turbo` | `ghcr.io/reaxlabs/reax-engine-ref-d3turbo` | `vllm-sr/d3-turbo@cfa0d41d253467e4742d5aedb46d4f0ef3b84b41` |
| `pplx` (s1-pro-2) | `pplx-decider-v1.1-27b` | `ghcr.io/reaxlabs/reax-engine-ref-pplx` | `perplexity-ai/pplx-decider-v1.1-27b@6195aa55a72ae21e503c8a60198f7b390d42adb5` |
| `reax1` | `reax-1` | `ghcr.io/reaxlabs/reax-engine-ref-reax1` | `decisionmodels/reax-1@ca2a0cc54a537c1f5cf24e5b09387cfb357b6518` |
| `blink` (s1-fast, s1-vision) | `blink-v0.3-26b-a4b` | `ghcr.io/reaxlabs/reax-engine-ref-blink` | `PixilabAI/Blink-v0.3-26B-A4B-NVFP4@08caad20c1574fb76341ca809ff38fdb081f7d4a` |

`d3turbo`, `pplx` and `reax1` are the same image (`docker/ref-qwen35`, one digest, three names). No weights are inside.

## Contract

Entrypoint `python -m engine.serve` (blink: `docker/vllm-readout/entrypoint.sh`), model from `$REAX_MODEL`, weights from `/model`.
`POST /internal/v1/decide` (header `X-S1M-Internal-Token`, one question per call) and `GET /health` on the unix socket
`$RUNTIME_UDS`; the token comes from `$S1M_INTERNAL_TOKEN_FILE`. With `RUNTIME_UDS` set the REAX server listens on no TCP port.
(Blink: vLLM itself listens on `127.0.0.1:8000` inside the container's own network namespace; the contract runs with `--network none`.)
Runs read-only with `/tmp` as the only writable path (caches live there) as uid 10002.

## `REAX_PROFILE=reference` (set in all four images)

Starts the model with the options of its production recipe (`production_opts` in `engine/registry.py`; `--opt key=value` still overrides):

| model | options |
|---|---|
| `d3-turbo` | FP8, `torch.compile`, step budget 16384, 8192-token cap per question, **state-once for every call**: `shared_state` + `shared_group`, `shared_until 64`, `shared_mixq 0`, fp32 readout, lazy load (`CIE_LAZY_LOAD=1`), 200 warm-up steps + the shared sweep |
| `pplx-decider-v1.1-27b` | FP8, `torch.compile`, step budget 4096, 8192-token cap, lazy load, 200 warm-up steps |
| `reax-1` | FP8, `torch.compile`, joint protocol v3 read at the question marker (one question per sequence, as the production node's `--v12-split single`), step budget 8192, 8192-token cap (422 above), lazy load, 200 warm-up steps |
| `blink-v0.3-26b-a4b` | vLLM 0.30.0, preset `ref-prod`: the s1-fast production flags (NVFP4 `modelopt_fp4`, FP8 KV, `VLLM_BATCH_INVARIANT=1`, prefix caching, 8192 context, 32 seqs, 8192-token batches) with `--gpu-memory-utilization 0.45` so that two engines fit on one card |

## d3-turbo (prefix_bidir) and state-once

`vllm-sr/d3-turbo` sets `attention_mode: prefix_bidir` in `decision_config.json`: in the full-attention layers the state block
attends to itself only, the question block reads the whole row. `PplxEngine` runs that as two varlen segments per row
(`_blocks`, checked against the dense block mask in `tests/test_prefix_bidir_blocks.py`); `noncausal_full_attention` models (d3,
pplx, reax-1) take the old code path unchanged. Because the state block never sees the question and DeltaNet is causal, the state
can be encoded once and shared by the questions of a request (`PplxEngine.forward_groups`). The reference routes **every** call through
that path (prefix = state block, one suffix) and merges queued calls whose whole state block is token-identical into one group, so an
answer does not depend on which sibling calls arrive with it. A call whose state block is shorter than the 3-token conv window runs the
ordinary forward.

## REAX 1

`decisionmodels/reax-1` uses the pplx-decider v1.2 joint format. `PplxEngine.forward_marked` reads the 255-way code at the question
marker of a packed sequence; tokenisation and validation are the author's own `Encoder` from the checkpoint's `source/src/pplx_decider`
(foreign code, read from the read-only weights mount; `joint_config.json` must equal the author's `PROTOCOL`).
