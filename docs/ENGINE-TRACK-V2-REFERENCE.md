# Engine v2.0 lineup reference update (25 October 2026)

Current scoring slots: **mammoth 0.50, blink 0.20, reax1 0.20, jeffan 0.05, vision 0.05**.
The historical recipes below remain for provenance; Quyet and pplx are outside this lineup.

- `docker/ref-qwen35` now publishes only `ghcr.io/reaxlabs/reax-engine-ref-mammoth`.
  Its model label is `d3,pplx,reax1,mammoth`, but only mammoth uses this new build.
  Existing d3/pplx/reax1 slots retain digest
  `sha256:503cd9e00114dada655691de44b5a4ff1a08a985bcaf0021121967a73e436eac`.
- Mammoth weights: `baselabs/mammoth-1@61cec44c281c693992e3d6ebeb44dd3402dfa186`.
  Reference options: FP8, compile=1, gva=1, budget/cap=16384, shared_state/group=1,
  shared_until=64, shared_mixq=0, `CIE_LAZY_LOAD=1`. qev-labels-v2 prompt, prefix_bidir
  full attention, causal GatedDeltaNet, slot_lm fp32, temperature=1.0. The qev contract
  and loader delta come from the owned production port 5117ec3; no other private files copied.
  The pinned card declares MIT but supplies no LICENSE; `/licenses/mammoth/NOTICE`
  explicitly documents this, alongside the declaration and standard MIT text.
- `docker/ref-jeffan` publishes `ghcr.io/reaxlabs/reax-engine-ref-jeffan`, model/alias
  `jeffan-31b-1710`/`jeffan`, weights `tattrongvu/Jeffan-31b-1710@cebfa86c2bed2b5c83b12e44de4374b20c87fd79`.
  Backend from validated branch c992305: preset `fp8-kv-colo-bi`, utilization=0.44,
  FP8 weights/KV, context=16448, prefix cache, batch-invariant kernels, conditional reread.
  First read public limit=8192 tokens; reread limit=16440 (not an 8192-token sequence).
  One question per call; 255 options. Apache-2.0 LICENSE/attribution in `/licenses/jeffan`.
  Measured alone 49.7 GiB peak: two Jeffan engines may exceed 96 GB, so pair fit is unproven.
  Lowering context to 8192 would truncate valid rereads and is not an equivalent fallback.
- Blink text stays pinned to
  `sha256:ab02e141ffb5672dc68e137736be0a6fa15445c7cc824a51f6ad2a6664c0e10a`.
  The new Blink build is a **vision candidate only** until GPU checks pass. Its installed
  vLLM 0.30.0 `Gemma4ForConditionalGeneration._encoder_chunk` has an opt-in
  `REAX_VISION_ENCODER_BUDGET=total10`: budget=`total_bytes//10`, independent of free
  memory. Reference-profile Blink enables it at startup. `dynamic` restores upstream
  `min(free_bytes//2,total_bytes//10)`. Text never calls this encoder function.
  Build-time source validation fails on drift. This is a determinism hypothesis, not
  a measured fix; co-resident equality and production parity <=0.02 are still required.

Sampling provenance: the cited node-letters production `letters/serve_letters.py`
currently sends completions `temperature=1.0, top_p=1.0` for Quyet and Blink text;
`serve.py` sends chat completions `temperature=1.0, top_p=1.0` for vision. Quyet reference
keeps both. Blink text restores the **calibrated** client (`temperature=0`, no explicit
`top_p`); the 33db8c3 addition of top_p changed 42/256 text answers. This distinction
must be checked in the GPU gate; no claim of wire-identical current production text.
Jeffan retains the validated branch's `temperature=0`, no explicit `top_p`, followed
by restricted softmax T=1.0. Vision chat remains `temperature=1.0, top_p=1.0`, T=1.3.

CPU tests cover private-port render/token-split fixtures, slot labels, backend options,
FP8 anchor semantics and the vision budget patch. Real tokenizer parity is offline-only:
`python tools/lineup_qev_parity.py --production-contract <5117ec3>/engine/qev_contract.py --tokenizer <mammoth>`.
GPU parity, repeated-process determinism and pair fit are release gates.

---

# Engine track v2: reference images

The v2 engine competition measures one candidate image on each production model against a pinned reference engine for that
model. The reference of a slot is numerically the engine that serves the model in production. Five image names, built by
`.github/workflows/images.yml` (job `ref`) and published to GHCR by digest:

| slot | `REAX_MODEL` | image | weights (read-only at `/model`) |
|---|---|---|---|
| `d3turbo` (s1-pro) | `d3-turbo` | `ghcr.io/reaxlabs/reax-engine-ref-d3turbo` | `vllm-sr/d3-turbo@cfa0d41d253467e4742d5aedb46d4f0ef3b84b41` |
| `pplx` (s1-pro-2) | `pplx-decider-v1.1-27b` | `ghcr.io/reaxlabs/reax-engine-ref-pplx` | `perplexity-ai/pplx-decider-v1.1-27b@6195aa55a72ae21e503c8a60198f7b390d42adb5` |
| `reax1` | `reax-1` | `ghcr.io/reaxlabs/reax-engine-ref-reax1` | `decisionmodels/reax-1@ca2a0cc54a537c1f5cf24e5b09387cfb357b6518` |
| `blink` / `vision` (s1-fast / s1-vision) | `blink-v0.3-26b-a4b` | `ghcr.io/reaxlabs/reax-engine-ref-blink` | `PixilabAI/Blink-v0.3-26B-A4B-NVFP4@08caad20c1574fb76341ca809ff38fdb081f7d4a` |
| `quyet` (Quyet-1.0-Large) | `quyet-1-large` (alias of `quyet-1.0-large`) | `ghcr.io/reaxlabs/reax-engine-ref-quyet` | `chinhnc/Quyet-1.0-Large@d997d58951e3c0da4486cf0884ca9d4d916e95dc` |

`d3turbo`, `pplx` and `reax1` are the same image (`docker/ref-qwen35`, one digest, three names). No weights are inside.

## Contract

Entrypoint `python -m engine.serve` (blink/quyet: `docker/vllm-readout/entrypoint.sh`), model from `$REAX_MODEL`, weights from `/model`.
`POST /internal/v1/decide` (header `X-S1M-Internal-Token`, one question per call) and `GET /health` on the unix socket
`$RUNTIME_UDS`; the token comes from `$S1M_INTERNAL_TOKEN_FILE`. With `RUNTIME_UDS` set the REAX server listens on no TCP port.
(Blink: vLLM itself listens on `127.0.0.1:8000` inside the container's own network namespace; the contract runs with `--network none`.)
Runs read-only with `/tmp` as the only writable path (caches live there) as uid 10002.

## `REAX_PROFILE=reference` (set in all reference images)

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

## Quyet and Blink vision

Quyet's `ref-prod` preset uses vLLM 0.30.0 at the same base digest as Blink: online FP8 weights, FP8 KV,
prefix caching, 8192 context, max 8000 prompt tokens, 10 options, 64 sequences, max-logprobs 256,
per-type temperatures choice 1.3007 / score 1.3159 / noul 1.4957. Reference requests are never truncated.
The only runtime budget change from node-letters is GPU memory utilization 0.40 -> 0.45, following Blink:
two engines reserve at most 90% of one 96 GB card. This is a budget choice, not GPU fit evidence; cold-start
online quantization peaks and the 8192-context KV allocation must be checked together on the GPU.
Quyet LICENSE and NOTICE from the pinned checkpoint are at `/licenses/quyet/` (including "Quyet by Chinh Nguyen").

Blink uses ONE image and registry id for the text and vision slots: `blink-v0.3-26b-a4b`, label `blink,vision`.
An optional `images` list carries exactly one PNG/JPEG base64 data URL. Multiple images, external URLs,
malformed base64, mismatched MIME and non-image bytes get HTTP 422. Text-only backends return 422
`unsupported_images` with `fallback: "author"`. Missing/null/empty-list images use the unchanged text path.
The vision message body is the production Rune template with an image_url part followed by the text part.
Vision strings use production's raw state string; text requests retain their existing JSON string rendering.
The readout calls `/v1/chat/completions` with one option-letter token, temperature 1.0 and top_p 1.0;
input-token usage includes image tokens from vLLM. Prompts above 8000 tokens get HTTP 422. The preset allows
`{"image":2,"audio":0,"video":0}` internally (production flags), while the public request contract allows ONE image;
`--chat-template-content-format string` is absent. A GPU numerical and two-engine fit check is still required.

CPU regression goldens in `tests/fixtures/node_letters_golden.json` come from the production node-letters
`letters/templates.py` (SHA256 recorded in the fixture) and `serve.py` message-part construction. They cover
choice/score/noul, structured and raw string states, Unicode and missing option descriptions. Blink text goldens
are from parent commit 4230e50. Tests use a character tokenizer; actual HF tokenizer parity remains a GPU-check item.
