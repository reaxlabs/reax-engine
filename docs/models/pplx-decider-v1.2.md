# pplx-decider v1.2 (27B and Flash 4B): joint request format

perplexity-ai/pplx-decider-v1.2 (27B, @225c913d) and perplexity-ai/pplx-decider-flash-v1.2 (4B, @6d771ad7) read a whole request
as **one sequence**: the state, then every question, each followed by a readout marker (`<|fim_suffix|>`). Full-attention
layers are noncausal over the whole request (questions see the state and each other), the linear-attention layers stay causal,
and the readout head is applied at every marker. The checkpoint declares this in `joint_config.json` (protocol version 3).

## What the engine does

- `PplxEngine.forward_marked(ids, lens, reads)`: packed sequences (one per request), readout rows at the marker positions
  (`reads[j]` = marker offsets of sequence j). Last-layer pruning computes Q, o_proj, MLP and the readout only at the
  markers; every marker attends to its whole sequence. Exact computation, no approximation (unlike the v1.1 shared-state mode); on 27B bf16 matches the author at 100 % argmax, on Flash
  bf16 it is at bf16-noise level (99.4 % packed argmax, mean max |dp| 0.0025).
- `engine/server.py` detects a v1.2 checkpoint (`joint_config.json`) and tokenizes with the author's own `Encoder`
  (`source/src/pplx_decider/protocol.py` of the checkpoint): same prompt, same marker placement, same escaping of literal
  special tokens, same request validation (1-255 options, 2-255 score levels).
- `--v12-split joint` (default, the author's default) sends one sequence per request; `--v12-split single` sends every question
  as its own request (state repeated), like v1.1.
- `--max-length 8192` (default): the per-sequence token limit of this server. The author's library defaults to 262,144 tokens, where
  a long joint request is still ONE sequence in which all questions see each other; the model was trained on sequences up to ~32k.
  The benchmark runs (RESULTS.md) used 8,192 on both sides (harness/v12_server.py), and the canary uses the default 8,192.
- `--v12-overlength reject|split` (default `reject`): a joint request that needs more than `--max-length` tokens in one sequence
  - `reject`: HTTP 422 `{"error": "maximum context length exceeded: request needs N tokens in one sequence ...", "code":
    "context_length", "fallback": "author"}`. A router sends it to the author path (decision: requests over 8,192 tokens stay on
    the author path). A single question longer than the limit is also a 422 `context_length`.
  - `split`: answered in question chunks exactly as the author's `Encoder.plan` does (every chunk keeps the full state, questions
    of different chunks do not see each other, so this differs from the author default). Never run on GPU against the author.
- `usage.sequences` = number of sequences the request ran as (author parity); `> 1` only in `split` mode.
- Text only: requests with `images` get HTTP 422 `{"code": "unsupported_images", "fallback": "author"}` (use the author's
  `pplx_decider` for vision). This applies to v1.2 checkpoints only; v1.1 serving is unchanged (images are ignored as on main).
- A missing `state` is HTTP 400 (as the author's `RequestError`). Router rule: 422 and 5xx go to the author path, 400 back to the
  client.
- `joint_config.json` must equal the author's `PROTOCOL` (asserted at startup). The server imports `<model>/source/src`, so
  only point it at the reviewed checkpoint revision.
- `--served-name NAME`: the response `model` (default: HF id from the checkpoint's release manifest/config, else the directory
  name). Set it explicitly to what the router/comparator expects, e.g. `pplx-decider-flash-v1.2`.

## Run

```bash
# 27B: FP8 (needs the explicit flag; the server default is bf16)
python engine/server.py --model /path/to/pplx-decider-v1.2 --linear fp8 --compile 1 --gva 1 --warm-steps 120 \
    --served-name pplx-decider-v1.2 --port 8741
# Flash 4B: bf16 (the quality-safe choice; FP8 only after a full-set check, see RESULTS.md)
python engine/server.py --model /path/to/pplx-decider-flash-v1.2 --linear bf16 --compile 1 --gva 1 --warm-steps 120 \
    --served-name pplx-decider-flash-v1.2 --port 8741
# one question per request (v1.1-style comparability mode): add --v12-split single
```

`GET /health` reports `model`/`served_name`, `linear` (bf16|fp8), `max_length`, `v12_overlength` and `v12_split`, so a canary can
assert that Flash runs bf16.

Warm-up takes about 5 minutes (every Triton/compile specialisation is triggered before `/health` is ready).

## Check

`tools/exact_v12.py author|engine MODEL ROWS N OUT.json [--linear bf16|fp8]` compares the engine with the author's exact
reference path (`Decider(fast=False)`, one request per call). `EXACT_MULTI=1` uses only multi-question requests.
Measured results: RESULTS.md, results/pplx-decider-v1.2/.

CPU test of the server path (fake GPU engine, no weights): `PPLX_V12_MODEL_DIR=/path/to/v1.2-checkpoint-dir pytest
tests/test_pplx_v12_server_cpu.py` (needs torch, transformers 5.x, torchvision; skipped without the directory; the directory
needs the small files only: configs, tokenizer/processor files, `chat_template.jinja`, `source/src/pplx_decider/`).
