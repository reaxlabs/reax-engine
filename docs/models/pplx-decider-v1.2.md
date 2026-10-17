# pplx-decider v1.2 (27B and Flash 4B): joint request format

perplexity-ai/pplx-decider-v1.2 (27B, @225c913d) and perplexity-ai/pplx-decider-flash-v1.2 (4B, @6d771ad7) read a whole request
as **one sequence**: the state, then every question, each followed by a readout marker (`<|fim_suffix|>`). Full-attention
layers are noncausal over the whole request (questions see the state and each other), the linear-attention layers stay causal,
and the readout head is applied at every marker. The checkpoint declares this in `joint_config.json` (protocol version 3).

## What the engine does

- `PplxEngine.forward_marked(ids, lens, reads)`: packed sequences (one per request), readout rows at the marker positions
  (`reads[j]` = marker offsets of sequence j). Last-layer pruning computes Q, o_proj, MLP and the readout only at the
  markers; every marker attends to its whole sequence. Exact computation, no approximation (unlike the v1.1 shared-state mode).
- `engine/server.py` detects a v1.2 checkpoint (`joint_config.json`) and tokenizes with the author's own `Encoder`
  (`source/src/pplx_decider/protocol.py` of the checkpoint): same prompt, same marker placement, same escaping of literal
  special tokens, same request validation (1-255 options, 2-255 score levels).
- `--v12-split joint` (default, the author's default) sends one sequence per request; `--v12-split single` sends every question
  as its own request (state repeated), like v1.1.
- `--max-length 8192` (default): a request longer than this is split into question chunks exactly as the author's
  `Encoder.plan` does (every chunk keeps the full state); a single question longer than the limit gets HTTP 422 (the v1.1
  server rule). The author's library defaults to 262,144 tokens; the model was trained on sequences up to ~32k.
- Text only: requests with `images` get HTTP 422 (use the author's `pplx_decider` for vision).

## Run

```bash
python engine/server.py --model /path/to/pplx-decider-v1.2 --linear fp8 --compile 1 --gva 1 --warm-steps 120 --port 8741
# Flash 4B: same flags, --model /path/to/pplx-decider-flash-v1.2
# one question per request (v1.1-style comparability mode): add --v12-split single
```

Warm-up takes about 5 minutes (every Triton/compile specialisation is triggered before `/health` is ready).

## Check

`tools/exact_v12.py author|engine MODEL ROWS N OUT.json [--linear bf16|fp8]` compares the engine with the author's exact
reference path (`Decider(fast=False)`, one request per call). `EXACT_MULTI=1` uses only multi-question requests.
Measured results: RESULTS.md, results/pplx-decider-v1.2/.
