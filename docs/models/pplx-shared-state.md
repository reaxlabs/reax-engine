# Perplexity Decider v1.1: shared-state mode for multi-question requests (experimental)

`engine/server.py --shared-state 1` (Perplexity Decider only). A `/v1/systemone` request asks several questions about one
state. In the normal mode every question is its own sequence (system prompt + state + question), so the state is encoded once
per question. In shared-state mode the common token prefix of the request's questions (system prompt + state) is encoded
**once**, and every question continues from it.

**This is an approximation, not a lossless optimisation.** The model's full-attention layers are noncausal: in the original
model the state tokens also attend to the question. When the prefix is shared, they cannot see "their" question:
- Before the first attention layer (layers 0-2, DeltaNet, causal) the shared prefix is exact; from layer 3 on everything downstream
  differs slightly, which the measurements below quantify.
- In the 16 attention layers (3, 7, ..., 63) the state tokens attend either to the state only (`PPLX_SHARED_MIXQ=0`) or to the state
  plus **all** questions of the request (`PPLX_SHARED_MIXQ=1`), instead of the state plus their own question.
- Each question attends to the state and to itself, as in the original model; the readout is unchanged.
- `PPLX_SHARED_UNTIL=L` (default 16) shares only layers 0..L-1 and then switches to full per-question sequences.

Single-question requests (and requests whose questions share fewer than `--shared-min` = 64 prefix tokens) take the normal path:
same results as the non-shared server up to batch-composition noise. Consequence for users: a question's probabilities can depend
on the other questions in its request (top-1 changed for 10 of 108 questions asked alone vs in their request with FP8 + shared state; 2 of 108 without sharing).

## Measured drift (bf16, 206 decisions in 34 multi-question requests, vs the unshared engine)
| sharing | state sees | top-1 agreement | mean abs dp |
|---|---|---|---|
| until layer 8 | state only | 99.0 % | 0.0045 |
| until layer 16 | state only | 98.1 % | 0.0082 |
| until layer 32 | state only | 96.6 % | 0.033 |
| all layers | state only | 95.6 % | 0.037 |
| until layer 32 | state + all questions | 98.5 % | 0.017 |
| all layers | state + all questions | 97.6 % | 0.023 |
(batch-composition noise floor of the same engine: 99.5 % / 0.002.)

## Recommended settings and results
- FP8 (any FP8 GPU; the quality-safe setting, 6.0x the author reference on DI-1000, 5.8x on DI-5k, no measurable DI change):
  `python engine/server.py --model /model --linear fp8 --compile 1 --gva 1 --shared-state 1 --shared-until 64 --shared-mixq 1`
- NVFP4 mid-layers (Blackwell; 8.1x, small non-significant score loss, as the NVFP4 mode itself): add
  `--nvfp4 gu@4-59,down@4-59,in_qkvz@4-59,out@4-59 --nvfp4-scales engine/scales/nvfp4-act-amax-max512.json`.
The flags override `PPLX_SHARED_UNTIL` / `PPLX_SHARED_MIXQ`; the server prints the active mode at start. Results and the quality gate:
RESULTS.md, section "17 Oct 2026".

## Implementation
`PplxEngine.forward_groups(groups)` takes `(prefix_ids, [suffix_ids, ...])` groups: prefixes are one packed varlen stream, suffixes
a second one; DeltaNet suffixes start from the prefix's recurrent state and the last 3 pre-conv rows; attention suffix keys are the
prefix rows + the suffix rows (gathered). Under causal attention the shared path is exact, which `tools/test_shared.py` uses as
its correctness check (shared vs full at the batch-noise floor). The server splits requests larger than the step budget into
several groups with the same prefix, and its warm-up sweeps every step-size bucket (fla's conv kernel specialises per 1024-token
bucket; without the sweep a cold server stalls for tens of seconds on unseen step sizes).
