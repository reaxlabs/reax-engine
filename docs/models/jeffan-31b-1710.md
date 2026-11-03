# Jeffan-31b-1710 on the REAX Engine (family `jeffan`, vLLM-backed)

Model: [tattrongvu/Jeffan-31b-1710](https://huggingface.co/tattrongvu/Jeffan-31b-1710) @ `cebfa86c2bed2b5c83b12e44de4374b20c87fd79`,
Apache-2.0, Gemma-4-31B base (`dm_config.json`: readout `code`, layout `state_first`, attention `causal`).

## What the backend does
- Prompt: the author's `dm_decision_engine.render.CodeRenderer` transcribed token for token (system line, `State:` /
  `Question:` / `Options:` user turn, codes `A..Z, AA..` that stay one token after the assistant turn opens, at most 255).
  Measured: identical token ids for 5,929 of 5,929 questions (DI-1000, P300, the 500 public parity items), both the
  first read and the read-twice prompt, and the same code set.
- Readout: vLLM `/v1/completions`, `max_tokens=1`, `allowed_token_ids` = the codes, `processed_logprobs`; softmax with T = 1.
- Conditional re-read (the author's policy): a question whose top-1 probability is below 0.9 is read again with the
  "read twice" prompt (the input shown twice); that distribution replaces the first. The read-twice prompt starts with
  the first read's tokens, so the prefix cache serves the first copy. If the read-twice prompt is longer than
  `max_reread_tokens` (16,440), the first read stands (the author's `reread_skipped`).
- Limits: first read up to 8,192 tokens (the gateway's question limit; longer -> HTTP 422), never truncated.
  `usage.input_tokens` = first-read prompt tokens (the author's usage also counts the re-read prompts; see below).
- `/health` adds `rereads` and `reread_tokens`.

## Presets
| preset | flags | use |
|---|---|---|
| `bf16` | bf16 weights/KV, util 0.9, 64 seqs, prefix cache, max-model-len 16,448 | port check |
| `fp8-kv` | online FP8 weights + FP8 KV, util 0.9 | single-model card |
| `fp8-kv-colo` | `fp8-kv` at util 0.44 (0.42 cannot hold one 16,448-token sequence) | shared card, NOT deterministic |
| `fp8-kv-colo-bi` | `fp8-kv-colo` + `VLLM_BATCH_INVARIANT=1` | **canary recipe** (golden gate run1 == run2) |

## Measured (25 Oct 2026, one RTX PRO 6000 Blackwell 96 GB, RunPod, vLLM 0.30.0; author arm vLLM 0.29.0)
- Parity vs the author engine (DI-1000, P300): bf16 DI -0.13 [-0.34, +0.12], agreement 99.84 %; FP8 + FP8 KV DI +0.16
  [-0.84, +1.07], agreement 99.38 %, P300 capability -0.75 [-2.05, +2.18], agreement 97.64 %.
- Without batch-invariant kernels the 500-item golden gate is not bit-identical between two passes (13-14 items, max
  |dp| 0.010, all on first reads with top-1 > 0.9). `VLLM_BATCH_INVARIANT=1` or `--no-enable-prefix-caching` fixes it
  (0/500); batch-invariant keeps the prefix cache the re-reads need, at about -26 % throughput. Across a fresh restart
  the node golden gate passes (run1 == run2; max |dp| vs the stored reference 0.0071 on 64 items, 0.0113 on 500; limit 0.02).
- Throughput, DI-1000 (8 clients): bf16 9.61, fp8-kv 16.50, fp8-kv-colo 16.18, fp8-kv-colo-bi 11.97 decisions/s
  (author engine 9.18). c1 p50 / p95: fp8-kv-colo 91 / 240 ms, fp8-kv-colo-bi 110 / 322 ms.
- Memory (whole GPU peak): fp8-kv-colo 48.3 GiB, fp8-kv-colo-bi 49.7 GiB; next to Quyet (util 0.40) on one 96 GB
  card: 85.3 GiB peak under load on both.
- Re-reads: about 19 % of DI-1000 questions and 36 % of P300 questions take the second (read-twice) pass.
