# ref-rune: reference engine of the s1-pro pool model (Surogate Rune 26B-A4B v3)

Image `ghcr.io/reaxlabs/reax-engine-ref-rune` (`docker/ref-rune/Dockerfile`), `REAX_MODEL=rune-26b-a4b-v3`, text only.

* **Recipe = the REAX s1-pro pool runtime** (reax-subnet `runtimes/s1-pro`, commit in `upstream/rune/SUBNET-COMMIT`): same vLLM 0.30.0
  digest, `VLLM_BATCH_INVARIANT=1`, `--no-enable-chunked-prefill --max-num-batched-tokens 32768 -O0`, `--max-model-len 32768`,
  `--logprobs-mode processed_logits --max-logprobs 300`, `--no-enable-prefix-caching`, `--max-num-seqs 64`,
  `--gpu-memory-utilization 0.85`, `--mm-processor-cache-gb 0`, `--limit-mm-per-prompt {"image":1}`, the pool chat template.
  Preset `ref-pool`; `tests/test_rune_reference.py` compares it flag by flag with the pool's `start.sh` and manifest.
* **Readout** = engine `rune` family = the pool adapter's decision protocol (system line + `SHARED STATE (JSON string)`, option letters,
  one forward pass, `max_tokens=1`, `temperature=0`, `allowed_token_ids`, float64 softmax of the processed logits / T=2.0, prompts
  above 4,096 tokens rejected). The test loads the pool's own `rune_adapter.py` (fixture) and checks prompt text, option order,
  T and answer shape equal for choice/score/noul, odd states and unicode. The real-tokenizer codebook (two-letter codes past 26
  options) is part of the GPU gate. Images are **not** served (the pool adapter has an image path; this reference is text only).
* **Weights**: mount the gated snapshot read-only at `/model`. The entrypoint verifies all 18 files against the pool release
  manifest (`/opt/reax/s1-pro-v1.json`, manifest id 5b11440bf57b5434) before vLLM starts and exits 2 on any mismatch.
  `REAX_SKIP_VERIFY=1` skips it (tests only).
* **Memory share**: `REAX_GPU_MEM_UTIL=0.62` replaces the 0.85. The subnet's calibration (docs/CALIBRATION.md, docs/hardware.md) ran
  its reference at util 0.60 / max-model-len 8192 and found it bit-identical to the 0.85 / 32768 pool runs; the self-check
  reference was recorded at 8192. This must be re-confirmed in the GPU gate for this image.

## Memory (computed from configs; no GPU measurement of this image)

Config: google/gemma-4-26B-A4B-it (public; Rune's own config is behind the gate, its sha256 is in the manifest): 30 layers = 25 sliding
(window 1024, 8 KV heads x 256) + 5 full attention (2 KV heads x 512, `attention_k_eq_v`), bf16.

| item | size |
|---|---|
| weights bf16 (pool docs: 52 GB; 18 files, 11 shards) | 48.4 GiB |
| KV per sequence, hybrid KV manager, 32,768 tokens (25 x 1024 x 8 KiB sliding + 5 x 32768 x 4 KiB full) | 0.82 GiB (0.52 if K=V is shared) |
| KV per sequence at the 4,096-token input limit | 0.27 GiB |
| KV per sequence if vLLM allocates every layer to full length (upper bound) | 6.9 GiB at 32k, 0.86 GiB at 4k |
| activation peak of the profile run (32,768 batched tokens, eager) | ~3-5 GiB **estimate** |
| process floor (weights + activations + CUDA context + one 32k sequence) | ~54 GiB |
| pool setting 0.85 of 97,887 MiB (RTX PRO 6000) | 81.3 GiB reserved; ~27 GiB of it KV |
| subnet measurement, util 0.60, max-model-len 8192 | 58.7 GB |

**Reference + candidate on one 96 GB card**: depends on the candidate.
* candidate = a second bf16 Rune: **no**. 2 x 48.4 GiB = 96.8 GiB > 95.6 GiB, before any KV or CUDA context.
* candidate = FP8 (or smaller) Rune, ~26 GB weights: **yes, estimated**: reference 0.62 (59.3 GiB) + candidate 0.33 (31.5 GiB) = 0.95
  of the card. Not measured; start the reference first.
* otherwise sequential runs on one card (golden answers of the reference recorded first) or a 2-GPU box.
