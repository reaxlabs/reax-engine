# /opt/jeffan: team ruj line J5 changes (workflow v2-bj-5, 28-29 Oct 2026)

The Jeffan engine is **pjs's** (E61 / E62: `jeffan_engine.py`, `jef_pin.py`, `pins.json`, `alt/`, his e62v1z / v2a / v2b
tree with KV16, SA, LLP / LLPA; in-process vLLM 0.30.0 with the engine core in a spawned process). Team ruj's line-J4
host-side files sit under this layer's changes (`jef_host.py` = line B's NOMETA / LEAN / DSAMP / DFREE, ported; FT's
memo in `jef_levers.py`; FT / LF / EM in `jeffan_engine.py`; OC and the defaults in `sitecustomize.py`). This layer adds
three exact levers; in this tree they default on in `sitecustomize.py` (`_RUJ_DEFAULTS`; a variable set in the
container wins). With all three, the engine's answers are bit-identical to the engine without them (line J5 screens:
1,024 timed + 24 c1 + 32 c1 re-run answers, exact float equality, every run).

| switch | lever | file | what changes |
|---|---|---|---|
| `JEF_J5_FR=1` | FR, fast read path | `jeffan_engine.py` | each read's `EngineCoreRequest` is a copy of the request vLLM's own `InputProcessor.process_inputs` built for the first read with the same number of answer codes (same sampling parameters), with this read's token ids, a fresh request id and arrival time, and its own copy of the sampling parameters. It enters `AsyncLLM.generate` as an `EngineCoreRequest`: deprecated but supported in the pinned vLLM 0.30.0 (one warning in the log); prove FR again on any vLLM bump. CPU checks: the msgpack bytes the front-end sends equal vLLM's path for 2,288 / 2,288 screen reads and 216 synthetic reads (option counts 1..255, prompts up to 16,289 tokens), field by field including the sampling parameters' private fields |
| `JEF_J5_TF=1` | TF, tokenizer backend call | `jeffan_engine.py` | the checkpoint renderer's `tok.encode(s, add_special_tokens=False)` (inside EM's memo) goes to the tokenizer's Rust backend directly (`tok.backend_tokenizer.encode(s, add_special_tokens=False).ids`; transformers 5.17's `encode` is that call after resetting truncation / padding, both null in `tokenizer.json`). CPU checks: identical ids for 1,415 screen piece strings, 40,254 adversarial strings and one 60,000-character string |
| `JEF_J5_SUB=1` | SUB, sub-block prefix-cache hits (round 2) | `jef_sub.py` | a prefix-cache hit may end inside a 16-token block: the last two blocks of every finished read and the block where its own hit ended are indexed by the chain hash of the block before them plus their token ids; a new read's full-block hit is extended by the longest common prefix with an indexed block (capped at its length - 1), and vLLM 0.30.0's own copy-on-write path for sub-block hits gives the read a private copy of that block (`allocate_slots` -> `_apply_cow` -> `SchedulerOutput.kv_cache_block_copies` -> the V2 runner's `copy_kv_cache_blocks_inplace` before the step's forward). Read 2 no longer recomputes the ~7.5 tokens of read 1 that end inside a block, read 1 no longer the template tokens that end inside its first item block: 234.4 -> 226.5 computed prompt tokens per decision on the screen items (-3.4 %). Only partial last blocks are held (one reference, at most 64, all released when an allocation fails); full blocks are used only while they keep their block hash |

Measured (box RTX PRO 6000, runsc, the pinned jeffan reference resident and paused, p7 items, 4 x 256 at c8):
FR + TF is exact and worth about +0.6-0.7 % [est] (crossover of line J5's screen s1, x1.0138 with the control first,
and the verifier's reversed-order session v1, x1.0003 with FR + TF first; not resolved from 0). SUB: see line J5's
`ROUND2.md` (screen s3, A B B A order with a warm-up start).

Also in `jeffan_engine.py`, off unless set: `JEF_FRONTENDS` (several frontend processes: pjs's BLINK_FRONTENDS plumbing +
team ruj's #376 balancing, ported; measured slower here, default 1 = off).

Licence: our files Apache-2.0 (as pjs's engine).
