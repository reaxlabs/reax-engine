# Evaluation harness

Stdlib-only Python (runs on the host of the pod; the model servers run in containers).

| tool | what |
|---|---|
| `python -m bench.agreement --ref URL\|FILE --cand URL\|FILE --set p300\|public231\|di\|s1 --contract systemone\|s1` | top-choice agreement, mean/max abs dp and abs d log p, accuracy vs gold (P300, DI) with a paired-bootstrap CI of the candidate-minus-reference delta |
| `python -m bench.loadgen --url URL --set p300 --concurrency 1,16 --seconds 60 [--grouped]` | closed-loop decisions/s and request latency p50/p95 |
| `bench/pod_validate.sh <model-id>` | the whole pod run for one model (download, reference, REAX server, agreement, load, `results/<model>/<date>.json`) |
| `python -m bench.summarize` | merges the raw outputs of one pod run |
| `bench/ref/quyet_ref_server.py` | `/v1/systemone` shim around the author's `quyet` package (container only) |

Contracts: `systemone` = `POST /v1/systemone` (reference servers, legacy engine), `s1` = `POST /internal/v1/decide` with
`X-S1M-Internal-Token` from `$S1M_INTERNAL_TOKEN` (REAX multi-model server, Rune reference). One question per request.
A server answer of HTTP 400/413/422 marks the item *unsupported* for that server; other failures are errors; agreement is computed
on items both servers answered. `--ref` may be a response file written by `--ref-cache`, so the reference runs once, is stopped, and
the candidate gets the GPU (`--fetch-only` collects the reference responses and exits).

## Data (not in this repository)

Set `--data-dir` or `$REAX_BENCH_DATA` to a directory with the public JevBench / Decision Index files named in `bench/sets.py`
(`p300-input.jsonl` + `p300-gold.jsonl`, `jevbench_public231.jsonl`, `di5k-rows.jsonl` + `di5k-ids.json`). They come from the benchmark
owners; never put sealed or private items there. P300 gold has `expected` per item (choice label, noul yes/no, score level); the
DI-N subset is N evenly spaced states of `di5k-ids.json` (the "DI-1000" of the earlier studies is `--limit 1000`).

The accuracy delta printed by `agreement.py` is a proxy (argmax accuracy and mean gold probability). It is **not** the official
JevBench capability score; use the benchmark's own scorer on the saved response files for that.

## A pod run

```bash
git clone <this repo> && cd reax-engine
export REAX_BENCH_DATA=/data/eval            # p300 / di5k files
export HF_TOKEN=hf_...                       # needed for the gated Rune repo only
bash bench/pod_validate.sh pplx-decider-v1.1-27b
bash bench/pod_validate.sh d3
PRESET=fp8-kv bash bench/pod_validate.sh quyet-1.0-large
PRESET=fp8-kv-seqs64 REAX_SUBNET_DIR=/data/reax-subnet bash bench/pod_validate.sh rune-26b-a4b
```
Knobs: `CONC` (16), `LOAD_SECONDS` (60), `DI_LIMIT` (1000, `0` skips DI), `WORK`, `MIN_FREE_GB`, `RO=0` (drop `--read-only`),
`REAX_OPTS` (backend options). Reference server for Rune is the reax-subnet `s1-pro` runtime image (`REF_IMAGE`, or built from
`REAX_SUBNET_DIR`). The script rents and stops nothing on the pod.
