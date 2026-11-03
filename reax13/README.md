# REAX Engine: reax13

`Reax13Engine` is a custom engine for the public [Decision Index kit](https://github.com/apolinario/decision-index/blob/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d/docs/engines.md). It loads both parts of one model repository on **one NVIDIA RTX PRO 6000 Blackwell 96 GB**. The language-model linear weights use dynamic fine-grained FP8, block128. Heads, non-language towers and small gated-delta projections remain BF16. Both parts must fit together; this is not two-GPU inference.

For every request, two host workers launch the native Gemma/KnowLine and Qwen/d3 paths on separate CUDA streams. Every question goes to both parts. Their per-option probabilities are averaged with weights 0.5/0.5, checked for completeness and normalized for rounding error. The result contains every supplied option. External requests are serialized. A host lock protects Triton’s shared mutable autotuner during tuning and dispatch; GPU kernels remain asynchronous after launch. Prompts, native temperature policies and option readouts are retained; no truncation or benchmark-specific rules. A request above the declared input cap (`max_input_proxy_tokens` in `reax13.json`, 32,000; size proxy = characters of the compact request JSON divided by 4) is refused as unsupported before any model work. A request has `request_timeout` = 1800 s to answer; rows of 24k-32k proxy tokens take minutes (measured 242-258 s on a stand-in box), so a client must allow at least that (the kit's `http` engine defaults to 600 s and needs a larger `timeout`). If the native backend dies (request timeout `request_timeout`, GPU memory failure, crash), later requests fail with an error and `/healthz` answers 503 until the container is restarted; nothing in this bundle restarts it (`docker run --rm`, no restart policy, no HEALTHCHECK), so the operator must. A bad or partial reply line also closes the backend.

The default `reax13` mode enables the official causal-conv1d kernel and within-request Qwen prefix/state reuse for multi-question prefixes up to24576 tokens. Prefixes above16384 wait for the Gemma forward before the cached Qwen pass to bound peak memory. An allocation failure retries the native uncached Qwen path. The cache is cleared between requests; no answers are cached. The conservative mode does not require miner projection kernels, CUDA graphs or RMSNorm compilation.

## Model layout

```text
model-repo/
  part-g/   # complete Gemma-4-31B / KnowLine-compatible model + tokenizer
  part-q/   # complete Qwen3.5-27B / d3-compatible model + tokenizer
            # keep modeling_d3.py, d3_runtime.py, d3_fast.py, d3_kernels.py,
            # decision_config.json, readout.safetensors and model manifests
```

Both directories need complete checkpoints, including upstream attribution. Keep a native `temperatures.json` in part-g (the bundled KnowLine table is the compatibility fallback). Native d3 policy comes from part-q/decision_config.json. `reax13.json` specifies the model root and the two relative subfolders; `--model /model` overrides the root. The package does not fetch weights during serving.

## Build and run

Tested dependency stack: Python3.12, PyTorch2.14.0/CUDA13, Transformers5.17.0, FLA0.5.2, causal-conv1d1.7.0 and kernels0.16.0. Exact versions are in the Dockerfiles. GPU driver must support CUDA13 and SM120. Build and download on a clean GPU host with >=400 GB disk and >=64 GiB host RAM; >=128 GiB is preferable while quantizing both checkpoints. No credentials or gated pulls are needed for the public stand-ins.

From this directory on the GPU host:

```bash
docker build -t reax13-base -f Dockerfile .
docker build -t reax13-runtime -f Dockerfile.conv .
git clone https://github.com/apolinario/decision-index.git decision-index
git -C decision-index checkout 25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d
mkdir -p results
```

For public stand-ins, download complete pinned checkpoint files without a token:

```bash
mkdir -p model-repo
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 hf download PelaAI/KnowLine-31B-Gen1 \
  --revision 8f052a8130e599b53e5324c34f0f5ff90696e1c6 --local-dir model-repo/part-g \
  --exclude '*.bin' '*.png' '*.jpg' '*.mp4' '*.pt' 'evaluation/*' '.eval_results/*'
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 hf download vllm-sr/d3-ultra \
  --revision 52ba9e7a00402f9560c7fba8c039608e7ddee7fb --local-dir model-repo/part-q \
  --exclude '*.bin' '*.png' '*.jpg' '*.mp4' '*.pt' 'evaluation/*' '.eval_results/*'
```

For actual REAX weights, use the assembled model repository instead. Place a public sample in `sample.jsonl` using the official kit row format. Run the official harness: 

```bash
docker run --rm --network none --gpus device=0 --read-only \
  --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges --shm-size 8g \
  --tmpfs /tmp:rw,exec,size=8g \
  -v "$PWD/model-repo:/model:ro" \
  -v "$PWD/decision-index/decision_index:/kit/decision_index:ro" \
  -v "$PWD/sample.jsonl:/inputs/sample.jsonl:ro" \
  -v "$PWD/results:/out:rw" \
  --entrypoint env reax13-runtime -i \
  PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp PYTHONPATH=/engine:/kit \
  DUO_ISOLATED=1 OPENAI_API_KEY= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/tmp/hf KERNELS_CACHE=/opt/kernels \
  LOCAL_KERNELS=kernels-community/finegrained-fp8=/opt/fp8-kernel \
  D3_STATE_ONCE=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m decision_index run --engine reax13_engine:Reax13Engine \
  --option mode=reax13 --option config=/engine/reax13.json \
  --rows /inputs/sample.jsonl --out /out/run --fresh
```

For index-style latency, pass `--option warmup_rows=/inputs/sample.jsonl`. This adds three longest rows and the first ten sample rows to the kit's two warmups (rows above the input cap are skipped before the longest rows are chosen), then measures each request once, sequentially. The official kit supports choice/noul. For a mixed sample containing native score questions, use the included compatibility driver with the same Engine API and timing boundaries: replace the final command above with `python /engine/benchmark.py --config /engine/reax13.json --rows /inputs/sample.jsonl --out /out/latency`. It adds score-distribution and expected-value validation without changing inputs. Startup/loading/warmup are excluded. The measured wall time includes IPC, both prompt renderers/tokenizers, all native model work, synchronization and probability combination. Engine raw output includes internal times and peak allocated/reserved GPU bytes. Peak GPU process memory is also sampled externally in the validation run.

A Python caller with the pinned kit available can construct `Reax13Engine(config=..., model=...)`, then call `engine(state, questions)` to get `(response, diagnostics)`. Use `engine.close()` when finished. The supplied adapter requires a container isolation marker. The experimental `optimization=plain` configuration is a sequential FP8 reference for parity diagnostics, not the fast mode. The public kit at this pin validates choice and noul; the backend also handles native score questions.

## HTTP server

`serve.py` loads the same `Reax13Engine` with `reax13.json` defaults, performs the normal warmup, then serves `GET /healthz` and `POST /v1/systemone` on port 8080. The POST body uses the JevBench System One `{state, questions}` format; a request can contain one question or a joint map of up to 64 questions. The JSON response preserves the Engine API answer and adds a `probabilities` map to noul answers (`false = 1 - noul`, `true = noul`). See [SEALED-CONTRACT.md](SEALED-CONTRACT.md) for request examples, offline launch steps, prompt/context facts, and verification status.

The HTTP wrapper does not invoke model-specific code itself: it calls `Reax13Engine(state, questions)`, checks that every requested option is present and normalized, and serializes that result. CPU contract tests use a mock native backend:

```bash
python3 -m unittest -v test_probabilities test_serve test_bundle
```

`test_bundle` also runs on CPU: a stub backend replaces the GPU process to test the input cap, dead-backend handling and the kernel dispatch guards. `test_fast_fp8_gemm.py` needs the GPU and the runtime image.

## Public stand-in checkpoints

Validation uses these pinned public repositories, downloaded and hash-checked on the disposable pod:

- `PelaAI/KnowLine-31B-Gen1@8f052a8130e599b53e5324c34f0f5ff90696e1c6` → part-g
- `vllm-sr/d3-ultra@52ba9e7a00402f9560c7fba8c039608e7ddee7fb` → part-q

They match the target layouts. Stand-in smoke and latency do not measure released REAX1.3 blended/fine-tuned weights or establish its quality. Release weights require their own end-to-end smoke and quality gates. No publication or production deployment is performed by this validation.

Code: MIT (LICENSE), except fast_fp8_gemm.py, which stays Apache-2.0 as a derivative of finegrained-fp8 (full text in third_party/finegrained-fp8/LICENSE). Third-party files and external weights retain their licences; see THIRD_PARTY_NOTICES.md and third_party/.
