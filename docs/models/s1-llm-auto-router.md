# s1-llm-auto-router: why it stays a separate classifier

`s1-llm-auto-router` (Hugging Face `decisionmodels/s1-llm-auto-router`, MIT) is **not** a decision model in the sense of
this engine, so it has no backend here.

## What the engine's backends do

A decision model answers *any* question about *any* state: the caller sends `state` plus a question
(`type` choice / noul / score, free-text `instructions`, per-option `criteria`), the model sees all of it in its prompt,
and the answer is a softmax over the answer-option logits at the last prompt position.
The `Backend` API (`encode(state, question)`, `run(batch)`, `options(question)`) assumes exactly that: the options
come from the question and the prompt changes with every question.

## What the router does

An encoder (jhu-clsp/mmBERT-small, pruned to 11 layers, ONNX int8 embeddings, about 183 MB) reads
`{"request", "context"}` and, in **one pass**, emits seven fixed classification heads:

| question | type | classes |
|---|---|---|
| `category` | choice | 9 fixed categories |
| `difficulty` | score | 5 levels |
| `stakes` | score | 4 levels |
| `needs_tools`, `needs_vision`, `needs_long_context`, `follow_up` | noul | yes/no |

There is no prompt, no question text and no option input; the heads are trained for these seven questions only. Its own
server (`POST /v1/systemone`) rejects every other question name or type with HTTP 400. So it cannot take "arbitrary
state + question + options" and return "option probabilities" for that question; a backend would be a lookup that
accepts seven hard-wired question names and ignores `instructions` and `criteria`.

## Decision

* No `engine/backends/onnx_router.py`. Wrapping it would pass the type signature but not the contract: a caller who
  sends a different question would get an error, or worse, a silently wrong answer from an unrelated head.
* The router's own server (onnxruntime on CPU, 4 threads) is the right way to run it. Its published latency is about
  14 ms p50 / 47 ms p99 on 4 pinned cores of a modern EPYC and 26 ms p50 on an older Xeon (model card; not re-measured
  for this repository). For comparison, the GGUF decision models in this engine need roughly 1 to 10 s per decision on
  8 shared CPU threads (see `docs/models/gguf-cpu.md`), because they run a full decoder over a prompt of a few hundred tokens.

## How it relates to the decision models

It sits **in front of** them. A gateway classifies an incoming request (category, difficulty, stakes, tool and vision
needs) in about 14 to 26 ms and uses that to pick which model, such as one of the decision models served by this engine
or a general LLM, handles the request. The classifier chooses a model; the engine runs one.

If a "fixed-head classifier" readout is ever wanted here (for a model family that really has fixed heads and no prompt),
it should be a separate readout kind with its own request schema, not a `Backend` for arbitrary questions.
