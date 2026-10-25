#!/usr/bin/env python3
"""/v1/systemone server for pplx-decider-v1.1-27b.

Backends
  engine  our prefill-only engine (pplx_engine.PplxEngine) behind a continuous batcher: every question of every
          in-flight request is tokenised in the request thread and queued; one GPU thread packs queued decisions up
          to --budget tokens per forward (no padding) and resolves them. Requests are never serialised behind a lock.
  ref     the author's DecisionModel, served like the author's server (one request at a time, questions in chunks of 8).
Protocol (as the author's server + harness needs): POST /v1/systemone {model, state, questions{id: question}} ->
{model, answers{id: answer}, usage{input_tokens, output_tokens:0}}. Noul answers additionally carry
probabilities {false, true}. A question branch > 8192 tokens -> HTTP 422 "maximum context length ..." (the author
refuses those too; the marker makes the Decision Index kit count it as unsupported). GET /health.
v1.2 checkpoints: usage also carries `sequences`; 422 bodies carry `code` (unsupported_images | context_length) and
`fallback: "author"` so a router can send the request to the author path (see docs/models/pplx-decider-v1.2.md).
Usage: server.py --model ./model --backend engine [--linear fp8 --compile 1 --budget 16384] --port 8741
"""
from __future__ import annotations

import argparse, json, os, queue, sys, threading, time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from pplx_engine import MAX_OPTIONS, Prompter, answer, options  # noqa: E402

MAX_TOKENS = 8192
NAME = "pplx-decider-v1.1-27b"


class Capacity(Exception):
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code  # v1.2 only: machine-readable reason in the 422 body


def served_name(model, override=None):
    """Response `model` of a v1.2 server: --served-name, else the HF id of the release (release manifest / config), else the dir name."""
    if override:
        return override
    d = Path(model).resolve()
    for fn in ("release_manifest.json", "release-manifest.json", "manifest.json", "config.json"):
        try:
            m = json.loads((d / fn).read_text())
        except (OSError, ValueError):
            continue
        for k in ("hf_id", "hf_repo", "repo_id", "model_id", "_name_or_path"):
            v = m.get(k) if isinstance(m, dict) else None
            if isinstance(v, str) and v.strip() and not v.startswith(("/", ".")):
                return v.rsplit("/", 1)[-1]
    return d.name


def validate(questions):
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object")
    for q in questions.values():
        if not isinstance(q, dict) or q.get("type") not in ("choice", "score", "noul"):
            raise ValueError("each question needs type choice|score|noul")
        n = len(options(q)[0])
        if q["type"] == "choice" and not 1 <= n <= MAX_OPTIONS:
            raise Capacity(f"choice questions take 1 to {MAX_OPTIONS} options per choice")
        if q["type"] == "score" and not 2 <= n <= 10:
            raise Capacity("a score takes 2 to 10 levels")


def shape(q, probs):
    a = answer(q, probs)
    if q["type"] == "noul":
        a["probabilities"] = {"false": 1.0 - a["noul"], "true": a["noul"]}
    return a


class EngineBackend:
    def __init__(self, a):
        from pplx_engine import PplxEngine
        self.pr = Prompter(a.model)
        self.eng = PplxEngine(a.model, linear_mode=a.linear, compile=a.compile, fp8_skip=tuple(x for x in a.fp8_skip.split(",") if x),
                              gva_native=a.gva, nvfp4_groups=tuple(g for g in a.nvfp4.split(",") if g), nvfp4_scales=a.nvfp4_scales or None)
        self.budget = a.budget
        # pplx-decider v1.2 (joint_config.json protocol v3): one sequence per request, one readout per question at its
        # marker token. Tokenisation is the author's own Encoder (source/src/pplx_decider/protocol.py of the checkpoint).
        self.v12 = (Path(a.model) / "joint_config.json").exists()
        if self.v12:
            if a.shared_state == "1":
                raise SystemExit("--shared-state is a v1.1 approximation; v1.2 checkpoints share the state natively (joint format)")
            sys.path.insert(0, str(Path(a.model) / "source" / "src"))
            from transformers import AutoProcessor
            from pplx_decider.protocol import PROTOCOL, Encoder, LengthError, RequestError
            joint = json.loads((Path(a.model) / "joint_config.json").read_text())
            assert joint == PROTOCOL, f"joint_config.json does not match the author's PROTOCOL: {joint} != {PROTOCOL}"
            proc = AutoProcessor.from_pretrained(a.model)
            self.encoder = Encoder(proc, self.pr.codes)
            self.v12_errors = (LengthError, RequestError)
            self.v12_split = a.v12_split
            self.v12_overlength = a.v12_overlength
            self.linear = a.linear
            self.served = served_name(a.model, a.served_name)
            self.max_length = a.max_length
            assert self.max_length <= self.budget, "--max-length must fit one step (--budget)"
            print(f"v1.2 joint protocol: {a.v12_split} requests, max_length {self.max_length}, overlength {a.v12_overlength}, "
                  f"linear {a.linear}, served as {self.served}", flush=True)
        self.shared = a.shared_state == "1"  # encode a request's common prefix (system + state) once, see forward_groups
        self.shared_min = a.shared_min
        if a.shared_until is not None:
            self.eng.shared_until = a.shared_until
        if a.shared_mixq is not None:
            self.eng.shared_mixq = a.shared_mixq == "1"
        if self.shared:
            print(f"shared-state mode: until layer {self.eng.shared_until} of {self.eng.n_layers}, "
                  f"state rows see {'all questions' if self.eng.shared_mixq else 'the state only'} (approximation)", flush=True)
        self.max_wait = a.max_wait_ms / 1000
        a_warm_steps = a.warm_steps
        self.q: queue.Queue = queue.Queue()
        self.steps = self.decisions = 0
        self.idle_s, self.step_tokens = 0.0, 0
        threading.Thread(target=self._loop, daemon=True).start()
        # Warm-up: trigger every torch.compile specialization / Triton autotune BEFORE serving (otherwise the first
        # requests of each new batch shape stall the GPU loop for seconds). Covers 1..many sequences per step, short
        # and long sequences, and both DeltaNet conv paths (split path for steps >= 8192 tokens).
        t0 = time.time()
        shapes = [[n] for n in (17, 64, 300, 700, 1500, 3000, 6000, 8192)]
        shapes += [[300] * k for k in (2, 3, 4, 5, 7, 8, 12, 16, 24, 32)]
        shapes += [[700] * 11 + [64], [4000, 4000, 4000, 4000], [2000] * 8, [8192, 8000], [500] * 32, [16000]]
        for rep in range(2):
            for lens in shapes:
                lens = [min(n, MAX_TOKENS) for n in lens]
                while sum(lens) > self.budget:
                    lens.pop()
                fs = []
                for n in lens:
                    f = Future()
                    fs.append(f)
                    self.q.put(([1] * n, 2, f))
                for f in fs:
                    f.result()
        # Randomised realistic steps (seeded): many sequence counts x length mixes, so every size bucket of the generated
        # kernels is tuned before serving (first-pass cost measured 9.2 vs 20.7 items/s without this).
        import random
        rng = random.Random(7)
        for _ in range(a_warm_steps):
            lens, n = [], rng.choice([1, 1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 40])
            for _ in range(n):
                L = int(min(MAX_TOKENS, max(16, rng.lognormvariate(6.3, 0.8))))
                if sum(lens) + L > self.budget:
                    break
                lens.append(L)
            fs = []
            for L in lens or [64]:
                f = Future()
                fs.append(f)
                self.q.put(([1] * L, rng.randint(2, 12), f))
            for f in fs:
                f.result()
        # Deterministic sweep over every step size bucket (fla's conv kernel specialises and re-autotunes per 1024-token
        # bucket of the step, NB = cdiv(T, 1024), and per conv path): otherwise a step size first seen while serving
        # stalls the GPU loop for tens of seconds (measured: 45 s stalls with fresh Triton caches).
        for T in range(512, self.budget + 1, 512):
            for n_seq in (1, 3, 9):
                lens = [max(16, T // n_seq)] * n_seq
                if max(lens) > MAX_TOKENS:
                    continue
                fs = []
                for L in lens:
                    f = Future()
                    fs.append(f)
                    self.q.put(([1] * L, 2, f))
                for f in fs:
                    f.result()
                if self.shared and n_seq > 1:  # one shared group of this full size: prefix + n_seq suffixes
                    P = T // n_seq - 40 if self.eng.shared_until < self.eng.n_layers else T - 40 * n_seq
                    if P < 64:
                        continue
                    subs = [([1] * 40, 2, Future()) for _ in range(n_seq)]
                    self.q.put(([0] * T, None, ([1] * P, subs)))
                    for x in subs:
                        x[2].result()
        if self.shared:  # the suffix-path kernels (conv/DeltaNet/attention on short continuation sequences) too
            for _ in range(max(40, a_warm_steps // 2)):
                items, tot = [], 0
                for _ in range(rng.choice([1, 2, 3, 4, 6, 8, 12])):
                    P = int(min(MAX_TOKENS - 600, max(64, rng.lognormvariate(6.8, 0.8))))
                    subs = [([1] * int(max(8, rng.lognormvariate(4.5, 0.7))), rng.randint(2, 12), Future())
                            for _ in range(rng.choice([1, 2, 3, 5, 8, 12, 32, 64]))]
                    n = (P * len(subs) if self.eng.shared_until < self.eng.n_layers else P) + sum(len(x[0]) for x in subs)
                    if tot + n > self.budget:
                        break
                    items.append(([0] * n, None, ([1] * P, subs)))
                    tot += n
                for it in items or []:
                    self.q.put(it)
                for it in items:
                    for x in it[2][1]:
                        x[2].result()
        print(f"warm-up done in {time.time() - t0:.0f}s ({self.steps} steps)", flush=True)

    def _loop(self):
        carry = None
        while True:
            t_idle = time.perf_counter()
            batch = [carry] if carry else [self.q.get()]
            self.idle_s += time.perf_counter() - t_idle
            carry = None
            n = len(batch[0][0])
            deadline = time.perf_counter() + self.max_wait
            while n < self.budget:
                try:
                    item = self.q.get(timeout=max(0.0, deadline - time.perf_counter())) if self.max_wait else self.q.get_nowait()
                except queue.Empty:
                    break
                if n + len(item[0]) > self.budget:
                    carry = item
                    break
                batch.append(item)
                n += len(item[0])
            self.step_tokens += n
            if self.shared:
                self._run_groups(batch)
                continue
            if self.v12:
                self._run_marked(batch)
                continue
            try:
                ids = torch.tensor([t for b in batch for t in b[0]], device="cuda")
                logits = self.eng.forward(ids, [len(b[0]) for b in batch])
                probs = self.eng.probs(logits, [b[1] for b in batch])
                for b, p in zip(batch, probs):
                    b[2].set_result(p)
            except Exception as e:  # noqa: BLE001
                for b in batch:
                    if not b[2].done():
                        b[2].set_exception(e)
            self.steps += 1
            self.decisions += len(batch)

    def _run_marked(self, batch):
        """v1.2 step: items (ids, (markers, counts), fut) -> fut gets one probability list per marker; warm-up items
        (ids, n, fut) read their last token."""
        reads, counts = [], []
        for b in batch:
            m, c = b[1] if isinstance(b[1], tuple) else ([len(b[0]) - 1], [b[1]])
            reads.append(m)
            counts.append(c)
        try:
            ids = torch.tensor([t for b in batch for t in b[0]], device="cuda")
            probs = self.eng.probs(self.eng.forward_marked(ids, [len(b[0]) for b in batch], reads),
                                   [c for cs in counts for c in cs])
            i = 0
            for b, cs in zip(batch, counts):
                b[2].set_result(probs[i:i + len(cs)] if isinstance(b[1], tuple) else probs[i])
                i += len(cs)
        except Exception as e:  # noqa: BLE001
            for b in batch:
                if not b[2].done():
                    b[2].set_exception(e)
        self.steps += 1
        self.decisions += sum(map(len, counts))

    def _decide_v12(self, state, questions):
        reqs = [{"state": state, "questions": questions}] if self.v12_split == "joint" else \
            [{"state": state, "questions": {k: q}} for k, q in questions.items()]
        try:
            plans = [self.encoder.plan(r, self.max_length) for r in reqs]
            chunks = [c for p in plans for c in p]
            if self.v12_overlength == "reject" and any(len(p) > 1 for p in plans):
                need = sum(self.encoder.encode(r).length for r, p in zip(reqs, plans) if len(p) > 1)
                raise Capacity(f"maximum context length exceeded: request needs {need} tokens in one sequence, more than "
                               f"--max-length {self.max_length} (it would be split into {len(chunks)} sequences; "
                               f"--v12-overlength reject)", "context_length")
        except self.v12_errors[0] as e:
            raise Capacity(f"maximum context length exceeded: {e}", "context_length") from None
        except self.v12_errors[1] as e:
            raise ValueError(str(e)) from None
        futs = []
        for c in chunks:
            f = Future()
            futs.append((c, f))
            self.q.put((c.input_ids.tolist(), (c.markers.tolist(), list(c.counts)), f))
        answers = {}
        for c, f in futs:
            for key, p in zip(c.keys, f.result()):
                answers[key] = shape(questions[key], p)
        return {k: answers[k] for k in questions}, sum(c.length for c in chunks), len(chunks)

    def _run_groups(self, batch):
        """shared-state step: queue items are (ids, n, fut) single decisions or (tokens, None, [(prefix, [(suffix, n, fut)...])])
        request groups whose questions share a token prefix (see PplxEngine.forward_groups)."""
        groups, counts, futs = [], [], []
        for b in batch:
            if b[1] is None:
                prefix, subs = b[2]
                groups.append((prefix, [x[0] for x in subs]))
                counts += [x[1] for x in subs]
                futs += [x[2] for x in subs]
            else:
                groups.append((b[0], []))
                counts.append(b[1])
                futs.append(b[2])
        try:
            probs = self.eng.probs(self.eng.forward_groups(groups), counts)
            for f, p in zip(futs, probs):
                f.set_result(p)
        except Exception as e:  # noqa: BLE001
            for f in futs:
                if not f.done():
                    f.set_exception(e)
        self.steps += 1
        self.decisions += len(futs)

    def decide(self, state, questions):
        if self.v12:
            return self._decide_v12(state, questions)
        enc = {k: self.pr.ids(state, q) for k, q in questions.items()}
        if any(len(ids) > MAX_TOKENS for ids, _ in enc.values()):
            raise Capacity(f"maximum context length exceeded: a question branch is longer than {MAX_TOKENS} tokens")
        futs = {}
        if self.shared and len(enc) >= 2:
            seqs = [ids for ids, _ in enc.values()]
            m = min(map(len, seqs)) - 1  # every suffix keeps >= 1 token
            k = 0
            while k < m and all(x[k] == seqs[0][k] for x in seqs):
                k += 1
            if k >= self.shared_min:
                # partial sharing (PPLX_SHARED_UNTIL < layers): the step expands to full sequences after the switch layer,
                # so the full token count is what the step budget has to bound. A request larger than the budget is split
                # into several shared groups (same prefix) so that no step exceeds the budget (kernel shapes stay inside
                # the warmed-up range, no int32 offset overflow).
                full = self.eng.shared_until < self.eng.n_layers
                cost = (lambda n_sfx: k + n_sfx) if full else (lambda n_sfx: n_sfx)
                chunks, cur, tot = [], [], 0 if full else k
                for key, (ids, n) in enc.items():
                    futs[key] = Future()
                    c = cost(len(ids) - k)
                    if cur and tot + c > self.budget:
                        chunks.append((cur, tot))
                        cur, tot = [], 0 if full else k
                    cur.append((ids[k:], n, futs[key]))
                    tot += c
                chunks.append((cur, tot))
                for subs, ntok in chunks:
                    self.q.put(([0] * ntok, None, (seqs[0][:k], subs)))  # len(item[0]) = the step's token count
                answers = {key: shape(questions[key], futs[key].result()) for key in questions}
                return answers, sum(len(ids) for ids, _ in enc.values())
        for k, (ids, n) in enc.items():
            futs[k] = Future()
            self.q.put((ids, n, futs[k]))
        answers = {k: shape(questions[k], futs[k].result()) for k in questions}
        return answers, sum(len(ids) for ids, _ in enc.values())


class MammothBackend:
    """Legacy HTTP surface over the same mammoth backend used by engine.serve.

    Existing pplx/v1.2 scheduling stays unchanged. The reference image uses
    engine.serve's continuous scheduler, not this compatibility surface.
    """
    def __init__(self, a):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from engine.backends.qwen35 import Qwen35Backend
        self.be = Qwen35Backend(a.model, family="mammoth", linear=a.linear, compile=a.compile, gva=a.gva,
                                fp8_skip=a.fp8_skip, budget=a.budget, max_input_tokens=a.max_tokens,
                                warm_steps=a.warm_steps, shared_state=a.shared_state == "1",
                                shared_group=a.shared_group == "1", shared_until=a.shared_until,
                                shared_mixq=None if a.shared_mixq is None else a.shared_mixq == "1")
        self.lock = threading.Lock()
        self.steps = self.decisions = 0

    def decide(self, state, questions):
        try:
            batch = [self.be.encode(state, q) for q in questions.values()]
        except ValueError as exc:
            if "maximum context length" in str(exc):
                raise Capacity(str(exc)) from None
            raise
        with self.lock:
            probs, current, tokens = [], [], 0
            for item in batch:
                if current and tokens + item.n_tokens > self.be.max_batch_tokens:
                    probs.extend(self.be.run(current))
                    self.steps += 1
                    current, tokens = [], 0
                current.append(item)
                tokens += item.n_tokens
            if current:
                probs.extend(self.be.run(current))
                self.steps += 1
            self.decisions += len(batch)
        return {k: self.be.answer(q, p) for (k, q), p in zip(questions.items(), probs)}, sum(e.n_tokens for e in batch)


class RefBackend:
    def __init__(self, a):
        sys.path.insert(0, str(Path(a.model) / "source" / "src"))
        from autojev.model import DecisionModel
        self.m = DecisionModel(a.model, device="cuda")
        self.lock = threading.Lock()
        self.steps = self.decisions = 0

    def decide(self, state, questions):
        rows = [{"state": state, "question": q} for q in questions.values()]
        with self.lock, torch.inference_mode():
            probs, tokens = [], 0
            for s in range(0, len(rows), 8):
                try:
                    b = self.m.prepare(rows[s:s + 8])
                except ValueError as e:
                    raise Capacity(f"maximum context length exceeded: {e}") from None
                probs += [p[:c] for p, c in zip((self.m(b) / self.m.temperature).softmax(-1).cpu().tolist(), b.counts)]
                tokens += b.input_tokens
                self.steps += 1
            self.decisions += len(rows)
        return {k: shape(q, p) for (k, q), p in zip(questions.items(), probs)}, tokens


def handler(backend):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            raw = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":
                h = {"status": "ready", "model": NAME, "steps": backend.steps, "decisions": backend.decisions,
                     "idle_s": round(getattr(backend, "idle_s", 0.0), 3), "step_tokens": getattr(backend, "step_tokens", 0)}
                if getattr(backend, "v12", False):
                    h.update(model=backend.served, served_name=backend.served, linear=backend.linear, v12=True,
                             max_length=backend.max_length, v12_overlength=backend.v12_overlength, v12_split=backend.v12_split)
                return self._send(200, h)
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                v12 = getattr(backend, "v12", False)
                if v12 and body.get("images"):  # v1.1 keeps its text-only behaviour (images ignored), as on main
                    raise Capacity("images are not supported by this engine (text-only); use the author's pplx_decider",
                                   "unsupported_images")
                if not v12 and not isinstance(backend, MammothBackend):  # v1.2: the author's Encoder validates (1-255 options, 2-255 levels)
                    validate(body.get("questions"))
                elif "state" not in body or not isinstance(body.get("questions"), dict) or not body["questions"]:
                    raise ValueError("A request needs 'state' and 'questions' (questions must be a non-empty object)")
                answers, tokens, *seqs = backend.decide(body.get("state"), body["questions"])
                usage = {"input_tokens": tokens, "output_tokens": 0}
                if seqs:  # v1.2: number of sequences (author parity, types.py usage.sequences)
                    usage["sequences"] = seqs[0]
                self._send(200, {"model": backend.served if v12 else NAME, "answers": answers, "usage": usage})
            except Capacity as e:
                self._send(422, {"error": str(e), **({"code": e.code, "fallback": "author"} if e.code else {})})
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": f"{type(e).__name__}: {e}"[:300]})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", choices=["pplx", "mammoth"], default="pplx")
    ap.add_argument("--max-tokens", type=int, default=16384, help="mammoth per-question token cap")
    ap.add_argument("--shared-group", default="0", help="mammoth: merge token-identical state blocks")
    ap.add_argument("--backend", choices=["engine", "ref"], default="engine")
    ap.add_argument("--linear", default="bf16")
    ap.add_argument("--compile", default="1")
    ap.add_argument("--fp8-skip", default="")
    ap.add_argument("--gva", default="1", help="native grouped-value heads in fla (bit-identical, faster)")
    ap.add_argument("--nvfp4", default="", help="NVFP4 groups, e.g. gu@4-59,down@4-59 (see fast_run.py)")
    ap.add_argument("--nvfp4-scales", default="", help="JSON of calibrated activation amax per layer.group")
    ap.add_argument("--budget", type=int, default=16384)
    ap.add_argument("--max-wait-ms", type=float, default=0.0, help="extra wait to fill a step (0 = take what is queued)")
    ap.add_argument("--port", type=int, default=8741)
    ap.add_argument("--shared-state", default="0", help="1: questions of one request share the encoded state prefix (approximation: "
                    "state tokens do not attend to the question); measured separately, see RESULTS.md")
    ap.add_argument("--shared-min", type=int, default=64, help="minimum common-prefix tokens for a shared-state group")
    ap.add_argument("--shared-until", type=int, default=None, help="share layers [0, N) (default: $PPLX_SHARED_UNTIL or 16; 64 = all)")
    ap.add_argument("--shared-mixq", default=None, help="1: in the shared attention layers the state rows attend to the state and all "
                    "questions of the request (default: $PPLX_SHARED_MIXQ or 0)")
    ap.add_argument("--v12-split", choices=["joint", "single"], default="joint", help="v1.2 checkpoints: joint = one sequence "
                    "per request (author default); single = every question as its own request (v1.1 style)")
    ap.add_argument("--max-length", type=int, default=8192, help="v1.2: per-sequence token limit (a single over-long question -> 422)")
    ap.add_argument("--v12-overlength", choices=["reject", "split"], default="reject", help="v1.2: a joint request needing more than "
                    "--max-length tokens: reject = HTTP 422 context_length (a router sends it to the author path, which runs it as one "
                    "sequence); split = answer it in question chunks by the author's Encoder.plan (questions of different chunks do not "
                    "see each other; usage.sequences > 1)")
    ap.add_argument("--served-name", default=None, help="v1.2: `model` in responses and /health (default: HF id from the checkpoint's "
                    "release manifest/config, else the directory name)")
    ap.add_argument("--warm-steps", type=int, default=200, help="randomised warm-up steps before /health is ready")
    a = ap.parse_args()
    global NAME
    if a.family == "mammoth":
        if a.backend != "engine":
            ap.error("mammoth requires --backend engine")
        NAME = "mammoth-1"
        backend = MammothBackend(a)
    else:
        backend = EngineBackend(a) if a.backend == "engine" else RefBackend(a)
    class Srv(ThreadingHTTPServer):
        request_queue_size = 512
    srv = Srv(("127.0.0.1", a.port), handler(backend))
    srv.daemon_threads = True
    print(f"ready on :{a.port} backend={a.backend}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
