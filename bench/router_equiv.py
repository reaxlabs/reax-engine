"""Equivalence and load tests for the classifier_heads path (s1-llm-auto-router).

  python -m bench.router_equiv samples [--inputs FILE.jsonl] [--n 320]            print the sample size / composition
  python -m bench.router_equiv inproc --model-dir DIR [--inputs FILE.jsonl]       engine backend vs bench/router_reference.py
  python -m bench.router_equiv http --ref-url URL --engine-url URL [--inputs ..]  two running servers, same requests
  python -m bench.router_equiv load --url URL --path /v1/systemone --concurrency 1,4 --seconds 30

The sample is deterministic: synthetic requests (several languages, code, emoji, very long text that is truncated to
512 tokens, empty and whitespace-only requests, long and non-ASCII contexts) plus, optionally, real requests from a
JSONL file with a "request" (and optional "context") field. Equivalence means: per-head argmax identical in 100 % of the
inputs and probabilities identical (max abs difference is reported; 0.0 = bit-identical).
Servers: the engine answers /v1/systemone with the production shape; the reference may be the production engine node
(`--ref-path /internal/v1/decide`, needs the same token header) or another REAX server.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

HEADS = ("category", "difficulty", "stakes", "needs_tools", "needs_vision", "needs_long_context", "follow_up")
CATS = ["coding", "agentic", "math", "knowledge", "long_context", "tool_use", "design", "summarisation", "general"]
QUESTIONS = {
    "category": {"type": "choice", "instructions": "", "criteria": {c: "" for c in CATS}},
    "difficulty": {"type": "score", "instructions": "", "criteria": ["trivial", "easy", "moderate", "hard", "frontier"]},
    "stakes": {"type": "score", "instructions": "", "criteria": ["negligible", "low", "medium", "high"]},
    **{f: {"type": "noul", "instructions": ""} for f in HEADS[3:]},
}

SEED_TEXTS = [
    "Refactor this ABAP report to use CDS views", "What is the capital of Australia?", "Prove that sqrt(2) is irrational.",
    "Summarise the attached meeting notes in five bullet points", "Draw a landing page for a bakery with a warm colour palette",
    "Book me a flight to Lisbon next Friday and add it to my calendar", "Why is my React component re-rendering on every keystroke?",
    "Write a SQL query that returns the top 5 customers by revenue per region", "hi", "thanks, that worked!",
    "Explain the difference between TCP and UDP to a ten year old", "Review this contract clause for liability risks before I sign",
    "Translate the following paragraph into French and keep the tone formal", "Integrate (x^2 + 1) / (x^3 + 3x) dx",
    "Look at this screenshot and tell me why the button is misaligned", "Plan a 3 day agentic workflow that triages my inbox every morning",
    "My dog ate chocolate an hour ago, what should I do?", "Can you continue where we stopped yesterday?",
]
MULTILINGUAL = [
    "Wie kann ich in Python eine Liste nach dem zweiten Element sortieren?", "Quelle est la différence entre un prêt fixe et variable ?",
    "¿Puedes resumir este artículo sobre cambio climático en tres frases?", "Напиши функцию на Rust, которая разворачивает строку",
    "请帮我把这段代码改成异步的，并解释为什么这样更快", "この文章を丁寧な敬語に直してください", "اشرح لي كيف تعمل الشبكات العصبية",
    "मुझे एक छोटा सा वेब ऐप बनाना है, कहाँ से शुरू करूँ?", "서울에서 부산까지 가는 가장 저렴한 방법은 무엇인가요?",
    "Ποια είναι η διαφορά ανάμεσα σε στοίβα και ουρά;", "Jak mogę zoptymalizować to zapytanie do bazy danych?",
]
CODE = [
    "def f(xs):\n    return sorted(xs, key=lambda x: (x[1], -x[0]))\n# why is this slow for 1e6 items?",
    "```rust\nfn main() { let v: Vec<u32> = (0..10).collect(); println!(\"{:?}\", v); }\n```\nexplain the ownership here",
    "SELECT u.id, COUNT(*) FROM users u JOIN orders o ON o.uid = u.id GROUP BY 1 HAVING COUNT(*) > 3; -- optimise",
    "<div class=\"card\"><span>{{ item.name }}</span></div>  how do I center this vertically?",
    "kubectl get pods -A | grep -v Running   # but I want only those pending for more than 10 minutes",
]
EMOJI = [
    "😀😀😀 can you make this email sound friendlier 🙏", "🚀🔥💯", "👨‍👩‍👧‍👦 family trip planner for 🇯🇵 in spring 🌸",
    "Fix my regex ^(?:\\d{3}-)?\\d{4}$ 😭", "​​zero‍width⁠ joiners ‮and RTL override",
]
CONTEXTS = ["", "(new conversation)", "User asked about invoices earlier; assistant produced a table of Q3 revenue.",
            "Long running coding session in a TypeScript monorepo. " * 30, "日本語の要約: ユーザーはAPIの設計について相談している。" * 40,
            "é" * 700]


def make_samples(n: int = 320, seed: int = 7, max_chars: int = 12000):
    rnd = random.Random(seed)
    s = [("", ""), (" ", ""), ("\n\t \n", ""), ("a", ""), ("?", "(new conversation)"), ("x" * 5000, ""),
         ("word " * 4000, ""), ("The quick brown fox jumps over the lazy dog. " * 400, "ctx"),
         ("日本語のテキスト。" * 1500, ""), ("\U0001f600" * 3000, ""), ("line\n" * 3000, "c" * 5000),
         ("normal question about math: what is 17 * 23?", "z" * 599), ("normal question about math: what is 17 * 23?", "z" * 601),
         ("same request", "ü" * 600), ("same request", "ü" * 601), ("\x00\x01\x02 control chars", ""), ("<s> </s> [CLS] [SEP] <pad>", ""), ("context that cannot be truncated", "e\u0301" * 700)]
    pool = SEED_TEXTS + MULTILINGUAL + CODE + EMOJI
    while len(s) < n:
        k = rnd.random()
        if k < 0.7:
            req = rnd.choice(pool)
        elif k < 0.9:
            req = rnd.choice(pool) + " " + rnd.choice(pool)
        else:
            req = " ".join(rnd.choice(pool) for _ in range(rnd.randint(8, 80)))
        s.append((req[:max_chars], rnd.choice(CONTEXTS)))
    return s


def load_samples(path, n, limit_real, max_chars=12000):
    s = make_samples(n)
    if path:
        for line in open(path):
            if limit_real and len(s) >= n + limit_real:
                break
            r = json.loads(line)
            s.append((str(r["request"])[:max_chars], str(r.get("context") or "")))
    return s


def body(req, ctx, model="s1-llm-auto-router"):
    return {"request_id": "bench" + str(abs(hash((req, ctx))) % 10**10).rjust(10, "0"), "deadline_unix_ms": int(time.time() * 1000) + 100_000,
            "model": model, "state": {"request": req, "context": ctx}, "questions": QUESTIONS}


def post(url, path, obj, token, timeout=60):
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json", "X-S1M-Internal-Token": token})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None


def probs_of(ans):
    out = {}
    for h in HEADS:
        p = ans[h]["probabilities"]
        out[h] = [p[k] for k in (CATS if h == "category" else sorted(p, key=lambda k: (k != "false", k)) if h in HEADS[3:] else sorted(p))]
    return out


def top(h, p):
    return max(range(len(p)), key=p.__getitem__)


def report(name, n, rows, extra=None):
    mism = sum(1 for r in rows if r["argmax_diff"])
    res = {"check": name, "n": n, "argmax_identical": n - mism, "argmax_mismatches": mism,
           "max_abs_prob_diff": max([r["max_diff"] for r in rows] or [0.0]),
           "bit_identical": sum(1 for r in rows if r["max_diff"] == 0.0), **(extra or {})}
    print(json.dumps(res))
    return res


def cmd_inproc(a):
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))
    from backends.onnx_heads import OnnxHeadsBackend
    from bench.router_reference import Reference
    be = OnnxHeadsBackend(a.model_dir, threads=a.threads)
    ref = Reference(a.model_dir, threads=a.threads)
    rows, ntok_diff = [], 0
    for req, ctx in load_samples(a.inputs, a.n, a.limit_real):
        try:
            pr, nr = ref.predict(req, ctx)
        except Exception:
            pr = None
        try:
            pe, ne = be.classify(req, ctx)
        except ValueError:
            pe = None
        if pr is None or pe is None:   # both must reject (tokenizer cannot truncate the pair to max_len)
            rows.append({"max_diff": 0.0, "argmax_diff": (pr is None) != (pe is None), "rejected": True})
            continue
        ntok_diff += nr != ne
        rows.append({"max_diff": max(float(np.abs(pr[h] - pe[h]).max()) for h in HEADS),
                     "argmax_diff": any(int(pr[h].argmax()) != int(pe[h].argmax()) for h in HEADS)})
    return report("inproc backend vs reference", len(rows), rows,
                  {"token_count_mismatches": ntok_diff, "rejected_by_both": sum(1 for r in rows if r.get("rejected"))})


def cmd_http(a):
    token = os.environ.get("S1M_INTERNAL_TOKEN", "")
    rows, status_diff = [], 0
    for req, ctx in load_samples(a.inputs, a.n, a.limit_real):
        sr, jr = post(a.ref_url, a.ref_path, body(req, ctx), token)
        se, je = post(a.engine_url, a.engine_path, body(req, ctx), token)
        if (sr == 200) != (se == 200):   # both reject (400 vs 503 for unfittable input) or both answer
            status_diff += 1
            print("STATUS DIFF", repr(req[:40]), sr, se)
            continue
        if sr != 200:
            rows.append({"max_diff": 0.0, "argmax_diff": False, "rejected": True})
            continue
        pr, pe = probs_of(jr["answers"]), probs_of(je["answers"])
        rows.append({"max_diff": max(abs(x - y) for h in HEADS for x, y in zip(pr[h], pe[h])),
                     "argmax_diff": any(top(h, pr[h]) != top(h, pe[h]) for h in HEADS) or
                     any(jr["answers"][h].get("choice", jr["answers"][h].get("score")) != je["answers"][h].get("choice", je["answers"][h].get("score")) for h in HEADS[:3]),
                     "tok": jr["usage"]["input_tokens"] != je["usage"]["input_tokens"]})
    return report("http engine vs reference server", len(rows) + status_diff, rows,
                  {"status_mismatches": status_diff, "rejected_by_both": sum(1 for r in rows if r.get("rejected")),
                   "input_token_mismatches": sum(1 for r in rows if r.get("tok"))})


def cmd_load(a):
    token = os.environ.get("S1M_INTERNAL_TOKEN", "")
    items = [(r, c) for r, c in load_samples(a.inputs, a.n, a.limit_real) if r.strip() and len(r) < 3000]
    out = []
    for conc in [int(x) for x in a.concurrency.split(",")]:
        lat, stop, lock, idx = [], time.time() + a.seconds, threading.Lock(), [0]

        def worker():
            while time.time() < stop:
                with lock:
                    r, c = items[idx[0] % len(items)]
                    idx[0] += 1
                t = time.perf_counter()
                st, _ = post(a.url, a.path, body(r, c), token)
                dt = (time.perf_counter() - t) * 1000
                if st == 200:
                    lat.append(dt)
        for _ in range(a.warmup):
            post(a.url, a.path, body(*items[0]), token)
        t0 = time.time()
        ts = [threading.Thread(target=worker) for _ in range(conc)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        wall = time.time() - t0
        q = statistics.quantiles(lat, n=100, method="inclusive")
        out.append({"concurrency": conc, "n": len(lat), "p50_ms": round(q[49], 2), "p95_ms": round(q[94], 2),
                    "throughput_rps": round(len(lat) / wall, 1), "label": a.label})
        print(json.dumps(out[-1]), flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--inputs", help="JSONL with request[, context] added to the synthetic sample")
        p.add_argument("--n", type=int, default=320, help="size of the synthetic sample")
        p.add_argument("--limit-real", type=int, default=300)
    p = sub.add_parser("samples"); common(p)
    p = sub.add_parser("inproc"); common(p); p.add_argument("--model-dir", required=True); p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("http"); common(p)
    p.add_argument("--ref-url", required=True); p.add_argument("--engine-url", required=True)
    p.add_argument("--ref-path", default="/internal/v1/decide"); p.add_argument("--engine-path", default="/v1/systemone")
    p = sub.add_parser("load"); common(p)
    p.add_argument("--url", required=True); p.add_argument("--path", default="/v1/systemone"); p.add_argument("--label", default="")
    p.add_argument("--concurrency", default="1,4"); p.add_argument("--seconds", type=float, default=30); p.add_argument("--warmup", type=int, default=20)
    a = ap.parse_args(argv)
    if a.cmd == "samples":
        s = load_samples(a.inputs, a.n, a.limit_real)
        print(len(s), "samples;", sum(1 for r, _ in s if not r.strip()), "empty/blank;", sum(1 for r, _ in s if len(r) > 3000), "long")
    else:
        globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
