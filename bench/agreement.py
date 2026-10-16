#!/usr/bin/env python3
"""Agreement of a candidate decision server with a reference server (or a saved response file) on one evaluation set.

  python -m bench.agreement --ref http://127.0.0.1:8001 --cand http://127.0.0.1:8002 --set p300 --contract systemone \
      --data-dir DATA --workers 8 --out agreement.json

--contract systemone  POST /v1/systemone     (the reference servers and the legacy engine)
           s1         POST /internal/v1/decide with X-S1M-Internal-Token ($S1M_INTERNAL_TOKEN) (REAX multi-model server)
--ref / --cand are URLs, or paths of response files written by an earlier run (--ref-cache / --cand-cache), so a reference
server can be run once, dumped and stopped before the candidate starts (one GPU, one server at a time).

One question per request (agreement is a per-decision property). Reported: top-choice agreement, mean/max |dp| and
|d log p|, accuracy vs gold where the set has gold (P300, DI), and a paired bootstrap 95% CI of the accuracy and
gold-probability deltas (candidate - reference). The accuracy delta is a score proxy, NOT the official JevBench capability
score (which needs the benchmark's own scorer).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import sets, stats  # noqa: E402

UNSUPPORTED = (400, 413, 422)   # deterministic capacity / validation rejections: the item is "unsupported", not an error


class Endpoint:
    def __init__(self, target, contract, model="model", token=None, timeout=600):
        self.target, self.contract, self.model, self.timeout = target, contract, model, timeout
        self.is_file = not target.startswith(("http://", "https://"))
        self.token = token or os.environ.get("S1M_INTERNAL_TOKEN", "")

    def ask(self, item):
        body = json.dumps({"model": self.model, "state": item.state, "questions": {item.qid: item.question}}).encode()
        path = "/v1/systemone" if self.contract == "systemone" else "/internal/v1/decide"
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-S1M-Internal-Token"] = self.token
        req = urllib.request.Request(self.target.rstrip("/") + path, data=body, headers=headers)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                code, payload = r.status, json.load(r)
        except urllib.error.HTTPError as e:
            code, payload = e.code, {"error": e.read().decode("utf-8", "replace")[:300]}
        except Exception as e:  # noqa: BLE001
            code, payload = 0, {"error": f"{type(e).__name__}: {e}"[:300]}
        ms = (time.perf_counter() - t0) * 1000
        ans = (payload.get("answers") or {}).get(item.qid) if code == 200 and isinstance(payload, dict) else None
        if code == 200 and ans is None:
            code = 502
        return {"key": item.key, "http": code, "ms": ms, "answer": ans, "error": None if code == 200 else payload.get("error")}


def fetch(ep: Endpoint, items, workers, cache: str | None):
    """-> {key: record}. Records with http 200 or a deterministic rejection are kept in the cache file and not re-asked."""
    done = {}
    if ep.is_file:
        for line in open(ep.target):
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r
        return done
    if cache and Path(cache).exists():
        for line in open(cache):
            if line.strip():
                r = json.loads(line)
                if r["http"] == 200 or r["http"] in UNSUPPORTED:
                    done[r["key"]] = r
    todo = [it for it in items if it.key not in done]
    lock = threading.Lock()
    out = open(cache, "a") if cache else None

    def one(it):
        r = ep.ask(it)
        with lock:
            done[it.key] = r
            if out:
                out.write(json.dumps(r) + "\n")
                out.flush()
        return r

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    if out:
        out.close()
    return done


def status_counts(recs, items):
    c = {"answered": 0, "unsupported": 0, "errors": 0, "missing": 0}
    for it in items:
        r = recs.get(it.key)
        if r is None:
            c["missing"] += 1
        elif r["http"] == 200:
            c["answered"] += 1
        elif r["http"] in UNSUPPORTED:
            c["unsupported"] += 1
        else:
            c["errors"] += 1
    return c


def evaluate(items, ref, cand, reps=1000):
    pairs, by_type, gold_rows = [], {}, []
    for it in items:
        a, b = ref.get(it.key), cand.get(it.key)
        if not (a and b and a["http"] == 200 and b["http"] == 200):
            continue
        keys = sets.option_keys(it.question)
        pa, pb = stats.probs_dict(a["answer"], keys), stats.probs_dict(b["answer"], keys)
        if pa is None or pb is None:
            continue
        m = stats.compare_pair(pa, pb)
        pairs.append(m)
        by_type.setdefault(it.qtype, []).append(m)
        if it.gold in pa and it.gold in pb:
            gold_rows.append((it, stats.gold_scores(pa, it.gold), stats.gold_scores(pb, it.gold)))
    out = {"paired": stats.summarize(pairs), "by_type": {t: stats.summarize(v) for t, v in sorted(by_type.items())}}
    if gold_rows:
        ra, ca = [r[1][0] for r in gold_rows], [r[2][0] for r in gold_rows]
        rp, cp = [r[1][1] for r in gold_rows], [r[2][1] for r in gold_rows]
        grp = [r[0].req for r in gold_rows]
        out["gold"] = {"n": len(gold_rows), "ref_accuracy": sum(ra) / len(ra), "cand_accuracy": sum(ca) / len(ca),
                       "ref_mean_gold_prob": sum(rp) / len(rp), "cand_mean_gold_prob": sum(cp) / len(cp),
                       "accuracy_delta": stats.paired_bootstrap(ra, ca, grp, reps),
                       "gold_prob_delta": stats.paired_bootstrap(rp, cp, grp, reps),
                       "note": "accuracy/gold-probability proxy, not the official JevBench capability score"}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand")
    ap.add_argument("--fetch-only", action="store_true", help="only collect --ref responses into --ref-cache, then exit")
    ap.add_argument("--set", dest="set_name", required=True, choices=sets.SETS)
    ap.add_argument("--contract", default="systemone", choices=["systemone", "s1"])
    ap.add_argument("--ref-contract", choices=["systemone", "s1"], help="contract of the reference if different")
    ap.add_argument("--data-dir")
    ap.add_argument("--data-file", help="jsonl of {id,state,question} for --set s1")
    ap.add_argument("--limit", type=int, default=0, help="first N items (DI: N evenly spaced states, e.g. 1000)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ref-model", default="model")
    ap.add_argument("--cand-model", default="model")
    ap.add_argument("--ref-cache", help="jsonl: responses of the reference are appended here / read from here")
    ap.add_argument("--cand-cache")
    ap.add_argument("--reps", type=int, default=1000, help="bootstrap replicates")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    items = sets.load(a.set_name, a.data_dir, a.limit, a.data_file)
    ref = fetch(Endpoint(a.ref, a.ref_contract or a.contract, a.ref_model), items, a.workers, a.ref_cache)
    if a.fetch_only:
        print(json.dumps({"set": a.set_name, "n_items": len(items), "ref": status_counts(ref, items)}))
        return 0
    if not a.cand:
        ap.error("--cand is required unless --fetch-only")
    cand = fetch(Endpoint(a.cand, a.contract, a.cand_model), items, a.workers, a.cand_cache)
    rep = {"set": a.set_name, "contract": a.contract, "n_items": len(items), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "ref": {"target": a.ref if not a.ref.startswith("http") else "url", **status_counts(ref, items)},
           "cand": {"target": a.cand if not a.cand.startswith("http") else "url", **status_counts(cand, items)},
           **evaluate(items, ref, cand, a.reps)}
    text = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
