"""Evaluation sets as flat lists of single-question items.

Data files are NOT part of this repository. Point --data-dir (or $REAX_BENCH_DATA) at a directory holding:

  p300     p300-input.jsonl  rows {task_id, state, question, labels}   (300 open-split items of JevBench v1.6)
           p300-gold.jsonl   rows {opaque_id (= task_id), type, labels, expected}
  public231  jevbench_public231.jsonl  rows {id, state, question}      (231 public text items, no gold)
  di       di5k-rows.jsonl   rows {id, state, questions{qid: question}, expected{qid: gold}, _evaluation{run_id}}
           di5k-ids.json     {"ids": [run_id, ...]}  (order used to pick the DI-N subset)
  s1       any jsonl of rows {id, state, question} (e.g. generated s1 items); give the file with --data-file

These are the public JevBench / Decision Index samples distributed by their owners; this repo only reads them.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

SETS = ("p300", "public231", "di", "s1")


@dataclass
class Item:
    key: str            # unique id, "<run>::<qid>" for DI
    state: object
    qid: str
    question: dict
    gold: str | None = None   # gold option key in our key space (noul: "true"/"false"; score: "0".."k-1")
    req: str = ""             # items with the same req share a state (one /v1/systemone request in production)
    src: str = ""

    @property
    def qtype(self):
        return self.question.get("type")


def option_keys(q: dict) -> list[str]:
    """Option keys of a question in the canonical order choice: criteria order, score: "0".., noul: false,true."""
    t = q.get("type")
    if t == "choice":
        c = q.get("criteria")
        return [str(k) for k in (c if isinstance(c, (dict, list)) else [])]
    if t == "score":
        return [str(i) for i in range(len(q.get("criteria") or []))]
    return ["false", "true"]


def norm_gold(q: dict, expected) -> str | None:
    if expected is None:
        return None
    if q.get("type") == "noul":
        return {True: "true", False: "false", "yes": "true", "no": "false", "true": "true", "false": "false"}.get(
            expected if not isinstance(expected, str) else expected.lower())
    g = str(expected)
    return g if g in option_keys(q) else None


def _jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def data_dir(arg=None) -> Path:
    d = arg or os.environ.get("REAX_BENCH_DATA")
    if not d:
        raise SystemExit("no data directory: pass --data-dir or set REAX_BENCH_DATA (see bench/sets.py for the file layout)")
    return Path(d)


def load(name: str, data=None, limit: int = 0, file=None) -> list[Item]:
    items: list[Item]
    if name == "p300":
        d = data_dir(data)
        gold = {g["opaque_id"]: g for g in _jsonl(d / "p300-gold.jsonl")} if (d / "p300-gold.jsonl").exists() else {}
        items = []
        for r in _jsonl(d / "p300-input.jsonl"):
            g = gold.get(r["task_id"])
            items.append(Item(r["task_id"], r["state"], "q", r["question"], norm_gold(r["question"], g and g.get("expected")),
                              req=r["task_id"], src="p300"))
    elif name in ("public231", "s1"):
        path = Path(file) if file else data_dir(data) / "jevbench_public231.jsonl"
        items = [Item(r["id"], r["state"], "q", r["question"], norm_gold(r["question"], r.get("expected")), req=r["id"], src=name)
                 for r in _jsonl(path)]
    elif name == "di":
        d = data_dir(data)
        rows = _jsonl(d / "di5k-rows.jsonl")
        by_id = {r["_evaluation"]["run_id"]: r for r in rows}
        if (d / "di5k-ids.json").exists():
            ids = json.loads((d / "di5k-ids.json").read_text())["ids"]
        else:
            ids = list(by_id)
        n = limit or len(ids)
        step = len(ids) / n
        sel = [ids[int(k * step)] for k in range(n)]   # evenly spaced, identical to the DI-N subset of earlier runs
        items = []
        for rid in sel:
            r = by_id[rid]
            for qid, q in r["questions"].items():
                items.append(Item(f"{rid}::{qid}", r["state"], qid, q, norm_gold(q, (r.get("expected") or {}).get(qid)),
                                  req=rid, src="di"))
        return items
    else:
        raise SystemExit(f"unknown set {name!r}; one of {SETS}")
    return items[:limit] if limit else items
