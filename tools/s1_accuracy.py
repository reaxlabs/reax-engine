#!/usr/bin/env python3
"""Accuracy of an s1_loadgen.py --dump against P300 gold (choice/noul/score argmax = expected). usage: s1_accuracy.py DUMP GOLD"""
import json
import sys

d, g = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ok, n = {}, {}
for i, gold in enumerate(g):
    p = d.get(str(i))
    if p is None:
        continue
    keys = list(p)
    best = max(p, key=p.get)
    want = {"yes": "true", "no": "false"}.get(gold["expected"], str(gold["expected"])) if gold["type"] == "noul" else str(gold["expected"])
    n[gold["type"]] = n.get(gold["type"], 0) + 1
    ok[gold["type"]] = ok.get(gold["type"], 0) + (best == want)
tot = sum(n.values())
print(json.dumps({"n": tot, "accuracy": round(sum(ok.values()) / tot, 4), **{t: round(ok[t] / n[t], 4) for t in n}, "answered": n}))
