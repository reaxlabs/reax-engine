#!/usr/bin/env python3
"""Agreement of two answer dumps (s1_loadgen.py --dump): top-choice agreement and the competition's logit distance
(temperature * max |centred log p difference| per item). usage: s1_compare.py REF.json CAND.json [--gold GOLD.json]"""
import json
import math
import sys

T = 2.07


def centred(p):
    ls = [math.log(max(x, 1e-30)) for x in p]
    m = sum(ls) / len(ls)
    return [x - m for x in ls]


def dist(a, b):
    return T * max(abs(x - y) for x, y in zip(centred(a), centred(b)))


def argmax(p):
    return max(range(len(p)), key=p.__getitem__)


ref, cand = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ids = sorted(set(ref) & set(cand), key=int)
ds, agree = [], 0
for i in ids:
    a, b = list(ref[i].values()), list(cand[i].values())
    ds.append(dist(a, b))
    agree += argmax(a) == argmax(b)
print(json.dumps({"n": len(ids), "top_agreement": round(agree / len(ids), 4), "mean_distance": round(sum(ds) / len(ds), 5),
                  "max_distance": round(max(ds), 4), "p99_distance": round(sorted(ds)[int(0.99 * (len(ds) - 1))], 4)}))
