#!/usr/bin/env python3
"""Compare a candidate's per-decision probabilities to the reference: top-1 agreement, |dp| stats, by source.
Usage: compare.py REF.jsonl CAND.jsonl [--json]"""
import json, sys


def load(p):
    return {r["id"]: r for r in (json.loads(l) for l in open(p) if l.strip())}


ref, cand = load(sys.argv[1]), load(sys.argv[2])
ids = [i for i in ref if i in cand]
out = {"n": len(ids), "missing": len(ref) - len(ids)}
for name, sel in (("all", ids), ("p300", [i for i in ids if ref[i]["src"] == "p300"]), ("di", [i for i in ids if ref[i]["src"] == "di"])):
    if not sel:
        continue
    agree, dps, maxes, tokdiff = 0, [], [], 0
    for i in sel:
        p, q = ref[i]["probs"], cand[i]["probs"]
        agree += max(range(len(p)), key=p.__getitem__) == max(range(len(q)), key=q.__getitem__)
        d = [abs(x - y) for x, y in zip(p, q)]
        dps.extend(d)
        maxes.append(max(d))
        tokdiff += ref[i].get("tokens") != cand[i].get("tokens")
    maxes.sort()
    out[name] = {"n": len(sel), "top1_agree": agree / len(sel), "mean_abs_dp": sum(dps) / len(dps),
                 "p99_max_dp": maxes[int(0.99 * (len(maxes) - 1))], "max_dp": maxes[-1],
                 "frac_maxdp_gt_0.005": sum(m > 0.005 for m in maxes) / len(maxes), "token_count_mismatch": tokdiff}
print(json.dumps(out, indent=None if "--json" in sys.argv else 1))
