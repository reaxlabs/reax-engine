"""Cross-question dependence: answer each question of N multi-question DI requests alone (single-question request) and inside
its full request (one at a time, c1), compare probabilities. In the unshared engine both are the same computation up to
batch noise; in shared-state mode an answer may depend on the other questions of its request.
usage: python3 tools/indep.py URL DI_ROWS DI_IDS [N=50]"""
import json, sys, urllib.request
url, rows_p, ids_p = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 50
ids = json.load(open(ids_p))["ids"]
step = len(ids) / 1000
sel = set(ids[int(k * step)] for k in range(1000))
rows = [r for r in map(json.loads, open(rows_p)) if r["_evaluation"]["run_id"] in sel and 2 <= len(r["questions"]) <= 16][:N]
def ask(state, qs):
    body = json.dumps({"model": "x", "state": state, "questions": qs}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", body, {"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=600).read())["answers"]
    except Exception:  # noqa: BLE001
        return None
def probs(a):
    return a.get("probabilities") or {"true": a.get("noul")}
n = flips = 0; dps = []
for r in rows:
    full = ask(r["state"], r["questions"])
    if full is None:
        continue
    for k, q in r["questions"].items():
        alone = ask(r["state"], {k: q})
        if alone is None:
            continue
        pa, pb = probs(alone[k]), probs(full[k])
        n += 1
        flips += max(pa, key=pa.get) != max(pb, key=pb.get)
        dps.append(max(abs(pa[x] - pb[x]) for x in pa))
dps.sort()
print(json.dumps({"requests": len(rows), "questions": n, "top1_flips_alone_vs_in_request": flips,
                  "mean_max_abs_dp": sum(dps) / max(1, n), "p95_max_abs_dp": dps[int(0.95 * len(dps))] if dps else None}))
