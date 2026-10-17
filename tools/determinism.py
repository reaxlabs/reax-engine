"""c1 determinism: send the same N DI requests (multi-question) one at a time, R times, to a running /v1/systemone server;
report how many answers are bit-identical across repeats and the max probability difference.
usage: python3 tools/determinism.py URL DI_ROWS DI_IDS [N=60] [R=3]"""
import json, sys, urllib.request
url, rows_p, ids_p = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 60
R = int(sys.argv[5]) if len(sys.argv) > 5 else 3
ids = json.load(open(ids_p))["ids"]
step = len(ids) / 1000
sel = [ids[int(k * step)] for k in range(0, 1000, max(1, 1000 // N))][:N]
want = set(sel)
rows = [r for r in map(json.loads, open(rows_p)) if r["_evaluation"]["run_id"] in want]
def ask(r):
    body = json.dumps({"model": "x", "state": r["state"], "questions": r["questions"]}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", body, {"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=600).read())["answers"]
    except Exception as e:  # noqa: BLE001  (422 = unsupported, skipped)
        return None
runs = [[ask(r) for r in rows] for _ in range(R)]
n = same = 0; maxd = 0.0; nmulti = sum(len(r["questions"]) > 1 for r in rows)
for i in range(len(rows)):
    if any(run[i] is None for run in runs):
        continue
    for k in runs[0][i]:
        n += 1
        ps = [run[i][k].get("probabilities") or {"noul": run[i][k].get("noul")} for run in runs]
        same += all(p == ps[0] for p in ps)
        maxd = max([maxd] + [abs(p[x] - ps[0][x]) for p in ps for x in ps[0]])
print(json.dumps({"requests": len(rows), "multi_question_requests": nmulti, "repeats": R, "answers": n,
                  "bit_identical_answers": same, "max_abs_dp": maxd}))
