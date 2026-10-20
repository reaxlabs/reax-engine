import json, sys
def ans(f): return {r["key"]: json.dumps(r["answer"], sort_keys=True) for r in map(json.loads, open(f)) if r.get("http") == 200}
O = "/workspace/out"
for a in sys.argv[1].split(","):
    row = [a]
    for s in ("p300", "di"):
        row += [(l["concurrency"], l["decisions_per_s"], l["req_p50_ms"]) for l in json.load(open(f"{O}/{a}/load-{s}.json"))["levels"]]
    print(*row)
for pair in sys.argv[2:]:
    x, y = pair.split(":")
    for s in ("p300", "di"):
        A, B = ans(f"{O}/{x}/{s}.jsonl"), ans(f"{O}/{y}/{s}.jsonl"); k = set(A) & set(B)
        print(pair, s, "n", len(k), "bitwise-different", sum(A[i] != B[i] for i in k))
