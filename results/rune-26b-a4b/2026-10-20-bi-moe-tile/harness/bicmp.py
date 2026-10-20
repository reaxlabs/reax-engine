"""bicmp.py DIR...: per arm, bitwise compare of the P300 answer files (w32 / w1 / w32), plus load numbers."""
import glob, json, sys
def load(f):
    return {r["key"]: json.dumps(r["answer"], sort_keys=True) for r in map(json.loads, open(f)) if r.get("http") == 200}
for d in sys.argv[1:]:
    fs = sorted(glob.glob(f"{d}/p300-w*.jsonl"), key=lambda f: int(f.rsplit("-", 1)[1].split(".")[0]))
    a = [load(f) for f in fs]
    keys = set(a[0]) & set(a[1]) & set(a[2])
    diff = lambda x, y: sum(x[k] != y[k] for k in keys)
    loads = []
    for s in ("p300", "di"):
        try:
            loads += [(s, l["concurrency"], l["decisions_per_s"], l["req_p50_ms"]) for l in json.load(open(f"{d}/load-{s}.json"))["levels"]]
        except Exception: pass
    print(d.rsplit("/", 1)[1], "n", len(keys), "w32a!=w1", diff(a[0], a[1]), "w32a!=w32b", diff(a[0], a[2]), loads)
