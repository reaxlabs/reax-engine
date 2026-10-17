#!/usr/bin/env python3
"""Pack runs/ (pod A) and runs-podB/ into repo/results/pplx-decider-v1.2/2026-10-17/: summary.json (every run: config,
dec/s, c1 latency, paired comparisons) + raw/<pod>/<run>/ (small files, gzipped DI/P300 responses and server log)."""
import gzip, json, shutil
from pathlib import Path

J = Path(__file__).resolve().parent.parent
OUT = J / "repo/results/pplx-decider-v1.2/2026-10-17"
SMALL = ("args.txt", "env.txt", "caches.txt", "gpu.txt", "commit.txt", "startup.txt", "health_start.json", "health_end.json",
         "latency.json", "p300.jsonl.run.json", "determinism.json", "di-partial.json")


def main():
    summary = {"models": {"27b": "perplexity-ai/pplx-decider-v1.2@225c913ddc8761a8264f04f64548e1f97e443eb3",
                          "flash": "perplexity-ai/pplx-decider-flash-v1.2@6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d",
                          "proxy_v1.1": "perplexity-ai/pplx-decider-v1.1-27b@6195aa55a72ae21e503c8a60198f7b390d42adb5"},
               "gpu": "1x RTX PRO 6000 Blackwell Server Edition (96 GB)",
               "pods": {"A": "dsvlv19xcsttee (RunPod SECURE)", "B": "mx84h42rt37389 (RunPod COMMUNITY)"},
               "naming": "a12* author v1.2 fast path (a12L = recompile limit re-applied in the serving thread), f12* author Flash, "
                         "r12* REAX on v1.2 27B, rf12* REAX on Flash, reax-* REAX v1.1 engine on v1.1 weights (Phase 0 proxy)",
               "runs": {}}
    if OUT.exists():
        shutil.rmtree(OUT)
    for pod, base in (("A", J / "runs"), ("B", J / "runs-podB")):
        if not base.exists():
            continue
        for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name != "exact"):
            r = {"pod": pod}
            if (d / "args.txt").exists():
                r["args"] = (d / "args.txt").read_text().strip()
            for sub in ("di-native",):
                if (d / sub / "run.json").exists():
                    rj = json.loads((d / sub / "run.json").read_text())
                    r["di"] = {k: rj.get(k) for k in ("rows", "questions", "wall_s", "decisions_per_s", "workers", "status")}
            if (d / "latency.json").exists():
                r["c1_latency"] = json.loads((d / "latency.json").read_text())
            if (d / "determinism.json").exists():
                r["determinism"] = json.loads((d / "determinism.json").read_text())
            r["paired"] = {}
            for c in sorted(d.glob("compare-vs-*.json")):
                cj = json.loads(c.read_text())
                r["paired"][c.stem.removeprefix("compare-vs-")] = {k: cj[k] for k in ("di5k", "p300") if k in cj}
            summary["runs"][d.name] = r
            o = OUT / "raw" / pod / d.name
            o.mkdir(parents=True, exist_ok=True)
            for f in SMALL + tuple(x.name for x in d.glob("compare-vs-*.json")):
                if (d / f).exists():
                    shutil.copy(d / f, o / f)
            for src, dst in (("di-native/results.jsonl", "di-results.jsonl.gz"), ("p300.jsonl", "p300.jsonl.gz"),
                             ("server.log", "server.log.gz")):
                if (d / src).exists():
                    with open(d / src, "rb") as fi, gzip.open(o / dst, "wb") as fo:
                        shutil.copyfileobj(fi, fo)
    ex = J / "runs/exact"
    if ex.exists():
        o = OUT / "exactness"
        o.mkdir(parents=True, exist_ok=True)
        for f in ("exact.out", "exactM-m12.out", "exactM-m12f.out"):
            if (ex / f).exists():
                lines = [l for l in (ex / f).read_text().splitlines() if l.startswith("{") or l.startswith("author logits")]
                (o / f).write_text("\n".join(lines) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(summary['runs'])} runs -> {OUT}")


if __name__ == "__main__":
    main()
