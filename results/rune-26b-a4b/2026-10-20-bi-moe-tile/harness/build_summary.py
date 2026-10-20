"""Builds summary.json from the per-arm run directories (runs/<arm>/{p300,di}.jsonl, load-*.json). Usage: build_summary.py RUNS_DIR OUT.json
answers_sha256 = sha256 of the sorted lines "key<TAB>json(answer, sort_keys)" over HTTP-200 rows. fixed_workload = DI-1000 fetch
(8 workers, all 5,129 questions) wall time = mtime(di.jsonl) - mtime(p300.jsonl) (the fetches run back to back)."""
import hashlib, json, os, sys
R, OUT = sys.argv[1], sys.argv[2]
def ans(f): return {r["key"]: json.dumps(r["answer"], sort_keys=True) for r in map(json.loads, open(f)) if r.get("http") == 200}
arms = {"rA1": "deterministic (unpatched)", "rA2": "deterministic (unpatched)", "rB1": "deterministic + REAX_BI_MOE_CONFIG (manual env, pod patch)",
        "rB2": "deterministic + REAX_BI_MOE_CONFIG (manual env, pod patch)", "rC": "deterministic, this repo's vllm_patch module + preset env",
        "fA1": "fp8-kv-seqs64 (unpatched)", "fB1": "fp8-kv-seqs64 + REAX_BI_MOE_CONFIG (manual env)"}
out = {"date": "2026-10-20", "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition 97,887 MiB, driver 595.91.07, RunPod SECURE pod hpxzhvybmcv8p8",
       "vllm": "0.30.0 (vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90), torch 2.13.0+cu130",
       "model": "surogate/rune-26b-a4b-GGUF@c6b360d47895bb77bdf3805a13ee5a5557ab1921 (bf16 safetensors)",
       "harness": "harness/arm.sh: engine.serve (s1 contract) spawning vLLM; bench.agreement --fetch-only P300 then DI-1000 (8 workers); bench.loadgen 1,32 clients x 90 s on P300 and DI-1000; fresh Triton cache per arm",
       "order": ["rA1", "rB1", "rA2", "rB2", "rC", "fA1", "fB1"],
       "tile": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "SPLIT_K": 1, "num_warps": 8, "num_stages": 3},
       "arms": {}, "bitwise_answer_comparison": {}}
for a, desc in arms.items():
    d = f"{R}/{a}"; row = {"config": desc}
    for s in ("p300", "di"):
        for l in json.load(open(f"{d}/load-{s}.json"))["levels"]:
            row[f"load_{s}_c{l['concurrency']}"] = {k: l.get(k) for k in ("decisions_per_s", "requests", "wall_s", "req_p50_ms", "req_p95_ms", "http_codes")}
        row[f"{s}_answers_sha256"] = hashlib.sha256("\n".join(sorted(f"{k}\t{v}" for k, v in ans(f"{d}/{s}.jsonl").items())).encode()).hexdigest()
    w = os.path.getmtime(f"{d}/di.jsonl") - os.path.getmtime(f"{d}/p300.jsonl"); n = len(ans(f"{d}/di.jsonl"))
    row["fixed_workload_di1000_8clients"] = {"answered": n, "wall_s": round(w, 1), "decisions_per_s": round(n / w, 2)}
    out["arms"][a] = row
for x, y in [("rA1", "rA2"), ("rA1", "rB1"), ("rA1", "rB2"), ("rA1", "rC"), ("fA1", "fB1"), ("rA1", "fA1")]:
    for s in ("p300", "di"):
        A, B = ans(f"{R}/{x}/{s}.jsonl"), ans(f"{R}/{y}/{s}.jsonl"); k = set(A) & set(B)
        out["bitwise_answer_comparison"][f"{x}-vs-{y}-{s}"] = {"n": len(k), "different": sum(A[i] != B[i] for i in k)}
json.dump(out, open(OUT, "w"), indent=1)
