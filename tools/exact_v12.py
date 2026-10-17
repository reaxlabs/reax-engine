#!/usr/bin/env python3
"""Exactness check of the v1.2 joint port: author reference path vs PplxEngine.forward_marked on the same requests.

  exact_v12.py author MODEL ROWS N OUT.json   # author Decider (fast=False = the evaluated computation), one request per call
  exact_v12.py engine MODEL ROWS N OUT.json [--linear bf16|fp8]  # engine, then compares with OUT.json's author logits
ROWS: di5k-rows.jsonl (request rows with state + questions; the first N with >= 1 question and <= 8192 tokens are used).
Prints argmax agreement and probability drift (temperature-scaled softmax over valid options).
"""
import json, os, sys
from pathlib import Path

import torch


def rows(path, n):
    out = []
    for line in open(path):
        r = json.loads(line)
        req = r.get("request") or r.get("payload") or r
        multi = os.environ.get("EXACT_MULTI") == "1"  # only requests with >= 2 questions (joint readouts)
        if isinstance(req, dict) and "state" in req and isinstance(req.get("questions"), dict) and \
                len(req["questions"]) >= (2 if multi else 1):
            out.append({"state": req["state"], "questions": req["questions"]})
        if len(out) >= 4 * n:
            break
    return out


def main():
    mode, model, path, n, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])
    sys.path.insert(0, str(Path(model) / "source" / "src"))
    from pplx_decider import Decider
    if mode == "author":
        d = Decider.from_pretrained(model, device="cuda", fast=False, max_length=8192)
        res, keep = [], []
        for req in rows(path, n):
            try:
                plan = d.encoder.plan(req, 8192)
            except Exception:  # noqa: BLE001
                continue
            if len(plan) != 1:
                continue
            res.append({"request": req, "logits": d.logits([req])[0]})
            if len(res) >= n:
                break
        out.write_text(json.dumps({"temperature": d.temperature, "items": res}))
        print(f"author logits for {len(res)} requests -> {out}")
        return
    linear = sys.argv[sys.argv.index("--linear") + 1] if "--linear" in sys.argv else "bf16"
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
    from transformers import AutoProcessor
    from pplx_decider.protocol import Encoder
    from pplx_engine import PplxEngine
    ref = json.loads(out.read_text())
    cfg = json.loads((Path(model) / "decision_config.json").read_text())
    enc = Encoder(AutoProcessor.from_pretrained(model), cfg["codes"])
    eng = PplxEngine(model, linear_mode=linear, compile=True, gva_native=True)
    t = ref["temperature"]
    n_q = same = 0
    drift, worst = [], 0.0
    for single in (True, False):  # one request per step, then all requests packed into steps of <= 16k tokens
        n_q = same = 0
        drift, worst = [], 0.0
        chunks = [enc.plan(it["request"], 8192)[0] for it in ref["items"]]
        groups, cur, tot = [], [], 0
        for i, c in enumerate(chunks):
            if cur and (single or tot + c.length > 16384):
                groups.append(cur)
                cur, tot = [], 0
            cur.append(i)
            tot += c.length
        groups.append(cur)
        got = {}
        for g in groups:
            ids = torch.cat([chunks[i].input_ids for i in g]).cuda()
            lg = eng.forward_marked(ids, [chunks[i].length for i in g], [chunks[i].markers.tolist() for i in g]).cpu()
            k = 0
            for i in g:
                got[i] = [lg[k + j, :cnt] for j, cnt in enumerate(chunks[i].counts)]
                k += len(chunks[i].counts)
        for i, it in enumerate(ref["items"]):
            for j, key in enumerate(chunks[i].keys):
                a = (torch.tensor(it["logits"][key]) / t).softmax(-1)
                b = (got[i][j].float() / t).softmax(-1)
                n_q += 1
                same += int(a.argmax() == b.argmax())
                dd = (a - b).abs().max().item()
                drift.append(dd)
                worst = max(worst, dd)
        print(json.dumps({"linear": linear, "packed": not single, "requests": len(chunks), "questions": n_q,
                          "argmax_agree": same / n_q, "mean_max_abs_dp": sum(drift) / len(drift), "max_abs_dp": worst}))


if __name__ == "__main__":
    main()
