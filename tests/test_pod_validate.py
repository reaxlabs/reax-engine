"""Dry run of bench/pod_validate.sh with stand-in docker/nvidia-smi and a fake decision server (no GPU, no weights)."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("curl") is None, reason="needs bash and curl")


def write(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_data(d):
    q1 = {"type": "choice", "instructions": "i", "criteria": {"a": "x", "b": "y"}}
    q2 = {"type": "noul", "instructions": "i"}
    write(d / "p300-input.jsonl", [{"task_id": f"t{i}", "state": f"s{i}", "question": q1 if i % 2 else q2, "labels": []} for i in range(6)])
    write(d / "p300-gold.jsonl", [{"opaque_id": f"t{i}", "expected": "a" if i % 2 else "yes"} for i in range(6)])
    write(d / "di5k-rows.jsonl", [{"id": f"r{i}", "state": f"st{i}", "questions": {"u": q1, "v": q2}, "expected": {"u": "a", "v": True},
                                   "_evaluation": {"run_id": f"run{i}"}} for i in range(8)])
    (d / "di5k-ids.json").write_text(json.dumps({"ids": [f"run{i}" for i in range(8)]}))


@pytest.mark.parametrize("model,contract", [("quyet-1.0-large", "systemone"), ("rune-26b-a4b", "s1")])
def test_dry_run(tmp_path, model, contract):
    data, work, stub = tmp_path / "data", tmp_path / "work", tmp_path / "stub"
    data.mkdir()
    make_data(data)
    rev = json.loads(subprocess.check_output(["python3", "-m", "engine.registry", model], cwd=ROOT))["revision"]
    (work / "weights" / model).mkdir(parents=True)
    (work / "weights" / model / f".complete-{rev}").touch()
    env = dict(os.environ, PATH=f"{ROOT / 'tests' / 'stubs'}:{os.environ['PATH']}", STUB_DIR=str(stub),
               FAKE_SERVER=str(ROOT / "tests" / "fake_decide_server.py"), WORK=str(work), REAX_BENCH_DATA=str(data),
               CONC="2", LOAD_SECONDS="0.3", DI_LIMIT="4", REAX_SUBNET_DIR=str(tmp_path), MIN_FREE_GB="1", RESULTS_DIR=str(tmp_path / "results"))
    r = subprocess.run(["bash", str(ROOT / "bench" / "pod_validate.sh"), model], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    out = tmp_path / "results" / model
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    res = json.loads(next(out.glob("*.json")).read_text())
    assert res["model"] == model and res["env"]["gpu"].startswith("Fake GPU")
    assert res["headline"]["p300_top_choice_agreement"] == 1.0
    assert res["agreement"]["di"]["paired"]["n"] == 8
    assert set(res["headline"]["speedup_by_concurrency_p300"]) == {"1", "2"}
    calls = (stub / "calls.log").read_text()
    assert "--read-only" in calls and "--cap-drop ALL" in calls and "OPENAI_API_KEY=" in calls and ":/model:ro" in calls
    assert ("RUNE_PORT=8001" in calls) == (model == "rune-26b-a4b")
    assert f"REAX_MODEL={model}" in calls
