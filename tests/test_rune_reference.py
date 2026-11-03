"""CPU checks of the ref-rune image (s1-pro pool model, Surogate Rune 26B-A4B v3): flags, manifest, prompt/answer parity with the
pool's reference adapter, entrypoint verification, Dockerfile. No GPU, no weights, no network."""
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from engine import registry
from engine.backends import vllm_readout as V
from tests.fake_tok import FakeTok

ROOT = Path(__file__).resolve().parent.parent
UP = ROOT / "upstream" / "rune"
MANIFEST = json.loads((ROOT / "manifests" / "s1-pro-v1.json").read_text())


def test_registry_reference_options():
    r = registry.get("rune-26b-a4b-v3")
    assert r.id == "rune-26b-a4b" and registry.get("s1-pro") is r and registry.get("rune") is r
    assert r.production_opts == {"preset": "ref-pool", "max_input_tokens": 4096, "max_options": 64}
    assert (r.hf_repo, r.revision) == (MANIFEST["weights"]["hf_repo"], MANIFEST["weights"]["revision"])
    cfg = V.load_family_config("rune", None, max_options=64, max_prompt_tokens=4096)
    assert cfg.temperatures == {"choice": 2.0, "score": 2.0, "noul": 2.0}      # POOL_PARAMS s1-pro temperature=2.0
    assert (cfg.max_options, cfg.max_prompt_tokens) == (MANIFEST["limits"]["max_options"], MANIFEST["limits"]["max_input_tokens"])


def test_manifest_is_the_pool_manifest():
    files = MANIFEST["weights"]["files"]
    assert len(files) == 18 and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in files.values())
    assert sum(n.endswith(".safetensors") for n in files) == 11
    assert MANIFEST["model"] == "s1-pro" and MANIFEST["readout"]["temperature"] == 2.0


def _flags(argv):
    """argv -> {flag: value or True}; a value is the next token unless it starts with '-' followed by a letter or dash."""
    out, i = {}, 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("-") and not re.fullmatch(r"-\d+", a):
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                out[a] = argv[i + 1]; i += 2
            else:
                out[a] = True; i += 1
        else:
            i += 1
    return out


def test_ref_pool_flags_equal_pool_start_sh_and_manifest():
    sh = (UP / "start.sh").read_text()
    # the pool's deterministic defaults, read from the pool script itself
    maxlen = re.search(r'MAXLEN=\$\{RUNE_MAX_MODEL_LEN:-(\d+)\}', sh).group(1)
    seqs = re.search(r'--max-num-seqs "\$\{RUNE_MAX_NUM_SEQS:-(\d+)\}"', sh).group(1)
    util = re.search(r'--gpu-memory-utilization "\$\{RUNE_GPU_MEM_UTIL:-([\d.]+)\}"', sh).group(1)
    assert "export VLLM_BATCH_INVARIANT=1" in sh and "RUNE_DETERMINISTIC:-1" in sh
    assert 'DET_ARGS=(--no-enable-chunked-prefill --max-num-batched-tokens "$MAXLEN" -O0)' in sh
    expect = {"--max-model-len": maxlen, "--max-num-batched-tokens": maxlen, "--max-num-seqs": seqs, "--gpu-memory-utilization": util,
              "--logprobs-mode": "processed_logits", "--max-logprobs": "300", "--mm-processor-cache-gb": "0",
              "--limit-mm-per-prompt": '{"image":1}', "--no-enable-chunked-prefill": True, "-O0": True,
              "--no-enable-prefix-caching": True}
    for k, v in expect.items():   # every expected flag is literally in the pool script
        assert k in sh, k
    p = V.preset_for("rune", "ref-pool")
    got = _flags(list(p.args))
    chat = got.pop("--chat-template")
    assert chat == V.RUNE_POOL_CHAT_TEMPLATE and f"--chat-template /opt/rune/chat_template_nospace.jinja" in sh
    assert got == expect, (got, expect)
    assert (maxlen, seqs, util) == ("32768", "64", "0.85")
    assert p.env == {"VLLM_BATCH_INVARIANT": "1"}
    assert MANIFEST["runtime"]["env"] == p.env
    man = _flags(MANIFEST["runtime"]["args"])
    for k in ("--max-model-len", "--max-num-batched-tokens", "--max-num-seqs", "--gpu-memory-utilization", "--logprobs-mode",
              "--max-logprobs", "-O0", "--no-enable-prefix-caching", "--no-enable-chunked-prefill"):
        assert man[k] == expect[k], k
    assert registry.get("rune").production_opts["preset"] == "ref-pool"
    argv = V.vllm_argv(p, "/model")
    assert argv[:3] == ["vllm", "serve", "/model"]


def _load_pool_adapter(monkeypatch):
    """Import the pool's reference adapter (upstream/rune/rune_adapter.py) with stubbed heavy deps and the character FakeTok."""
    for name in ("httpx", "uvicorn", "fastapi", "fastapi.responses", "transformers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    class App:
        def get(self, *a, **k): return lambda f: f
        def post(self, *a, **k): return lambda f: f
        def on_event(self, *a, **k): return lambda f: f

    sys.modules["fastapi"].FastAPI = App
    sys.modules["fastapi"].Request = object
    sys.modules["fastapi.responses"].JSONResponse = dict
    tf = sys.modules["transformers"]
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *_a, **_k: FakeTok())
    monkeypatch.setattr(sys, "argv", ["rune_adapter.py", "--allow-no-auth"])
    spec = importlib.util.spec_from_file_location("pool_rune_adapter", UP / "rune_adapter.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


QUESTIONS = [
    {"type": "choice", "instructions": "Pick one.", "criteria": {"x": "first", "y": None, "z": {"k": 1}}},
    {"type": "noul", "instructions": "Is it?", "criteria": {"true": "yes it is"}},
    {"type": "noul"},
    {"type": "noul", "instructions": "ü", "criteria": {"false": "nö", "true": "jä"}},
    {"type": "score", "instructions": "How much?", "criteria": ["none", "some", "lots"]},
    {"type": "choice", "criteria": {"a": "A", "b": "B"}},
]
STATES = [{"a": 1}, "raw string", None, {"ü": ["x", {"y": 2}]}, ""]


@pytest.mark.parametrize("q", QUESTIONS)
@pytest.mark.parametrize("state", STATES)
def test_prompt_and_answer_equal_the_pool_adapter(monkeypatch, q, state):
    ad = _load_pool_adapter(monkeypatch)
    enc = V.ReadoutEncoder(FakeTok(), V.load_family_config("rune", None, max_options=64, max_prompt_tokens=4096))
    system, branch, ids = ad.render(q)
    state_text = "SHARED STATE (JSON string):\n" + json.dumps(state, ensure_ascii=False) + "\n\n"
    want = FakeTok().apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": state_text + branch}],
                                                      add_generation_prompt=True, tokenize=False, enable_thinking=False)
    item = enc.encode(state, q)
    assert FakeTok().decode(item.payload.ids) == want
    assert item.payload.letter_ids == ids and item.payload.temperature == ad.args.temperature == 2.0
    assert enc.options(q) == ad.options(q)[0]
    z = [0.3 * i - 0.2 for i in range(len(ids))]
    probs = V.softmax_t(z, item.payload.temperature)
    assert enc.answer(q, probs) == ad.answer(q, z)       # same softmax, same answer shape (exact, float64)


def test_two_letter_codes_past_26_options(monkeypatch):
    ad = _load_pool_adapter(monkeypatch)
    q = {"type": "choice", "instructions": "many", "criteria": {f"k{i}": f"o{i}" for i in range(30)}}
    system, branch, _ = ad.render(q)   # FakeTok has no single-token two-letter codes: the codebook itself needs the real tokenizer (GPU gate)
    assert system == V.RUNE_SYSTEM_EXT
    assert "Answer with one option code only." in branch and V.RUNE_SYSTEM_EXT.count("option code") == 1


def test_input_limit_is_4096_tokens(monkeypatch):
    enc = V.ReadoutEncoder(FakeTok(), V.load_family_config("rune", None, max_options=64, max_prompt_tokens=4096))
    with pytest.raises(ValueError, match="maximum context length"):
        enc.encode("w" * 5000, QUESTIONS[0])


def _run_entrypoint(tmp_path, files, manifest_files, **env):
    d = tmp_path / "model"
    d.mkdir()
    for n, c in files.items():
        (d / n).write_bytes(c)
    man = tmp_path / "man.json"
    man.write_text(json.dumps({"weights": {"files": manifest_files}}))
    e = {"PATH": os.environ["PATH"], "PYTHONPATH": str(ROOT), "REAX_MODEL": "rune-26b-a4b-v3", "REAX_PROFILE": "reference",
         "REAX_MODEL_DIR": str(d), "REAX_VLLM_PORT": "9", "REAX_S1_MANIFEST": str(man), **env}
    return subprocess.run(["bash", str(ROOT / "docker/vllm-readout/entrypoint.sh")], env=e, capture_output=True, text=True, timeout=120)


def test_entrypoint_refuses_weights_that_do_not_match_the_manifest(tmp_path):
    h = hashlib.sha256(b"config").hexdigest()
    bad = _run_entrypoint(tmp_path, {"config.json": b"config", "w.safetensors": b"tampered"},
                          {"config.json": h, "w.safetensors": hashlib.sha256(b"original").hexdigest()})
    assert bad.returncode == 2 and "do not match the release manifest" in bad.stdout
    (tmp_path / "x").mkdir()
    missing = _run_entrypoint(tmp_path / "x", {"config.json": b"config"}, {"config.json": h, "w.safetensors": hashlib.sha256(b"x").hexdigest()})
    assert missing.returncode == 2 and "do not match" in missing.stdout


def test_entrypoint_accepts_matching_weights_and_goes_on_to_vllm(tmp_path):
    ok = _run_entrypoint(tmp_path, {"config.json": b"config", "w.safetensors": b"data"},
                         {"config.json": hashlib.sha256(b"config").hexdigest(), "w.safetensors": hashlib.sha256(b"data").hexdigest()})
    assert "verifying Rune weights" in ok.stdout and "do not match" not in ok.stdout
    assert ok.returncode == 3 and "vLLM died during startup" in ok.stdout      # no vllm binary here: the next stage was reached


def test_entrypoint_chooses_ref_pool_for_the_reference_profile():
    sh = (ROOT / "docker/vllm-readout/entrypoint.sh").read_text()
    assert re.search(r'FAMILY" = rune \]; then REAX_PRESET=ref-pool', sh)
    for alias in ("rune", "s1-pro", "rune-26b-a4b", "rune-26b-a4b-v3"):
        assert alias in re.search(r"(rune\|[^)]*\)) FAMILY=rune", sh).group(1)


def test_dockerfile_and_workflow():
    df = (ROOT / "docker/ref-rune/Dockerfile").read_text()
    other = (ROOT / "docker/ref-jeffan/Dockerfile").read_text()
    digest = re.search(r"sha256:[0-9a-f]{64}", df).group(0)
    assert digest == re.search(r"sha256:[0-9a-f]{64}", other).group(0) == "sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90"
    for src in re.findall(r"^COPY (\S+(?: \S+)*) \S+$", df, re.M):
        for s in src.split(" "):
            assert (ROOT / s).exists(), s
    assert "REAX_MODEL=rune-26b-a4b-v3" in df and "REAX_PROFILE=reference" in df and 'ai.reax.engine.models="rune,s1-pro"' in df
    assert "/licenses/rune/" in df and "weights" in df.lower()
    wf = (ROOT / ".github/workflows/images.yml").read_text()
    assert "docker/ref-rune/Dockerfile" in wf and 'names: "rune"' in wf
    sums = dict(l.split()[::-1] for l in (UP / "SOURCE-SHA256SUMS").read_text().splitlines())
    for n, h in sums.items():
        assert hashlib.sha256((UP / n).read_bytes()).hexdigest() == h, n
    assert "Apache License" in (UP / "LICENSE").read_text()


def test_gpu_memory_share_override(capsys, monkeypatch):
    monkeypatch.setenv("REAX_GPU_MEM_UTIL", "0.62")
    assert V.main(["rune", "ref-pool", "/model"]) == 0
    out = capsys.readouterr().out
    assert "--gpu-memory-utilization 0.62" in out and "0.85" not in out and "VLLM_BATCH_INVARIANT=1" in out
    monkeypatch.delenv("REAX_GPU_MEM_UTIL")
    V.main(["rune", "ref-pool", "/model"])
    assert "--gpu-memory-utilization 0.85" in capsys.readouterr().out
    assert V.preset_for("rune", "ref-pool").args.count("--gpu-memory-utilization") == 1   # the preset itself is untouched
