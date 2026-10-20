import json

import pytest

from engine.backends import vllm_patch as P
from engine.backends import vllm_readout as V

# the get_default_config head of vLLM 0.30.0 fused_moe.py (the part the patch anchors on)
SRC = '''def get_default_config(
    M: int,
    E: int,
) -> dict[str, int]:
    if envs.VLLM_BATCH_INVARIANT:
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "SPLIT_K": 1,
        }
'''


def test_patch_inserts_env_branch_and_is_idempotent():
    out = P.patch_text(SRC)
    assert 'os.environ.get("REAX_BI_MOE_CONFIG")' in out and '"BLOCK_SIZE_M": 64,' in out
    assert P.patch_text(out) == out
    compile(out, "fused_moe.py", "exec")


def test_patch_refuses_unknown_source():
    with pytest.raises(RuntimeError):
        P.patch_text(SRC.replace('"BLOCK_SIZE_M": 64,', '"BLOCK_SIZE_M": 32,'))


def test_patched_function_reads_env(monkeypatch):
    class envs:
        VLLM_BATCH_INVARIANT = True
    ns = {"envs": envs}
    exec(P.patch_text(SRC) + "\n", ns)
    assert ns["get_default_config"](1, 2)["BLOCK_SIZE_M"] == 64
    monkeypatch.setenv(P.ENV, P.env_value())
    assert ns["get_default_config"](1, 2) == P.BI_MOE_TILE


@pytest.mark.parametrize("name", ["deterministic", "fp8-kv-seqs64"])
def test_rune_measured_presets_set_tile(name):
    p = V.preset_for("rune", name)
    assert p.env["VLLM_BATCH_INVARIANT"] == "1" and json.loads(p.env[P.ENV]) == P.BI_MOE_TILE


def test_unmeasured_and_non_invariant_presets_unchanged():
    assert P.ENV not in V.preset_for("rune", "previous-f").env and P.ENV not in V.preset_for("blink", "bi1").env


def test_main_patches_file_and_guards_version(tmp_path, monkeypatch, capsys):
    f = tmp_path / "fused_moe.py"
    f.write_text(SRC)
    monkeypatch.setattr(P, "_target", lambda: ("0.29.9", str(f)))
    assert P.main([]) == 2 and f.read_text() == SRC          # other vLLM version: untouched
    assert P.main(["--check"]) == 1
    monkeypatch.setattr(P, "_target", lambda: (P.VLLM_VERSION, str(f)))
    assert P.main([]) == 0 and P.ENV in f.read_text()
    assert P.main(["--check"]) == 0 and P.main([]) == 0        # idempotent
