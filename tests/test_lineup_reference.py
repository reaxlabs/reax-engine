"""CPU contract and drift checks for the new reference slots; no model execution."""
import ast
import json
import os
from pathlib import Path

import pytest

from engine import qev_contract as Q, registry
from engine.backends import qwen35, vllm_readout as V
from engine.vision_budget import patch_source
from tests.fake_tok import FakeTok

FIXTURES = Path(__file__).parent / 'fixtures/lineup_qev.json'


def test_qev_contract_fixed_private_engine_goldens():
    # Captured from private engine 5117ec3 with FakeTok. Checked-in expectations
    # deliberately avoid a runtime dependency on the private repository.
    for row in json.loads(FIXTURES.read_text()):
        tok = FakeTok()
        qs = [Q.normalize_question(q) for q in row['questions']]
        full, prefix = Q.encode_request(tok, row['state'], qs)
        assert [Q.render(tok, row['state'], q.suffix) for q in qs] == row['texts']
        assert full == row['ids'] and prefix == row['prefix']
        assert [Q.product_answer(q, [1 / len(q.keys)] * len(q.keys)) for q in qs] == row['answers']


def test_qev_slots_and_schema():
    assert len(Q.SLOT_LABELS) == 256 and len(set(Q.SLOT_LABELS)) == 256
    assert Q.SLOT_LABELS[:27] == tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ('AA',)
    assert 'BQ' not in Q.SLOT_LABELS and 'BZ' not in Q.SLOT_LABELS
    q = Q.normalize_question({'type': 'choice', 'criteria': {str(i): None for i in range(255)}})
    assert len(q.keys) == 255 and 'Answer with one label:' in q.suffix
    with pytest.raises(ValueError):
        Q.normalize_question({'type': 'choice', 'criteria': {str(i): None for i in range(256)}})


def test_mammoth_backend_split_and_answer():
    be = object.__new__(qwen35.Qwen35Backend)
    be.family, be._tok, be.max_input_tokens = 'mammoth', FakeTok(), 16384
    q = {'type': 'choice', 'criteria': {'a': 'first', 'b': 'second'}}
    e = be.encode({'x': 1}, q)
    full, prefix = Q.encode_request(be._tok, {'x': 1}, [Q.normalize_question(q)])
    assert (e.payload, e.aux, e.n_options) == (full[0], prefix, 2)
    assert be.options(q) == ['a', 'b']
    assert be.answer(q, [0.1, 0.9]) == Q.product_answer(Q.normalize_question(q), [0.1, 0.9])
    be.max_input_tokens = 1
    with pytest.raises(ValueError, match='maximum context length'):
        be.encode('s', q)


def test_mammoth_and_jeffan_reference_options():
    m = registry.get('mammoth')
    assert m.revision == '61cec44c281c693992e3d6ebeb44dd3402dfa186'
    assert m.production_opts == dict(linear='fp8', compile='1', gva='1', budget=16384, max_input_tokens=16384,
                                     warm_steps=200, shared_state=True, shared_group=True, shared_until=64, shared_mixq=False)
    j = registry.get('jeffan')
    assert j.production_opts == dict(preset='fp8-kv-colo-bi', max_input_tokens=8192, max_options=255)
    p = V.preset_for('jeffan', j.production_opts['preset'])
    assert p.env['VLLM_BATCH_INVARIANT'] == '1'
    argv = V.vllm_argv(p, '/model')
    assert argv[argv.index('--gpu-memory-utilization') + 1] == '0.44'
    assert argv[argv.index('--max-model-len') + 1] == '16448'


def test_fp8_skip_anchor_expression_from_loader():
    tree = ast.parse(Path('engine/pplx_engine.py').read_text())
    lin = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'lin')
    expr = next(n for n in lin.body if isinstance(n, ast.Assign) and n.targets[0].id == 'mode').value
    code = compile(ast.Expression(expr), '<fp8_skip>', 'eval')
    for skips, name, expected in [(['^3.'], '3.down', 'bf16'), (['^3.'], '13.down', 'fp8'),
                                  (['down'], '13.down', 'bf16'), (['^63.'], '63.qkv', 'bf16'),
                                  ([], '63.qkv', 'fp8')]:
        assert eval(code, dict(fp8_skip=skips, name=name, linear_mode='fp8')) == expected


def test_text_sampling_and_image_sampling():
    for temp, top_p in [(0, None), (1.0, 1.0)]:
        c = V.VllmClient('http://localhost', completion_temperature=temp)
        bodies = []
        def post(path, body):
            bodies.append(body)
            return {'choices': [{'logprobs': {'top_logprobs': [{'token_id:7': -0.1}]}}]}
        c._post = post
        assert c.option_values([1], [7]) == [-0.1]
        assert bodies[0].get('top_p') == top_p
        assert bodies[0]['temperature'] == temp


def test_vision_pin_only_changes_budget(monkeypatch):
    source = '''class Encoder:
    @staticmethod
    def _encoder_chunk(patches_per_item, free_bytes, total_bytes, position_embedding_size):
        if patches_per_item <= 0:
            return 1
        budget = min(free_bytes // 2, total_bytes // 10)
        if budget <= 0:
            return 1
        cost = patches_per_item * 4 * position_embedding_size * 8
        return max(1, budget // cost) if cost > 0 else 1
'''
    ns = {}; exec(patch_source(source), ns)
    chunk = ns['Encoder']._encoder_chunk
    monkeypatch.delenv('REAX_VISION_ENCODER_BUDGET', raising=False)
    assert chunk(1, 640, 64000, 1) == 10
    monkeypatch.setenv('REAX_VISION_ENCODER_BUDGET', 'total10')
    assert chunk(1, 640, 64000, 1) == chunk(1, 20000, 64000, 1) == 200
    assert chunk(0, 0, 64000, 1) == 1
    with pytest.raises(ValueError, match='source changed'):
        patch_source(patch_source(source))


def test_mammoth_tokenizer_parity_offline():
    model_dir = os.environ.get('MAMMOTH_TOKENIZER_DIR')
    if not model_dir:
        pytest.skip('GPU check: set MAMMOTH_TOKENIZER_DIR to pinned offline mammoth tokenizer')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    assert len(Q.slot_ids(tok)) == 256
    # Full private-engine comparison is performed by tools/lineup_qev_parity.py.
    for row in json.loads(FIXTURES.read_text()):
        full, prefix = Q.encode_request(tok, row['state'], [Q.normalize_question(q) for q in row['questions']])
        assert 0 < prefix < min(map(len, full))
