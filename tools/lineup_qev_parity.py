#!/usr/bin/env python3
"""Compare the public qev contract to the owned production port on an offline tokenizer.

Run only on the GPU-check host with the private engine 5117ec3 source supplied
by its owner; no upstream model Python is loaded. No weights or network needed.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import qev_contract as public


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--production-contract', required=True)
    ap.add_argument('--tokenizer', required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    spec = importlib.util.spec_from_file_location('production_qev', args.production_contract)
    prod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = prod
    spec.loader.exec_module(prod)
    rows = json.loads((Path(__file__).resolve().parent.parent / 'tests/fixtures/lineup_qev.json').read_text())
    assert public.SLOT_LABELS == prod.SLOT_LABELS
    assert public.slot_ids(tok) == prod.slot_ids(tok)
    count = 0
    for row in rows:
        a = [public.normalize_question(q) for q in row['questions']]
        b = [prod.normalize_question(q) for q in row['questions']]
        assert public.encode_request(tok, row['state'], a) == prod.encode_request(tok, row['state'], b)
        for qa, qb in zip(a, b):
            assert public.render(tok, row['state'], qa.suffix) == prod.render(tok, row['state'], qb.suffix)
            count += 1
    print(json.dumps({'branches': count, 'slot_ids': len(public.slot_ids(tok)), 'render_and_token_split': 'identical'}))


if __name__ == '__main__':
    main()
