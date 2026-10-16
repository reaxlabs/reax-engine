"""Token ids of the native Plumb prompt == jevk5.runtime.JevK5.encode on a JSONL of {"state","question"} items.
Needs jevk5 + transformers (the reference image has both); no GPU.  usage: check_plumb_ids.py MODEL_DIR ITEMS.jsonl"""
import json
import sys
from pathlib import Path

import transformers
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from jevk5.runtime import decision_options as ref_options, messages as ref_messages  # noqa: E402
from plumb_engine import decision_options, prompt_text  # noqa: E402

model_dir, items = sys.argv[1], sys.argv[2]
hf = transformers.AutoTokenizer.from_pretrained(model_dir)
fast = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
n = bad = 0
for line in open(items):
    row = json.loads(line)
    st, q = row["state"], row["question"]
    q.setdefault("instructions", "")
    opts = ref_options(q)
    assert opts == decision_options(q)
    text = hf.apply_chat_template(ref_messages(st, q["instructions"], [t for _, t in opts]), tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    ref = hf.encode(text, add_special_tokens=False)
    mine_text = prompt_text(st, q["instructions"], [t for _, t in opts])
    mine = fast.encode(mine_text, add_special_tokens=False).ids
    n += 1
    if mine_text != text or mine != ref:
        bad += 1
        print("MISMATCH", n, mine_text == text, len(mine), len(ref))
print(f"{n} items, {bad} mismatches")
sys.exit(1 if bad else 0)
