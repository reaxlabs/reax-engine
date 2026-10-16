"""A character-level stand-in for a HF tokenizer (one token per character; two-letter codes are two tokens)."""


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text)}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=None):
        out = "".join(f"<{m['role']}>{m['content']}</>" for m in msgs)
        return out + ("<model>" if add_generation_prompt else "")
