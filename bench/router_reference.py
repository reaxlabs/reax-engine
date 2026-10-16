"""Reference execution of s1-llm-auto-router, written from the model card (example.py of the model repo) with the
two details of the serving path made explicit: attention mask of ones and float64 softmax / sigmoid.
Used by bench/router_equiv.py as the in-process reference for engine/backends/onnx_heads.py."""
import json
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

CATS = ["coding", "agentic", "math", "knowledge", "long_context", "tool_use", "design", "summarisation", "general"]
FLAGS = ["needs_tools", "needs_vision", "needs_long_context", "follow_up"]


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class Reference:
    def __init__(self, root, threads=4):
        self.temps = json.load(open(os.path.join(root, "meta.json")))["temperatures"]
        self.tok = Tokenizer.from_file(os.path.join(root, "tokenizer", "tokenizer.json"))
        self.tok.enable_truncation(512, strategy="only_second")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(os.path.join(root, "model.gq.onnx"), so, providers=["CPUExecutionProvider"])

    def predict(self, request, context=""):
        """-> ({head: probability array}, n_tokens); binary heads are (false, true)."""
        enc = self.tok.encode(context[:600], request)
        ids = np.asarray([enc.ids], dtype=np.int64)
        cat, dif, stk, flg = self.sess.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
        t = self.temps
        out = {"category": _softmax(cat[0].astype(np.float64) / t["category"]),
               "difficulty": _softmax(dif[0].astype(np.float64) / t["difficulty"]),
               "stakes": _softmax(stk[0].astype(np.float64) / t["stakes"])}
        pf = 1 / (1 + np.exp(-flg[0].astype(np.float64) / t["flags"]))
        for f, p in zip(FLAGS, pf):
            out[f] = np.array([1.0 - p, p])
        return out, len(enc.ids)
