# SPDX-License-Identifier: Apache-2.0
"""Per-M launch configs for vLLM's batch-invariant dense matmul (BLINK_BI_TILES=1).

With VLLM_BATCH_INVARIANT=1 every unquantized linear layer (and the LM head) runs vLLM's Triton
matmul_kernel_persistent with one fixed config (128x128x64, 8 warps, 3 stages) for every batch size: under
torch.compile the launch config is fixed when the graph is traced. Each output element's accumulation is the same
k-ordered chain of tensor-core MMAs for any BLOCK_M / BLOCK_N / BLOCK_K / warps / stages / grid (bit-identical in the
E58/E59 kernel probes), so the config may follow the row count. The linear layers here call an opaque custom op
(blink::bi_mm) that picks the config from TABLE by (N, K) and the row count when it runs (in every captured CUDA graph:
the padded batch size of that graph), and launches the same kernel. Shapes or sizes not in TABLE use vLLM's config.
"""

import importlib.abc
import json
import os
import sys

ON = os.environ.get("BLINK_BI_TILES", "0") == "1"
# BLINK_LABEL_HEAD=1: the LM head computes only the 26 option-letter logits. Every request carries
# allowed_token_ids (a subset of the letters), and vLLM's sampler masks every other logit to -inf before it computes
# the (processed) logprobs, so the other logits never matter; each letter logit is the same k-ordered MMA chain as in
# the full-vocabulary matmul, so it is bit-identical. The other entries are filled with 0.
LABEL_HEAD = os.environ.get("BLINK_LABEL_HEAD", "0") == "1"
# K1: the exact k1gemm dense GEMMs (BLINK_K1=1, per-M table k1_tiles.json) and the fused gate_up + GELU(tanh)*up
# (BLINK_K1_GLU=1, installed through blink_par's gemma4 hook); blink_k1.py
_K1 = None
if os.environ.get("BLINK_K1", "0") == "1" or os.environ.get("BLINK_K1_GLU", "0") == "1":
    try:
        import blink_k1 as _K1
    except Exception as _exc:  # noqa: BLE001
        print(f"[blink-bi] blink_k1 unavailable ({_exc!r})", file=sys.stderr, flush=True)
        _K1 = None
GEMM_DIR = os.environ.get("BLINK_GEMM_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemm"))
TABLE_PATH = os.environ.get("BLINK_BI_TABLE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bi_tiles.json"))
# E64 B7: BLINK_LAST=1 restricts the last decoder layer's GEMMs to the rows that reach the logits (blink_last.py);
# imported here, before vLLM's model runner, so that its load_model hook is in place
_LAST = None
if os.environ.get("BLINK_LAST", "0") == "1":
    try:
        import blink_last as _LAST
    except Exception as _exc:  # noqa: BLE001
        print(f"[blink-bi] blink_last unavailable ({_exc!r})", file=sys.stderr, flush=True)
        _LAST = None
DEFAULT = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "num_warps": 8, "num_stages": 3}
_T = {}


def _table():
    if "t" not in _T:
        try:
            with open(TABLE_PATH) as fh:
                raw = json.load(fh)
            # {"N,K": [[max_m, {cfg}], ...]} sorted by max_m
            _T["t"] = {tuple(int(x) for x in k.split(",")): sorted(((int(m), c) for m, c in v), key=lambda x: x[0])
                       for k, v in raw.get("shapes", {}).items()}
        except Exception as exc:  # noqa: BLE001
            print(f"[blink-bi] no tile table at {TABLE_PATH}: {exc!r}", file=sys.stderr, flush=True)
            _T["t"] = {}
    return _T["t"]


def _cfg(m, n, k):
    if not ON:
        return DEFAULT
    for max_m, c in _table().get((n, k), ()):
        if m <= max_m:
            return DEFAULT if c == "default" else c
    return DEFAULT


def _install():
    import torch
    import triton
    from vllm.model_executor.determinism import batch_invariant as bi
    from vllm.utils.platform_utils import num_compute_units

    nsm = {}
    ext = {}

    def _cutlass():
        """the prebuilt CUTLASS sm80 bf16 GEMM extension (blink_gemm.cu, built by build_gemm.py at image build)"""
        if "m" not in ext:
            import glob
            import importlib.util
            so = sorted(glob.glob(os.path.join(GEMM_DIR, "blink_gemm*.so")))
            if not so:
                raise RuntimeError(f"blink_gemm extension not found in {GEMM_DIR}")
            spec = importlib.util.spec_from_file_location("blink_gemm", so[0])
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            ext["m"] = mod
        return ext["m"]

    def _launch(a, b):          # a [M, K] (row-major), b = weight.t() [K, N]
        M, K = a.shape
        N = b.shape[1]
        if M == 0:
            return torch.empty((M, N), device=a.device, dtype=a.dtype)
        if _K1 is not None:
            out = _K1.mm(a, b.t())          # K1 table entry for (N, K, M): k1gemm (same K chain), else None
            if out is not None:
                return out
        cfg = dict(_cfg(M, N, K))
        if ("CUTLASS" in cfg and a.stride(1) == 1 and b.stride(0) == 1 and a.stride(0) % 8 == 0
                and b.stride(1) % 8 == 0 and a.data_ptr() % 16 == 0 and b.data_ptr() % 16 == 0 and N % 8 == 0):
            return _cutlass().mm(a, b.t(), int(cfg["CUTLASS"]))   # 16-byte aligned operands (CUTLASS default)
        cfg.pop("CUTLASS", None)
        if "BLOCK_SIZE_M" not in cfg:
            cfg = dict(DEFAULT)
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        dev = a.device.index or 0
        if dev not in nsm:
            nsm[dev] = num_compute_units(dev)
        progs = nsm[dev] * int(cfg.pop("GRIDX", 1))

        def grid(meta):
            return (min(progs, triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"])),)
        bi.matmul_kernel_persistent[grid](a, b, c, None, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                          c.stride(0), c.stride(1), NUM_SMS=progs, A_LARGE=a.numel() > 2 ** 31,
                                          B_LARGE=b.numel() > 2 ** 31, C_LARGE=c.numel() > 2 ** 31, HAS_BIAS=False,
                                          GROUP_SIZE_M=8, **cfg)
        return c

    def _launch_w(x, w):        # x @ w.T for a [N, K] weight (or a row slice of one)
        return _launch(x, w.t())

    @torch.library.custom_op("blink::bi_mm", mutates_args=())
    def bi_mm(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if _LAST is not None:
            out = _LAST.mm(a, weight, _launch_w)
            if out is not None:
                return out
        return _launch(a, weight.t())

    @bi_mm.register_fake
    def _(a, weight):
        return a.new_empty((a.shape[0], weight.shape[0]))

    letters = {}

    def _letter_rows(weight):
        key = (weight.data_ptr(), tuple(weight.shape))
        if key not in letters:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(os.environ.get("REAX_MODEL_DIR", "/model"))
            ids = []
            for i in range(26):
                t = tok.encode(chr(65 + i), add_special_tokens=False)
                if len(t) != 1:
                    raise RuntimeError(f"letter {chr(65 + i)} is not one token")
                ids.append(t[0])
            idx = torch.tensor(ids, dtype=torch.long, device=weight.device)
            letters[key] = (idx, weight.index_select(0, idx).contiguous())
            print(f"[blink-bi] label-only LM head: {len(ids)} letter rows of {weight.shape[0]}", file=sys.stderr,
                  flush=True)
        return letters[key]

    def linear_batch_invariant(input, weight, bias=None):
        if input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or weight.ndim != 2 or input.ndim < 2:
            return bi.linear_batch_invariant(input, weight, bias)
        is_head = LABEL_HEAD and bias is None and input.ndim == 2 and weight.shape[0] >= 100000
        if not ON and not is_head:
            return bi.linear_batch_invariant(input, weight, bias)
        if (LABEL_HEAD and bias is None and input.ndim == 2 and weight.shape[0] >= 100000
                and ((weight.data_ptr(), tuple(weight.shape)) in letters
                     or not torch.cuda.is_current_stream_capturing())):
            idx, w_l = _letter_rows(weight)
            out = torch.zeros((input.shape[0], weight.shape[0]), device=input.device, dtype=input.dtype)
            out.index_copy_(1, idx, torch.ops.blink.bi_mm(input, w_l))
            return out
        if input.ndim == 2:
            out = torch.ops.blink.bi_mm(input, weight)
        else:
            lead = input.shape[:-1]
            out = torch.ops.blink.bi_mm(input.reshape(-1, input.shape[-1]), weight).reshape(*lead, weight.shape[0])
        if bias is not None:
            out = out + bias
        return out

    return linear_batch_invariant


_TARGETS = ("vllm.model_executor.layers.linear", "vllm.model_executor.layers.vocab_parallel_embedding")
_FN = {}


def _patch(mod):
    # the per-M tiles change the compiled graph (an opaque op instead of vLLM's traced Triton launch), which renames
    # the Inductor kernels around it; with tiles off only the LM head (eager, outside torch.compile) is patched
    if not ON and mod.__name__ == "vllm.model_executor.layers.linear":
        return
    if "f" not in _FN:
        _FN["f"] = _install()
        print(f"[blink-bi] per-M tile table {TABLE_PATH}: {len(_table())} shapes", file=sys.stderr, flush=True)
    mod.linear_batch_invariant = _FN["f"]


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS:
            return None
        spec = None
        for f in sys.meta_path:
            if f is self:
                continue
            try:
                spec = f.find_spec(fullname, path, target)
            except Exception:  # noqa: BLE001
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


if ON or LABEL_HEAD:
    for _n in _TARGETS:
        if _n in sys.modules:
            _patch(sys.modules[_n])
    sys.meta_path.insert(0, _Finder())
