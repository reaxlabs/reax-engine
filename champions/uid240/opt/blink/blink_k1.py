# SPDX-License-Identifier: Apache-2.0
"""K1 (28 Oct 2026): Blink's dense bf16 GEMMs on k1gemm (BLINK_K1=1) and the dense MLP's gate_up GEMM with the
GELU(tanh)*up epilogue fused into it (BLINK_K1_GLU=1).

k1gemm.cu is an exact bf16 GEMM for sm_120a: one TMA producer warp + 8 mma.sync m16n8k16 warps, persistent CTAs and a
tile width BN that can be any multiple of 16 from 64 to 256, so each CUDA-graph row count can use a tile count that
fits the 188 SMs. Every output element is the same single fp32 chain as vLLM's batch-invariant matmul and E59's CUTLASS
kernel (k = 0, 16, ..., K-16 in increasing order, one mma.sync m16n8k16 per step, one cvt.rn to bf16), so the choice of
kernel per row count cannot change a bit (checked by k1probe.py: k1gemm == blink_gemm == vLLM's matmul_persistent).

The fused gate_up: the tile computes BN/2 gate columns and the BN/2 matching up columns with that chain, rounds both
to bf16 (the values the bi_mm op stores today), and applies Inductor's GELU(tanh)*up kernel arithmetic
(triton_poi_fused_bi_mm_gelu_mul_slice_1: the same PTX operations, ftz flags and constants; checked over all 65,536
bf16 gate values) before the single bf16 rounding of the product. The op blink::k1_glu replaces gate_up_proj + act_fn
in Gemma4MLP.forward (installed by blink_par's gemma4 hook through patch_mlp); row counts without a GLU table entry
run today's GEMM and the same exact epilogue as a separate kernel.

Tables: k1_tiles.json {"gemm": {"N,K": [[max_m, {"BN": bn, "CTA": cap} | null], ...]}, "glu": {...}}; null or a
missing shape keeps today's path (bi_tiles.json).
"""
import glob
import importlib.util
import json
import os
import sys

ON = os.environ.get("BLINK_K1", "0") == "1"
GLU = os.environ.get("BLINK_K1_GLU", "0") == "1"
SO_GLOB = os.environ.get("BLINK_K1_SO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemm", "k1gemm*.so"))
TABLE_PATH = os.environ.get("BLINK_K1_TABLE",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "k1_tiles.json"))
_S = {"mod": None, "tab": None, "n": {}}


def _log(msg):
    print(f"[blink-k1] {msg}", file=sys.stderr, flush=True)


def ext():
    if _S["mod"] is None:
        so = sorted(glob.glob(SO_GLOB))
        if not so:
            raise RuntimeError(f"k1gemm extension not found ({SO_GLOB})")
        name = os.path.basename(so[0]).split(".")[0]
        spec = importlib.util.spec_from_file_location(name, so[0])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _S["mod"] = mod
    return _S["mod"]


def _table():
    if _S["tab"] is None:
        tab = {"gemm": {}, "glu": {}}
        try:
            with open(TABLE_PATH) as fh:
                raw = json.load(fh)
            for part in ("gemm", "glu"):
                for k, v in raw.get(part, {}).items():
                    tab[part][tuple(int(x) for x in k.split(","))] = sorted(((int(m), c) for m, c in v),
                                                                           key=lambda x: x[0])
        except Exception as exc:  # noqa: BLE001
            _log(f"no K1 table at {TABLE_PATH}: {exc!r}")
        _S["tab"] = tab
        _log(f"table {TABLE_PATH}: {len(tab['gemm'])} GEMM shapes, {len(tab['glu'])} GLU shapes")
    return _S["tab"]


def cfg(part, n, k, m):
    """the K1 entry for an m-row (n, k) GEMM, or None (today's path)"""
    for max_m, c in _table()[part].get((n, k), ()):
        if m <= max_m:
            return c
    return None


def _ok(a, w):
    return (a.dim() == 2 and w.dim() == 2 and a.stride(1) == 1 and w.stride(1) == 1 and a.stride(0) % 8 == 0
            and w.stride(0) % 8 == 0 and a.data_ptr() % 16 == 0 and w.data_ptr() % 16 == 0 and w.shape[0] % 16 == 0
            and a.shape[1] % 32 == 0)


def _moe_w13_sms(m):
    """the w13 grouped GEMM's SM count for an m-token step from blink_moe's table (BLINK_MOE_GEMM_TABLE, rows
    [max tokens, variant, raster, sms]; BLINK_MOE_GEMM_SMS > 0 overrides), or None"""
    if "moe" not in _S:
        rows = []
        try:
            path = os.environ.get("BLINK_MOE_GEMM_TABLE",
                                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_gemm.json"))
            with open(path) as fh:
                rows = sorted((int(r[0]), int(r[3])) for r in json.load(fh)["gemms"]["w13"])
        except Exception as exc:  # noqa: BLE001
            _log(f"no MoE GEMM table: {exc!r}")
        _S["moe"] = rows
    over = int(os.environ.get("BLINK_MOE_GEMM_SMS", "0"))
    if over > 0:
        return over
    for mx, sms in _S["moe"]:
        if m <= mx:
            return sms
    return None


def _cta(c, m):
    """the CTA cap of a table entry: an int, or "moe" = the SMs the MoE's w13 GEMM leaves free (the dense MLP then never
    occupies the SMs w13 is about to take; scheduling only)"""
    v = c.get("CTA", 0)
    if v == "moe":
        import torch
        nsm = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        sms = _moe_w13_sms(m)
        return max(1, nsm - sms) if sms and sms < nsm else 0
    return int(v)


def mm(a, w):
    """a @ w.T on k1gemm when the table has an entry for this (N, K, M), else None (blink_bi runs today's kernel)"""
    if not ON:
        return None
    c = cfg("gemm", w.shape[0], w.shape[1], a.shape[0])
    if c is None or not _ok(a, w):
        return None
    _count(f"gemm {w.shape[0]}x{w.shape[1]}")
    return ext().mm(a, w, int(c["BN"]), _cta(c, a.shape[0]))


def _count(key):
    n = _S["n"].get(key, 0) + 1
    _S["n"][key] = n
    if n == 1:
        _log(f"first k1gemm call: {key}")


_OPS = {}


def _install_glu_op():
    if "glu" in _OPS:
        return
    import torch

    def _gate_up_stock(x, w):
        bi = sys.modules.get("blink_bi")
        if bi is not None and getattr(bi, "_FN", {}).get("f") is not None:
            return torch.ops.blink.bi_mm(x, w)       # today's per-M kernel (CUTLASS / Triton BI)
        from vllm.model_executor.determinism import batch_invariant as bimod
        return bimod.matmul_persistent(x, w.t())

    def _glu_rows(x, w):
        c = cfg("glu", w.shape[0], w.shape[1], x.shape[0])
        if c is not None and _ok(x, w):
            _count(f"glu {w.shape[0]}x{w.shape[1]}")
            return ext().glu(x, w, int(c["BN"]), _cta(c, x.shape[0]))
        return ext().gelu_mul(_gate_up_stock(x, w), False)

    def _last_rows(x, w):
        """E64 B7 (blink_last): the selected rows when w is the last layer's gate_up weight in a restricted forward"""
        last = sys.modules.get("blink_last")
        st = getattr(last, "_S", None) if last is not None else None
        if not st or not st.get("w"):
            return None
        kind = st["w"].get(w.data_ptr())
        cur = st.get("cur")
        if kind is None or kind[0] != "rows" or kind[1] != "gu" or cur is None or cur[0] != x.shape[0]:
            return None
        return cur[1]

    @torch.library.custom_op("blink::k1_glu", mutates_args=())
    def k1_glu(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        idx = _last_rows(x, w)
        if idx is not None:                          # the last layer computes only the rows that reach the logits
            out = torch.zeros((x.shape[0], w.shape[0] // 2), device=x.device, dtype=x.dtype)
            out.index_copy_(0, idx, _glu_rows(x.index_select(0, idx), w))
            return out
        return _glu_rows(x, w)

    @k1_glu.register_fake
    def _(x, w):
        return x.new_empty((x.shape[0], w.shape[0] // 2))

    _OPS["glu"] = k1_glu


def patch_mlp(mod):
    """Gemma4MLP.forward: gate_up_proj + GeluAndMul(tanh) -> blink::k1_glu (called from blink_par's gemma4 hook)"""
    if not GLU or getattr(mod, "_blink_k1_mlp", False):
        return
    import torch
    cls = getattr(mod, "Gemma4MLP", None)
    if cls is None:
        _log("no Gemma4MLP: GLU fusion off")
        return
    _install_glu_op()
    orig = cls.forward

    def forward(self, x):
        gu = self.gate_up_proj
        act = self.act_fn
        w = getattr(gu, "weight", None)
        if (w is None or w.dtype != torch.bfloat16 or x.dtype != torch.bfloat16 or x.dim() != 2 or
                getattr(gu, "bias", None) is not None or getattr(gu, "tp_size", 1) != 1 or
                type(act).__name__ != "GeluAndMul" or getattr(act, "approximate", None) != "tanh" or
                w.shape[0] != 2 * self.down_proj.input_size_per_partition):
            return orig(self, x)
        h = torch.ops.blink.k1_glu(x, w)
        out, _ = self.down_proj(h)
        return out

    cls.forward = forward
    mod._blink_k1_mlp = True
    _log("Gemma4MLP: gate_up GEMM + GELU(tanh)*up fused (blink::k1_glu)")
