# SPDX-License-Identifier: Apache-2.0
"""Router GEMM on our own CUTLASS kernel (BLINK_RTR=1).

Blink's MoE router projects the [M, 2816] router input onto 128 experts with bf16 inputs and fp32 output. In the
compiled graph Inductor emits it as `extern_kernels.mm_dtype(a, w_t, out_dtype=torch.float32, out=buf)`, i.e.
torch.mm -> cuBLASLt (vLLM's batch-invariant settings: 1-byte workspace, so no split-K): every output element is a
k-ordered chain of 16-wide tensor-core MMAs accumulated in fp32. blink_rgemm.cu computes the same chain with a CUTLASS
sm80 tensor-op GEMM whose tiles suit N = 128 (bit-identical on every checked row count, proto/rgemm59.py). This module
replaces the `mm_dtype` entry of Inductor's extern-kernel namespace with a function that uses our kernel for exactly
that case and calls torch.mm for everything else; the compiled graph and its kernels are unchanged.
Installed by sitecustomize.py (patches torch._inductor.kernel.mm when it is imported).
"""
import importlib.abc
import json
import os
import sys

ON = os.environ.get("BLINK_RTR", "0") == "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
GEMM_DIR = os.environ.get("BLINK_GEMM_DIR", os.path.join(_HERE, "gemm"))
TABLE = os.environ.get("BLINK_RTR_TABLE", os.path.join(_HERE, "rtr_gemm.json"))
_TARGET = "torch._inductor.kernel.mm"
_S = {}


def _ext():
    if "m" not in _S:
        import glob
        import importlib.util
        so = sorted(glob.glob(os.path.join(GEMM_DIR, "blink_rgemm*.so")))
        if not so:
            raise RuntimeError(f"blink_rgemm extension not found in {GEMM_DIR}")
        spec = importlib.util.spec_from_file_location("blink_rgemm", so[0])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _S["m"] = mod
    return _S["m"]


def _table():
    """{"N,K": [[max_m, cfg], ...]}: our kernel config per row-count bucket; shapes not listed use torch.mm"""
    if "t" not in _S:
        try:
            with open(TABLE) as fh:
                raw = json.load(fh)
            _S["t"] = {tuple(int(x) for x in k.split(",")): sorted((int(m), int(c)) for m, c in v)
                       for k, v in raw["shapes"].items()}
        except Exception as exc:  # noqa: BLE001
            print(f"[blink-rtr] no table at {TABLE}: {exc!r}", file=sys.stderr, flush=True)
            _S["t"] = {}
    return _S["t"]


def _patch(mod):
    import torch
    from torch._inductor.select_algorithm import extern_kernels
    orig = getattr(extern_kernels, "mm_dtype", None)
    if orig is None:
        print("[blink-rtr] extern_kernels.mm_dtype not found: not installed", file=sys.stderr, flush=True)
        return

    def mm_dtype(a, b, *args, out_dtype=None, out=None, **kwargs):
        # a [M, K] bf16 row-major, b = weight.t() [K, N] (column-major weight), out [M, N] fp32
        if (not args and not kwargs and out is not None and out_dtype == torch.float32 and a.dtype == torch.bfloat16
                and b.dtype == torch.bfloat16 and a.dim() == 2 and b.dim() == 2 and out.dtype == torch.float32
                and a.stride(1) == 1 and b.stride(0) == 1 and out.stride(1) == 1 and a.is_cuda):
            M, K = a.shape
            N = b.shape[1]
            rows = _table().get((N, K))
            if rows and M > 0 and a.stride(0) % 8 == 0 and b.stride(1) % 8 == 0 and a.data_ptr() % 16 == 0 \
                    and b.data_ptr() % 16 == 0 and out.data_ptr() % 16 == 0 and out.stride(0) % 4 == 0:
                for max_m, cfg in rows:
                    if M <= max_m:
                        if cfg >= 0:
                            _ext().mm_f32_out(a, b.t(), out, cfg)
                            return out
                        break
        return orig(a, b, *args, out_dtype=out_dtype, out=out, **kwargs)

    extern_kernels.mm_dtype = mm_dtype
    print(f"[blink-rtr] router GEMM (mm_dtype bf16->fp32) on blink_rgemm for {sorted(_table())}", file=sys.stderr,
          flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
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


if ON:
    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
    else:
        sys.meta_path.insert(0, _Finder())
