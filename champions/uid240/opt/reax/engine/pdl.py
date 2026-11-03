"""pdl.py (kai lever N2, KAI_PDL; unset / 0 = off): programmatic dependent launch (PDL) inside a forward step.

Why: at c8 the GPU is the bottleneck (host 51 ms vs GPU 114 ms per step), yet it idles ~3.6 ms per step between the ~1,550
kernels of a step (~2.3 us per kernel boundary: the previous grid drains, the next one is launched and ramps up). With
PDL a kernel launched with cudaLaunchAttributeProgrammaticStreamSerialization may be scheduled while its predecessor's
last wave still runs; it then waits in `griddepcontrol.wait` until the predecessor has completed and its memory is
visible, so the launch latency hides behind the predecessor's tail.

Exact by construction: the same kernels in the same order on the same inputs. Every PDL-launched kernel executes
griddepcontrol.wait before ANY global memory access (ours: as its first statement; Inductor's: before the first access
that depends on the previous kernel, and before it triggers its own dependents, see torch/_inductor/codegen/triton.py
_handle_pdl_*), so no kernel reads or writes memory before every earlier kernel of the stream has completed;
`griddepcontrol.launch_dependents` only lets the NEXT kernel be scheduled early. The arithmetic of every kernel is unchanged
(the twin of a kernel differs from it by the two griddepcontrol instructions at its top only).

Covered kernel families (KAI_PDL=1):
  triton  our Triton kernels and fla's pinned ones (TRITON_KERNELS): each JITFunction gets a twin whose body starts with
          tl.extra.cuda.gdc_wait(); tl.extra.cuda.gdc_launch_dependents(), launched with launch_pdl=True. The original
          JITFunction object stays in place (its `run` is routed per call), so every reference to it - module globals,
          fla's autotuner / heuristics wrappers, `from x import k` copies - follows the switch.
  inductor  the torch.compile'd kernel table pplx_engine.K: twins compiled with Inductor's own PDL code generation
          (options={"triton.enable_pdl": True}, torch 2.14; off by default) from copies of the same functions.
  f8g     rgh's FP8 GEMMs and row quant: libf8p.so (f8p.cu: f8g.cu's kernels with the griddepcontrol pair at the top,
          launched with cudaLaunchKernelEx + the PDL attribute); f8g._lib is routed per call.
  nv4     rgh's NVFP4 CUTLASS GEMM (nv4g variants = nv4t tiles 0-5) and kai lever B's tiles: libnv4p.so (nv4p.cu: nv4t.cu
          compiled with CUTLASS's own GDC support, CUTLASS_ENABLE_GDC_FOR_SM100, run with launch_with_pdl=true).
Not covered: cuBLAS (bf16 in_ba), flash attention, aten copies / elementwise, memcpy: they launch as before, and a PDL
kernel after one of them simply starts after it as usual.

switch(on) is a run-time switch (the lever bench flips it between passes); apply(env, pp) is the start-time hook
(backends/qwen35.py, before the warm-up, so the warm-up compiles the twins).
"""
from __future__ import annotations

import ast
import ctypes
import os
import sys
import types

_DIR = os.path.dirname(os.path.abspath(__file__))
ON = False
WAIT = "gdc_wait()"
LAUNCH = "gdc_launch_dependents()"
# (module, kernel) of every host-launched Triton kernel of the mammoth / reax1 forward (not device helpers: a twin is
# only made for kernels launched with kernel[grid](...))
TRITON_KERNELS = (
    ("pplx_engine", "_gnq_kernel"), ("pplx_engine", "_arq_kernel"), ("pplx_engine", "_smqs_kernel"),
    ("pplx_engine", "_gnq2_kernel"), ("pplx_engine", "_sconv_kernel"), ("pplx_engine", "_sconv3_kernel"),
    ("actq", "_amax_scales_kernel"), ("actq", "_quant_deq_kernel"), ("actq", "_silu_amax_kernel"),
    ("nv4q", "_amax_kernel"), ("nv4q", "_quant_dyn_kernel"),
    ("gdnps", "gdnps_pre_kernel"), ("gdnps", "gdnps_seq_kernel"),
    ("gdnfuse", "gdnfp_pre_kernel"), ("gdnfuse", "sconv3n_kernel"),
    ("gdnchain", "gdn_fwd_h_chain_kernel"),
    ("fla.modules.l2norm", "l2norm_fwd_kernel"),
    ("fla.ops.utils.cumsum", "chunk_local_cumsum_scalar_kernel"),
    ("fla.ops.gated_delta_rule.chunk_fwd", "chunk_gated_delta_rule_fwd_kkt_solve_kernel"),
    ("fla.ops.common.chunk_o", "chunk_fwd_kernel_o"),
    ("fla.ops.common.chunk_delta_h", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"),
    ("fla.ops.gated_delta_rule.wy_fast", "recompute_w_u_fwd_kernel"),
)
STATE = {"triton": [], "triton_skipped": [], "inductor": None, "f8g": None, "nv4": None}


def resolve(env) -> bool:
    v = (env.get("KAI_PDL") or "").strip()
    if v in ("", "0"):
        return False
    if v == "1":
        return True
    raise ValueError(f"KAI_PDL={v!r}: 1 (on) or 0 / unset (off)")


# ------------------------------------------------------------------------------------------------ Triton twins
def _tl_name(glb) -> str:
    """the name under which the kernel's module holds triton.language (fla, ours: tl)"""
    for k, v in glb.items():
        if isinstance(v, types.ModuleType) and v.__name__ == "triton.language":
            return k
    raise ValueError("pdl: the kernel's module does not import triton.language")


def twin_src(src: str, tl: str = "tl") -> str:
    """the kernel source with `<tl>.extra.cuda.gdc_wait()` and `<tl>.extra.cuda.gdc_launch_dependents()` as the first two
    statements of its body (after the docstring); everything else byte-identical"""
    tree = ast.parse(src)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("pdl: one function definition expected")
    fn = tree.body[0]
    body = fn.body
    first = body[0]
    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
            and len(body) > 1):
        first = body[1]
    lines = src.splitlines(keepends=True)
    if first.lineno == fn.lineno:
        raise ValueError("pdl: the body must start on its own line")
    i = first.lineno - 1
    ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    if ind != " " * first.col_offset:
        raise ValueError("pdl: unexpected indentation")
    ins = f"{ind}{tl}.extra.cuda.{WAIT}\n{ind}{tl}.extra.cuda.{LAUNCH}\n"
    return "".join(lines[:i]) + ins + "".join(lines[i:])


def make_twin(jf):
    """a new JITFunction of jf's Python function with twin_src(jf.src) (same specialisation options); jf is unchanged"""
    from triton.runtime import jit as J
    key = f"{jf.module}:{jf.fn.__qualname__}"
    reg = J._triton_jit_function_registry.get(key)
    t = J.JITFunction(jf.fn, version=jf.version, do_not_specialize=list(jf.do_not_specialize),
                      do_not_specialize_on_alignment=list(jf.do_not_specialize_on_alignment), debug=jf.debug,
                      noinline=jf.noinline, repr=jf._repr, launch_metadata=jf.launch_metadata)
    if reg is not None:
        J._triton_jit_function_registry[key] = reg  # the twin must not replace jf in triton's registry
    t._unsafe_update_src(twin_src(jf.src, _tl_name(jf.__globals__)))
    t.pre_run_hooks = list(jf.pre_run_hooks)
    return t


def _inner_jit(obj):
    """(holder, JITFunction) of a kernel object: a JITFunction itself, or fla's heuristics / autotuner wrappers"""
    seen = 0
    while not hasattr(obj, "device_caches"):
        nxt = getattr(obj, "fn", None)
        if nxt is None or seen > 4:
            raise TypeError(f"pdl: {obj!r} is not a Triton kernel")
        obj, seen = nxt, seen + 1
    return obj


def route_run(jf, twin) -> None:
    """jf.run per call: the PDL twin (launch_pdl=True) while ON, else jf's own run. Instance attribute: every reference to
    jf (kernel[grid] proxies, autotuner.fn.run) follows."""
    if getattr(jf, "_kai_pdl_twin", None) is not None:
        return
    base = jf.run  # the bound class method

    def run(*args, **kwargs):
        if ON:
            return twin.run(*args, launch_pdl=True, **kwargs)
        return base(*args, **kwargs)
    jf._kai_pdl_twin = twin
    jf._kai_pdl_base = base
    jf.run = run


def install_triton(kernels=TRITON_KERNELS, log=print):
    """route every listed kernel that is importable (fla / gdnps may be absent in a stack); idempotent"""
    import importlib
    done, skipped = [], []
    for mod, name in kernels:
        try:
            m = sys.modules.get(mod) or importlib.import_module(mod)
            k = getattr(m, name)
        except (ImportError, AttributeError) as e:
            skipped.append(f"{mod}.{name} ({type(e).__name__})")
            continue
        jf = _inner_jit(k)
        if getattr(jf, "_kai_pdl_twin", None) is None:
            route_run(jf, make_twin(jf))
        done.append(name)
    STATE["triton"], STATE["triton_skipped"] = done, skipped
    return done


# ------------------------------------------------------------------------------------------------ Inductor twins
INDUCTOR_OPTIONS = {"triton.enable_pdl": True}


def _clone_fn(f):
    """a distinct function (and code object) with f's behaviour: dynamo keeps its compiled code per code object, so the twin
    compiles and caches separately from f's own compiled copy"""
    c = f.__code__.replace()
    if c is f.__code__:  # pragma: no cover (CPython always copies)
        raise RuntimeError("pdl: code.replace() returned the same object")
    g = types.FunctionType(c, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
    g.__kwdefaults__ = f.__kwdefaults__
    g.__qualname__ = f.__qualname__
    return g


class InductorTwins:
    """pplx_engine.K's compiled table and its PDL twins; set(on) swaps the class attributes (the forward reads K.<name> per
    call). Twins are compiled lazily by torch.compile on their first call (the warm-up when KAI_PDL=1 at start)."""

    def __init__(self, K, compile_fn=None):
        raw = getattr(K, "_raw", None)
        if not raw:
            raise RuntimeError("pdl: K was not compiled (PplxEngine(compile=1)) or has no _raw table")
        if os.environ.get("PPLX_COMPILE_MODE"):
            raise RuntimeError("pdl: PPLX_COMPILE_MODE set (torch.compile takes mode or options, not both)")
        if compile_fn is None:
            import torch

            def compile_fn(f):
                return torch.compile(f, dynamic=True, fullgraph=True, options=dict(INDUCTOR_OPTIONS))
        self.K = K
        self.base = {n: K.__dict__[n] for n in raw}
        self.twin = {n: staticmethod(compile_fn(_clone_fn(f))) for n, f in raw.items()}

    def set(self, on: bool):
        for n, v in (self.twin if on else self.base).items():
            setattr(self.K, n, v)


# ------------------------------------------------------------------------------------------------ native routers
def open_f8p():
    lib = ctypes.CDLL(os.path.join(_DIR, "libf8p.so"))
    lib.f8p_gemm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    lib.f8p_gemm.restype = ctypes.c_int
    lib.f8p_rowquant_launch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_void_p]
    lib.f8p_rowquant_launch.restype = ctypes.c_int
    return lib


def open_nv4p():
    lib = ctypes.CDLL(os.path.join(_DIR, "libnv4p.so"))
    lib.nv4p_gemm.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_void_p,
                                                                                               ctypes.c_size_t,
                                                                                               ctypes.c_void_p]
    lib.nv4p_gemm.restype = ctypes.c_int
    lib.nv4p_ws.argtypes = [ctypes.c_int] * 6
    lib.nv4p_ws.restype = ctypes.c_size_t
    lib.nv4p_has.argtypes = [ctypes.c_int]
    lib.nv4p_has.restype = ctypes.c_int
    return lib


class F8Router:
    """stands in for f8g._lib: f8_gemm / f8_rowquant_launch go to libf8p.so's PDL launches while ON (same kernels)"""

    def __init__(self, base, plib):
        self.base, self.plib = base, plib
        self.calls = 0

    def f8_gemm(self, *a):
        if ON:
            self.calls += 1
            return self.plib.f8p_gemm(*a)
        return self.base.f8_gemm(*a)

    def f8_rowquant_launch(self, *a):
        if ON:
            self.calls += 1
            return self.plib.f8p_rowquant_launch(*a)
        return self.base.f8_rowquant_launch(*a)

    def __getattr__(self, name):
        return getattr(self.base, name)


class Nv4Router:
    """stands in for nv4's ctypes library (rgh's libnv4g.so): nv4_gemm(variant) runs libnv4p.so tile = variant with
    nv4g's scheduler defaults (raster 0 = heuristic, swizzle 1; nv4t.cu: tiles 0-5 are nv4g's variants 0-5) while ON"""

    def __init__(self, base, plib):
        self.base, self.plib = base, plib
        self.calls = 0

    def nv4_gemm(self, v, A, B, SFA, SFB, alpha, D, M, N, K, ldd, wsp, wsb, st):
        if ON and 0 <= v <= 5:
            self.calls += 1
            return self.plib.nv4p_gemm(v, 0, 1, A, B, SFA, SFB, alpha, D, M, N, K, ldd, wsp, wsb, st)
        return self.base.nv4_gemm(v, A, B, SFA, SFB, alpha, D, M, N, K, ldd, wsp, wsb, st)

    def __getattr__(self, name):
        return getattr(self.base, name)


class Nv4tRouter:
    """stands in for nv4t.Router.tlib (kai lever B's libnv4t.so): nv4t_gemm(tile, raster, swz) runs libnv4p.so's twin"""

    def __init__(self, base, plib):
        self.base, self.plib = base, plib
        self.calls = 0

    def nv4t_gemm(self, t, raster, swz, *a):
        if ON:
            self.calls += 1
            return self.plib.nv4p_gemm(t, raster, swz, *a)
        return self.base.nv4t_gemm(t, raster, swz, *a)

    def __getattr__(self, name):
        return getattr(self.base, name)


def install_f8g(f8g, plib=None):
    """route f8g._lib (idempotent); returns the router"""
    if isinstance(f8g._lib, F8Router):
        return f8g._lib
    f8g._lib = F8Router(f8g._lib, plib if plib is not None else open_f8p())
    STATE["f8g"] = f8g._lib
    return f8g._lib


def install_nv4(nv4, nv4t=None, plib=None):
    """route nv4's library: rgh's nv4_gemm and, under lever B's router, its libnv4t.so too (idempotent)"""
    lib = nv4.lib()
    plib = plib if plib is not None else open_nv4p()
    if nv4t is not None and isinstance(lib, nv4t.Router):
        if not isinstance(lib.base, Nv4Router):
            lib.base = Nv4Router(lib.base, plib)
        if not isinstance(lib.tlib, Nv4tRouter):
            lib.tlib = Nv4tRouter(lib.tlib, plib)
        STATE["nv4"] = (lib.base, lib.tlib)
        return lib
    if not isinstance(lib, Nv4Router):
        nv4._lib = Nv4Router(lib, plib)
    STATE["nv4"] = (nv4._lib,)
    return nv4._lib


# ------------------------------------------------------------------------------------------------ switch
def switch(on: bool, pp=None, log=print) -> None:
    """switch PDL on / off at run time. On: installs (once) the Triton twins, the f8g / nv4 routers for the modules
    loaded now (and re-routes libraries swapped since), the Inductor twins of pp.K (when K is compiled)."""
    global ON
    on = bool(on)
    if on:
        install_triton(log=log)
        if "f8g" in sys.modules:
            install_f8g(sys.modules["f8g"])
        if "nv4" in sys.modules:
            install_nv4(sys.modules["nv4"], sys.modules.get("nv4t"))
        if pp is not None and getattr(pp.K, "_compiled", False) and STATE["inductor"] is None:
            STATE["inductor"] = InductorTwins(pp.K)
    if STATE["inductor"] is not None:
        STATE["inductor"].set(on)
    ON = on


def describe() -> dict:
    f8 = STATE["f8g"]
    nv = STATE["nv4"] or ()
    return {"on": ON, "triton": len(STATE["triton"]), "triton_skipped": list(STATE["triton_skipped"]),
            "inductor": None if STATE["inductor"] is None else sorted(STATE["inductor"].twin),
            "f8g_calls": None if f8 is None else f8.calls, "nv4_calls": [r.calls for r in nv] or None}


def apply(env, pp, log=print) -> bool:
    """start-time hook (KAI_PDL): on -> set(True) before the warm-up; returns the switch"""
    if not resolve(env):
        return False
    switch(True, pp, log=log)
    d = describe()
    log(f"kai pdl: on (triton kernels {d['triton']}, skipped {d['triton_skipped'] or 'none'}; inductor twins "
        f"{len(d['inductor'] or [])}; f8g {'routed' if STATE['f8g'] is not None else 'not loaded'}; nv4 "
        f"{'routed' if STATE['nv4'] is not None else 'not loaded'})")
    return True
