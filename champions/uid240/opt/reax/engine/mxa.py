"""mxa.py (rgh lane MXA, 28 Oct 2026): teammate kai's EXACT engine levers on the rgh m2 stacks (mammoth /opt/reax-m,
REAX-1 /opt/reax-r), each behind its own switch, each with a start-up bit-equality check on the serving GPU (a failing
check logs and keeps the reference path). ON[...] is read at call time (in-process A/B tools flip it).

  MXA_D=1   kai lever D (gdnps.py, kai/combo@0eecdb1b, adopted verbatim): in PplxEngine._gdn_chain (mammoth state-once
            steps whose every group has one suffix) recompute_w_u + gdnchain + chunk_fwd_o are replaced by gdnps pass P +
            pass S (h / v_new stay in registers; FF-PEEL), cfg (BV 32, 4 warps, 1 stage, pass P 2 warps) = kai's default.
            Check: gdnps.check() (kai's: random chain layouts of serving shapes vs the reference stages, every element).
  MXA_D1=1  rgh extension of lever D: the same two passes with nseg=1 (independent sequences, h0 = +0) in the plain DeltaNet
            path (PplxEngine._deltanet, MAM_SCONV lag table + MAM_GDNDIRECT branch = REAX-1's forward_marked c8 / c1 steps)
            in place of fla 0.5.2 chunk_gated_delta_rule_fwd's intra (kkt_solve kept) + fwd_h + chunk_fwd_o.
            Check: check_d1() = chain(nseg=1) vs chunk_gated_delta_rule_fwd itself on random varlen layouts.
  MXA_K=1   kai lever K (f8k.py / f8k_sched.py / libf8k.so, kai/combo@0eecdb1b, adopted verbatim): exact chained stream-K
            FP8 GEMMs (f8g's mainloop / epilogues, split tiles continue from the head's exact fp32 accumulators) for the
            shapes in MAM_F8SK_SHAPES (kai default "down,out"); calls f8k refuses (too few tiles, e.g. c1) run f8g.
            Check: check_k() = stream-K vs f8g (MAM_F8TAB table plan) on random e4m3 operands at c8 step sizes.
"""
import os

ON = {"D": os.environ.get("MXA_D", "0") == "1", "D1": os.environ.get("MXA_D1", "0") == "1",
      "K": os.environ.get("MXA_K", "0") == "1"}
OK = {}  # lever -> start-up check result (True = bit-equal on this GPU)
D_CFG = (32, 4, 1, 2)  # kai's default KAI_GDNPS_CFG: BV, warps, stages, pass-P warps
_f8 = {}


def _log(msg):
    print(f"rgh mxa: {msg}", flush=True)


def d_ok():
    if "D" not in OK:
        import gdnps
        try:
            why = gdnps.check(cfg=D_CFG, peel=True)
        except Exception as e:  # noqa: BLE001
            why = f"check raised {type(e).__name__}: {e}"
        OK["D"] = not why
        _log(f"lever D (gdnps chain, cfg {D_CFG}): " + ("check bit-equal, on" if OK["D"] else f"check FAILED ({why}), off"))
    return OK["D"]


def chain(qn, kn, v, gc, beta, cu, ci, nseq, scale, nseg):
    import gdnps
    bv, nw, ns, pw = D_CFG
    return gdnps.chain(qn, kn, v, gc, beta, cu, ci, nseq, scale, nseg=nseg, BV=bv, num_warps=nw, num_stages=ns,
                       pre_warps=pw, peel=True)


def plain(qn, kn, v, g, beta, cu, ci, scale):
    """== chunk_gated_delta_rule_fwd(q=qn, k=kn, v=v, g=g, beta=beta, scale=scale, initial_state=None,
    output_final_state=False, cu_seqlens=cu, chunk_indices=ci, chunk_size=64)[1] (fla 0.5.2), via gdnps nseg=1"""
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2
    gc = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2, cu_seqlens=cu, chunk_indices=ci)
    return chain(qn, kn, v, gc, beta, cu, ci, cu.numel() - 1, scale, 1)


def check_d1(seed=91):
    """chain(nseg=1) vs fla's chunk_gated_delta_rule_fwd on random varlen layouts of serving shapes (H 16, HV 48, K = V =
    128): lengths 1, 63, 64, 65, 128, partial chunks, 1..16 sequences. '' when every output element is bit-equal."""
    import torch
    import torch.nn.functional as F
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
    from fla.ops.utils.index import prepare_chunk_indices
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen, device="cuda")  # noqa: E731
    layouts = [[64], [1], [65, 3], [128, 1, 63], [191, 200, 17, 64, 129, 2, 250, 190],
               torch.randint(1, 330, (16,), generator=gen, device="cuda").tolist()]
    for lens in layouts:
        cu_l = [0]
        for n in lens:
            cu_l.append(cu_l[-1] + n)
        T = cu_l[-1]
        cu = torch.tensor(cu_l, dtype=torch.int64, device="cuda")
        q, k = r(1, T, 16, 128).to(torch.bfloat16), r(1, T, 16, 128).to(torch.bfloat16)
        v = r(1, T, 48, 128).to(torch.bfloat16)
        g = (-F.softplus(r(1, T, 48)) * 0.3).float()
        beta = torch.sigmoid(r(1, T, 48)).to(torch.bfloat16)
        qn, kn = l2norm_fwd(q)[0], l2norm_fwd(k)[0]
        ci = prepare_chunk_indices(cu, 64)
        sc = 128 ** -0.5
        ref = chunk_gated_delta_rule_fwd(q=qn, k=kn, v=v, g=g, beta=beta, scale=sc, initial_state=None,
                                         output_final_state=False, cu_seqlens=cu, chunk_indices=ci, chunk_size=64)[1]
        got = plain(qn, kn, v, g, beta, cu, ci, sc)
        if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
            bad = int((got.view(torch.int16) != ref.view(torch.int16)).sum())
            return f"lens {lens[:6]}... T={T}: {bad} output elements differ"
    return ""


def d1_ok():
    if "D1" not in OK:
        try:
            why = check_d1()
        except Exception as e:  # noqa: BLE001
            why = f"check raised {type(e).__name__}: {e}"
        OK["D1"] = not why
        _log("lever D1 (gdnps nseg=1, plain path): " + ("check bit-equal, on" if OK["D1"] else f"check FAILED ({why}), off"))
    return OK["D1"]


def install_k(f8g):
    """wrap f8g.rowwise / f8g.swiglu_q: ON['K'] (and the check) -> kai's stream-K wrappers, else f8g's own functions"""
    if getattr(f8g, "_mxa_k", False):
        return
    try:
        import f8k
    except OSError as e:  # libf8k.so missing / not loadable: lever off
        _log(f"lever K unavailable: {e}")
        OK["K"] = False
        return
    r0, s0 = f8g.rowwise, f8g.swiglu_q
    f8k.install(f8g)
    rk, sk = f8g.rowwise, f8g.swiglu_q
    _f8.update(r0=r0, s0=s0, rk=rk, sk=sk, f8k=f8k)

    def rowwise(x8, xs, lin):
        return (rk if ON["K"] and k_ok() else r0)(x8, xs, lin)

    def swiglu_q(x8, xs, lin, H):
        return (sk if ON["K"] and k_ok() else s0)(x8, xs, lin, H)
    f8g.rowwise, f8g.swiglu_q = rowwise, swiglu_q
    f8g._mxa_k = True


class _Lin:
    def __init__(self, N, K, gen):
        import torch
        w = torch.randn(N, K, generator=gen, device="cuda") * 0.05
        s = w.abs().amax(1, keepdim=True).clamp(min=1e-12) / 448.0
        self.w8 = (w / s).clamp(-448, 448).to(torch.float8_e4m3fn).t()
        self.ws = s.view(1, -1).contiguous()
        self.out_features, self.in_features = N, K


def check_k(seed=5):
    """stream-K (f8k) vs f8g on random operands, every listed shape at c8 step sizes (+ the SwiGLU kind if listed)"""
    import torch
    f8k, S = _f8["f8k"], _f8["f8k"].S
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    n = 0
    for (kind, N, K) in sorted(f8k.SHAPES | {S.NAMES["down"], S.NAMES["out"]}):
        lin = _Lin(N, K, gen)
        for M in (961, 1186, 1586, 1897, 2433):
            x = torch.randn(M, K, generator=gen, device="cuda")
            xs = x.abs().amax(1, keepdim=True).clamp(min=1e-12) / 448.0
            x8 = (x / xs).clamp(-448, 448).to(torch.float8_e4m3fn)
            if kind == 0:
                a, b = _f8["r0"](x8, xs, lin), _f8["rk"](x8, xs, lin)
                same = torch.equal(a.view(torch.int16), b.view(torch.int16))
            else:
                a, b = _f8["s0"](x8, xs, lin, N // 2), _f8["sk"](x8, xs, lin, N // 2)
                same = torch.equal(a[0].view(torch.uint8), b[0].view(torch.uint8)) and torch.equal(a[1], b[1])
            n += 1
            if not same:
                return f"shape {(kind, N, K)} M {M}: stream-K != f8g"
        del lin
    torch.cuda.synchronize()
    return ""


def k_ok():
    if "K" not in OK:
        if not _f8:
            OK["K"] = False
            return False
        try:
            why = check_k()
        except Exception as e:  # noqa: BLE001
            why = f"check raised {type(e).__name__}: {e}"
        OK["K"] = not why
        _log(f"lever K (f8k stream-K, shapes {sorted(_f8['f8k'].SHAPES)}): " + ("check bit-equal, on" if OK["K"] else
                                                                               f"check FAILED ({why}), off"))
    return OK["K"]


def startup():
    """PplxEngine.__init__ end: run the checks of the levers switched on at start (never inside a CUDA graph capture)"""
    if ON["D"]:
        d_ok()
    if ON["D1"]:
        d1_ok()
    if ON["K"]:
        k_ok()
    _log(f"levers {ON} checks {OK}")
