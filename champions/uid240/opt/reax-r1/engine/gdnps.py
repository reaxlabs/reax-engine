"""gdnps.py (kai, lever D, KAI_GDNPS): FF ps mode on mammoth's state-once chain layout.

Port of rgh's v1 FF ps mode (team image rgh-g9r8k07, s1eng/fla_fused.py: _ps_pre_kernel = pass P, _ps_seq_kernel = pass
S, and kai's FF-PEEL _ps_seq_x_kernel; team code, adopted) from Plumb's padded layout (H=16, HV=32, one sequence per
row of a [B, T] batch, no initial state) to the varlen layout cu_a of PplxEngine._gdn_chain (mammoth-1: H=16 key heads,
HV=48 value heads, K=V=128, chunk 64; cu_a = [prefix sequences..., suffix sequences...], suffix i continues from the
final state of prefix i, MAM_CHAIN / gdnchain.py).

Reference it replaces (fla 0.5.2 with rgh's pins, as _gdn_chain runs it after l2norm and chunk_local_cumsum):
  kkt_solve (A, bf16, upper blocks from torch.zeros) -> recompute_w_u (w = A (k beta 2^g), u = A (v beta), bf16)
  -> gdn_fwd_h_chain_kernel (per chunk: store h (bf16), v_new = u - w h (stored bf16),
     h = h 2^g_last + k^T (v_new 2^(g_last - g)); the prefix's final state carried into its suffix as h + 0.0)
  -> chunk_fwd_kernel_o (BK 64: o = (q h) 2^g scale + (masked gated q k^T, bf16) v_new scale).
Here: kkt_solve unchanged (fla's pinned kernel) into torch.empty (pass P masks the never-written upper 16 x 16 blocks to
+0, the value the zero fill gives), then
  pass P (gdnps_pre_kernel, grid chunks x value heads): u, w and the gated masked q k^T (Aq), in fla's expressions,
         stored in bf16 where fla rounds them (u, w as recompute_w_u stores them; Aq as chunk_fwd_o's b_A.to(bf16));
  pass S (gdnps_seq_kernel, grid V blocks x (sequences x value heads)): the state chain of gdnchain with chunk_fwd_o's
         output fused in; h and v_new never leave registers (their bf16 rounds are applied in registers).
Every tl.dot has the reference's operands, K order (two 64-wide halves of K where fla splits K, BT=64 otherwise) and
accumulator start, so each output element gets the same mma sequence; only the N tiling (BV) and the warp count differ,
which do not change a per-element K-sequential mma.sync accumulation. Elementwise expressions are copied as written (same
fma contraction). PEEL (kai's FF-PEEL): the first chunk of a chain starts from h = +0, so its w h and q h dots are the
+0 they evaluate to (all products +-0, accumulator +0) and are skipped; the h update after the chain's last chunk is
never read and is skipped. Bit equality is a GPU property (Triton lowers tl.dot per architecture): check() compares
against the reference on this GPU at engine start and the engine falls back to the reference path on any mismatch.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp2


@triton.jit(do_not_specialize=['T'])
def gdnps_pre_kernel(q, k, v, beta, A, g, w, u, Aq, cu, chunk_indices, T, H: tl.constexpr, HV: tl.constexpr,
                     K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr):
    """pass P (rgh _ps_pre_kernel, varlen): one program per (chunk of cu, value head)"""
    i_c, i_h = tl.program_id(0), tl.program_id(1)
    i_n = tl.load(chunk_indices + i_c * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_c * 2 + 1).to(tl.int32)
    bos = tl.load(cu + i_n).to(tl.int32)
    Tn = tl.load(cu + i_n + 1).to(tl.int32) - bos
    qk = (bos * H + i_h // (HV // H)).to(tl.int64) * K
    vo = (bos * HV + i_h).to(tl.int64) * V
    wo = (bos * HV + i_h).to(tl.int64) * K
    ao = (bos * HV + i_h).to(tl.int64) * BT
    go = (bos * HV + i_h).to(tl.int64)
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_A = tl.arange(0, BT)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < Tn
    p_b = beta + go + o_t * HV
    b_b = tl.load(p_b, mask=m_t, other=0.0)
    p_A = A + ao + o_t[:, None] * (HV * BT) + o_A[None, :]
    # kkt_solve writes the 16 x 16 blocks on and below the diagonal; the ones above read as the zero fill's +0
    m_blk = (o_A[:, None] // 16) >= (o_A[None, :] // 16)
    b_A = tl.load(p_A, mask=m_t[:, None] & (o_A[None, :] < BT) & m_blk, other=0.0)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        p_v = v + vo + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        p_u = u + vo + o_t[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])
    p_g = g + go + o_t * HV
    b_gw = exp2(tl.load(p_g, mask=m_t, other=0.0))
    p_k = k + qk + o_t[:, None] * (H * K) + o_k1[None, :]
    b_k = tl.load(p_k, mask=m_t[:, None] & m_k1[None, :], other=0.0)
    b_kb = b_k * b_b[:, None]
    b_kb *= b_gw[:, None]
    b_w1 = tl.dot(b_A, b_kb.to(b_k.dtype)).to(k.dtype.element_ty)
    p_w = w + wo + o_t[:, None] * (HV * K) + o_k1[None, :]
    tl.store(p_w, b_w1, mask=m_t[:, None] & m_k1[None, :])
    p_k = k + qk + o_t[:, None] * (H * K) + o_k2[None, :]
    b_k = tl.load(p_k, mask=m_t[:, None] & m_k2[None, :], other=0.0)
    b_kb = b_k * b_b[:, None]
    b_kb *= b_gw[:, None]
    b_w2 = tl.dot(b_A, b_kb.to(b_k.dtype)).to(k.dtype.element_ty)
    p_w = w + wo + o_t[:, None] * (HV * K) + o_k2[None, :]
    tl.store(p_w, b_w2, mask=m_t[:, None] & m_k2[None, :])
    b_Aq = tl.zeros([BT, BT], dtype=tl.float32)
    p_q = q + qk + o_t[:, None] * (H * K) + o_k1[None, :]
    p_kt1 = k + qk + o_k1[:, None] + o_t[None, :] * (H * K)
    b_q = tl.load(p_q, mask=m_t[:, None] & m_k1[None, :], other=0.0)
    b_kt1 = tl.load(p_kt1, mask=m_k1[:, None] & m_t[None, :], other=0.0)
    b_Aq += tl.dot(b_q, b_kt1)
    p_q = q + qk + o_t[:, None] * (H * K) + o_k2[None, :]
    p_kt2 = k + qk + o_k2[:, None] + o_t[None, :] * (H * K)
    b_q = tl.load(p_q, mask=m_t[:, None] & m_k2[None, :], other=0.0)
    b_kt2 = tl.load(p_kt2, mask=m_k2[:, None] & m_t[None, :], other=0.0)
    b_Aq += tl.dot(b_q, b_kt2)
    b_go = tl.load(p_g, mask=m_t, other=0.0)
    b_Aq = b_Aq * exp2(b_go[:, None] - b_go[None, :])
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_Aq = tl.where(m_A, b_Aq, 0)
    p_Aq = Aq + ao + o_t[:, None] * (HV * BT) + o_A[None, :]
    tl.store(p_Aq, b_Aq.to(v.dtype.element_ty), mask=m_t[:, None] & (o_A[None, :] < BT))


@triton.jit
def _ps_chunk(b_h1, b_h2, i_t, Tn, q, k, w, u, Aq, g, o, scale, qk, vo, wo, ao, go, o_v, m_v, H: tl.constexpr,
              HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
              HAS_H: tl.constexpr, UPDATE: tl.constexpr):
    """one chunk of pass S (rgh _ps_seq_kernel's loop body, varlen offsets). HAS_H False: h = +0 (chain start, FF-PEEL):
    the w h and q h dots are replaced by the +0 they evaluate to. UPDATE False: the dead state update after the chain's
    last chunk is skipped (FF-PEEL)."""
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_A = tl.arange(0, BT)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < Tn
    p_u = u + vo + o_t[:, None] * (HV * V) + o_v[None, :]
    b_u = tl.load(p_u, mask=m_t[:, None] & m_v[None, :], other=0.0)
    p_g = g + go + o_t * HV
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    if HAS_H:
        p_w = w + wo + o_t[:, None] * (HV * K) + o_k1[None, :]
        b_w1 = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        p_w = w + wo + o_t[:, None] * (HV * K) + o_k2[None, :]
        b_w2 = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
        b_hs1 = b_h1.to(k.dtype.element_ty)
        b_hs2 = b_h2.to(k.dtype.element_ty)
        b_vn = tl.dot(b_w1, b_hs1.to(b_w1.dtype))
        b_vn += tl.dot(b_w2, b_hs2.to(b_w1.dtype))
        b_vn = b_u - b_vn
        p_q = q + qk + o_t[:, None] * (H * K) + o_k1[None, :]
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_o += tl.dot(b_q, b_hs1)
        p_q = q + qk + o_t[:, None] * (H * K) + o_k2[None, :]
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k2[None, :], other=0.0)
        b_o += tl.dot(b_q, b_hs2)
    else:
        b_vn = b_u - tl.zeros([BT, BV], dtype=tl.float32)
    b_vns = b_vn.to(u.dtype.element_ty)
    b_go = tl.load(p_g, mask=m_t, other=0.0)
    b_o = b_o * exp2(b_go)[:, None]
    p_Aq = Aq + ao + o_t[:, None] * (HV * BT) + o_A[None, :]
    b_Aq = tl.load(p_Aq, mask=m_t[:, None] & (o_A[None, :] < BT), other=0.0)
    # chunk_fwd_o's b_o * scale + dot * scale compiles to fma(scale, dot, round(b_o * scale)) (LLVM's fma contraction
    # picks the product); left to the compiler, pass S's h != 0 chunks fused the other product (1-ulp bf16 differences
    # in ~1e-5 of the outputs). Spelled out here so the rounding is the reference's in every instantiation.
    b_o = tl.fma(tl.dot(b_Aq, b_vns), scale, b_o * scale)
    p_o = o + vo + o_t[:, None] * (HV * V) + o_v[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])
    if UPDATE:
        p_kt1 = k + qk + o_k1[:, None] + o_t[None, :] * (H * K)
        b_kt1 = tl.load(p_kt1, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        p_kt2 = k + qk + o_k2[:, None] + o_t[None, :] * (H * K)
        b_kt2 = tl.load(p_kt2, mask=m_k2[:, None] & m_t[None, :], other=0.0)
        last_idx = tl.minimum((i_t + 1) * BT, Tn) - 1
        b_g_last = tl.load(g + (go + last_idx * HV)).to(tl.float32)
        b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
        b_vd = b_vn * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
        b_g_last = exp2(b_g_last)
        b_h1 *= b_g_last
        b_h2 *= b_g_last
        b_vd = b_vd.to(k.dtype.element_ty)
        b_h1 += tl.dot(b_kt1, b_vd)
        b_h2 += tl.dot(b_kt2, b_vd)
    return b_h1, b_h2


@triton.jit(do_not_specialize=['T'])
def gdnps_seq_kernel(q, k, w, u, Aq, g, o, cu, NSEQ, scale, T, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                     V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, NSEG: tl.constexpr, PEEL: tl.constexpr):
    """pass S: one program per (V block, chain x value head); a chain = NSEG sequences of cu (i_n, i_n + NSEQ, ...),
    the state carried from one to the next as gdnchain carries it (h + 0.0 = the reference's zeros + load(ht))"""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    for seg in tl.static_range(NSEG):
        i_s = i_n + seg * NSEQ
        bos = tl.load(cu + i_s).to(tl.int32)
        Tn = tl.load(cu + i_s + 1).to(tl.int32) - bos
        NT = tl.cdiv(Tn, BT)
        qk = (bos * H + i_h // (HV // H)).to(tl.int64) * K
        vo = (bos * HV + i_h).to(tl.int64) * V
        wo = (bos * HV + i_h).to(tl.int64) * K
        ao = (bos * HV + i_h).to(tl.int64) * BT
        go = (bos * HV + i_h).to(tl.int64)
        if PEEL:
            if seg == 0:
                if seg == NSEG - 1:
                    if NT > 1:
                        b_h1, b_h2 = _ps_chunk(b_h1, b_h2, 0, Tn, q, k, w, u, Aq, g, o, scale, qk, vo, wo, ao, go, o_v, m_v,
                                               H, HV, K, V, BT, BV, False, True)
                    else:
                        b_h1, b_h2 = _ps_chunk(b_h1, b_h2, 0, Tn, q, k, w, u, Aq, g, o, scale, qk, vo, wo, ao, go, o_v, m_v,
                                               H, HV, K, V, BT, BV, False, False)
                else:
                    b_h1, b_h2 = _ps_chunk(b_h1, b_h2, 0, Tn, q, k, w, u, Aq, g, o, scale, qk, vo, wo, ao, go, o_v, m_v,
                                           H, HV, K, V, BT, BV, False, True)
                lo = 1
            else:
                lo = 0
            if seg == NSEG - 1:
                for i_t in range(lo, NT - 1):
                    b_h1, b_h2 = _ps_chunk(b_h1, b_h2, i_t, Tn, q, k, w, u, Aq, g, o, scale, qk, vo, wo, ao, go, o_v, m_v,
                                           H, HV, K, V, BT, BV, True, True)
                if NT - 1 >= lo:
                    b_h1, b_h2 = _ps_chunk(b_h1, b_h2, NT - 1, Tn, q, k, w, u, Aq, g, o, scale, qk, vo, wo, ao, go, o_v,
                                           m_v, H, HV, K, V, BT, BV, True, False)
            else:
                for i_t in range(lo, NT):
                    b_h1, b_h2 = _ps_chunk(b_h1, b_h2, i_t, Tn, q, k, w, u, Aq, g, o, scale, qk, vo, wo, ao, go, o_v, m_v,
                                           H, HV, K, V, BT, BV, True, True)
        else:
            for i_t in range(0, NT):
                b_h1, b_h2 = _ps_chunk(b_h1, b_h2, i_t, Tn, q, k, w, u, Aq, g, o, scale, qk, vo, wo, ao, go, o_v, m_v,
                                       H, HV, K, V, BT, BV, True, True)
        if seg < NSEG - 1:
            b_h1 = b_h1 + 0.0
            b_h2 = b_h2 + 0.0


def chain(qn, kn, v, gc, beta, cu, ci, nseq, scale, nseg=2, BV=32, num_warps=4, num_stages=1, pre_warps=2,
          peel=True):
    """o [1, T, HV, V] of the chain layout: qn / kn [1, T, H, K] l2-normalised, v [1, T, HV, V], gc [1, T, HV] (chunk-local
    cumsum, log2 scale), beta [1, T, HV], cu int64 [nseg * nseq + 1] (chain i = sequences i, nseq + i, ...), ci =
    prepare_chunk_indices(cu, 64). nseg=2: _gdn_chain's prefix -> suffix layout; nseg=1: independent sequences, h0 = 0."""
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
    B, T, H, K = kn.shape
    HV, V = v.shape[2], v.shape[3]
    assert B == 1 and K == 128 and HV % H == 0 and cu.numel() == nseg * nseq + 1, (kn.shape, v.shape, cu.numel(), nseq)
    BT = 64
    NC = ci.shape[0]
    A = torch.empty(B, T, HV, BT, device=kn.device, dtype=kn.dtype)
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NC, B * HV)](k=kn, g=gc, beta=beta, A=A, cu_seqlens=cu, chunk_indices=ci,
                                                              T=T, H=H, HV=HV, K=K, BT=BT, BC=16)
    w = kn.new_empty(B, T, HV, K)
    u = torch.empty_like(v)
    Aq = torch.empty(B, T, HV, BT, device=kn.device, dtype=v.dtype)
    gdnps_pre_kernel[(NC, HV)](qn, kn, v, beta, A, gc, w, u, Aq, cu, ci, T, H=H, HV=HV, K=K, V=V, BT=BT, BV=BV,
                               num_warps=pre_warps, num_stages=1)
    o = torch.empty_like(v)
    gdnps_seq_kernel[(triton.cdiv(V, BV), nseq * HV)](qn, kn, w, u, Aq, gc, o, cu, nseq, scale, T, H=H, HV=HV, K=K, V=V,
                                                      BT=BT, BV=BV, NSEG=nseg, PEEL=peel, num_warps=num_warps,
                                                      num_stages=num_stages)
    return o


def reference(qn, kn, v, gc, beta, cu, ci, nseq, scale):
    """_gdn_chain's stages after l2norm / cumsum without GDNPS (fla intra -> gdnchain -> chunk_fwd_o), same pins"""
    import gdnchain
    from fla.ops.common.chunk_o import chunk_fwd_o
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.utils.index import prepare_chunk_offsets
    w, u, _A = chunk_gated_delta_rule_fwd_intra(k=kn, v=v, g=gc, beta=beta, cu_seqlens=cu, chunk_indices=ci, chunk_size=64)
    h, v_new = gdnchain.chain_fwd_h(kn, w, u, gc, cu, prepare_chunk_offsets(cu, 64), ci.shape[0], nseq)
    return chunk_fwd_o(q=qn, k=kn, v=v_new, h=h, g=gc, scale=scale, cu_seqlens=cu, chunk_indices=ci, state_v_first=False,
                       chunk_size=64)


def random_chain(nseq, H=16, HV=48, K=128, V=128, p_lens=None, s_lens=None, gen=None, device="cuda"):
    """random chain-layout inputs of serving shapes (as _gdn_chain gets them after l2norm and cumsum)"""
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.index import prepare_chunk_indices
    import torch.nn.functional as F
    r = lambda *s: torch.randn(*s, generator=gen, device=device)  # noqa: E731
    if p_lens is None:
        p_lens = torch.randint(1, 300, (nseq,), generator=gen, device=device).tolist()
    if s_lens is None:
        s_lens = torch.randint(1, 130, (nseq,), generator=gen, device=device).tolist()
    cum = lambda ls: [0] + torch.tensor(ls).cumsum(0).tolist()  # noqa: E731
    cu_p, cu_s = cum(p_lens), cum(s_lens)
    Tp, T = cu_p[-1], cu_p[-1] + cu_s[-1]
    cu = torch.tensor(cu_p + [Tp + x for x in cu_s[1:]], dtype=torch.int64, device=device)
    q, k = r(1, T, H, K).to(torch.bfloat16), r(1, T, H, K).to(torch.bfloat16)
    v = r(1, T, HV, V).to(torch.bfloat16)
    g = (-F.softplus(r(1, T, HV)) * 0.3).float()
    beta = torch.sigmoid(r(1, T, HV)).to(torch.bfloat16)
    qn, kn = l2norm_fwd(q)[0], l2norm_fwd(k)[0]
    ci = prepare_chunk_indices(cu, 64)
    gc = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2, cu_seqlens=cu, chunk_indices=ci)
    return dict(qn=qn, kn=kn, v=v, gc=gc, beta=beta, cu=cu, ci=ci, nseq=nseq, scale=K ** -0.5, Tp=Tp)


@torch.no_grad()
def check(cfg=(32, 4, 1, 2), peel=True, seed=47) -> str:
    """start-up gate on THIS GPU: chain() vs the reference stages (reference()) on random chain layouts of serving
    shapes, incl. lengths 1, 64, 128 and partial chunks. "" when every output element is bit-equal, else the reason."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    layouts = [(1, [64], [1]), (2, [1, 128], [64, 3]), (3, None, None), (8, None, None), (16, None, None),
               (5, [300, 17, 64, 129, 2], [130, 1, 65, 64, 33])]
    for nseq, pl, sl in layouts:
        x = random_chain(nseq, p_lens=pl, s_lens=sl, gen=gen)
        ref = reference(x["qn"], x["kn"], x["v"], x["gc"], x["beta"], x["cu"], x["ci"], nseq, x["scale"])
        got = chain(x["qn"], x["kn"], x["v"], x["gc"], x["beta"], x["cu"], x["ci"], nseq, x["scale"], BV=cfg[0],
                    num_warps=cfg[1], num_stages=cfg[2], pre_warps=cfg[3], peel=peel)
        if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
            bad = int((got.view(torch.int16) != ref.view(torch.int16)).sum())
            return f"nseq={nseq} T={ref.shape[1]}: {bad} output elements differ from the reference"
    return ""
