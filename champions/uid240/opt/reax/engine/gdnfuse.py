"""gdnfuse.py (kai, round-6 lever #5, KAI_GDNFUSE): two exact GDN fusions on mammoth's state-once chain path.

  pre   fla's pinned chunk_gated_delta_rule_fwd_kkt_solve_kernel (BK 64) fused into lever D's pass P (gdnps_pre_kernel):
        one program per (chunk, value head) computes (I + A)^-1 in registers with fla's code (same dots, same tf32 block
        merge, same forward substitution, same bf16 rounding of the stored A) and feeds it to pass P's u / w dots, which
        read it from memory in lever D. A never leaves registers: no A buffer, no kkt launch, no A store / load.
        The 4 x 4 grid of 16 x 16 blocks is assembled into pass P's [64, 64] operand by tl.join / permute / reshape (data
        movement only); the blocks above the diagonal and the rows past the sequence end are +0, the values pass P's
        masked load gives. Acts where lever D acts (KAI_GDNPS=1, its start-up check passed), opt-in KAI_GDNFUSE_PRE=1.
        NOT EXACT on the RTX PRO 6000 (gdnfusebench + debug, 28 Oct): the assembled A is bit-equal to kkt_solve's stored A
        at 1 / 2 / 4 warps, but pass P's u / w dots differ in ~1e-5 of the elements (1 bf16 ulp) when their A operand
        comes from registers instead of a memory load (the Aq dot, which does not read A, is equal); its check fails.
        It is also not faster at c8 (pass P 1 / 2 / 4 warps: +0.00 / -0.44 / -1.70 ms per step; c1 +0.3-0.7, c16 0..+0.9).
  conv  q / k l2norm (fla l2norm_fwd_kernel, pinned BT16 / w4) fused into the q / k column blocks of rgh's _sconv3_kernel
        (one 128-wide block = one head = one l2norm row; the conv output is rounded to bf16 exactly as it was stored and
        normalised in fp32 with fla's expressions), and fla's chunk_local_cumsum_scalar_kernel (g -> chunk-local cumsum
        x RCP_LN2, the pinned w2 kernel) run by extra programs of the same launch (one per (chunk, value head), fla's
        body verbatim). Removes 3 launches per GDN layer and the q / k round trip. The DeltaNet gates (K.dn_gates) are
        computed before the conv instead of after it (independent inputs, same values). Chain path only.

Exact by construction only up to the GPU: Triton may pick another register layout inside a fused kernel, and a row sum /
scan / substitution over a different layout can round differently. check() therefore compares every output element of
both fusions against the unfused kernels on THIS GPU at engine start (random chain layouts of serving shapes, incl. lengths
1, 63, 64, 65, 128 and partial chunks); a part with any differing bit stays off (the engine keeps the unfused path).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.gated_delta_rule.chunk_fwd import SOLVE_TRIL_DOT_PRECISION
from fla.ops.utils.op import exp2

BC = 16  # fla's kkt_solve sub-chunk
BK = 64  # rgh's kkt_solve pin (_FLA_PINS: {"BK": 64})


@triton.jit
def _cat_cols(a, b):
    """[R, C], [R, C] -> [R, 2C]: a's columns, then b's (data movement only)"""
    R: tl.constexpr = a.shape[0]
    C: tl.constexpr = a.shape[1]
    return tl.reshape(tl.permute(tl.join(a, b), (0, 2, 1)), (R, 2 * C))


@triton.jit
def _cat_rows(a, b):
    """[R, C], [R, C] -> [2R, C]: a's rows, then b's (data movement only)"""
    R: tl.constexpr = a.shape[0]
    C: tl.constexpr = a.shape[1]
    return tl.reshape(tl.permute(tl.join(a, b), (2, 0, 1)), (2 * R, C))


@triton.jit
def _kkt_solve(k, g, beta, bos, i_t, T, i_h, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
               BC: tl.constexpr, BK: tl.constexpr):
    """fla 0.5.2 chunk_gated_delta_rule_fwd_kkt_solve_kernel (USE_G, IS_VARLEN), steps 1-4 verbatim (MIT); returns the
    ten blocks of (I + A)^-1 in fp32 instead of storing them (step 5)"""
    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    k += (bos * H + i_h // (HV // H)) * K

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    p_b0 = beta + bos * HV + i_h + (i_tc0 + o_i) * HV
    p_b1 = beta + bos * HV + i_h + (i_tc1 + o_i) * HV
    p_b2 = beta + bos * HV + i_h + (i_tc2 + o_i) * HV
    p_b3 = beta + bos * HV + i_h + (i_tc3 + o_i) * HV
    b_b0 = tl.load(p_b0, mask=m_tc0, other=0.0).to(tl.float32)
    b_b1 = tl.load(p_b1, mask=m_tc1, other=0.0).to(tl.float32)
    b_b2 = tl.load(p_b2, mask=m_tc2, other=0.0).to(tl.float32)
    b_b3 = tl.load(p_b3, mask=m_tc3, other=0.0).to(tl.float32)

    p_g0 = g + bos * HV + i_h + (i_tc0 + o_i) * HV
    p_g1 = g + bos * HV + i_h + (i_tc1 + o_i) * HV
    p_g2 = g + bos * HV + i_h + (i_tc2 + o_i) * HV
    p_g3 = g + bos * HV + i_h + (i_tc3 + o_i) * HV
    b_g0 = tl.load(p_g0, mask=m_tc0, other=0.0).to(tl.float32)
    b_g1 = tl.load(p_g1, mask=m_tc1, other=0.0).to(tl.float32)
    b_g2 = tl.load(p_g2, mask=m_tc2, other=0.0).to(tl.float32)
    b_g3 = tl.load(p_g3, mask=m_tc3, other=0.0).to(tl.float32)

    b_A00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A33 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_k0 = k + (i_tc0 + o_i)[:, None] * (H*K) + o_k[None, :]
        b_k0 = tl.load(p_k0, mask=m_tc0[:, None] & (o_k[None, :] < K), other=0.0)
        b_A00 += tl.dot(b_k0, tl.trans(b_k0))

        if i_tc1 < T:
            p_k1 = k + (i_tc1 + o_i)[:, None] * (H*K) + o_k[None, :]
            b_k1 = tl.load(p_k1, mask=m_tc1[:, None] & (o_k[None, :] < K), other=0.0)
            b_A11 += tl.dot(b_k1, tl.trans(b_k1))
            b_A10 += tl.dot(b_k1, tl.trans(b_k0))

            if i_tc2 < T:
                p_k2 = k + (i_tc2 + o_i)[:, None] * (H*K) + o_k[None, :]
                b_k2 = tl.load(p_k2, mask=m_tc2[:, None] & (o_k[None, :] < K), other=0.0)
                b_A22 += tl.dot(b_k2, tl.trans(b_k2))
                b_A20 += tl.dot(b_k2, tl.trans(b_k0))
                b_A21 += tl.dot(b_k2, tl.trans(b_k1))

                if i_tc3 < T:
                    p_k3 = k + (i_tc3 + o_i)[:, None] * (H*K) + o_k[None, :]
                    b_k3 = tl.load(p_k3, mask=m_tc3[:, None] & (o_k[None, :] < K), other=0.0)
                    b_A33 += tl.dot(b_k3, tl.trans(b_k3))
                    b_A30 += tl.dot(b_k3, tl.trans(b_k0))
                    b_A31 += tl.dot(b_k3, tl.trans(b_k1))
                    b_A32 += tl.dot(b_k3, tl.trans(b_k2))

    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_A00 *= tl.where(m_d & m_tc0[:, None] & m_tc0[None, :], exp2(b_g0[:, None] - b_g0[None, :]), 0.)
    b_A11 *= tl.where(m_d & m_tc1[:, None] & m_tc1[None, :], exp2(b_g1[:, None] - b_g1[None, :]), 0.)
    b_A22 *= tl.where(m_d & m_tc2[:, None] & m_tc2[None, :], exp2(b_g2[:, None] - b_g2[None, :]), 0.)
    b_A33 *= tl.where(m_d & m_tc3[:, None] & m_tc3[None, :], exp2(b_g3[:, None] - b_g3[None, :]), 0.)

    b_A10 *= tl.where(m_tc1[:, None] & m_tc0[None, :], exp2(b_g1[:, None] - b_g0[None, :]), 0.)
    b_A20 *= tl.where(m_tc2[:, None] & m_tc0[None, :], exp2(b_g2[:, None] - b_g0[None, :]), 0.)
    b_A21 *= tl.where(m_tc2[:, None] & m_tc1[None, :], exp2(b_g2[:, None] - b_g1[None, :]), 0.)
    b_A30 *= tl.where(m_tc3[:, None] & m_tc0[None, :], exp2(b_g3[:, None] - b_g0[None, :]), 0.)
    b_A31 *= tl.where(m_tc3[:, None] & m_tc1[None, :], exp2(b_g3[:, None] - b_g1[None, :]), 0.)
    b_A32 *= tl.where(m_tc3[:, None] & m_tc2[None, :], exp2(b_g3[:, None] - b_g2[None, :]), 0.)

    b_A00 = b_A00 * b_b0[:, None]
    b_A11 = b_A11 * b_b1[:, None]
    b_A22 = b_A22 * b_b2[:, None]
    b_A33 = b_A33 * b_b3[:, None]

    b_A10 = b_A10 * b_b1[:, None]
    b_A20 = b_A20 * b_b2[:, None]
    b_A21 = b_A21 * b_b2[:, None]
    b_A30 = b_A30 * b_b3[:, None]
    b_A31 = b_A31 * b_b3[:, None]
    b_A32 = b_A32 * b_b3[:, None]

    b_Ai00 = -b_A00
    b_Ai11 = -b_A11
    b_Ai22 = -b_A22
    b_Ai33 = -b_A33

    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.), 0)
        b_a00 = tl.where(o_i < i, b_a00, 0.)
        b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(2, min(BC, T - i_tc1)):
        b_a11 = tl.sum(tl.where((o_i == i)[:, None], -b_A11, 0.), 0)
        b_a11 = tl.where(o_i < i, b_a11, 0.)
        b_a11 = b_a11 + tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i)[:, None], b_a11, b_Ai11)
    for i in range(2, min(BC, T - i_tc2)):
        b_a22 = tl.sum(tl.where((o_i == i)[:, None], -b_A22, 0.), 0)
        b_a22 = tl.where(o_i < i, b_a22, 0.)
        b_a22 = b_a22 + tl.sum(b_a22[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where((o_i == i)[:, None], b_a22, b_Ai22)
    for i in range(2, min(BC, T - i_tc3)):
        b_a33 = tl.sum(tl.where((o_i == i)[:, None], -b_A33, 0.), 0)
        b_a33 = tl.where(o_i < i, b_a33, 0.)
        b_a33 = b_a33 + tl.sum(b_a33[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where((o_i == i)[:, None], b_a33, b_Ai33)

    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_A10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_A21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_A32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_A20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_A21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_A31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_A32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_A30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_A31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_A32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    # step 5 (the store) is replaced by the assembly in gdnfp_pre_kernel
    z = tl.zeros([BC, BC], dtype=tl.float32)
    r0 = _cat_cols(_cat_cols(b_Ai00, z), _cat_cols(z, z))
    r1 = _cat_cols(_cat_cols(b_Ai10, b_Ai11), _cat_cols(z, z))
    r2 = _cat_cols(_cat_cols(b_Ai20, b_Ai21), _cat_cols(b_Ai22, z))
    r3 = _cat_cols(_cat_cols(b_Ai30, b_Ai31), _cat_cols(b_Ai32, b_Ai33))
    return _cat_rows(_cat_rows(r0, r1), _cat_rows(r2, r3))


@triton.jit(do_not_specialize=['T'])
def gdnfp_pre_kernel(q, k, v, beta, g, w, u, Aq, cu, chunk_indices, T, H: tl.constexpr, HV: tl.constexpr,
                     K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
                     BK: tl.constexpr):
    """kkt_solve + pass P (gdnps_pre_kernel) in one program per (chunk of cu, value head); A stays in registers"""
    i_c, i_h = tl.program_id(0), tl.program_id(1)
    i_n = tl.load(chunk_indices + i_c * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_c * 2 + 1).to(tl.int32)
    bos = tl.load(cu + i_n).to(tl.int32)
    Tn = tl.load(cu + i_n + 1).to(tl.int32) - bos
    # (I + A)^-1 of this chunk as fla's kkt_solve computes it (its i_t is int64, its T the sequence length)
    b_Af = _kkt_solve(k, g, beta, bos, i_t.to(tl.int64), Tn, i_h, H, HV, K, BT, BC, BK)
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
    # lever D's load of A: kkt_solve's bf16 store (round to nearest even), +0 above the 16 x 16 block diagonal (the zero
    # fill / masked load) and in rows past the sequence end (masked load)
    m_blk = (o_A[:, None] // 16) >= (o_A[None, :] // 16)
    b_A = tl.where(m_t[:, None] & (o_A[None, :] < BT) & m_blk, b_Af, 0.0).to(k.dtype.element_ty)
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


def chain(qn, kn, v, gc, beta, cu, ci, nseq, scale, nseg=2, BV=32, num_warps=4, num_stages=1, pre_warps=2, peel=True):
    """== gdnps.chain (same arguments, same output bits) with kkt_solve fused into pass P"""
    import gdnps
    B, T, H, K = kn.shape
    HV, V = v.shape[2], v.shape[3]
    assert B == 1 and K == 128 and HV % H == 0 and cu.numel() == nseg * nseq + 1, (kn.shape, v.shape, cu.numel(), nseq)
    BT = 64
    NC = ci.shape[0]
    w = kn.new_empty(B, T, HV, K)
    u = torch.empty_like(v)
    Aq = torch.empty(B, T, HV, BT, device=kn.device, dtype=v.dtype)
    gdnfp_pre_kernel[(NC, HV)](qn, kn, v, beta, gc, w, u, Aq, cu, ci, T, H=H, HV=HV, K=K, V=V, BT=BT, BV=BV, BC=BC, BK=BK,
                               num_warps=pre_warps, num_stages=1)
    o = torch.empty_like(v)
    gdnps.gdnps_seq_kernel[(triton.cdiv(V, BV), nseq * HV)](qn, kn, w, u, Aq, gc, o, cu, nseq, scale, T, H=H, HV=HV, K=K,
                                                            V=V, BT=BT, BV=BV, NSEG=nseg, PEEL=peel,
                                                            num_warps=num_warps, num_stages=num_stages)
    return o


# ------------------------------------------------------------------------------------------------ conv + l2norm + cumsum
@triton.jit
def _cumsum_chunk(s, o, scale, cu_seqlens, chunk_indices, i_c, i_h, H: tl.constexpr, BT: tl.constexpr):
    """fla 0.5.2 chunk_local_cumsum_scalar_kernel (IS_VARLEN, HAS_SCALE, not REVERSE / HEAD_FIRST, B = 1) for program
    (i_c, i_h), verbatim (MIT)"""
    i_n, i_t = tl.load(chunk_indices + i_c * 2).to(tl.int32), tl.load(chunk_indices + i_c * 2 + 1).to(tl.int64)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_s = s + bos*H + i_h + o_t * H
    p_o = o + bos*H + i_h + o_t * H
    b_s = tl.load(p_s, mask=m_t, other=0.0).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t)


@triton.jit
def sconv3n_kernel(x, src, weight, yq, yk, yv, T, stride_x_t, eps, gs, go, cu, ci, NGC, gscale, KD: tl.constexpr,
                   VD: tl.constexpr, W: tl.constexpr, BT: tl.constexpr, BW: tl.constexpr, BD: tl.constexpr,
                   HV: tl.constexpr, CBT: tl.constexpr):
    """programs i_d < ND: rgh's _sconv3_kernel (q / k / v = silu(causal conv) through the lag table), q and k blocks
    l2-normalised per row as fla's l2norm_fwd_kernel; programs i_d >= ND: fla's chunk-local cumsum of g, one
    (chunk, value head) each (NGC = chunks x HV of them)"""
    i_d, i_t = tl.program_id(0), tl.program_id(1)
    ND: tl.constexpr = (2 * KD + VD + BD - 1) // BD
    if i_d >= ND:
        f = (i_d - ND) * tl.num_programs(1) + i_t
        if f < NGC:
            _cumsum_chunk(gs, go, gscale, cu, ci, f // HV, f % HV, HV, CBT)
    else:
        c0 = i_d * BD
        if c0 < KD:
            y = yq
            oc = c0
            DO = KD
        elif c0 < 2 * KD:
            y = yk
            oc = c0 - KD
            DO = KD
        else:
            y = yv
            oc = c0 - 2 * KD
            DO = VD
        D = 2 * KD + VD
        o_d = c0 + tl.arange(0, BD)
        o_t = i_t * BT + tl.arange(0, BT)
        o_w = tl.arange(0, BW) + W - BW
        m_d = o_d < D
        m_w = o_w >= 0
        m_t = o_t < T
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)
        b_y = tl.zeros((BT, BD), dtype=tl.float32)
        for i_w in tl.static_range(-W + 1, 1):
            if i_w == 0:
                o_x = o_t.to(tl.int64)
                m_x = m_t
            else:
                o_x = tl.load(src + o_t * (W - 1) + (-i_w - 1), mask=m_t, other=-1).to(tl.int64)
                m_x = o_x >= 0
            p_yi = x + o_x[:, None] * stride_x_t + o_d[None, :]
            b_yi = tl.load(p_yi, mask=m_x[:, None] & m_d[None, :], other=0.0).to(tl.float32)
            b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
        b_y = b_y * tl.sigmoid(b_y)
        o_o = oc + tl.arange(0, BD)
        p_y = y + o_t[:, None] * DO + o_o[None, :]
        m_y = m_t[:, None] & (o_o < DO)[None, :]
        if c0 < 2 * KD:
            # the conv's bf16 store, re-read as fla's l2norm_fwd_kernel reads it (one head = one row of BD = 128)
            b_x = tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding='rtne').to(tl.float32)
            b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
            b_n = b_x * b_rstd[:, None]
            tl.store(p_y, b_n.to(p_y.dtype.element_ty), mask=m_y)
        else:
            tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_y)


def sconv3n(mixed, src, w, kd, vd, T, g, cu, ci, eps=1e-6, gscale=None, head_dim=128):
    """(qn, kn, v, gc): qn / kn = l2norm_fwd(silu(conv) q / k) [T, kd], v = silu(conv) [T, vd] (as sconv3_triton),
    gc = chunk_local_cumsum(g, 64, scale=gscale (default RCP_LN2), cu_seqlens=cu, chunk_indices=ci) [1, T, HV] fp32"""
    from fla.ops.utils.constant import RCP_LN2
    BT, BD, CBT = 64, 128, 64
    assert head_dim == BD and kd % BD == 0, (kd, head_dim)
    assert g.dim() == 3 and g.shape[0] == 1 and g.shape[1] == T and g.is_contiguous(), g.shape
    HV = g.shape[2]
    yq = torch.empty(T, kd, device=mixed.device, dtype=mixed.dtype)
    yk = torch.empty(T, kd, device=mixed.device, dtype=mixed.dtype)
    yv = torch.empty(T, vd, device=mixed.device, dtype=mixed.dtype)
    gc = torch.empty_like(g, dtype=torch.float32)
    ntb = triton.cdiv(T, BT)
    nd = triton.cdiv(2 * kd + vd, BD)
    ngc = ci.shape[0] * HV
    sconv3n_kernel[(nd + triton.cdiv(ngc, ntb), ntb)](mixed, src, w, yq, yk, yv, T, mixed.stride(0), eps, g, gc, cu, ci,
                                                      ngc, RCP_LN2 if gscale is None else gscale, KD=kd, VD=vd,
                                                      W=w.shape[1], BT=BT, BW=triton.next_power_of_2(w.shape[1]), BD=BD,
                                                      HV=HV, CBT=CBT, num_warps=4)
    return yq, yk, yv, gc


def conv_reference(mixed, src, w, kd, vd, T, g, cu, ci, head_dim=128):
    """the unfused stages sconv3n replaces: rgh's sconv3_triton, fla's l2norm_fwd (pinned) on q and k, fla's
    chunk_local_cumsum (pinned) on g"""
    import pplx_engine as P
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2
    q, k, v = P.sconv3_triton(mixed, src, w, kd, vd, T)
    qn = l2norm_fwd(q.view(1, T, kd // head_dim, head_dim))[0].view(T, kd)
    kn = l2norm_fwd(k.view(1, T, kd // head_dim, head_dim))[0].view(T, kd)
    gc = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2, cu_seqlens=cu, chunk_indices=ci)
    return qn, kn, v, gc


# ------------------------------------------------------------------------------------------------ start-up check
CHECKED = {"pre": 0, "conv": 0}  # output elements the last check compared (0 = nothing: the check fails, never passes)


def _nontrivial(ref) -> bool:
    """a reference output the check may count: elements present and not all zero (an empty or all-zero reference means a
    broken harness, and two equal empty / zero tensors must not pass the check)"""
    return ref.numel() > 0 and bool((ref != 0).any())


def _bits_equal(a, b) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


_LAYOUTS = [(1, [64], [1]), (2, [1, 128], [64, 3]), (3, None, None), (8, None, None), (16, None, None),
            (5, [300, 17, 64, 129, 63], [130, 1, 65, 64, 33]), (4, [65, 4, 191, 255], [63, 2, 127, 192])]


def check_pre(cfg=(32, 4, 1, 2), peel=True, seed=53, layouts=None, H=16, HV=48, device="cuda") -> str:
    """the pre fusion vs lever D's chain (gdnps.chain: fla's kkt_solve + pass P + pass S) on random chain layouts of
    serving shapes: "" when every output element is bit-equal, else the reason. Also compares pass P's w, u, Aq."""
    import gdnps
    CHECKED["pre"] = 0
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    n = 0
    for nseq, pl, sl in (layouts or _LAYOUTS):
        x = gdnps.random_chain(nseq, H=H, HV=HV, p_lens=pl, s_lens=sl, gen=gen, device=device)
        a = (x["qn"], x["kn"], x["v"], x["gc"], x["beta"], x["cu"], x["ci"], nseq, x["scale"])
        kw = dict(BV=cfg[0], num_warps=cfg[1], num_stages=cfg[2], pre_warps=cfg[3], peel=peel)
        ref = gdnps.chain(*a, **kw)
        got = chain(*a, **kw)
        if not _nontrivial(ref):
            return f"pre: nseq={nseq}: nothing compared (empty or all-zero reference output)"
        if not _bits_equal(got, ref):
            bad = int((got.view(torch.int16) != ref.view(torch.int16)).sum())
            return f"pre: nseq={nseq} T={ref.shape[1]}: {bad} output elements differ from lever D's chain"
        n += ref.numel()
    if n == 0:
        return "pre: nothing compared (no layouts)"
    CHECKED["pre"] = n
    return ""


def random_conv(nseq, kd=2048, vd=6144, HV=48, W=4, p_lens=None, s_lens=None, gen=None, device="cuda", extra=1024):
    """random inputs of _deltanet_shared_sc's conv at serving shapes: mixed [T, 2kd + vd] as a row-strided view (the
    projection output has more columns), the state-once lag table, conv weights [2kd + vd, W], gates g [1, T, HV]"""
    import numpy as np
    import pplx_engine as P
    from fla.ops.utils.index import prepare_chunk_indices
    r = lambda *s: torch.randn(*s, generator=gen, device=device)  # noqa: E731
    if p_lens is None:
        p_lens = torch.randint(W, 300, (nseq,), generator=gen, device=device).tolist()
    if s_lens is None:
        s_lens = torch.randint(1, 130, (nseq,), generator=gen, device=device).tolist()
    cum = lambda ls: [0] + torch.tensor(ls).cumsum(0).tolist()  # noqa: E731
    cu_p, cu_s = cum(p_lens), cum(s_lens)
    Tp, T = cu_p[-1], cu_p[-1] + cu_s[-1]
    lag = P.shared_lag_table(np.asarray(cu_p), np.asarray(p_lens), np.asarray(cu_s), np.asarray(s_lens),
                             np.arange(nseq), W - 1)
    D = 2 * kd + vd
    big = (r(T, D + extra) * 2).to(torch.bfloat16)
    mixed = big[:, extra // 2: extra // 2 + D]
    w = (r(D, W) * 0.5).to(torch.bfloat16)
    g = (-torch.nn.functional.softplus(r(1, T, HV)) * 0.3).float().contiguous()
    cu = torch.tensor(cu_p + [Tp + s for s in cu_s[1:]], dtype=torch.int64, device=device)
    return dict(mixed=mixed, src=torch.from_numpy(lag).to(device), w=w, kd=kd, vd=vd, T=T, g=g, cu=cu,
                ci=prepare_chunk_indices(cu, 64))


def check_conv(seed=59, layouts=None, kd=2048, vd=6144, HV=48, device="cuda") -> str:
    """sconv3n vs conv_reference on random serving-shape layouts: "" when qn, kn, v and gc are bit-equal, else the
    reason"""
    CHECKED["conv"] = 0
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    n = 0
    for nseq, pl, sl in (layouts or _LAYOUTS):
        pl = None if pl is None else [max(4, p) for p in pl]  # mammoth state blocks: every prefix >= the conv width
        x = random_conv(nseq, kd=kd, vd=vd, HV=HV, p_lens=pl, s_lens=sl, gen=gen, device=device)
        a = (x["mixed"], x["src"], x["w"], kd, vd, x["T"], x["g"], x["cu"], x["ci"])
        ref = conv_reference(*a)
        got = sconv3n(*a)
        if len(ref) != 4 or len(got) != 4:
            return f"conv: nseq={nseq}: {len(got)} outputs vs {len(ref)} reference outputs (expected qn, kn, v, gc)"
        for name, r_, g_ in zip(("qn", "kn", "v", "gc"), ref, got):
            if not _nontrivial(r_):
                return f"conv: nseq={nseq}: {name}: nothing compared (empty or all-zero reference output)"
            if not _bits_equal(g_, r_):
                if g_.shape != r_.shape or g_.dtype != r_.dtype:
                    return f"conv: nseq={nseq}: {name} {tuple(g_.shape)} {g_.dtype} vs {tuple(r_.shape)} {r_.dtype}"
                bad = int((g_.reshape(-1).view(torch.int16 if g_.dtype != torch.float32 else torch.int32)
                           != r_.reshape(-1).view(torch.int16 if r_.dtype != torch.float32 else torch.int32)).sum())
                return f"conv: nseq={nseq} T={x['T']}: {bad} {name} elements differ from the unfused kernels"
            n += r_.numel()
    if n == 0:
        return "conv: nothing compared (no layouts)"
    CHECKED["conv"] = n
    return ""


@torch.no_grad()
def check(pre_cfg=None, peel=True) -> dict:
    """start-up gate on THIS GPU: {"pre": reason or "" (or None when lever D is off), "conv": reason or ""}"""
    out = {"pre": None, "conv": None}
    if pre_cfg is not None:
        try:
            out["pre"] = check_pre(cfg=pre_cfg, peel=peel)
        except Exception as e:  # noqa: BLE001 - a compile error must only switch the part off
            out["pre"] = f"pre: check raised {type(e).__name__}: {e}"
    try:
        out["conv"] = check_conv()
    except Exception as e:  # noqa: BLE001
        out["conv"] = f"conv: check raised {type(e).__name__}: {e}"
    return out
