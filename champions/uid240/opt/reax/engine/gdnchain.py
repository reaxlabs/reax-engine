"""gdnchain.py (rgh f8m, MAM_CHAIN): the state recurrence of the state-once DeltaNet path for steps whose every group has
exactly ONE suffix, in order (unique states): one launch walks each decision's prefix chunks and then its suffix chunks,
with the recurrent state kept in registers in between.

Reference behaviour it replaces (fla 0.5.2 chunk_gated_delta_rule_fwd_h, called twice by PplxEngine._gdn_merged):
  call 1 (prefix segments, h0 = None, output_final_state=True): zero-fills ht [N, HV, K, V] fp32, runs the recurrence,
         stores ht;   call 2 (suffix segments, initial_state = ht[s_par]): b_h = zeros + load(ht), runs the recurrence.
The kernel below is chunk_gated_delta_rule_fwd_kernel_h_blockdim64 (fla 0.5.2, MIT, Songlin Yang, Yu Zhang, Zhiyuan Li et
al.) specialised to that call pattern (K = 128 -> two 64-row state blocks, STATE_V_FIRST False, USE_G True, USE_GK False,
SAVE_NEW_VALUE True, IS_VARLEN True), with the per-chunk body copied op for op, so every h / v_new store and the state
carried into the suffix are the same fp32 values; the fp32 ht store + reload of the reference is lossless.
Segments live in ONE combined varlen layout cu_a = [prefix sequences..., suffix sequences...] (the layout the merged
chunk-local stages already use), so h / v_new come out in cu_a chunk order and a single chunk_fwd_o call covers both.
"""
import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp2


@triton.jit(do_not_specialize=['T'])
def gdn_fwd_h_chain_kernel(k, v, w, v_new, g, h, cu_a, chunk_offsets, NSEQ, T, H: tl.constexpr, HV: tl.constexpr,
                           K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    for seg in tl.static_range(2):
        i_s = i_n + seg * NSEQ  # prefix i_n, then its suffix (sequence NSEQ + i_n of cu_a)
        bos, eos = tl.load(cu_a + i_s).to(tl.int32), tl.load(cu_a + i_s + 1).to(tl.int32)
        Tn = eos - bos
        NT = tl.cdiv(Tn, BT)
        boh = tl.load(chunk_offsets + i_s).to(tl.int32)
        hs = h + (boh * HV + i_h).to(tl.int64) * K*V
        vs = v + (bos * HV + i_h).to(tl.int64) * V
        ks = k + (bos * H + i_h // (HV // H)).to(tl.int64) * K
        ws = w + (bos * HV + i_h).to(tl.int64) * K
        vns = v_new + (bos * HV + i_h).to(tl.int64) * V
        for i_t in range(NT):
            i_t_int64 = i_t.to(tl.int64)
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < Tn
            p_h1 = hs + i_t_int64 * HV*K*V + o_k1[:, None] * V + o_v[None, :]
            m_h1 = m_k1[:, None] & m_v[None, :]
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
            p_h2 = hs + i_t_int64 * HV*K*V + o_k2[:, None] * V + o_v[None, :]
            m_h2 = m_k2[:, None] & m_v[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)

            p_w = ws + o_t[:, None] * (HV*K) + o_k1[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
            p_w = ws + o_t[:, None] * (HV*K) + o_k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
            p_v = vs + o_t[:, None] * (HV*V) + o_v[None, :]
            b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v

            p_v = vns + o_t[:, None] * (HV*V) + o_v[None, :]
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

            last_idx = min((i_t + 1) * BT, Tn) - 1
            b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
            p_g = g + (bos * HV + i_h).to(tl.int64) + o_t * HV
            b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
            b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = exp2(b_g_last)
            b_h1 *= b_g_last
            b_h2 *= b_g_last
            b_v = b_v.to(k.dtype.element_ty)

            p_k = ks + o_k1[:, None] + o_t[None, :] * (H*K)
            b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
            b_h1 += tl.dot(b_k, b_v)
            p_k = ks + o_k2[:, None] + o_t[None, :] * (H*K)
            b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            b_h2 += tl.dot(b_k, b_v)
        if seg == 0:
            # the reference's suffix call starts from zeros + load(ht): + 0.0 maps a -0.0 state entry to +0.0 like it
            b_h1 = b_h1 + 0.0
            b_h2 = b_h2 + 0.0


def chain_fwd_h(k, w, u, g, cu_a, chunk_offsets, n_chunks, nseq, BV=32, num_warps=2, num_stages=2):
    """k [1, T, H, K] (l2-normed), w [1, T, HV, K], u [1, T, HV, V], g [1, T, HV] (chunk-local cumsum, log2 scale),
    cu_a int64 [2*nseq + 1]: prefix i = sequence i, its suffix = sequence nseq + i. Returns h [1, n_chunks, HV, K, V] (in
    cu_a chunk order, dtype of k) and v_new [1, T, HV, V]. Config = the pinned fla config of the reference kernel."""
    B, T, H, K = k.shape
    HV, V = u.shape[2], u.shape[-1]
    assert B == 1 and K == 128 and cu_a.numel() == 2 * nseq + 1
    h = k.new_empty(B, n_chunks, HV, K, V)
    v_new = torch.empty_like(u)
    gdn_fwd_h_chain_kernel[(triton.cdiv(V, BV), nseq * HV)](
        k, u, w, v_new, g, h, cu_a, chunk_offsets, nseq, T, H=H, HV=HV, K=K, V=V, BT=64, BV=BV,
        num_warps=num_warps, num_stages=num_stages)
    return h, v_new
