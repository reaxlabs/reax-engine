# SPDX-License-Identifier: LicenseRef-REAX-Engine-1.1 AND Apache-2.0 AND MIT
# E63 mammoth engine (REAX engine track v2, slot mammoth = baselabs/mammoth-1).
# Derived from the REAX Engine (reax-engine 546728669f91, engine/pplx_engine.py: PplxEngine and its kernels, MIT-licensed
# REAX Engine code, copyright the REAX Engine licensor; see NOTICE). The numerics are the reference's: W8A8 FP8 GEMMs with
# per-token activation / per-channel weight scales, the same compiled rmsnorm+quant helpers, the same FLA kernels (fla-core
# 0.5.2, MIT) and torch varlen attention, prefix_bidir attention (state block attends to itself, the question reads the row),
# last-layer pruning and the fp32 slot readout. What E63 changes:
#   * the whole 64-layer stack of one step is replayed as ONE CUDA graph per (token bucket, sequence bucket): every
#     per-step index tensor (cu_seqlens, FLA chunk indices / offsets, prefix_bidir key gather, readout rows, positions)
#     lives in a static device buffer that the host fills before the replay;
#   * the step is batch-invariant: the only batch-size-dependent GEMM of the reference (the bf16 in_proj_b|a projection,
#     cuBLASLt picks its kernel by M) runs in a fixed-tile Triton kernel, the readout always on 16 rows, attention with
#     num_splits=1 and every FLA autotuned kernel pinned to one config, so an answer never depends on its batch;
#   * a decision is one packed sequence [state block | question] (the reference's plain prefix_bidir forward), which is
#     mathematically identical to the reference's state-once forward_groups path for one question per call.
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, "/opt/reax/engine")
sys.path.insert(0, "/opt/reax")


# ------------------------------------------------------------------------------------------------ in_proj_b|a GEMM
@triton.jit
def _ba_kernel(x_ptr, w_ptr, y_ptr, M, sx, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rm = pid * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mm = rm[:, None] < M
    mn = rn[None, :] < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(x_ptr + rm[:, None] * sx + (k0 + rk)[None, :], mask=mm, other=0.0)
        b = tl.load(w_ptr + rn[None, :] * K + (k0 + rk)[:, None], mask=mn, other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=mm & mn)


def ba_linear(x, w):
    """y = x @ w.T (bf16 in/out, fp32 accumulation over K in a fixed order): every row's bits are independent of M."""
    M, K = x.shape
    N = w.shape[0]
    assert x.stride(1) == 1 and w.is_contiguous() and K % 128 == 0 and N <= 128
    y = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    _ba_kernel[(triton.cdiv(M, 32),)](x, w, y, M, x.stride(0), K=K, N=N, BM=32, BN=128, BK=128, num_warps=4,
                                      num_stages=3)
    return y


READ_ROWS = 16  # the readout GEMM always runs on this many rows (cuBLAS picks its kernel by n)


# ------------------------------------------------------------------------------------------------ FLA pins
def fla_autotuners():
    """Every FLA autotuner on the mammoth forward path (also behind @triton.heuristics): name -> Autotuner."""
    from triton.runtime.autotuner import Autotuner
    from fla.modules import l2norm
    from fla.modules.conv.triton import kernels as convk
    from fla.ops.common import chunk_delta_h, chunk_o
    from fla.ops.gated_delta_rule import chunk_fwd, wy_fast
    from fla.ops.utils import cumsum
    out = {}
    for mod in (l2norm, convk, chunk_delta_h, chunk_o, chunk_fwd, wy_fast, cumsum):
        for name in dir(mod):
            o, depth = getattr(mod, name, None), 0
            while o is not None and depth < 4:
                if isinstance(o, Autotuner):
                    out[f"{mod.__name__.split('.')[-1]}.{name}"] = o
                    break
                o, depth = getattr(o, "fn", None), depth + 1
    return out


def pin_fla(pins: dict) -> list:
    """pins: name -> {"kwargs": {...}, "num_warps": w, "num_stages": s}; replaces each autotuner's candidate list by the one
    pinned config (no timing at run time, the same bits at every start and every token count)."""
    tuners = fla_autotuners()
    done = []
    for name, c in pins.items():
        t = tuners.get(name)
        if t is None:
            continue
        t.configs = [triton.Config(dict(c["kwargs"]), num_warps=c["num_warps"], num_stages=c["num_stages"])]
        t.cache.clear()
        for attr in ("_cache", "cached_configs"):
            if hasattr(t, attr) and isinstance(getattr(t, attr), dict):
                getattr(t, attr).clear()
        done.append(name)
    return done


def dump_fla_choices() -> dict:
    out = {}
    for name, t in fla_autotuners().items():
        if t.cache:
            out[name] = {str(k): {"kwargs": dict(c.kwargs), "num_warps": c.num_warps, "num_stages": c.num_stages}
                         for k, c in t.cache.items()}
    return out


# ------------------------------------------------------------------------------------------------ the graph engine
class Bucket:
    """Static device buffers of one (T tokens, S sequence slots) layout and its CUDA graph."""

    def __init__(self, T, NS, dev):
        self.T, self.NS = T, NS
        self.NSEQ = NS + 2  # real slots + the padding sequence + one empty sequence (target of the unused chunk entries)
        self.NT = T // 64 + self.NSEQ  # chunk entries: an upper bound for any layout of T tokens in NSEQ sequences
        self.KX = 2 * T
        self.NQ = 2 * NS + 1  # prefix_bidir query segments: [state_i | question_i] per slot, then the padding sequence
        i64, i32 = torch.int64, torch.int32
        z = lambda n, dt: torch.zeros(n, dtype=dt, device=dev)  # noqa: E731
        self.ids = z(T, i64)
        self.pos = z(T, torch.float32)
        self.cu64 = z(self.NSEQ + 1, i64)
        self.ci = z((self.NT, 2), i64)
        self.coff = z(self.NSEQ + 1, i64)
        self.kidx = z(self.KX, i64)
        self.cu_q = z(self.NQ + 1, i32)
        self.cu_k = z(self.NQ + 1, i32)
        self.last = z(max(NS, READ_ROWS), i64)
        self.cu_lk = z(max(NS, READ_ROWS) + 1, i32)
        self.cu_lq = torch.arange(max(NS, READ_ROWS) + 1, dtype=i32, device=dev)
        n_int = T + T + self.NSEQ + 1 + 2 * self.NT + self.NSEQ + 1 + self.KX + 2 * (self.NQ + 1) + max(NS, READ_ROWS) * 2 + 1
        self.host = torch.empty(n_int, dtype=torch.int64).pin_memory()
        self.graph = None
        self.out = None


class E63Graph:
    """CUDA-graph forward over the weights of a loaded reference PplxEngine (prefix_bidir, FP8, fused quant)."""

    def __init__(self, eng, t_buckets, s_buckets):
        self.e = eng
        assert eng.prefix_bidir and eng.linear_mode == "fp8" and eng.fused_quant and eng.prune_last and eng.gva_native
        assert not eng.triton_arq and eng.mlp_chunk == 0
        Lin = sys.modules[type(eng).__module__].Linear
        assert all(l.mode == "fp8" for L in eng.layers for k, l in L.items() if isinstance(l, Lin) and k != "in_ba")
        self.dev = torch.device(eng.device)
        self.t_buckets = sorted(t_buckets)
        self.s_buckets = sorted(s_buckets)
        self.buckets = {}
        self.pool = None
        self.stream = None
        self._fla_patch()
        from torch.nn.attention.varlen import varlen_attn
        self._varlen = varlen_attn
        self.pp = sys.modules[type(eng).__module__]  # the module the engine was built from (its K is the compiled one)
        self.K = self.pp.K
        self.ba_w = [L["in_ba"].w if L["type"] == "linear_attention" else None for L in eng.layers]

    # -------------------------------------------------------------- FLA: chunk offsets from our static buffer
    def _fla_patch(self):
        from fla.ops.common import chunk_delta_h
        orig = chunk_delta_h.prepare_chunk_offsets
        self._coff = None

        def prepare_chunk_offsets(cu_seqlens, chunk_size):
            if self._coff is not None and chunk_size == 64:
                return self._coff
            return orig(cu_seqlens, chunk_size)
        chunk_delta_h.prepare_chunk_offsets = prepare_chunk_offsets

    # -------------------------------------------------------------- layout -> host metadata
    def pick(self, lens):
        T = sum(lens)
        n = len(lens)
        for t in self.t_buckets:
            if t >= T:
                for s in self.s_buckets:
                    if s >= n:
                        return t, s
                return None
        return None

    def fill(self, b, lens, states, ids_list):
        """Host-side metadata for real sequences `lens` (state blocks `states`) in bucket b; one H2D copy."""
        T, NS = b.T, b.NS
        n = len(lens)
        assert n <= NS
        offs = np.zeros(n + 1, np.int64)
        np.cumsum(lens, out=offs[1:])
        Tr = int(offs[-1])
        assert Tr <= T
        ids = np.zeros(T, np.int64)
        ids[:Tr] = np.concatenate([np.asarray(x, np.int64) for x in ids_list]) if n else []
        pos = np.zeros(T, np.int64)
        for o, L in zip(offs[:-1], lens):
            pos[o:o + L] = np.arange(L)
        pad = T - Tr
        if pad:
            pos[Tr:] = np.arange(pad)
        # sequences: n real, then the padding sequence [Tr, T), then empty ones up to NSEQ
        cu = np.full(b.NSEQ + 1, T, np.int64)
        cu[:n + 1] = offs
        cu[n + 1:] = T  # padding sequence ends at T; the rest are empty at T
        seq_lens = np.diff(cu)
        cnt = (seq_lens + 63) // 64
        ci = np.empty((b.NT, 2), np.int64)
        tot = int(cnt.sum())
        assert tot <= b.NT
        seg = np.repeat(np.arange(b.NSEQ), cnt)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        intra = np.arange(tot) - np.repeat(start, cnt)
        ci[:tot, 0], ci[:tot, 1] = seg, intra
        ci[tot:, 0], ci[tot:, 1] = b.NSEQ - 1, 0  # empty last sequence: masked no-op programs
        coff = np.zeros(b.NSEQ + 1, np.int64)
        np.cumsum(cnt, out=coff[1:])
        # prefix_bidir attention segments + key gather
        kidx = np.zeros(b.KX, np.int64)
        cq = np.zeros(b.NQ + 1, np.int64)
        ck = np.zeros(b.NQ + 1, np.int64)
        kp, qi = 0, 0
        for j in range(NS):
            if j < n:
                o, L, s = int(offs[j]), int(lens[j]), int(states[j])
                kidx[kp:kp + s] = np.arange(o, o + s)
                kidx[kp + s:kp + s + L] = np.arange(o, o + L)
                cq[qi + 1], cq[qi + 2] = o + s, o + L
                ck[qi + 1], ck[qi + 2] = kp + s, kp + s + L
                kp += s + L
            else:
                cq[qi + 1] = cq[qi + 2] = cq[qi]
                ck[qi + 1] = ck[qi + 2] = ck[qi]
            qi += 2
        # padding sequence: one noncausal segment over itself
        cq[qi + 1] = T
        if pad:
            kidx[kp:kp + pad] = np.arange(Tr, T)
        ck[qi + 1] = kp + pad
        R = max(NS, READ_ROWS)
        last = np.zeros(R, np.int64)
        last[:n] = offs[1:] - 1
        clk = np.full(R + 1, offs[-1], np.int64)
        clk[:n + 1] = offs
        h = b.host.numpy()
        parts = (ids, pos, cu, ci.reshape(-1), coff, kidx, cq, ck, last, clk)
        o = 0
        for p in parts:
            h[o:o + p.size] = p
            o += p.size
        hd = b.host[:o].to(self.dev, non_blocking=True)
        o = 0
        b.ids.copy_(hd[o:o + T]); o += T
        b.pos.copy_(hd[o:o + T]); o += T
        b.cu64.copy_(hd[o:o + b.NSEQ + 1]); o += b.NSEQ + 1
        b.ci.view(-1).copy_(hd[o:o + 2 * b.NT]); o += 2 * b.NT
        b.coff.copy_(hd[o:o + b.NSEQ + 1]); o += b.NSEQ + 1
        b.kidx.copy_(hd[o:o + b.KX]); o += b.KX
        b.cu_q.copy_(hd[o:o + b.NQ + 1]); o += b.NQ + 1
        b.cu_k.copy_(hd[o:o + b.NQ + 1]); o += b.NQ + 1
        b.last.copy_(hd[o:o + R]); o += R
        b.cu_lk.copy_(hd[o:o + R + 1]); o += R + 1

    # -------------------------------------------------------------- the layer stack on static buffers
    def _attend(self, q, k, v, cu_q, cu_k, max_q, max_k):
        e = self.e
        return self._varlen(q.contiguous(), k.contiguous(), v.contiguous(), cu_q, cu_k, max_q, max_k,
                            scale=e.hd ** -0.5, window_size=(-1, -1), enable_gqa=True, num_splits=1)

    def _deltanet(self, L, li, x, xq, b):
        from fla.modules.conv.causal_conv1d import causal_conv1d
        from fla.modules.l2norm import l2norm_fwd
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
        e = self.e
        T = x.shape[0]
        qkvz = L["in_qkvz"](x, xq)
        ba = ba_linear(x, self.ba_w[li])
        conv_dim = 2 * e.key_dim + e.value_dim
        mixed, z = qkvz[:, :conv_dim], qkvz[:, conv_dim:]
        mixed, _ = causal_conv1d(mixed[None].contiguous(), weight=L["conv_w"], activation="silu", cu_seqlens=b.cu64,
                                 chunk_indices=b.ci)
        mixed = mixed[0]
        q, k, v = torch.split(mixed, [e.key_dim, e.key_dim, e.value_dim], dim=-1)
        q = q.reshape(1, T, e.lnk, e.lk)
        k = k.reshape(1, T, e.lnk, e.lk)
        v = v.reshape(1, T, e.lnv, e.lv)
        KK = self.K
        beta, g = KK.dn_gates(ba, L["A_neg"], L["dt_bias"], e.lnv)
        beta, g = beta[None], g[None]
        # == fla chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu): its autograd
        # Function's forward with our static chunk indices
        q = q.contiguous(); k = k.contiguous(); v = v.contiguous(); g = g.contiguous(); beta = beta.contiguous()
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
        self._coff = b.coff
        _, o, _, _, _, _ = chunk_gated_delta_rule_fwd(q=q, k=k, v=v, g=g, beta=beta, scale=k.shape[-1] ** -0.5,
                                                      initial_state=None, output_final_state=False, cu_seqlens=b.cu64,
                                                      chunk_indices=b.ci)
        self._coff = None
        return e._deltanet_out(L, o.to(q.dtype), z, T, xq)

    def _attention(self, L, x, cos, sin, xq, b, last):
        KK = self.K
        e = self.e
        T = x.shape[0]
        nq = e.nh * e.hd * 2
        if not last:
            qkv = L["qkv"](x, xq)
            qg, k, v = qkv[:, :nq], qkv[:, nq:nq + e.nkv * e.hd], qkv[:, nq + e.nkv * e.hd:]
        else:
            qg = L["q_only"](x[b.last], (xq[0][b.last], xq[1][b.last]))
            kv = L["kv_only"](x, xq)
            k, v = kv[:, : e.nkv * e.hd], kv[:, e.nkv * e.hd:]
        qg = qg.view(qg.shape[0], e.nh, 2 * e.hd)
        q, gate = qg[..., : e.hd], qg[..., e.hd:].reshape(qg.shape[0], -1)
        k = KK.norm_rope(k.view(T, e.nkv, e.hd), L["k_norm"], cos[:, None], sin[:, None], e.eps, e.rot)
        v = v.view(T, e.nkv, e.hd)
        if not last:
            q = KK.norm_rope(q, L["q_norm"], cos[:, None], sin[:, None], e.eps, e.rot)
            o = self._attend(q, k.index_select(0, b.kidx), v.index_select(0, b.kidx), b.cu_q, b.cu_k, T, T)
        else:
            q = KK.norm_rope(q, L["q_norm"], cos[b.last][:, None], sin[b.last][:, None], e.eps, e.rot)
            o = self._attend(q, k, v, b.cu_lq, b.cu_lk, 1, T)
        return L["o"](None, KK.gate_mul_q(o.reshape(o.shape[0], -1), gate))

    def forward_static(self, b):
        KK = self.K
        silu_mul_q_scaled_triton = self.pp.silu_mul_q_scaled_triton
        e = self.e
        h = F.embedding(b.ids, e.embed)
        freqs = b.pos[:, None] * e.inv_freq[None]
        emb = torch.cat([freqs, freqs], -1)
        cos, sin = emb.cos().to(e.dtype), emb.sin().to(e.dtype)
        x, x8, xs = KK.rms_q(h, e.layers[0]["in_norm"], e.eps)
        xq = (x8, xs)
        nL = e.n_layers
        for i, L in enumerate(e.layers):
            last = i == nL - 1
            if L["type"] == "linear_attention":
                d = self._deltanet(L, i, x, xq, b)
            else:
                d = self._attention(L, x, cos, sin, xq, b, last)
                if last:
                    h = h[b.last]
            h, x, x8, xs = KK.add_rms_q(h, d, L["post_norm"], e.eps)
            xq = (x8, xs)
            f8 = getattr(self.pp, "_F8", None)  # kai: rgh's fused gate/up FP8 GEMM + SwiGLU/quant epilogue (MAM_F8_GU=1)
            if getattr(self.pp, "_F8_GU", False) and f8.ok(x.shape[0], L["gu"].out_features, L["gu"].in_features):  # kai
                d = L["down"](None, f8.swiglu_q(xq[0], xq[1], L["gu"], e.inter))  # kai
            elif e.gu_tensorwise:  # kai: was `if`
                gu = torch._scaled_mm(xq[0], L["gu"].w8, scale_a=e._one, scale_b=e._one, out_dtype=torch.bfloat16)
                d = L["down"](None, silu_mul_q_scaled_triton(gu, xq[1], L["gu"].ws, e.inter))
            else:
                d = L["down"](None, KK.silu_mul_q(L["gu"](x, xq), e.inter))
            if not last:
                h, x, x8, xs = KK.add_rms_q(h, d, e.layers[i + 1]["in_norm"], e.eps)
                xq = (x8, xs)
            else:
                h, x = KK.add_rms(h, d, e.norm_w, e.eps)
        return x.float() @ e.readout.T  # [max(NS, 16), 255] fp32

    # -------------------------------------------------------------- capture / replay
    def bucket(self, T, NS):
        key = (T, NS)
        if key not in self.buckets:
            self.buckets[key] = Bucket(T, NS, self.dev)
        return self.buckets[key]

    def _synthetic(self, b):
        """A valid layout for warm-up / capture: NS sequences sharing T - pad tokens."""
        n = b.NS
        L = max(8, (b.T - 64) // n)
        L = min(L, b.T // n)
        lens = [L] * n
        states = [max(1, L // 2)] * n
        ids = [[1] * L for _ in range(n)]
        return lens, states, ids

    @torch.inference_mode()
    def capture(self, T, NS, warm=2):
        b = self.bucket(T, NS)
        lens, states, ids = self._synthetic(b)
        self.fill(b, lens, states, ids)
        for _ in range(warm):
            self.forward_static(b)
        torch.cuda.synchronize()
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = self.forward_static(b)
        torch.cuda.synchronize()
        b.graph, b.out = g, out
        return b

    @torch.inference_mode()
    def run(self, lens, states, ids_list):
        """logits [n, 255] fp32 (device) for n packed decisions; graph replay when a bucket fits, else eager."""
        key = self.pick(lens)
        n = len(lens)
        if key is not None and key in self.buckets and self.buckets[key].graph is not None:
            b = self.buckets[key]
            self.fill(b, lens, states, ids_list)
            b.graph.replay()
            return b.out[:n]
        T = -(-sum(lens) // 128) * 128
        b = Bucket(T, max(n, 1), self.dev)  # eager: same kernels, same bits
        self.fill(b, lens, states, ids_list)
        return self.forward_static(b)[:n]
