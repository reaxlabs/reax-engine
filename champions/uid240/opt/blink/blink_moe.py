# SPDX-License-Identifier: Apache-2.0
"""Bit-exact glue fusion around vLLM's CUTLASS NVFP4 MoE experts (run_cutlass_moe_fp4, the reference's MoE path).

  BLINK_MOE_HEAD=1  The stock path permutes the activations (shuffle_rows, one bf16 copy per token-expert pair) and
                    then NVFP4-quantises every permuted row with its expert's input scale. Blink's checkpoint has ONE
                    input scale per layer (all 128 experts equal), and the quantisation of a row depends only on the
                    row and that scale, so we quantise the m unpermuted rows once (vLLM's own quant kernel) and gather
                    the FP4 rows and their swizzled block scales into the per-expert layout the grouped GEMM reads.
                    The scale vector is checked once per tensor on the first eager call; anything else uses the stock
                    path.
  BLINK_MOE_TAIL=1  The stock path un-permutes the w2 output (shuffle_rows), multiplies by the bf16-rounded routing
                    weights (bf16 result) and sums the top-k rows (ATen sum, fp32 accumulation, bf16 out). One Triton
                    kernel does the same per output element in ATen's summation order (BLINK_MOE_TAIL_ORDER).
  BLINK_MOE_GEMM=1  The two NVFP4 grouped GEMMs run our own CUTLASS SM120 kernel (blink_moegemm.cu, built at image
                    build) with the tile / schedule chosen per GEMM and token count from BLINK_MOE_GEMM_TABLE; same
                    block-scaled MMA, K order and epilogue as vLLM's cutlass_fp4_moe_mm (bit-identical, kcheck59.py).
  BLINK_MOE_ACTQ=1  (E64) The GELU(tanh)-and-multiply of the w13 output and the NVFP4 experts quantisation of the result
                    run as one kernel (blink_actq.cu, built at image build) that replays vLLM's activation and
                    quantisation arithmetic: the same FP4 bytes and block scales, without the bf16 intermediate.
  BLINK_MOE_PLAN=1  (E64, with BLINK_MOE_HEAD's uniform-scale path) The routing data (vLLM's get_cutlass_moe_mm_data:
                    expert offsets, 128-row block-scale offsets, problem sizes, permutation) comes from two small Triton
                    kernels (an atomic histogram, then one block that scans it), and the quantised token rows are
                    scattered to their top-k slots (position inside an expert by atomic counter) instead of gathered:
                    the same FP4 bytes and block scales per row. Rows inside one expert are independent in the grouped
                    GEMM, and the tail reads every slot through c_map, so the row order inside an expert (vLLM's own
                    arg-sort also orders them by atomics) does not change any output.
  BLINK_MOE_FUSE=1  (team ruj, bj/moe "bmoe", ported by E64 on the user's instruction; see /opt/blink/NOTICE.bj) with
                    BLINK_MOE_GEMM=1 and the gelu_tanh activation: the w13 GEMM, vLLM's gelu_tanh_and_mul and vLLM's
                    scaled_fp4_experts_quant run as ONE kernel (blink_moegemm.so w13_fused, config BLINK_MOE_FUSE_CFG):
                    the bf16 [m*topk, 2n] intermediate is never written; the FP4 activations and scales are
                    bit-identical to the three-kernel path. Takes precedence over BLINK_MOE_ACTQ. bmoe's w13_fused and
                    its w2 variant 11 share one dynamic-scheduler counter per loaded .so, so they must never run at the
                    same time: every MoE kernel of this engine runs on one stream (blink_par's MoE stream).
  BLINK_MOE_FHEAD=1  (team ruj, bj/blinkm line M; with BLINK_MOE_PLAN's path) the same head outputs (routing data, FP4
                    rows + swizzled scales in the per-expert layout, c_map) from 3 kernels instead of 9: one program
                    computes pjs's histogram (Triton block histograms: exact integer counts, no global atomics, no
                    zero-fill kernel), pjs's plan and the [0, m] / [0, ceil128(m)] offset pairs of the quant; vLLM's
                    quant of the m rows is unchanged; the scatter runs one program per token (its top-k slots at once,
                    slot inside an expert from pjs's per-expert atomic counter) instead of one per (token, slot). The
                    head sits on the critical path at the MoE / MLP fork: the w13 GEMM can only take SMs as the dense
                    MLP's gate_up CTAs retire, and a head that ends before the first gate_up wave retires lets the w13
                    GEMM start ~50 us per layer earlier. Same bytes in the same expert segments (the order inside a
                    segment comes from atomics in PLAN as well and enters no output), so every output bit is unchanged.
  BLINK_MOE_DISCARD=1  (E69 C6, with BLINK_MOE_FUSE + TAIL) dead MoE temporaries never reach DRAM: the w2 GEMM runs
                    E69's bmoe69 build of team ruj's variant 11 (gemm/bmoe69.so from bmoe69.cu; bit-identical) with
                    L2 evict_last hints on its bf16 c3 stores, and discard.global.L2 drops the dirty L2 lines of the
                    expert-sorted FP4 input rows + the used prefix of their scales once w13 has read them, and of c3 +
                    the FP4 intermediate + scales once the tail has read them (only on the fused-tail path, where the
                    output does not alias c3). Values are never touched: hints and discards only (kbit69).
  BLINK_MOE_W2ZS=1  (E69) the w2 GEMM (bmoe69 build of ruj's v11) skips the MMAs of its all-zero 12th K chunk
                    (CUTLASS's zero-filled 256-wide tail tile). Exact: on sm_120 that chunk only maps an accumulator
                    of -0 to +0 (measured: finite / subnormal / inf values pass unchanged), which the epilogue's
                    fma(0, 0, alpha * acc) + 0 does anyway; NaN accumulators cannot occur (finite e2m1 x e4m3).
  BLINK_MOE_L2PERSIST=<bytes>  (E69) this process's persisting-L2 set-aside (cudaLimitPersistingL2CacheSize, clamped
                    to the device maximum), set on the first (eager) MoE call, before any capture: the evict_last hints
                    already in ruj's kernels (expert-sorted A rows) and C6's c3 stores only persist inside it. Caching only.
  BLINK_MOE_W16=1   (E69 C2-lite, with BLINK_MOE_FUSE) the w13 (fused GELU + quant) and w2 GEMMs run bmoe69's builds of
                    team ruj's kernels with 16 consumer warps (4 x 4 warps of 32 x 32 instead of 4 x 2 of 32 x 64; four
                    warps per SM sub-partition keep the block-scaled MMA pipe busier). Only which warp computes an
                    output element changes, not how: bit-identical (kbit69 / probe69 cons7: 0 bytes differ).
Everything else is the stock function. Installed by sitecustomize.py (patches the module when it is imported)."""
import importlib.abc
import json
import os
import sys

HEAD = os.environ.get("BLINK_MOE_HEAD", "0") == "1"
TAIL = os.environ.get("BLINK_MOE_TAIL", "0") == "1"
TAIL_ORDER = int(os.environ.get("BLINK_MOE_TAIL_ORDER", "1"))
TAIL_BLOCK = int(os.environ.get("BLINK_MOE_TAIL_BLOCK", "1024"))
GEMM = os.environ.get("BLINK_MOE_GEMM", "0") == "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
GEMM_DIR = os.environ.get("BLINK_GEMM_DIR", os.path.join(_HERE, "gemm"))
GEMM_TABLE = os.environ.get("BLINK_MOE_GEMM_TABLE", os.path.join(_HERE, "moe_gemm.json"))
GEMM_SMS = int(os.environ.get("BLINK_MOE_GEMM_SMS", "0"))   # > 0: persistent grid on that many SMs (all rows)
ACTQ = os.environ.get("BLINK_MOE_ACTQ", "0") == "1"
PLAN = os.environ.get("BLINK_MOE_PLAN", "0") == "1"
FUSE = os.environ.get("BLINK_MOE_FUSE", "0") == "1"           # team ruj bj/moe: fused w13 + gelu_tanh*up + NVFP4 quant
FUSE_CFG = int(os.environ.get("BLINK_MOE_FUSE_CFG", "1"))
DISCARD = os.environ.get("BLINK_MOE_DISCARD", "0") == "1"     # E69 C6: L2 evict_last c3 + discard of dead temporaries
W2ZS = os.environ.get("BLINK_MOE_W2ZS", "0") == "1"           # E69: w2 skips its all-zero 12th K chunk (exact)
W16 = os.environ.get("BLINK_MOE_W16", "0") == "1"             # E69 C2-lite: 16 consumer warps in w13 and w2 (exact)
L2PERSIST = int(os.environ.get("BLINK_MOE_L2PERSIST", "-1"))  # E69: persisting-L2 set-aside bytes (-1: untouched)
FHEAD = os.environ.get("BLINK_MOE_FHEAD", "0") == "1"          # team ruj bj/blinkm: fast exact MoE head (with PLAN)
FH_BLOCK = int(os.environ.get("BLINK_MOE_FHEAD_BLOCK", "2048"))
FH_WARPS = int(os.environ.get("BLINK_MOE_FHEAD_WARPS", "8"))
_K = {}
_G = {}


def _gemm_table():
    """{"w13": [[max_m, variant, raster(, sms)], ...], "w2": [...]} (m = tokens); variant -1 = vLLM's kernel"""
    if "t" not in _G:
        try:
            with open(GEMM_TABLE) as fh:
                raw = json.load(fh)
            _G["t"] = {k: sorted((int(r[0]), int(r[1]), int(r[2]), int(r[3]) if len(r) > 3 else 0) for r in v)
                       for k, v in raw["gemms"].items()}
        except Exception as exc:  # noqa: BLE001
            print(f"[blink-moe] no GEMM table at {GEMM_TABLE}: {exc!r}", file=sys.stderr, flush=True)
            _G["t"] = {}
    return _G["t"]


def _gemm_cfg(which, m):
    if not GEMM:
        return -1, 0, 0
    for max_m, v, r, sms in _gemm_table().get(which, ()):
        if m <= max_m:
            return v, r, (GEMM_SMS or sms)
    return -1, 0, 0


def _gemm_ext():
    if "m" not in _G:
        import glob
        import importlib.util
        so = sorted(glob.glob(os.path.join(GEMM_DIR, "blink_moegemm*.so")))
        if not so:
            raise RuntimeError(f"blink_moegemm extension not found in {GEMM_DIR}")
        spec = importlib.util.spec_from_file_location("blink_moegemm", so[0])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _G["m"] = mod
    return _G["m"]


def _x69():
    """E69's bmoe69 extension (team ruj's bmoe kernels + E69's L2 hints / discards / zero-chunk skip),
    gemm/bmoe69.so built from gemm/bmoe69.cu"""
    if "x" not in _G:
        import glob
        import importlib.util
        so = sorted(glob.glob(os.path.join(GEMM_DIR, "bmoe69*.so")))
        if not so:
            raise RuntimeError(f"bmoe69 extension not found in {GEMM_DIR}")
        spec = importlib.util.spec_from_file_location("bmoe69", so[0])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _G["x"] = mod
        if L2PERSIST >= 0:
            import torch
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("[blink-moe] L2PERSIST: first set while capturing")
            got = mod.l2_persist69(L2PERSIST)
            print(f"[blink-moe] persisting-L2 set-aside {got[0]} B (device max {got[1]}, L2 {got[2]})", file=sys.stderr,
                  flush=True)
    return _G["x"]


triton = tl = None      # bound by _kernels(): Triton resolves names (and string annotations) in the module globals


def _kernels():
    global triton, tl
    if _K:
        return _K
    import triton
    import triton.language as tl

    @triton.jit
    def _prod(c3_ptr, map_ptr, w_ptr, t, j: tl.constexpr, cols, msk, stride_c3, TOPK: tl.constexpr):
        row = tl.load(map_ptr + t * TOPK + j).to(tl.int64)
        x = tl.load(c3_ptr + row * stride_c3 + cols, mask=msk, other=0.0).to(tl.float32)
        wj = tl.load(w_ptr + t * TOPK + j).to(tl.bfloat16).to(tl.float32)
        return (x * wj).to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _tail(c3_ptr, map_ptr, w_ptr, out_ptr, K, stride_c3, stride_out, TOPK: tl.constexpr, ORDER: tl.constexpr,
              BLOCK: tl.constexpr):
        t = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        msk = cols < K
        p0 = _prod(c3_ptr, map_ptr, w_ptr, t, 0, cols, msk, stride_c3, TOPK)
        p1 = _prod(c3_ptr, map_ptr, w_ptr, t, 1, cols, msk, stride_c3, TOPK)
        p2 = _prod(c3_ptr, map_ptr, w_ptr, t, 2, cols, msk, stride_c3, TOPK)
        p3 = _prod(c3_ptr, map_ptr, w_ptr, t, 3, cols, msk, stride_c3, TOPK)
        p4 = _prod(c3_ptr, map_ptr, w_ptr, t, 4, cols, msk, stride_c3, TOPK)
        p5 = _prod(c3_ptr, map_ptr, w_ptr, t, 5, cols, msk, stride_c3, TOPK)
        p6 = _prod(c3_ptr, map_ptr, w_ptr, t, 6, cols, msk, stride_c3, TOPK)
        p7 = _prod(c3_ptr, map_ptr, w_ptr, t, 7, cols, msk, stride_c3, TOPK)
        if ORDER == 0:
            acc = ((((((p0 + p1) + p2) + p3) + p4) + p5) + p6) + p7
        elif ORDER == 1:
            acc = (((p0 + p4) + (p1 + p5)) + (p2 + p6)) + (p3 + p7)
        elif ORDER == 2:
            acc = ((p0 + p4) + (p1 + p5)) + ((p2 + p6) + (p3 + p7))
        elif ORDER == 3:
            acc = ((p0 + p4) + (p2 + p6)) + ((p1 + p5) + (p3 + p7))
        elif ORDER == 4:
            acc = ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7))
        elif ORDER == 5:
            acc = (((p0 + p2) + p4) + p6) + (((p1 + p3) + p5) + p7)
        elif ORDER == 6:
            acc = (p0 + p4) + ((p1 + p5) + ((p2 + p6) + (p3 + p7)))
        else:
            acc = (((p0 + p1) + (p2 + p3)) + (p4 + p5)) + (p6 + p7)
        tl.store(out_ptr + t * stride_out + cols, acc.to(tl.bfloat16), mask=msk)

    @triton.jit
    def _swz(row, kb, K4: tl.constexpr):
        return (row // 128) * (K4 * 512) + (kb // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + (kb % 4)

    @triton.jit
    def _gather(q_ptr, sf_ptr, amap_ptr, eoff_ptr, boff_ptr, outq_ptr, outsf_ptr, QB: tl.constexpr,
                K4: tl.constexpr, E: tl.constexpr, NSTEP: tl.constexpr, BQ: tl.constexpr, BS: tl.constexpr):
        i = tl.program_id(0)
        src = tl.load(amap_ptr + i)
        offs = tl.arange(0, BQ)
        m = offs < QB
        v = tl.load(q_ptr + src.to(tl.int64) * QB + offs, mask=m)
        tl.store(outq_ptr + i.to(tl.int64) * QB + offs, v, mask=m)
        lo = i * 0
        hi = lo + (E + 1)
        for _ in tl.static_range(NSTEP):      # the largest e with expert_offsets[e] <= i: the expert of row i
            mid = (lo + hi) // 2
            ok = (hi - lo > 1) & (tl.load(eoff_ptr + mid, mask=hi - lo > 1, other=0) <= i)
            lo = tl.where(ok, mid, lo)
            hi = tl.where((hi - lo > 1) & ~ok, mid, hi)
        d = tl.load(boff_ptr + lo) + (i - tl.load(eoff_ptr + lo))
        kb = tl.arange(0, BS)
        km = kb < K4 * 4
        s = tl.load(sf_ptr + _swz(src, kb, K4), mask=km)
        tl.store(outsf_ptr + _swz(d, kb, K4), s, mask=km)

    @triton.jit
    def _hist(ids_ptr, hist_ptr, NE, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        msk = offs < NE
        ex = tl.load(ids_ptr + offs, mask=msk, other=0)
        tl.atomic_add(hist_ptr + ex, 1, mask=msk)

    @triton.jit
    def _plan(hist_ptr, cnt_ptr, eoff_ptr, boff_ptr, ps1_ptr, ps2_ptr, N1, KD, N, E: tl.constexpr):
        r = tl.arange(0, E)
        h = tl.load(hist_ptr + r)
        c = tl.cumsum(h, 0)
        tl.store(eoff_ptr + r, c - h)                      # exclusive prefix: expert_offsets[0..E-1]
        tl.store(eoff_ptr + E, tl.sum(h, 0))
        hb = (h + 127) // 128 * 128
        cb = tl.cumsum(hb, 0)
        tl.store(boff_ptr + r, cb - hb)                    # 128-row rounded: blockscale_offsets[0..E-1]
        tl.store(boff_ptr + E, tl.sum(hb, 0))
        tl.store(ps1_ptr + r * 3, h)
        tl.store(ps1_ptr + r * 3 + 1, h * 0 + N1)
        tl.store(ps1_ptr + r * 3 + 2, h * 0 + KD)
        tl.store(ps2_ptr + r * 3, h)
        tl.store(ps2_ptr + r * 3 + 1, h * 0 + KD)
        tl.store(ps2_ptr + r * 3 + 2, h * 0 + N)
        tl.store(cnt_ptr + r, h * 0)                       # per-expert slot counters for the scatter

    @triton.jit
    def _scatter(q_ptr, sf_ptr, ids_ptr, eoff_ptr, boff_ptr, cnt_ptr, outq_ptr, outsf_ptr, cmap_ptr, QB: tl.constexpr,
                 K4: tl.constexpr, TOPK: tl.constexpr, BQ: tl.constexpr, BS: tl.constexpr):
        j = tl.program_id(0)
        t = j // TOPK
        ex = tl.load(ids_ptr + j)
        r = tl.atomic_add(cnt_ptr + ex, 1)
        pos = tl.load(eoff_ptr + ex) + r
        tl.store(cmap_ptr + j, pos)
        offs = tl.arange(0, BQ)
        m = offs < QB
        v = tl.load(q_ptr + t.to(tl.int64) * QB + offs, mask=m)
        tl.store(outq_ptr + pos.to(tl.int64) * QB + offs, v, mask=m)
        d = tl.load(boff_ptr + ex) + r
        kb = tl.arange(0, BS)
        km = kb < K4 * 4
        s = tl.load(sf_ptr + _swz(t, kb, K4), mask=km)
        tl.store(outsf_ptr + _swz(d, kb, K4), s, mask=km)

    @triton.jit
    def _fplan(ids_ptr, cnt_ptr, eoff_ptr, boff_ptr, ps1_ptr, ps2_ptr, e1_ptr, b1_ptr, NE, M, N1, KD, N,
               E: tl.constexpr, BLOCK: tl.constexpr):
        # team ruj bj/blinkm FHEAD: pjs's _hist + _plan in one program (block histograms: exact integer counts, no
        # global atomics, no zero-fill), plus the quant's [0, M] / [0, ceil128(M)] offset pairs (pjs: 4 tiny kernels)
        h = tl.zeros([E], dtype=tl.int32)
        for start in range(0, NE, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            msk = offs < NE
            ex = tl.load(ids_ptr + offs, mask=msk, other=0).to(tl.int32)
            h += tl.histogram(ex, E, mask=msk).to(tl.int32)
        r = tl.arange(0, E)
        c = tl.cumsum(h, 0)
        tl.store(eoff_ptr + r, c - h)                      # exclusive prefix: expert_offsets[0..E-1]  (= pjs's _plan)
        tl.store(eoff_ptr + E, tl.sum(h, 0))
        hb = (h + 127) // 128 * 128
        cb = tl.cumsum(hb, 0)
        tl.store(boff_ptr + r, cb - hb)                    # 128-row rounded: blockscale_offsets[0..E-1]
        tl.store(boff_ptr + E, tl.sum(hb, 0))
        tl.store(ps1_ptr + r * 3, h)
        tl.store(ps1_ptr + r * 3 + 1, h * 0 + N1)
        tl.store(ps1_ptr + r * 3 + 2, h * 0 + KD)
        tl.store(ps2_ptr + r * 3, h)
        tl.store(ps2_ptr + r * 3 + 1, h * 0 + KD)
        tl.store(ps2_ptr + r * 3 + 2, h * 0 + N)
        tl.store(cnt_ptr + r, h * 0)                       # per-expert slot counters for the scatter
        two = tl.arange(0, 2)
        tl.store(e1_ptr + two, two * M)                    # [0, m]
        tl.store(b1_ptr + two, two * ((M + 127) // 128 * 128))   # [0, ceil128(m)]

    @triton.jit
    def _tscatter(q_ptr, sf_ptr, ids_ptr, eoff_ptr, boff_ptr, cnt_ptr, outq_ptr, outsf_ptr, cmap_ptr,
                  QB: tl.constexpr, K4: tl.constexpr, TOPK: tl.constexpr, BQ: tl.constexpr, BS: tl.constexpr):
        # team ruj bj/blinkm FHEAD: pjs's _scatter, one program per token (its TOPK slots at once): the token's FP4
        # row and scale bytes are read once and written to each slot; slot inside an expert = pjs's atomic counter
        t = tl.program_id(0)
        j = tl.arange(0, TOPK)
        ex = tl.load(ids_ptr + t * TOPK + j)
        r = tl.atomic_add(cnt_ptr + ex, 1)                 # TOPK distinct experts per token
        pos = tl.load(eoff_ptr + ex) + r
        tl.store(cmap_ptr + t * TOPK + j, pos)
        offs = tl.arange(0, BQ)
        m = offs < QB
        v = tl.load(q_ptr + t.to(tl.int64) * QB + offs, mask=m)
        dst = pos.to(tl.int64)[:, None] * QB + offs[None, :]
        allj = j[:, None] < TOPK
        tl.store(outq_ptr + dst, tl.broadcast_to(v[None, :], (TOPK, BQ)), mask=m[None, :] & allj)
        d = tl.load(boff_ptr + ex) + r
        kb = tl.arange(0, BS)
        km = kb < K4 * 4
        s = tl.load(sf_ptr + _swz(t, kb, K4), mask=km)
        tl.store(outsf_ptr + _swz(d[:, None], kb[None, :], K4), tl.broadcast_to(s[None, :], (TOPK, BS)),
                 mask=km[None, :] & allj)

    _K.update(triton=triton, tail=_tail, gather=_gather, hist=_hist, plan=_plan, scatter=_scatter, fplan=_fplan,
              tscatter=_tscatter)
    return _K


def _actq_ext():
    """the fused activation + NVFP4 experts quantisation extension (blink_actq.cu)"""
    if "a" not in _G:
        import glob
        import importlib.util
        so = sorted(glob.glob(os.path.join(GEMM_DIR, "blink_actq*.so")))
        if not so:
            raise RuntimeError(f"blink_actq extension not found in {GEMM_DIR}")
        spec = importlib.util.spec_from_file_location("blink_actq", so[0])
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _G["a"] = m
    return _G["a"]


def _patch(mod):
    import torch
    import vllm.envs as envs
    orig = mod.run_cutlass_moe_fp4
    ops, _resize_cache, MoEActivation = mod.ops, mod._resize_cache, mod.MoEActivation
    apply_moe_activation = mod.apply_moe_activation
    uniform = {}

    def is_uniform(t):
        key = (t.data_ptr(), t.numel())
        v = uniform.get(key)
        if v is None:
            if torch.cuda.is_current_stream_capturing():
                return False
            v = bool(t.numel() > 0 and (t == t.reshape(-1)[:1]).all().item())
            uniform[key] = v
            print(f"[blink-moe] input scale tensor {key[0]:#x} ({t.numel()}): uniform={v}", file=sys.stderr,
                  flush=True)
        return v

    def head(a, gscale, a_map, eoff, boff, topk, e):
        k = _kernels()
        m, kk = a.shape
        dev = a.device
        e1 = torch.arange(2, dtype=torch.int32, device=dev) * m          # [0, m]; capture-safe (no host copy)
        b1 = torch.arange(2, dtype=torch.int32, device=dev) * ((m + 127) // 128 * 128)
        q1, sf1 = ops.scaled_fp4_experts_quant(a, gscale[:1], e1, b1, 1)
        mk = a_map.shape[0]
        outq = torch.empty(mk, kk // 2, device=dev, dtype=torch.uint8)
        outsf = torch.empty(sf1.shape[0] * topk, sf1.shape[1], device=dev, dtype=sf1.dtype)
        nstep = max(1, (e + 1 - 1).bit_length())
        k["gather"][(mk,)](q1, sf1.view(torch.uint8), a_map, eoff, boff, outq, outsf.view(torch.uint8), QB=kk // 2,
                           K4=sf1.shape[1] // 4, E=e, NSTEP=nstep, BQ=k["triton"].next_power_of_2(kk // 2),
                           BS=k["triton"].next_power_of_2(sf1.shape[1]), num_warps=4)
        return outq, outsf

    def plan_scatter(a, gscale, topk_ids, eoff, boff, ps1, ps2, c_map, topk, e, n1, kd, n):
        """routing data + quantised rows in the per-expert layout (BLINK_MOE_PLAN): histogram, scan, vLLM's quant of
        the m rows (as head()), scatter of each row's FP4 bytes and swizzled scales to its top-k slots"""
        k_ = _kernels()
        m, kk = a.shape
        dev = a.device
        ne = topk_ids.numel()
        hist = torch.zeros(e, dtype=torch.int32, device=dev)
        cnt = torch.empty(e, dtype=torch.int32, device=dev)
        k_["hist"][(k_["triton"].cdiv(ne, 1024),)](topk_ids, hist, ne, BLOCK=1024, num_warps=4)
        k_["plan"][(1,)](hist, cnt, eoff, boff, ps1, ps2, n1, kd, n, E=e, num_warps=4)
        e1 = torch.arange(2, dtype=torch.int32, device=dev) * m
        b1 = torch.arange(2, dtype=torch.int32, device=dev) * ((m + 127) // 128 * 128)
        q1, sf1 = ops.scaled_fp4_experts_quant(a, gscale[:1], e1, b1, 1)
        outq = torch.empty(ne, kk // 2, device=dev, dtype=torch.uint8)
        outsf = torch.empty(sf1.shape[0] * topk, sf1.shape[1], device=dev, dtype=sf1.dtype)
        k_["scatter"][(ne,)](q1, sf1.view(torch.uint8), topk_ids, eoff, boff, cnt, outq, outsf.view(torch.uint8), c_map,
                             QB=kk // 2, K4=sf1.shape[1] // 4, TOPK=topk, BQ=k_["triton"].next_power_of_2(kk // 2),
                             BS=k_["triton"].next_power_of_2(sf1.shape[1]), num_warps=4)
        return outq, outsf

    def fast_plan_scatter(a, gscale, topk_ids, eoff, boff, ps1, ps2, c_map, topk, e, n1, kd, n):
        """team ruj bj/blinkm FHEAD: plan_scatter's outputs from 3 kernels: _fplan (pjs's _hist + _plan + the
        quant's offset pairs), vLLM's quant of the m rows (as pjs), _tscatter (pjs's _scatter, one program per token)"""
        k_ = _kernels()
        m, kk = a.shape
        dev = a.device
        ne = topk_ids.numel()
        cnt = torch.empty(e, dtype=torch.int32, device=dev)
        e1 = torch.empty(2, dtype=torch.int32, device=dev)
        b1 = torch.empty(2, dtype=torch.int32, device=dev)
        k_["fplan"][(1,)](topk_ids, cnt, eoff, boff, ps1, ps2, e1, b1, ne, m, n1, kd, n, E=e, BLOCK=FH_BLOCK,
                          num_warps=FH_WARPS)
        q1, sf1 = ops.scaled_fp4_experts_quant(a, gscale[:1], e1, b1, 1)
        outq = torch.empty(ne, kk // 2, device=dev, dtype=torch.uint8)
        outsf = torch.empty(sf1.shape[0] * topk, sf1.shape[1], device=dev, dtype=sf1.dtype)
        k_["tscatter"][(m,)](q1, sf1.view(torch.uint8), topk_ids, eoff, boff, cnt, outq, outsf.view(torch.uint8),
                             c_map, QB=kk // 2, K4=sf1.shape[1] // 4, TOPK=topk,
                             BQ=k_["triton"].next_power_of_2(kk // 2), BS=k_["triton"].next_power_of_2(sf1.shape[1]),
                             num_warps=4)
        return outq, outsf

    def tail(c3, c_map, w, out, topk, kk):
        k = _kernels()
        m = out.shape[0]
        grid = (m, k["triton"].cdiv(kk, TAIL_BLOCK))
        k["tail"][grid](c3, c_map, w.contiguous(), out, kk, c3.stride(0), out.stride(0), TOPK=topk, ORDER=TAIL_ORDER,
                        BLOCK=TAIL_BLOCK, num_warps=4)

    def _overlap(x, y):
        xa, ya = x.data_ptr(), y.data_ptr()
        return xa < ya + y.numel() * y.element_size() and ya < xa + x.numel() * x.element_size()

    def run_cutlass_moe_fp4(output, a, a1_gscale, w1_fp4, w1_blockscale, w1_alphas, a2_gscale, w2_fp4,
                            w2_blockscale, w2_alphas, topk_weights, topk_ids, activation, workspace13, workspace2, m, n,
                            k, e, device, apply_router_weight_on_input=False, *, activation_config=None):
        num_topk = topk_ids.size(1)
        if L2PERSIST >= 0 and "x" not in _G:
            _x69()                                   # E69: the persisting-L2 set-aside, once, on the first (eager) call
        use_head = HEAD and a.dtype == torch.bfloat16 and a.is_contiguous() and is_uniform(a1_gscale)
        use_tail = (TAIL and num_topk == 8 and output.dtype == torch.bfloat16 and output.is_contiguous()
                    and not apply_router_weight_on_input)
        if apply_router_weight_on_input or m == 0 or not (use_head or use_tail or GEMM):
            return orig(output, a, a1_gscale, w1_fp4, w1_blockscale, w1_alphas, a2_gscale, w2_fp4, w2_blockscale,
                        w2_alphas, topk_weights, topk_ids, activation, workspace13, workspace2, m, n, k, e, device,
                        apply_router_weight_on_input, activation_config=activation_config)
        # ---- the stock function (vLLM 0.30.0 run_cutlass_moe_fp4, Apache-2.0) with the two fused sections ----
        is_gated = activation.is_gated
        w1_n = n * 2 if is_gated else n
        topk = num_topk
        out_dtype = a.dtype
        expert_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
        blockscale_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
        problem_sizes1 = torch.empty((e, 3), dtype=torch.int32, device=device)
        problem_sizes2 = torch.empty((e, 3), dtype=torch.int32, device=device)
        c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
        plan = (PLAN and use_head and e & (e - 1) == 0 and topk_ids.is_contiguous()
                and topk_ids.dtype in (torch.int32, torch.int64))
        if plan and FHEAD and num_topk & (num_topk - 1) == 0:      # team ruj bj/blinkm FHEAD: same outputs
            rep_a_fp4, rep_a_blockscale = fast_plan_scatter(a, a1_gscale, topk_ids, expert_offsets,
                                                            blockscale_offsets, problem_sizes1, problem_sizes2, c_map,
                                                            num_topk, e, w1_n, k, n)
        elif plan:
            rep_a_fp4, rep_a_blockscale = plan_scatter(a, a1_gscale, topk_ids, expert_offsets, blockscale_offsets,
                                                       problem_sizes1, problem_sizes2, c_map, num_topk, e, w1_n, k, n)
        else:
            a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
            ops.get_cutlass_moe_mm_data(topk_ids, expert_offsets, problem_sizes1, problem_sizes2, a_map, c_map, e, n,
                                        k, blockscale_offsets, is_gated=is_gated)
        if plan:
            pass
        elif use_head:
            rep_a_fp4, rep_a_blockscale = head(a, a1_gscale, a_map, expert_offsets, blockscale_offsets, num_topk, e)
        else:
            a_s = ops.shuffle_rows(a, a_map)
            rep_a_fp4, rep_a_blockscale = ops.scaled_fp4_experts_quant(a_s, a1_gscale, expert_offsets,
                                                                       blockscale_offsets, num_topk)
        c1 = _resize_cache(workspace13, (m * topk, w1_n))
        c2 = _resize_cache(workspace2, (m * topk, n))
        c3 = _resize_cache(workspace13, (m * topk, k))
        v, r, sms = _gemm_cfg("w13", m)
        fuse = FUSE and v >= 0 and activation == MoEActivation.GELU_TANH and k % 64 == 0 and n % 64 == 0
        if fuse:
            # team ruj bj/moe: one kernel = moe_mm(w13) -> torch.ops._C.gelu_tanh_and_mul -> ops.scaled_fp4_experts_quant
            int_fp4 = torch.empty((m * topk, n // 2), dtype=torch.uint8, device=device)
            int_blockscale = torch.empty(((m * topk + 128 * e + 127) // 128 * 128, n // 16), dtype=torch.uint8,
                                         device=device)
            if W16 and FUSE_CFG == 3:
                # E69 C2-lite: ruj's cfg 3 kernel (bmoe69 build) with 16 consumer warps; same bytes out
                _x69().w13_fused69(int_fp4, int_blockscale, rep_a_fp4, w1_fp4, rep_a_blockscale, w1_blockscale,
                                   w1_alphas, problem_sizes1, expert_offsets[:-1], blockscale_offsets[:-1], a2_gscale,
                                   10, 0, sms)
            else:
                _gemm_ext().w13_fused(int_fp4, int_blockscale, rep_a_fp4, w1_fp4, rep_a_blockscale, w1_blockscale,
                                      w1_alphas, problem_sizes1, expert_offsets[:-1], blockscale_offsets[:-1], a2_gscale,
                                      FUSE_CFG, sms)
            int_blockscale = int_blockscale.view(torch.float8_e4m3fn)
        elif v >= 0:
            _gemm_ext().moe_mm(c1, rep_a_fp4, w1_fp4, rep_a_blockscale, w1_blockscale, w1_alphas, problem_sizes1,
                               expert_offsets[:-1], blockscale_offsets[:-1], v, r, sms)
        else:
            ops.cutlass_fp4_moe_mm(c1, rep_a_fp4, w1_fp4, rep_a_blockscale, w1_blockscale, w1_alphas, problem_sizes1,
                                   expert_offsets[:-1], blockscale_offsets[:-1])
        disc = DISCARD and fuse and v >= 10
        if disc:
            # E69 C6: rep_a and the used prefix of its scale rows (vLLM's scale buffer has MAX_TOKENS x topk rows)
            srows = (m * topk + 128 * e + 127) // 128 * 128
            sbytes = min(srows * (k // 16), rep_a_blockscale.numel() * rep_a_blockscale.element_size())
            _x69().discard69n([rep_a_fp4, rep_a_blockscale.view(torch.uint8).reshape(-1)[:sbytes]])
        del rep_a_fp4, rep_a_blockscale
        if fuse:
            pass
        elif activation == MoEActivation.SILU and (activation_config is None or activation_config.clamp_limit is None):
            int_fp4, int_blockscale = ops.silu_and_mul_scaled_fp4_experts_quant(c1, a2_gscale, expert_offsets,
                                                                                blockscale_offsets, num_topk)
        elif (ACTQ and activation == MoEActivation.GELU_TANH and c1.dtype == torch.bfloat16 and c1.is_contiguous()
              and n % 16 == 0 and a2_gscale.dtype == torch.float32 and a2_gscale.is_contiguous()):
            # = gelu_tanh_and_mul(c2, c1) + scaled_fp4_experts_quant(c2, ...): same outputs, one kernel (blink_actq.cu)
            mnt = c1.shape[0]
            maxt = envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE
            assert mnt <= maxt * num_topk, f"m_numtopk {mnt} > VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE x topk"
            int_fp4 = torch.empty(mnt, n // 2, device=c1.device, dtype=torch.uint8)
            int_blockscale = torch.empty(maxt * num_topk, (n // 16 + 3) // 4, dtype=torch.int32, device=c1.device)
            _actq_ext().gelu_quant(c1, a2_gscale, expert_offsets, blockscale_offsets, int_fp4, int_blockscale)
            int_blockscale = int_blockscale.view(torch.float8_e4m3fn)
        else:
            apply_moe_activation(activation, c2, c1, activation_config=activation_config)
            int_fp4, int_blockscale = ops.scaled_fp4_experts_quant(c2, a2_gscale, expert_offsets, blockscale_offsets,
                                                                   num_topk)
        v, r, sms = _gemm_cfg("w2", m)
        disc = disc and v == 11 and int_blockscale.is_contiguous()
        zs = W2ZS and v == 11
        if zs:
            key = ("zs", w2_blockscale.data_ptr())
            ok_zs = _G.get(key)
            if ok_zs is None:
                # W2ZS is exact for every non-NaN accumulator; NaNs need a NaN e4m3 scale (0x7F / 0xFF): check the
                # layer's weight scales once (eager); the activation scales are satfinite by construction
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("[blink-moe] W2ZS: a layer's w2 scales were first seen while capturing")
                u = w2_blockscale.view(torch.uint8)
                ok_zs = _G[key] = not bool(((u & 0x7F) == 0x7F).any().item())
                if not ok_zs:
                    print("[blink-moe] W2ZS: NaN e4m3 in w2 scales: zero-chunk skip off for this layer", file=sys.stderr,
                          flush=True)
            zs = ok_zs
        w2fl = (1 if disc else 0) | (2 if zs else 0) | (8 if (W16 and v == 11) else 0)
        if w2fl:
            # E69: team ruj's variant 11 (bmoe69 build): flags 1 = L2 evict_last c3 stores (C6), 2 = zero-chunk skip
            _x69().moe_mm69x(c3, int_fp4, w2_fp4, int_blockscale, w2_blockscale, w2_alphas, problem_sizes2,
                             expert_offsets[:-1], blockscale_offsets[:-1], w2fl, 0, sms)
        elif v >= 0:
            _gemm_ext().moe_mm(c3, int_fp4, w2_fp4, int_blockscale, w2_blockscale, w2_alphas, problem_sizes2,
                               expert_offsets[:-1], blockscale_offsets[:-1], v, r, sms)
        else:
            ops.cutlass_fp4_moe_mm(c3, int_fp4, w2_fp4, int_blockscale, w2_blockscale, w2_alphas, problem_sizes2,
                                   expert_offsets[:-1], blockscale_offsets[:-1])
        if use_tail and not _overlap(output, c3):
            tail(c3, c_map, topk_weights, output, num_topk, k)
            if disc:
                _x69().discard69n([c3, int_fp4, int_blockscale])
            del int_fp4, int_blockscale
        else:
            del int_fp4, int_blockscale
            c3 = ops.shuffle_rows(c3, c_map)
            output.copy_((c3.view(m, num_topk, k) * topk_weights.view(m, num_topk, 1).to(out_dtype)).sum(dim=1),
                         non_blocking=True)

    mod._blink_orig_run_cutlass_moe_fp4 = orig       # for the kernel self-check (proto/kcheck59.py)
    mod._blinkm_heads = (plan_scatter, fast_plan_scatter)   # team ruj bj/blinkm: for the head check (test/headcheck2.py)
    mod.run_cutlass_moe_fp4 = run_cutlass_moe_fp4
    print(f"[blink-moe] installed: head={HEAD} tail={TAIL} order={TAIL_ORDER} block={TAIL_BLOCK} gemm={GEMM} actq={ACTQ} plan={PLAN} fuse={FUSE}/{FUSE_CFG} fhead={FHEAD} discard={DISCARD} w2zs={W2ZS} w16={W16} l2persist={L2PERSIST} "
          f"{_gemm_table() if GEMM else ''}", file=sys.stderr, flush=True)


_TARGET = "vllm.model_executor.layers.fused_moe.experts.cutlass_moe"


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


if HEAD or TAIL or GEMM or ACTQ or PLAN:
    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
    else:
        sys.meta_path.insert(0, _Finder())
