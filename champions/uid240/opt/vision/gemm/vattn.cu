// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 pjs team (vision engine)
//
// vattn: exact attention forward for the Gemma4 VISION tower (HF Gemma4VisionAttention under PyTorch SDPA).
//
// The eager reference runs torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=<bool padding mask>,
// scale=1.0) with q/k/v bf16 [B, 16 heads, S, 72]. PyTorch turns the bool mask into a bf16 0/-inf bias and dispatches
// its memory-efficient kernel fmha_cutlassF_bf16_aligned_64x128_rf_sm80 (PyTorch's copy of the xformers "cutlassF"
// forward, BSD-3-Clause; aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h, shipped in the
// torch wheel's include tree). This file is OUR implementation of the same arithmetic, written from that public source
// and from the SASS of the torch 2.13.0+cu130 binary (libtorch_cuda.so, sm_120 cubin), so that every output element
// goes through the identical sequence of float operations:
//   * S = Q K^T as a chain of mma.m16n8k8 bf16 -> fp32 tensor-core steps over the head dim in increasing k order
//     (the reference's InstructionShape 16x8x8; it also runs 3 all-zero steps for dims 72..95, which add exact zeros;
//     template flag ZQK reproduces them for the equivalence test);
//   * s = fmul(acc, scale); s = fadd(s, bias) for in-range keys (bias 0 or -inf; FMUL + predicated FADD in the SASS,
//     not contracted); s = fmul(s, (float)log2(e));
//   * online softmax over blocks of 128 keys (kKeysPerBlock): block max -> m_new = max(m_old, max); if m_old < m_new:
//     r = exp2f(fadd(m_old, -m_new)), l = fmul(l, r); else r = 1 (and m = 0 is used for the exponent when m_new is
//     -inf); O = fmul(O, r) from the second block on; p = exp2f(fadd(s, -m)) (exp2f = ex2.approx.f32 with the non-ftz
//     range fix, as in the reference SASS), 0 for keys >= S;
//   * row sum in the reference's order: each 32-key group (one reference warp column) is summed per lane over its 8
//     values (tile-major, then column) starting from 0, then lane^1 and lane^2 butterflies; l = (((l + w0) + w1) + w2)
//     + w3;
//   * P rounded to bf16 (round to nearest even), O += P V as mma.m16n8k8 steps over the block's keys in increasing
//     order, 4 * ceil(keys_in_block / 32) steps (the reference's 32-key mainloop iterations);
//   * out = bf16_rn(fmul(O, 1 / (l == 0 ? 1 : l))) with a correctly rounded reciprocal.
// Free choices (scheduling only): 16 query rows per warp with P kept in registers, no 128-wide padding of the value
// dim, cp.async K/V tiles, any number of query rows per block, and the order of independent work: the default kernel
// (fwd3_kernel, variant 0) post-processes each group of 4 key tiles right after its Q K^T MMAs and runs exp2 / row-sum
// terms / bf16 pack / P V MMAs tile by tile, with single K and V buffers so that two 4-warp CTAs share an SM.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace vattn {

constexpr int HD = 72;                 // head dim (q, k and v)
constexpr int KB = 128;                // keys per online-softmax block (reference kKeysPerBlock)
constexpr int NKT = KB / 8;            // 16 n8 key tiles per block
constexpr int NDT = HD / 8;            // 9 k8 steps over the head dim (QK) / 9 n8 tiles of the value dim (PV)
constexpr int ROWB = HD * 2;           // 144 bytes per smem row (conflict-free for ldmatrix: 36-word stride)
constexpr int CHUNKS = ROWB / 16;      // 9 x 16-byte chunks per row
constexpr float L2E = 1.4426950408889634074f;   // the reference's kLog2e as float

__device__ __forceinline__ uint32_t su32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool pred) {
  const int sz = pred ? 16 : 0;    // 0: zero fill (the reference's predicated loads also give zeros)
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm2(uint32_t& r0, uint32_t& r1, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm2t(uint32_t& r0, uint32_t& r1, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
// C += A(16x8 row) * B(8x8 col), bf16 in, fp32 accumulate: HMMA.1688.F32.BF16, the reference's instruction
__device__ __forceinline__ void mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(b));
}
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);    // cvt.rn.bf16x2.f32 = F2FP.BF16.F32.PACK_AB
  return *reinterpret_cast<uint32_t*>(&v);
}

struct Args {
  const __nv_bfloat16* q;
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  __nv_bfloat16* o;
  const uint8_t* kvalid;   // [B, Sk] (batch stride mvb), nonzero = key visible; nullptr = all visible
  int Sq, Sk;
  int64_t qsb, qss, qsh, ksb, kss, ksh, vsb, vss, vsh, osb, oss, osh, mvb;
  float scale;
};

template <int NW, bool ZQK>
__global__ void __launch_bounds__(NW * 32, 1) fwd_kernel(const Args a) {
  constexpr int BM = NW * 16;
  constexpr int NT = NW * 32;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sq = smem;                          // [BM][144]
  uint8_t* sk = sq + BM * ROWB;                // [2][128][144]
  uint8_t* sv = sk + 2 * KB * ROWB;            // [2][128][144]
  float* sb = reinterpret_cast<float*>(sv + 2 * KB * ROWB);   // [2][128] key bias (0 / -inf)

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int b = blockIdx.z, h = blockIdx.y;
  const int q0 = blockIdx.x * BM;
  const __nv_bfloat16* qb = a.q + b * a.qsb + h * a.qsh;
  const __nv_bfloat16* kb = a.k + b * a.ksb + h * a.ksh;
  const __nv_bfloat16* vb = a.v + b * a.vsb + h * a.vsh;
  const uint8_t* mb = a.kvalid ? a.kvalid + b * a.mvb : nullptr;
  const int Sk = a.Sk;
  const int nblk = (Sk + KB - 1) / KB;

  for (int i = tid; i < BM * CHUNKS; i += NT) {
    const int r = i / CHUNKS, c = i - r * CHUNKS;
    const int qi = q0 + r;
    const bool ok = qi < a.Sq;
    cp16(su32(sq + r * ROWB + c * 16), qb + (int64_t)(ok ? qi : 0) * a.qss + c * 8, ok);
  }
  cp_commit();
  auto load_kv = [&](int blk, int st) {
    const int j0 = blk * KB;
    for (int i = tid; i < KB * CHUNKS; i += NT) {
      const int r = i / CHUNKS, c = i - r * CHUNKS;
      const int kj = j0 + r;
      const bool ok = kj < Sk;
      const int kk = ok ? kj : 0;
      cp16(su32(sk + (st * KB + r) * ROWB + c * 16), kb + (int64_t)kk * a.kss + c * 8, ok);
      cp16(su32(sv + (st * KB + r) * ROWB + c * 16), vb + (int64_t)kk * a.vss + c * 8, ok);
    }
    for (int i = tid; i < KB; i += NT) {
      const int kj = j0 + i;
      sb[st * KB + i] = (mb != nullptr && kj < Sk && mb[kj] == 0) ? -INFINITY : 0.f;
    }
    cp_commit();
  };
  load_kv(0, 0);
  if (nblk > 1) load_kv(1, 1);
  if (nblk > 1) cp_wait<2>(); else cp_wait<1>();
  __syncthreads();

  uint32_t qa[NDT][2];
  {
    const int lr = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int lc = lane >> 4;
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
      uint32_t r[4];
      ldsm4(r, su32(sq + lr * ROWB + (j + lc) * 16));
      qa[j][0] = r[0]; qa[j][1] = r[1]; qa[j + 1][0] = r[2]; qa[j + 1][1] = r[3];
    }
    ldsm2(qa[8][0], qa[8][1], su32(sq + lr * ROWB + 8 * 16));
  }

  float o[NDT][4];
#pragma unroll
  for (int d = 0; d < NDT; ++d) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f;
  float mrow[2] = {-INFINITY, -INFINITY};
  float lrow[2] = {0.f, 0.f};
  const float scale = a.scale;

  for (int blk = 0; blk < nblk; ++blk) {
    const int st = blk & 1;
    if (blk + 1 < nblk) cp_wait<1>(); else cp_wait<0>();
    __syncthreads();
    const int nk = min(KB, Sk - blk * KB);
    const uint8_t* ks = sk + st * KB * ROWB;
    const uint8_t* vs = sv + st * KB * ROWB;
    const float* bs = sb + st * KB;

    float s[NKT][4];
#pragma unroll
    for (int n = 0; n < NKT; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
    {
      const int kr = (lane & 7) + (lane >> 3) * 8;
#pragma unroll
      for (int j = 0; j < NDT; ++j) {
#pragma unroll
        for (int n4 = 0; n4 < 4; ++n4) {
          uint32_t r[4];
          ldsm4(r, su32(ks + (n4 * 32 + kr) * ROWB + j * 16));
#pragma unroll
          for (int i = 0; i < 4; ++i) mma(s[n4 * 4 + i], qa[j][0], qa[j][1], r[i]);
        }
      }
      if (ZQK) {
#pragma unroll
        for (int j = NDT; j < 12; ++j)
#pragma unroll
          for (int n = 0; n < NKT; ++n) mma(s[n], 0u, 0u, 0u);
      }
    }
    // scale, key bias, log2(e)
#pragma unroll
    for (int n = 0; n < NKT; ++n) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int key = n * 8 + 2 * t + c;
        const float bias = bs[key];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          float x = __fmul_rn(s[n][2 * r + c], scale);
          if (key < nk) x = __fadd_rn(x, bias);
          s[n][2 * r + c] = __fmul_rn(x, L2E);
        }
      }
    }
    float muse[2], resc[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float mx = -INFINITY;
#pragma unroll
      for (int n = 0; n < NKT; ++n)
#pragma unroll
        for (int c = 0; c < 2; ++c)
          if (n * 8 + 2 * t + c < nk) mx = fmaxf(mx, s[n][2 * r + c]);
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float mn = fmaxf(mrow[r], mx);
      if (mrow[r] < mn) {
        resc[r] = exp2f(__fadd_rn(mrow[r], -mn));
        lrow[r] = __fmul_rn(lrow[r], resc[r]);
        muse[r] = mn;
      } else {
        resc[r] = 1.f;
        muse[r] = (mn == -INFINITY) ? 0.f : mn;
      }
      mrow[r] = mn;
    }
    if (blk > 0) {
#pragma unroll
      for (int d = 0; d < NDT; ++d) {
        o[d][0] = __fmul_rn(o[d][0], resc[0]);
        o[d][1] = __fmul_rn(o[d][1], resc[0]);
        o[d][2] = __fmul_rn(o[d][2], resc[1]);
        o[d][3] = __fmul_rn(o[d][3], resc[1]);
      }
    }
#pragma unroll
    for (int n = 0; n < NKT; ++n)
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const bool in = n * 8 + 2 * t + c < nk;
#pragma unroll
        for (int r = 0; r < 2; ++r)
          s[n][2 * r + c] = in ? exp2f(__fadd_rn(s[n][2 * r + c], -muse[r])) : 0.f;
      }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float w[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float acc = 0.f;
#pragma unroll
        for (int n = 4 * i; n < 4 * i + 4; ++n) {
          acc = __fadd_rn(acc, s[n][2 * r]);
          acc = __fadd_rn(acc, s[n][2 * r + 1]);
        }
        acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 1));
        acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 2));
        w[i] = acc;
      }
      lrow[r] = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(lrow[r], w[0]), w[1]), w[2]), w[3]);
    }
    uint32_t pa[NKT][2];
#pragma unroll
    for (int n = 0; n < NKT; ++n) {
      pa[n][0] = pack(s[n][0], s[n][1]);
      pa[n][1] = pack(s[n][2], s[n][3]);
    }
    const int npair = ((nk + 31) / 32) * 2;   // k8 steps / 2 (the reference runs 4 per 32 keys)
    {
      const int vr = lane & 7, vg = lane >> 3, v2 = lane & 15;
#pragma unroll
      for (int kp = 0; kp < NKT / 2; ++kp) {
        if (kp < npair) {
          uint32_t r[4], e0, e1;
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            const int kk = 2 * kp + u;
            ldsm4t(r, su32(vs + (kk * 8 + vr) * ROWB + vg * 16));
#pragma unroll
            for (int i = 0; i < 4; ++i) mma(o[i], pa[kk][0], pa[kk][1], r[i]);
            ldsm4t(r, su32(vs + (kk * 8 + vr) * ROWB + (4 + vg) * 16));
#pragma unroll
            for (int i = 0; i < 4; ++i) mma(o[4 + i], pa[kk][0], pa[kk][1], r[i]);
          }
          ldsm2t(e0, e1, su32(vs + (2 * kp * 8 + v2) * ROWB + 8 * 16));
          // dim tile 8 for the two steps, in step order (accumulators are independent per dim tile)
          mma(o[8], pa[2 * kp][0], pa[2 * kp][1], e0);
          mma(o[8], pa[2 * kp + 1][0], pa[2 * kp + 1][1], e1);
        }
      }
    }
    __syncthreads();
    if (blk + 2 < nblk) load_kv(blk + 2, st);
  }

  float alpha[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) alpha[r] = (lrow[r] != 0.f) ? __frcp_rn(lrow[r]) : 1.f;
  const int r0 = q0 + warp * 16 + g;
  __nv_bfloat16* ob = a.o + b * a.osb + h * a.osh;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int qi = r0 + 8 * r;
    if (qi < a.Sq) {
      uint32_t* dst = reinterpret_cast<uint32_t*>(ob + (int64_t)qi * a.oss + 2 * t);
#pragma unroll
      for (int d = 0; d < NDT; ++d)
        dst[d * 4] = pack(__fmul_rn(o[d][2 * r], alpha[r]), __fmul_rn(o[d][2 * r + 1], alpha[r]));
    }
  }
}

template <int NW, bool ZQK>
void launch(const Args& a, int B, int H, cudaStream_t stream) {
  constexpr int BM = NW * 16;
  const size_t smem = (size_t)BM * ROWB + 4 * KB * ROWB + 2 * KB * sizeof(float);
  auto kern = fwd_kernel<NW, ZQK>;
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess,
                "vattn: cannot set shared memory size");
    attr = true;
  }
  dim3 grid((a.Sq + BM - 1) / BM, H, B);
  kern<<<grid, NW * 32, smem, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------------------------------------------
// Faster schedule of the same arithmetic (variants 4 and 5). Per 128-key block and warp (16 query rows):
//   qk_part: Q K^T in 4 groups of 4 key tiles (each tile's k8 chain in increasing k order); each group's
//            scale / key bias / log2(e) and the running row max right after its MMAs, so they overlap the next group;
//   softmax_state: block max -> m_new, rescale factor, l *= r (as above);
//   pv_part: per key tile in increasing order: p = exp2f(s - m), its row-sum terms (the reference's per-32-key-group
//            lane order), its bf16 pack and its P V MMAs (the PV chain of every output stays in increasing key order).
// FULL: all 128 keys in range (no per-key range checks); MASK: a key mask is given (bias add; without a mask every
// key is visible and the reference's "+ 0" bias only turns -0 into +0, which cannot reach any output: max, s - m and
// exp2 give the same values for either zero); SCALE1: scale == 1 (x * 1 == x).
template <bool FULL, bool MASK, bool SCALE1>
__device__ __forceinline__ void qk_part(float (&s)[NKT][4], float (&mx)[2], const uint32_t (&qa)[NDT][2],
                                        const uint8_t* ks, const float* bs, int nk, float scale, int lane) {
  const int t = lane & 3;
#pragma unroll
  for (int n = 0; n < NKT; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
  mx[0] = mx[1] = -INFINITY;
  const int kr = (lane & 7) + (lane >> 3) * 8;
#pragma unroll
  for (int n4 = 0; n4 < 4; ++n4) {
#pragma unroll
    for (int j = 0; j < NDT; ++j) {
      uint32_t r[4];
      ldsm4(r, su32(ks + (n4 * 32 + kr) * ROWB + j * 16));
#pragma unroll
      for (int i = 0; i < 4; ++i) mma(s[n4 * 4 + i], qa[j][0], qa[j][1], r[i]);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int n = n4 * 4 + i;
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int key = n * 8 + 2 * t + c;
        const bool in = FULL || key < nk;
        const float bias = MASK ? bs[key] : 0.f;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          float x = s[n][2 * r + c];
          if (!SCALE1) x = __fmul_rn(x, scale);
          if (MASK && in) x = __fadd_rn(x, bias);
          x = __fmul_rn(x, L2E);
          s[n][2 * r + c] = x;
          if (in) mx[r] = fmaxf(mx[r], x);
        }
      }
    }
  }
}

__device__ __forceinline__ void softmax_state(const float (&mx)[2], float (&mrow)[2], float (&lrow)[2],
                                              float (&muse)[2], float (&resc)[2]) {
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    float m = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    const float mn = fmaxf(mrow[r], m);
    if (mrow[r] < mn) {
      resc[r] = exp2f(__fadd_rn(mrow[r], -mn));
      lrow[r] = __fmul_rn(lrow[r], resc[r]);
      muse[r] = mn;
    } else {
      resc[r] = 1.f;
      muse[r] = (mn == -INFINITY) ? 0.f : mn;
    }
    mrow[r] = mn;
  }
}

template <bool FULL>
__device__ __forceinline__ void pv_part(float (&o)[NDT][4], float (&lrow)[2], const float (&s)[NKT][4],
                                        const float (&muse)[2], const uint8_t* vs, int nk, int lane) {
  const int t = lane & 3;
  float acc[2] = {0.f, 0.f};
  float w[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
  const int npair = FULL ? NKT / 2 : ((nk + 31) / 32) * 2;
  const int vr = lane & 7, vg = lane >> 3, v2 = lane & 15;
#pragma unroll
  for (int kp = 0; kp < NKT / 2; ++kp) {
    if (FULL || kp < npair) {
      uint32_t pa[2][2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int kk = 2 * kp + u;
        float p[4];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const bool in = FULL || kk * 8 + 2 * t + c < nk;
#pragma unroll
          for (int r = 0; r < 2; ++r) p[2 * r + c] = in ? exp2f(__fadd_rn(s[kk][2 * r + c], -muse[r])) : 0.f;
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          acc[r] = __fadd_rn(acc[r], p[2 * r]);
          acc[r] = __fadd_rn(acc[r], p[2 * r + 1]);
          if ((kk & 3) == 3) {
            w[r][kk >> 2] = acc[r];
            acc[r] = 0.f;
          }
        }
        pa[u][0] = pack(p[0], p[1]);
        pa[u][1] = pack(p[2], p[3]);
        uint32_t rr[4];
        ldsm4t(rr, su32(vs + (kk * 8 + vr) * ROWB + vg * 16));
#pragma unroll
        for (int i = 0; i < 4; ++i) mma(o[i], pa[u][0], pa[u][1], rr[i]);
        ldsm4t(rr, su32(vs + (kk * 8 + vr) * ROWB + (4 + vg) * 16));
#pragma unroll
        for (int i = 0; i < 4; ++i) mma(o[4 + i], pa[u][0], pa[u][1], rr[i]);
      }
      uint32_t e0, e1;
      ldsm2t(e0, e1, su32(vs + (2 * kp * 8 + v2) * ROWB + 8 * 16));
      mma(o[8], pa[0][0], pa[0][1], e0);
      mma(o[8], pa[1][0], pa[1][1], e1);
    }
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float x = w[r][i];
      x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 1));
      x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 2));
      w[r][i] = x;
    }
    lrow[r] = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(lrow[r], w[r][0]), w[r][1]), w[r][2]), w[r][3]);
  }
}

__device__ __forceinline__ void rescale_o(float (&o)[NDT][4], const float (&resc)[2]) {
#pragma unroll
  for (int d = 0; d < NDT; ++d) {
    o[d][0] = __fmul_rn(o[d][0], resc[0]);
    o[d][1] = __fmul_rn(o[d][1], resc[0]);
    o[d][2] = __fmul_rn(o[d][2], resc[1]);
    o[d][3] = __fmul_rn(o[d][3], resc[1]);
  }
}

template <int NW>
__device__ __forceinline__ void load_q_frags(uint32_t (&qa)[NDT][2], const uint8_t* sq, int warp, int lane) {
  const int lr = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
  const int lc = lane >> 4;
#pragma unroll
  for (int j = 0; j < 8; j += 2) {
    uint32_t r[4];
    ldsm4(r, su32(sq + lr * ROWB + (j + lc) * 16));
    qa[j][0] = r[0]; qa[j][1] = r[1]; qa[j + 1][0] = r[2]; qa[j + 1][1] = r[3];
  }
  ldsm2(qa[8][0], qa[8][1], su32(sq + lr * ROWB + 8 * 16));
}

__device__ __forceinline__ void store_o(const Args& a, float (&o)[NDT][4], const float (&lrow)[2], int b, int h,
                                        int r0, int t) {
  float alpha[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) alpha[r] = (lrow[r] != 0.f) ? __frcp_rn(lrow[r]) : 1.f;
  __nv_bfloat16* ob = a.o + b * a.osb + h * a.osh;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int qi = r0 + 8 * r;
    if (qi < a.Sq) {
      uint32_t* dst = reinterpret_cast<uint32_t*>(ob + (int64_t)qi * a.oss + 2 * t);
#pragma unroll
      for (int d = 0; d < NDT; ++d)
        dst[d * 4] = pack(__fmul_rn(o[d][2 * r], alpha[r]), __fmul_rn(o[d][2 * r + 1], alpha[r]));
    }
  }
}

// variant 4: K/V double-buffered (one CTA of NW warps per SM)
template <int NW, bool MASK, bool SCALE1>
__global__ void __launch_bounds__(NW * 32, 1) fwd2_kernel(const Args a) {
  constexpr int BM = NW * 16;
  constexpr int NT = NW * 32;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sq = smem;
  uint8_t* sk = sq + BM * ROWB;
  uint8_t* sv = sk + 2 * KB * ROWB;
  float* sb = reinterpret_cast<float*>(sv + 2 * KB * ROWB);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int b = blockIdx.z, h = blockIdx.y;
  const int q0 = blockIdx.x * BM;
  const __nv_bfloat16* qb = a.q + b * a.qsb + h * a.qsh;
  const __nv_bfloat16* kb = a.k + b * a.ksb + h * a.ksh;
  const __nv_bfloat16* vb = a.v + b * a.vsb + h * a.vsh;
  const uint8_t* mb = MASK ? a.kvalid + b * a.mvb : nullptr;
  const int Sk = a.Sk;
  const int nblk = (Sk + KB - 1) / KB;
  for (int i = tid; i < BM * CHUNKS; i += NT) {
    const int r = i / CHUNKS, c = i - r * CHUNKS;
    const int qi = q0 + r;
    const bool ok = qi < a.Sq;
    cp16(su32(sq + r * ROWB + c * 16), qb + (int64_t)(ok ? qi : 0) * a.qss + c * 8, ok);
  }
  cp_commit();
  auto load_kv = [&](int blk, int st) {
    const int j0 = blk * KB;
    for (int i = tid; i < KB * CHUNKS; i += NT) {
      const int r = i / CHUNKS, c = i - r * CHUNKS;
      const int kj = j0 + r;
      const bool ok = kj < Sk;
      const int kk = ok ? kj : 0;
      cp16(su32(sk + (st * KB + r) * ROWB + c * 16), kb + (int64_t)kk * a.kss + c * 8, ok);
      cp16(su32(sv + (st * KB + r) * ROWB + c * 16), vb + (int64_t)kk * a.vss + c * 8, ok);
    }
    if (MASK) {
      for (int i = tid; i < KB; i += NT) {
        const int kj = j0 + i;
        sb[st * KB + i] = (kj < Sk && mb[kj] == 0) ? -INFINITY : 0.f;
      }
    }
    cp_commit();
  };
  load_kv(0, 0);
  if (nblk > 1) load_kv(1, 1);
  if (nblk > 1) cp_wait<2>(); else cp_wait<1>();
  __syncthreads();
  uint32_t qa[NDT][2];
  load_q_frags<NW>(qa, sq, warp, lane);
  float o[NDT][4];
#pragma unroll
  for (int d = 0; d < NDT; ++d) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f;
  float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
  for (int blk = 0; blk < nblk; ++blk) {
    const int st = blk & 1;
    if (blk + 1 < nblk) cp_wait<1>(); else cp_wait<0>();
    __syncthreads();
    const int nk = min(KB, Sk - blk * KB);
    const uint8_t* ks = sk + st * KB * ROWB;
    const uint8_t* vs = sv + st * KB * ROWB;
    const float* bs = sb + st * KB;
    float s[NKT][4], mx[2], muse[2], resc[2];
    if (nk == KB) {
      qk_part<true, MASK, SCALE1>(s, mx, qa, ks, bs, nk, a.scale, lane);
      softmax_state(mx, mrow, lrow, muse, resc);
      if (blk > 0) rescale_o(o, resc);
      pv_part<true>(o, lrow, s, muse, vs, nk, lane);
    } else {
      qk_part<false, MASK, SCALE1>(s, mx, qa, ks, bs, nk, a.scale, lane);
      softmax_state(mx, mrow, lrow, muse, resc);
      if (blk > 0) rescale_o(o, resc);
      pv_part<false>(o, lrow, s, muse, vs, nk, lane);
    }
    __syncthreads();
    if (blk + 2 < nblk) load_kv(blk + 2, st);
  }
  store_o(a, o, lrow, b, h, q0 + warp * 16 + (lane >> 2), lane & 3);
}

// variant 5: single K and V buffers with split pipelining (K of the next block loads during P V, V during Q K^T),
// so that two CTAs of NW warps fit on an SM
template <int NW, bool MASK, bool SCALE1>
__global__ void __launch_bounds__(NW * 32, 2) fwd3_kernel(const Args a) {
  constexpr int BM = NW * 16;
  constexpr int NT = NW * 32;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sq = smem;
  uint8_t* sk = sq + BM * ROWB;
  uint8_t* sv = sk + KB * ROWB;
  float* sb = reinterpret_cast<float*>(sv + KB * ROWB);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int b = blockIdx.z, h = blockIdx.y;
  const int q0 = blockIdx.x * BM;
  const __nv_bfloat16* qb = a.q + b * a.qsb + h * a.qsh;
  const __nv_bfloat16* kb = a.k + b * a.ksb + h * a.ksh;
  const __nv_bfloat16* vb = a.v + b * a.vsb + h * a.vsh;
  const uint8_t* mb = MASK ? a.kvalid + b * a.mvb : nullptr;
  const int Sk = a.Sk;
  const int nblk = (Sk + KB - 1) / KB;
  for (int i = tid; i < BM * CHUNKS; i += NT) {
    const int r = i / CHUNKS, c = i - r * CHUNKS;
    const int qi = q0 + r;
    const bool ok = qi < a.Sq;
    cp16(su32(sq + r * ROWB + c * 16), qb + (int64_t)(ok ? qi : 0) * a.qss + c * 8, ok);
  }
  auto load_k = [&](int blk) {
    const int j0 = blk * KB;
    for (int i = tid; i < KB * CHUNKS; i += NT) {
      const int r = i / CHUNKS, c = i - r * CHUNKS;
      const int kj = j0 + r;
      const bool ok = kj < Sk;
      cp16(su32(sk + r * ROWB + c * 16), kb + (int64_t)(ok ? kj : 0) * a.kss + c * 8, ok);
    }
    if (MASK) {
      for (int i = tid; i < KB; i += NT) {
        const int kj = j0 + i;
        sb[i] = (kj < Sk && mb[kj] == 0) ? -INFINITY : 0.f;
      }
    }
    cp_commit();
  };
  auto load_v = [&](int blk) {
    const int j0 = blk * KB;
    for (int i = tid; i < KB * CHUNKS; i += NT) {
      const int r = i / CHUNKS, c = i - r * CHUNKS;
      const int kj = j0 + r;
      const bool ok = kj < Sk;
      cp16(su32(sv + r * ROWB + c * 16), vb + (int64_t)(ok ? kj : 0) * a.vss + c * 8, ok);
    }
    cp_commit();
  };
  load_k(0);          // group: Q + K0 (+ bias0)
  load_v(0);          // group: V0
  cp_wait<1>();
  __syncthreads();
  uint32_t qa[NDT][2];
  load_q_frags<NW>(qa, sq, warp, lane);
  float o[NDT][4];
#pragma unroll
  for (int d = 0; d < NDT; ++d) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f;
  float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
  for (int blk = 0; blk < nblk; ++blk) {
    const int nk = min(KB, Sk - blk * KB);
    const bool more = blk + 1 < nblk;
    float s[NKT][4], mx[2], muse[2], resc[2];
    if (nk == KB) qk_part<true, MASK, SCALE1>(s, mx, qa, sk, sb, nk, a.scale, lane);
    else qk_part<false, MASK, SCALE1>(s, mx, qa, sk, sb, nk, a.scale, lane);
    __syncthreads();              // every warp is done with K (and the bias) of this block
    if (more) load_k(blk + 1);
    softmax_state(mx, mrow, lrow, muse, resc);
    if (blk > 0) rescale_o(o, resc);
    if (more) cp_wait<1>(); else cp_wait<0>();     // V of this block
    __syncthreads();
    if (nk == KB) pv_part<true>(o, lrow, s, muse, sv, nk, lane);
    else pv_part<false>(o, lrow, s, muse, sv, nk, lane);
    __syncthreads();              // every warp is done with V of this block
    if (more) {
      load_v(blk + 1);
      cp_wait<1>();               // K of the next block
      __syncthreads();
    }
  }
  store_o(a, o, lrow, b, h, q0 + warp * 16 + (lane >> 2), lane & 3);
}

template <int NW, bool MASK, bool SCALE1>
void launch2(const Args& a, int B, int H, cudaStream_t stream) {
  constexpr int BM = NW * 16;
  const size_t smem = (size_t)BM * ROWB + 4 * KB * ROWB + 2 * KB * sizeof(float);
  auto kern = fwd2_kernel<NW, MASK, SCALE1>;
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess,
                "vattn: cannot set shared memory size");
    attr = true;
  }
  dim3 grid((a.Sq + BM - 1) / BM, H, B);
  kern<<<grid, NW * 32, smem, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int NW, bool MASK, bool SCALE1>
void launch3(const Args& a, int B, int H, cudaStream_t stream) {
  constexpr int BM = NW * 16;
  const size_t smem = (size_t)BM * ROWB + 2 * KB * ROWB + KB * sizeof(float);
  auto kern = fwd3_kernel<NW, MASK, SCALE1>;
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess,
                "vattn: cannot set shared memory size");
    attr = true;
  }
  dim3 grid((a.Sq + BM - 1) / BM, H, B);
  kern<<<grid, NW * 32, smem, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace vattn

// q, k, v: [B, S, H, 72] bf16 views (any batch / row / head strides, dim stride 1, 16-byte aligned rows), e.g. the
// .transpose(1, 2) of the [B, H, S, 72] tensors HF passes to SDPA. kvalid: optional [B, Sk] bool/uint8 key mask
// (True = visible; = the 2-D padding mask behind HF's bidirectional mask). Returns out [B, Sq, H, 72] contiguous (the
// memory layout of SDPA's mem-efficient output). Every variant computes the same bits (byte-tested); they differ in
// scheduling only: 0 (default) = interleaved schedule, 4 warps, two CTAs per SM; 1 = interleaved schedule, 8 warps,
// double-buffered K/V; 2 = the first version (8 warps, block-wise phases); 3 = 2 + the reference's all-zero k-steps.
torch::Tensor fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, c10::optional<torch::Tensor> kvalid, double scale,
                  int64_t variant) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "vattn: CUDA tensors expected");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16,
              "vattn: bf16 expected");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "vattn: [B, S, H, D] expected");
  const int64_t B = q.size(0), Sq = q.size(1), H = q.size(2);
  const int64_t Sk = k.size(1);
  TORCH_CHECK(q.size(3) == vattn::HD && k.size(3) == vattn::HD && v.size(3) == vattn::HD, "vattn: head dim 72 only");
  TORCH_CHECK(k.size(0) == B && v.size(0) == B && k.size(2) == H && v.size(2) == H && v.size(1) == Sk,
              "vattn: shape mismatch");
  TORCH_CHECK(Sk > 0 && Sq > 0, "vattn: empty");
  for (const auto& t : {q, k, v}) {
    TORCH_CHECK(t.stride(3) == 1 && t.stride(0) % 8 == 0 && t.stride(1) % 8 == 0 && t.stride(2) % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "vattn: rows must be 16-byte aligned with unit dim stride");
  }
  c10::cuda::CUDAGuard guard(q.device());
  auto out = torch::empty({B, Sq, H, vattn::HD}, q.options());
  vattn::Args a;
  a.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  a.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  a.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  a.o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  a.kvalid = nullptr;
  a.mvb = 0;
  if (kvalid.has_value() && kvalid->defined()) {
    const auto& m = *kvalid;
    TORCH_CHECK(m.is_cuda() && (m.scalar_type() == at::kBool || m.scalar_type() == at::kByte) && m.dim() == 2 &&
                    m.size(0) == B && m.size(1) == Sk && m.stride(1) == 1,
                "vattn: kvalid must be [B, Sk] bool/uint8 with unit key stride");
    a.kvalid = reinterpret_cast<const uint8_t*>(m.data_ptr());
    a.mvb = m.stride(0);
  }
  a.Sq = (int)Sq;
  a.Sk = (int)Sk;
  a.qsb = q.stride(0); a.qss = q.stride(1); a.qsh = q.stride(2);
  a.ksb = k.stride(0); a.kss = k.stride(1); a.ksh = k.stride(2);
  a.vsb = v.stride(0); a.vss = v.stride(1); a.vsh = v.stride(2);
  a.osb = out.stride(0); a.oss = out.stride(1); a.osh = out.stride(2);
  a.scale = (float)scale;
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  const bool mask = a.kvalid != nullptr, one = a.scale == 1.0f;
  switch (variant) {
    case 0:   // default: interleaved schedule, 4 warps (64 query rows), single K/V buffers, two CTAs per SM
      if (mask) { if (one) vattn::launch3<4, true, true>(a, (int)B, (int)H, stream); else vattn::launch3<4, true, false>(a, (int)B, (int)H, stream); }
      else { if (one) vattn::launch3<4, false, true>(a, (int)B, (int)H, stream); else vattn::launch3<4, false, false>(a, (int)B, (int)H, stream); }
      break;
    case 1:   // interleaved schedule, 8 warps (128 query rows), double-buffered K/V, one CTA per SM
      if (mask) { if (one) vattn::launch2<8, true, true>(a, (int)B, (int)H, stream); else vattn::launch2<8, true, false>(a, (int)B, (int)H, stream); }
      else { if (one) vattn::launch2<8, false, true>(a, (int)B, (int)H, stream); else vattn::launch2<8, false, false>(a, (int)B, (int)H, stream); }
      break;
    case 2: vattn::launch<8, false>(a, (int)B, (int)H, stream); break;   // first version (A1 hand-off 9ab3c07c)
    case 3: vattn::launch<8, true>(a, (int)B, (int)H, stream); break;    // + the reference's 3 all-zero k-steps
    default: TORCH_CHECK(false, "vattn: unknown variant");
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &fwd, "exact Gemma4 vision attention forward (= PyTorch mem-efficient SDPA, bias variant), head dim 72",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("kvalid"), py::arg("scale"), py::arg("variant") = 0);
}
