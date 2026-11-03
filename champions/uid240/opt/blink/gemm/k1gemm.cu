// SPDX-License-Identifier: Apache-2.0
// k1gemm (K1, 28 Oct 2026): exact bf16 GEMM C[M,N] = A[M,K] * W[N,K]^T for Blink's dense layers on sm_120a
// (RTX PRO 6000 Blackwell), and the dense MLP's gate_up GEMM with the GELU(tanh)*up epilogue fused (glu).
//
// Exactness contract (the same as blink_gemm.cu / vLLM's batch-invariant matmul): every output element is ONE fp32
// accumulator that starts at +0 and takes mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 steps over k = 0, 16,
// 32, ..., K-16 in increasing order (no split-K, no stream-K, no sliced-K), then one cvt.rn to bf16. Which CTA / warp
// computes an element, the tile shape, the pipeline depth, TMA vs cp.async and the CTA schedule do not change any
// element's arithmetic.
//
// Design (v7): one TMA producer warp + 8 MMA warps, BM = 128, BK = 32 (64-byte rows, TMA 64B swizzle), a STAGES-deep
// full/empty mbarrier ring. Each MMA warp loads both k16 steps' fragments of a stage, issues their MMAs, then releases
// the stage with a proxy fence (its ldmatrix reads are complete and ordered before the TMA refill: without it ptxas
// hoisted the arrive above outstanding ldmatrix reads, a WAR race found in v1) and one arrive. Warp layout
// 2 x 4 (warp tile 64 rows x BN/4 columns; uneven [NJ, NJ, NJ-1, NJ-1] n8 split, balanced per SM sub-partition, when
// BN % 32 == 16); BN is any multiple of 16 from 64 to 256, so
// the tile count can be fitted to the 188 SMs (wave-aware tile choice per M). Persistent CTAs, static round-robin
// tile schedule (M-fastest raster). Out-of-range rows / columns are zero-filled by TMA and never stored.
#ifndef K1_NO_TORCH
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>

namespace k1 {

constexpr int BM = 128;
constexpr int BK = 32;
constexpr int NCW = 8;                     // MMA warps
constexpr int NTH = (NCW + 1) * 32;        // + 1 TMA producer warp
constexpr int SMEM_MAX = 101376;           // sm_120 opt-in limit per block (99 KB)

template <int BN, bool GLU>
struct Cfg {
  // 2 (M) x 4 (N) MMA warps, warp tile 64 rows x (8 * NJ or 8 * NJ_LO) columns: the BN/8 n8 tiles of the B tile are
  // split [NJ, NJ, NJ_LO, NJ_LO] over wn = 0..3 (NJ_LO = NJ - 1 when BN % 32 == 16), so that the two warps that share an
  // SM sub-partition (wn and wn + 2) always hold NJ + NJ_LO n8 tiles: balanced tensor work for every BN % 16 == 0
  static constexpr int WGM = 2, WGN = 4;
  static constexpr int WM = BM / WGM;      // 64
  static constexpr int MI = WM / 16;       // 4 m16 tiles per warp
  static constexpr int NT = BN / 8;        // n8 tiles of the B tile
  static constexpr int NJ = (NT + 3) / 4;  // n8 tiles of warps wn = 0, 1
  static constexpr int NJ_LO = NT - 2 * NJ - (NT - 2 * NJ) / 2;   // n8 tiles of warps wn = 2, 3
  static constexpr int WN = 8 * NJ;
  static constexpr int A_BYTES = BM * BK * 2;
  static constexpr int B_BYTES = BN * BK * 2;
  static constexpr int STAGE = A_BYTES + B_BYTES;
  static constexpr int ST0 = (SMEM_MAX - 1024 - 256) / STAGE;
  static constexpr int STAGES = ST0 > 8 ? 8 : ST0;
  static constexpr int SMEM = STAGES * STAGE + 1024 + 2 * STAGES * 8;
  static_assert(BN % 16 == 0 && BN >= 32 && BN <= 256, "BN must be a multiple of 16 in [32, 256]");
  static_assert(2 * NJ + 2 * NJ_LO == NT && (NJ_LO == NJ || NJ_LO == NJ - 1), "column split");
  static_assert(!GLU || (BN % 64 == 0 && NJ_LO == NJ && NJ % 2 == 0), "GLU tiles: BN % 64 == 0");
  static_assert(B_BYTES % 1024 == 0, "stage buffers must stay 1024-byte aligned");
  static_assert(STAGES >= 3, "not enough shared memory");
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  do {
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(ok)
        : "r"(bar), "r"(parity)
        : "memory");
  } while (!ok);
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* tm, uint32_t bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(tm)), "r"(bar), "r"(c0), "r"(c1)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
// release a stage to the TMA producer warp: the warp's ldmatrix reads of it are complete and ordered before the
// async-proxy (TMA) refill (proxy fence = per-thread memory barrier, so ptxas cannot hoist the arrive above them;
// without it ptxas hoisted v1's arrive above outstanding ldmatrix reads, a WAR race), then one arrive per warp.
// (Only the producer warp issues TMA: a MEMBAR in a thread with its own bulk copies in flight would wait for them.)
__device__ __forceinline__ void warp_release(uint32_t empty_bar, int lane) {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncwarp();
  if (lane == 0) mbar_arrive(empty_bar);
}
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr)
               : "memory");
}
__device__ __forceinline__ void ldsm_x2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr) : "memory");
}
// the instruction of the exactness contract: D = A(16x16 bf16, row) * B(16x8 bf16, col) + C, fp32
__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// byte offset of 16-byte chunk c (0..3) of row r in a tile of 64-byte rows written by TMA with CU_TENSOR_MAP_SWIZZLE_64B
// (16-byte chunk bits [5:4] XOR address bits [8:7]; tiles are 1024-byte aligned)
__device__ __forceinline__ uint32_t sw64(int r, int c) { return r * 64 + ((c ^ ((r >> 1) & 3)) << 4); }

// Exact replica of Inductor's GELU(tanh)*up kernel (triton_poi_fused_bi_mm_gelu_mul_slice_1 of the Blink graph, Triton
// 3.7.1, enable_fp_fusion, libdevice __nv_tanhf with the FTZ reflect on): the same PTX operations, ftz flags and constants
// in the same order; x = float(bf16 gate), u = float(bf16 up); returns the fp32 value the kernel rounds with cvt.rn.
// Triton's plain mul.f32 / add.f32 have no contraction partner in that kernel, so they are written with .rn here.
__device__ __forceinline__ float k1_gelu_tanh_mul(float x, float u) {
  float r;
  asm("{\n"
      ".reg .f32 x2, x3, t8, t10, a, s, p, th0, e, d, rc, y, y1, th1, th, t13, t3, t14;\n"
      ".reg .b32 sg, yb;\n"
      ".reg .pred psmall, pbig;\n"
      "mul.rn.f32 x2, %1, %1;\n"
      "mul.rn.f32 x3, x2, %1;\n"
      "fma.rn.f32 t8, x3, 0f3D372713, %1;\n"
      "mul.rn.f32 t10, t8, 0f3F4C422A;\n"
      "abs.ftz.f32 a, t10;\n"
      "setp.ltu.f32 psmall, a, 0f3F19999A;\n"
      "mul.rn.f32 s, t10, t10;\n"
      "fma.rn.ftz.f32 p, 0f3C80F082, s, 0fBD563CAE;\n"
      "fma.rn.ftz.f32 p, p, s, 0f3E085941;\n"
      "fma.rn.ftz.f32 p, p, s, 0fBEAAA9ED;\n"
      "fma.rn.ftz.f32 p, p, s, 0f00000000;\n"
      "fma.rn.ftz.f32 th0, p, t10, t10;\n"
      "mul.rn.f32 e, a, 0f4038AA3B;\n"
      "ex2.approx.ftz.f32 e, e;\n"
      "add.rn.f32 d, e, 0f3F800000;\n"
      "rcp.approx.ftz.f32 rc, d;\n"
      "fma.rn.ftz.f32 y, rc, 0fC0000000, 0f3F800000;\n"
      "setp.ge.f32 pbig, a, 0f41102CB4;\n"
      "selp.f32 y1, 0f3F800000, y, pbig;\n"
      "mov.b32 sg, t10;\n"
      "and.b32 sg, sg, -2147483648;\n"
      "mov.b32 yb, y1;\n"
      "or.b32 yb, yb, sg;\n"
      "mov.b32 th1, yb;\n"
      "selp.f32 th, th0, th1, psmall;\n"
      "add.rn.f32 t13, th, 0f3F800000;\n"
      "mul.rn.f32 t3, %1, 0f3F000000;\n"
      "mul.rn.f32 t14, t3, t13;\n"
      "mul.rn.f32 %0, t14, %2;\n"
      "}\n"
      : "=f"(r)
      : "f"(x), "f"(u));
  return r;
}

// first B-tile row (= tile column) of warp wn's n8 tiles, and the warp's n8 tile count
template <int BN, bool GLU>
__device__ __forceinline__ constexpr int col_base(int wn) {
  using G = Cfg<BN, GLU>;
  return 8 * (wn < 2 ? wn * G::NJ : 2 * G::NJ + (wn - 2) * G::NJ_LO);
}
template <int BN, bool GLU>
__device__ __forceinline__ constexpr int nj_of(int wn) {
  using G = Cfg<BN, GLU>;
  return wn < 2 ? G::NJ : G::NJ_LO;
}
// smem row (in the stage's B tile) of the warp's n8 tile j: plain GEMM = col_base(wn) + 8j; GLU = the warp's gate
// columns (first NJ/2 n8 tiles) and the matching up rows (last NJ/2) of a B tile laid out [BN/2 gate rows | BN/2 up rows]
template <int BN, bool GLU>
__device__ __forceinline__ constexpr int b_row(int wn, int j) {
  using G = Cfg<BN, GLU>;
  if constexpr (GLU) {
    constexpr int H = G::NJ / 2;
    return j < H ? wn * (G::WN / 2) + 8 * j : BN / 2 + wn * (G::WN / 2) + 8 * (j - H);
  } else {
    return col_base<BN, GLU>(wn) + 8 * j;
  }
}

template <int BN, bool GLU>
struct Frag {
  uint32_t a[Cfg<BN, GLU>::MI][4];
  uint32_t b[Cfg<BN, GLU>::NJ][2];
};

// ldmatrix of one k16 step (kk = 0 / 1 within the stage) into f, for a warp with NJW n8 tiles (NJ or NJ_LO)
template <int BN, bool GLU, int NJW, int J = 0>
__device__ __forceinline__ void load_b(Frag<BN, GLU>& f, uint32_t sB, int b_c, int wn, int lane) {
  if constexpr (J < NJW) {
    constexpr bool pair = (J + 1 < NJW) && (b_row<BN, GLU>(0, J + 1) == b_row<BN, GLU>(0, J) + 8);
    if constexpr (pair) {
      const int r = b_row<BN, GLU>(wn, J) + (lane & 7) + ((lane >> 4) << 3);
      ldsm_x4(sB + sw64(r, b_c), f.b[J][0], f.b[J][1], f.b[J + 1][0], f.b[J + 1][1]);
      load_b<BN, GLU, NJW, J + 2>(f, sB, b_c, wn, lane);
    } else {
      const int r = b_row<BN, GLU>(wn, J) + (lane & 7);
      ldsm_x2(sB + sw64(r, b_c), f.b[J][0], f.b[J][1]);
      load_b<BN, GLU, NJW, J + 1>(f, sB, b_c, wn, lane);
    }
  }
}

template <int BN, bool GLU, int NJW>
__device__ __forceinline__ void load_frag(Frag<BN, GLU>& f, uint32_t sA, uint32_t sB, int kk, int wm, int wn, int lane) {
  using G = Cfg<BN, GLU>;
  // A x4: rows 0-15 x (k 0-7 | k 8-15); B x4: (n 0-7 | n 8-15) x (k 0-7 | k 8-15); B x2: n 0-7 x (k 0-7 | k 8-15)
  const int a_r = wm * G::WM + (lane & 15), a_c = kk * 2 + (lane >> 4);
#pragma unroll
  for (int i = 0; i < G::MI; ++i) ldsm_x4(sA + sw64(a_r + i * 16, a_c), f.a[i][0], f.a[i][1], f.a[i][2], f.a[i][3]);
  load_b<BN, GLU, NJW>(f, sB, kk * 2 + ((lane >> 3) & 1), wn, lane);
}

template <int BN, bool GLU, int NJW>
__device__ __forceinline__ void mma_frag(float (&acc)[Cfg<BN, GLU>::MI][Cfg<BN, GLU>::NJ][4], const Frag<BN, GLU>& f) {
#pragma unroll
  for (int i = 0; i < Cfg<BN, GLU>::MI; ++i)
#pragma unroll
    for (int j = 0; j < NJW; ++j) mma16816(acc[i][j], f.a[i], f.b[j][0], f.b[j][1]);
}

// the MMA warps' persistent tile loop for a warp with NJW n8 tiles
template <int BN, bool GLU, int NJW>
__device__ __forceinline__ void mma_loop(__nv_bfloat16* __restrict__ C, int M, int N, int ldc, uint32_t sbase,
                                         uint32_t full0, uint32_t empty0, int T, int tm, int KT, int wm, int wn,
                                         int lane) {
  using G = Cfg<BN, GLU>;
  constexpr int STAGES = G::STAGES, NJ = G::NJ, MI = G::MI;
  constexpr int NOUT = GLU ? BN / 2 : BN;
  const int Nout = GLU ? N / 2 : N;
  int s = 0, g = 0;
  uint32_t ph = 0;
  Frag<BN, GLU> f0, f1;
#pragma unroll 1
  for (int t = blockIdx.x; t < T; t += gridDim.x) {
    const int m0 = (t % tm) * BM, n0 = (t / tm) * NOUT;
    float acc[MI][NJ][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
      for (int j = 0; j < NJ; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
#pragma unroll 1
    for (int kt = 0; kt < KT; ++kt, ++g) {
      mbar_wait(full0 + 8 * s, ph);
      const uint32_t sA = sbase + s * G::STAGE, sB = sA + G::A_BYTES;
      // both k16 steps' fragments of the stage, then their MMAs in increasing k (step 0, then step 1)
      load_frag<BN, GLU, NJW>(f0, sA, sB, 0, wm, wn, lane);
      load_frag<BN, GLU, NJW>(f1, sA, sB, 1, wm, wn, lane);
      mma_frag<BN, GLU, NJW>(acc, f0);
      mma_frag<BN, GLU, NJW>(acc, f1);
      warp_release(empty0 + 8 * s, lane);
      if (++s == STAGES) {
        s = 0;
        ph ^= 1;
      }
    }
    // ---- epilogue: one cvt.rn per GEMM element; bf16x2 stores of in-range elements ----
#pragma unroll
    for (int i = 0; i < MI; ++i) {
      const int r = m0 + wm * G::WM + i * 16 + (lane >> 2);
      if constexpr (GLU) {
#pragma unroll
        for (int j = 0; j < NJW / 2; ++j) {
          const int col = n0 + wn * (G::WN / 2) + j * 8 + (lane & 3) * 2;
          if (col < Nout) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              if (r + 8 * h < M) {
                const __nv_bfloat162 gg = __floats2bfloat162_rn(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
                const __nv_bfloat162 u =
                    __floats2bfloat162_rn(acc[i][j + NJW / 2][2 * h], acc[i][j + NJW / 2][2 * h + 1]);
                const float2 gf = __bfloat1622float2(gg), uf = __bfloat1622float2(u);
                *reinterpret_cast<__nv_bfloat162*>(C + size_t(r + 8 * h) * ldc + col) =
                    __floats2bfloat162_rn(k1_gelu_tanh_mul(gf.x, uf.x), k1_gelu_tanh_mul(gf.y, uf.y));
              }
            }
          }
        }
      } else {
#pragma unroll
        for (int j = 0; j < NJW; ++j) {
          const int col = n0 + col_base<BN, GLU>(wn) + j * 8 + (lane & 3) * 2;
          if (col < N) {
            if (r < M)
              *reinterpret_cast<__nv_bfloat162*>(C + size_t(r) * ldc + col) =
                  __floats2bfloat162_rn(acc[i][j][0], acc[i][j][1]);
            if (r + 8 < M)
              *reinterpret_cast<__nv_bfloat162*>(C + size_t(r + 8) * ldc + col) =
                  __floats2bfloat162_rn(acc[i][j][2], acc[i][j][3]);
          }
        }
      }
    }
  }
}

// GLU = false: C[M, N] = A W^T, N = rows of W, BN = output columns per tile.
// GLU = true (Blink's dense gate_up + GELU(tanh)*up): W = [gate rows; up rows] (N rows, Nh = N / 2 each); a tile loads
// BN/2 gate rows and the BN/2 matching up rows, computes both GEMM tiles with the unchanged K chain, rounds each element
// to bf16 (the value the bi_mm op stores) and writes C[M, Nh] = bf16(gelu_tanh(gate) * up) with Inductor's arithmetic.
template <int BN, bool GLU>
__global__ void __launch_bounds__(NTH, 1)
    k1_gemm_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmW,
                   __nv_bfloat16* __restrict__ C, int M, int N, int K, int ldc) {
  using G = Cfg<BN, GLU>;
  constexpr int STAGES = G::STAGES;
  constexpr int NOUT = GLU ? BN / 2 : BN;  // output columns per tile
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t sbase = smem_u32(smem);
  const uint32_t full0 = sbase + STAGES * G::STAGE;
  const uint32_t empty0 = full0 + STAGES * 8;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int Nout = GLU ? N / 2 : N;
  const int tm = (M + BM - 1) / BM, tn = (Nout + NOUT - 1) / NOUT, T = tm * tn, KT = K / BK;
  if (tid == 0) {
#pragma unroll 1
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(full0 + 8 * i, 1);
      mbar_init(empty0 + 8 * i, NCW);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == NCW) {  // ---- TMA producer (one thread) ----
    if (lane == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tmA)) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tmW)) : "memory");
      int s = 0;
      uint32_t ph = 0;
#pragma unroll 1
      for (int t = blockIdx.x; t < T; t += gridDim.x) {
        const int m0 = (t % tm) * BM, n0 = (t / tm) * NOUT;
#pragma unroll 1
        for (int kt = 0; kt < KT; ++kt) {
          mbar_wait(empty0 + 8 * s, ph ^ 1);
          const uint32_t st = sbase + s * G::STAGE, bar = full0 + 8 * s;
          mbar_expect_tx(bar, G::STAGE);
          tma_load_2d(st, &tmA, bar, kt * BK, m0);
          if constexpr (GLU) {
            tma_load_2d(st + G::A_BYTES, &tmW, bar, kt * BK, n0);
            tma_load_2d(st + G::A_BYTES + G::B_BYTES / 2, &tmW, bar, kt * BK, Nout + n0);
          } else {
            tma_load_2d(st + G::A_BYTES, &tmW, bar, kt * BK, n0);
          }
          if (++s == STAGES) {
            s = 0;
            ph ^= 1;
          }
        }
      }
    }
    return;
  }

  const int wm = warp % G::WGM, wn = warp / G::WGM;
  if (G::NJ_LO == G::NJ || wn < 2)
    mma_loop<BN, GLU, G::NJ>(C, M, N, ldc, sbase, full0, empty0, T, tm, KT, wm, wn, lane);
  else
    mma_loop<BN, GLU, G::NJ_LO>(C, M, N, ldc, sbase, full0, empty0, T, tm, KT, wm, wn, lane);
}

// elementwise reference of the fused epilogue on a stored gate_up [M, 2*Nh] (bf16) -> [M, Nh]; used by the bit tests
// against the Inductor kernel (and usable as a stand-alone exact GELU*up)
__global__ void k1_gelu_mul_kernel(const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ out, int M, int Nh,
                                   int ldi, int ldo, int f32out) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= int64_t(M) * Nh) return;
  const int r = int(i / Nh), c = int(i % Nh);
  const float g = __bfloat162float(gu[int64_t(r) * ldi + c]), u = __bfloat162float(gu[int64_t(r) * ldi + Nh + c]);
  const float v = k1_gelu_tanh_mul(g, u);
  if (f32out)
    reinterpret_cast<float*>(out)[int64_t(r) * ldo + c] = v;
  else
    out[int64_t(r) * ldo + c] = __float2bfloat16_rn(v);
}

#ifndef K1_NO_TORCH
// ---------------------------------------------------------------- host side ----------------------------------------
using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static EncodeFn encode_fn() {
  static EncodeFn fn = nullptr;
  static std::once_flag once;
  std::call_once(once, [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
    TORCH_CHECK(e == cudaSuccess && q == cudaDriverEntryPointSuccess && p, "k1gemm: cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<EncodeFn>(p);
  });
  return fn;
}

// rows x K bf16 matrix with row stride ld (elements); box = BK x box_rows, 64B swizzle, zero OOB fill
static CUtensorMap make_tm(const void* ptr, int64_t rows, int64_t K, int64_t ld, int box_rows) {
  CUtensorMap tm;
  cuuint64_t dims[2] = {cuuint64_t(K), cuuint64_t(rows)};
  cuuint64_t strides[1] = {cuuint64_t(ld) * 2};
  cuuint32_t box[2] = {cuuint32_t(BK), cuuint32_t(box_rows)};
  cuuint32_t es[2] = {1, 1};
  CUresult r = encode_fn()(&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, es,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "k1gemm: cuTensorMapEncodeTiled failed (", int(r), ")");
  return tm;
}

static int num_sms() {
  static int n = 0;
  if (!n) {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  }
  return n;
}

template <int BN, bool GLU>
static void launch(const torch::Tensor& a, const torch::Tensor& w, torch::Tensor& c, int64_t maxcta) {
  using G = Cfg<BN, GLU>;
  static std::once_flag once;
  std::call_once(once, [] {
    cudaFuncSetAttribute(k1_gemm_kernel<BN, GLU>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
  });
  const int M = int(a.size(0)), K = int(a.size(1)), N = int(w.size(0));
  constexpr int NOUT = GLU ? BN / 2 : BN;
  const int Nout = GLU ? N / 2 : N;
  const int T = ((M + BM - 1) / BM) * ((Nout + NOUT - 1) / NOUT);
  int grid = num_sms();
  if (maxcta > 0) grid = int(maxcta);
  if (grid > T) grid = T;
  CUtensorMap tA = make_tm(a.data_ptr(), M, K, a.stride(0), BM);
  CUtensorMap tW = make_tm(w.data_ptr(), N, K, w.stride(0), GLU ? BN / 2 : BN);
  k1_gemm_kernel<BN, GLU><<<grid, NTH, G::SMEM, c10::cuda::getCurrentCUDAStream().stream()>>>(
      tA, tW, reinterpret_cast<__nv_bfloat16*>(c.data_ptr()), M, N, K, int(c.stride(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void dispatch(const torch::Tensor& a, const torch::Tensor& w, torch::Tensor& c, int64_t bn, int64_t maxcta) {
  switch (bn) {
#define K1_BN(x)                       \
  case x:                              \
    launch<x, false>(a, w, c, maxcta); \
    break;
    K1_BN(64) K1_BN(80) K1_BN(96) K1_BN(112) K1_BN(128) K1_BN(144) K1_BN(160) K1_BN(176) K1_BN(192) K1_BN(208)
    K1_BN(224) K1_BN(240) K1_BN(256)
#undef K1_BN
    default: TORCH_CHECK(false, "k1gemm: unsupported BN ", bn);
  }
}

// GLU: bn = gate (= output) columns per tile: 64 / 96 / 128 (the tile computes 2*bn GEMM columns)
static void dispatch_glu(const torch::Tensor& a, const torch::Tensor& w, torch::Tensor& c, int64_t bn, int64_t maxcta) {
  switch (bn) {
    case 64: launch<128, true>(a, w, c, maxcta); break;
    case 96: launch<192, true>(a, w, c, maxcta); break;
    case 128: launch<256, true>(a, w, c, maxcta); break;
    default: TORCH_CHECK(false, "k1gemm: unsupported GLU tile ", bn);
  }
}

static void check(const torch::Tensor& a, const torch::Tensor& w) {
  TORCH_CHECK(a.is_cuda() && w.is_cuda() && a.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16,
              "k1gemm: bf16 CUDA tensors expected");
  TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && a.size(1) == w.size(1) && a.stride(1) == 1 && w.stride(1) == 1,
              "k1gemm: a [M, K], w [N, K] with unit inner stride expected");
  TORCH_CHECK(a.size(1) % BK == 0, "k1gemm: K must be a multiple of ", BK);
  TORCH_CHECK(a.stride(0) % 8 == 0 && w.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 && w.size(0) % 8 == 0,
              "k1gemm: 16-byte aligned operands with row strides a multiple of 8 and N % 8 == 0 expected");
}

#endif  // K1_NO_TORCH
}  // namespace k1

#ifndef K1_NO_TORCH
torch::Tensor mm(torch::Tensor a, torch::Tensor w, int64_t bn, int64_t maxcta) {
  k1::check(a, w);
  auto c = torch::empty({a.size(0), w.size(0)}, a.options());
  if (a.size(0) > 0) k1::dispatch(a, w, c, bn, maxcta);
  return c;
}

void mm_out(torch::Tensor a, torch::Tensor w, torch::Tensor c, int64_t bn, int64_t maxcta) {
  k1::check(a, w);
  TORCH_CHECK(c.is_cuda() && c.scalar_type() == at::kBFloat16 && c.dim() == 2 && c.size(0) == a.size(0) &&
                  c.size(1) == w.size(0) && c.stride(1) == 1 && c.stride(0) % 2 == 0 &&
                  reinterpret_cast<uintptr_t>(c.data_ptr()) % 4 == 0,
              "k1gemm.mm_out: bf16 output [M, N] with unit inner stride, 4-byte aligned rows expected");
  if (a.size(0) > 0) k1::dispatch(a, w, c, bn, maxcta);
}

// x [M, K] @ [gate; up]^T (w [2*Nh, K]) -> bf16(gelu_tanh(bf16 gate) * bf16 up) [M, Nh] with Inductor's exact epilogue
torch::Tensor glu(torch::Tensor a, torch::Tensor w, int64_t bn, int64_t maxcta) {
  k1::check(a, w);
  TORCH_CHECK(w.size(0) % 16 == 0, "k1gemm.glu: gate_up rows must be a multiple of 16");
  auto c = torch::empty({a.size(0), w.size(0) / 2}, a.options());
  if (a.size(0) > 0) k1::dispatch_glu(a, w, c, bn, maxcta);
  return c;
}

// the fused epilogue alone on a stored gate_up [M, 2*Nh] -> [M, Nh] (bf16, or the fp32 value before rounding)
torch::Tensor gelu_mul(torch::Tensor gu, bool f32out) {
  TORCH_CHECK(gu.is_cuda() && gu.scalar_type() == at::kBFloat16 && gu.dim() == 2 && gu.stride(1) == 1 &&
                  gu.size(1) % 2 == 0,
              "k1gemm.gelu_mul: bf16 [M, 2*Nh] expected");
  const int M = int(gu.size(0)), Nh = int(gu.size(1) / 2);
  auto out = torch::empty({M, Nh}, gu.options().dtype(f32out ? at::kFloat : at::kBFloat16));
  const int64_t n = int64_t(M) * Nh;
  if (n > 0)
    k1::k1_gelu_mul_kernel<<<int((n + 255) / 256), 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(gu.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, Nh,
        int(gu.stride(0)), Nh, f32out ? 1 : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mm", &mm, "exact bf16 GEMM a @ w.T (K1: TMA + mma.sync m16n8k16, BM 128, BN in 64..256 step 16)");
  m.def("mm_out", &mm_out, "the same GEMM into a preallocated (possibly column-sliced) output");
  m.def("glu", &glu, "gate_up GEMM with the exact GELU(tanh)*up epilogue (Blink dense MLP)");
  m.def("gelu_mul", &gelu_mul, "the exact GELU(tanh)*up epilogue alone (reference / tests)");
}
#endif
