// SPDX-License-Identifier: Apache-2.0
// bmoe69.cu - E69 (pjs's engine team, 28 Oct 2026): probe + prototype build of team ruj's bmoe NVFP4 MoE grouped GEMM
// (bj/moe track, Apache-2.0; the kernel below is ruj's bmoe_kernel.cuh / bmoe_host.cuh as shipped in
// /opt/blink/blink_moegemm.cu (sha256 f9080d0c...), with credit). Unchanged from ruj's code: the unit table, the dynamic
// unit scheduler, the A operand path (TMA from the expert-sorted rows, 64B swizzle, 1D bulk scale atoms), the
// consumer's MMA chain (mma.sync m16n8k64 kind::mxf4nvf4 scale_vec::4X ue4m3, fp32 accumulation over 64-wide K
// chunks in increasing order, all-zero chunks past K exactly as CUTLASS's zero-filled 256-wide tail k-tile) and both
// epilogues (MODE 0: bf16_rn(fma(0, 0, alpha * acc)); MODE 1: vLLM's gelu_tanh_and_mul + scaled_fp4_experts_quant).
// E69 adds two compile-time knobs and a weight re-pack kernel:
//   PACK = 1  the weights B and their block scales SFB are read from a pre-packed copy (pack69 below): per
//             (expert, n-tile) unit tile, one record per real 64-wide K chunk: B = BN rows x 32 bytes with the two
//             16-byte halves of each row swapped on rows with bit 2 set (conflict-free ldmatrix), SFB = one 512-byte
//             128x4 scale atom whose row n is smem B row n (MODE 1: gate rows 0-63, up rows 64-127). One cp.async.bulk
//             per stage moves a contiguous nreal x 4 KB of weights and one moves nreal x 512 B of scales, instead of
//             2D TMA boxes of 64-byte row slices from rows K/2 bytes apart. The ldmatrix / lds addresses change, the
//             bytes each MMA receives do not: exact by construction (checked byte for byte by kbit69.py).
//   PROBE = 1 producer only: consumers wait for each stage and release it without MMAs or epilogue (stream rate).
//   PROBE = 2 consumer only: after the first STAGES stages the producer releases stages without loading (MMA +
//             epilogue rate on stale data). PROBE = 3 (MODE 0): full kernel without the output stores.
//             PROBE 4 / 5 / 6 = consumer only (as 2) and: 4 no ldmatrix / lds after a unit's first chunk (MMA issue
//             rate), 5 no epilogue, 6 no scale-factor lds after a unit's first chunk; 7 = consumer only with a
//             checksum epilogue (keeps every MMA alive: ptxas deletes MMAs whose results are dead, which voids 5).
//             PROBE != 0 builds are for timing only; their outputs are meaningless.
//   OPOL = 1  (MODE 0) the bf16 output stores carry an L2 evict_last hint (ruj's st_b32_hint / st_v4_hint, unshipped in
//             bmoe): the c3 rows stay in L2 for the combine that reads them next. Values are unchanged.
//   ZS = 1    skip the MMAs of the all-zero chunks past K (w2: CUTLASS's zero-filled 12th chunk). Exact: on sm_120 that
//             chunk only maps an accumulator of -0 to +0 (measured: every finite, subnormal and infinite value passes
//             unchanged; only NaN payloads are canonicalised) and the epilogue fma(0, 0, alpha * acc) + 0 maps -0 to +0
//             anyway; NaN accumulators cannot occur (finite e2m1 x finite e4m3 products).
// discard69: discard.global.L2 over the whole 128-byte lines of a dead buffer (its dirty lines are dropped, not
// written back to DRAM). Only for buffers whose last reader has completed on the same stream.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace bmoe69 {

constexpr int kMaxE = 128;
constexpr int kNumABoxes = 8;   // A box heights 16, 32, ..., 128 rows

struct __align__(64) Params {
  CUtensorMap tmA[kNumABoxes];  // A  [rows, K/2] bytes, box (BK/2) x (16*(i+1)) rows
  CUtensorMap tmB;              // B  [E*N, K/2] bytes, box (BK/2) x BN rows (MODE 1: BN/2 rows); unused with PACK
  const uint8_t* a_sf;          // A scales, 128x4 atoms; expert e starts at row-block sf_offsets[e]/128
  const uint8_t* b_sf;          // B scales [E][N/128][K/64] atoms of 512 B; PACK: packed scale records
  const uint8_t* b_pk;          // PACK: packed weight records
  __nv_bfloat16* out;           // MODE 0: out rows of expert e start at expert_offsets[e]
  const float* alphas;          // [E]
  const int* problem_sizes;     // [E][3] (m, n, k)
  const int* expert_offsets;    // [E]
  const int* sf_offsets;        // [E]
  uint8_t* q_out;               // MODE 1: FP4 activations [rows, I/2]
  uint8_t* q_sf;                // MODE 1: their scales, swizzled, expert e at row sf_offsets[e]
  const float* q_gscale;        // MODE 1: [E] global scale of the quantisation (a2_gscale)
  int E, N, K, ldo;
  int n_chunks;                 // ceil(K/256)*4: the chunk count of CUTLASS's 256-wide k-tiles
  int n_real;                   // K/64: chunks that carry data
  int I;                        // MODE 1: N/2
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
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred P1;\n\t"
      "BMOE69_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra BMOE69_DONE;\n\t"
      "bra BMOE69_WAIT;\n\t"
      "BMOE69_DONE:\n\t}" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* tm, uint32_t bar, int c0, int c1,
                                            uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(tm)), "r"(bar), "r"(c0), "r"(c1), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar,
                                          uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(
          dst),
      "l"(src), "r"(bytes), "r"(bar), "l"(policy)
      : "memory");
}
template <int P>
__device__ __forceinline__ uint64_t make_policy() {
  uint64_t p;
  if constexpr (P == 0) asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p));
  else if constexpr (P == 1) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  else asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ uint32_t lds32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(addr));
  return v;
}
// stores with an L2 policy (OPOL: keep the GEMM output in L2 for the kernel that reads it next) - ruj's helper
__device__ __forceinline__ void st_b32_hint(void* ptr, uint32_t v, uint64_t pol) {
  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;" ::"l"(ptr), "r"(v), "l"(pol) : "memory");
}
__device__ __forceinline__ void st_v4_hint(void* ptr, uint4 v, uint64_t pol) {
  asm volatile("st.global.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;" ::"l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w), "l"(pol)
               : "memory");
}
// D = A x B + D, NVFP4 (e2m1 data, ue4m3 scales per 16 K), fp32 accumulation; the instruction CUTLASS's
// SM120_16x8x64_TN_VS<e2m1, e2m1, float, ue4m3, 16> issues (byte-id 0, thread-id 0 for both scale operands).
__device__ __forceinline__ void mma_nvf4(float* d, const uint32_t* a, const uint32_t* b, uint32_t sfa,
                                         uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "h"((uint16_t)0),
        "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

// smem byte offset of 16-byte chunk c of row r in a TMA tile with rows of RB bytes (64B or 128B swizzle)
template <int RB>
__device__ __forceinline__ uint32_t swz(int r, int c) {
  if constexpr (RB == 64) {
    return (uint32_t)(r * 64 + ((c ^ ((r >> 1) & 3)) << 4));
  } else {
    return (uint32_t)(r * 128 + ((c ^ (r & 7)) << 4));
  }
}
// PACK: smem byte offset of 16-byte half h of row n inside one packed 64-wide K chunk (rows of 32 bytes)
__device__ __forceinline__ uint32_t pk_off(int n, int h) { return (uint32_t)(n * 32 + ((h ^ ((n >> 2) & 1)) << 4)); }
// byte offset of (row r, 4-k-block word) inside a 128x4 scale atom of 512 B
__device__ __forceinline__ uint32_t atom_off(int r) { return (uint32_t)((r & 31) * 16 + ((r >> 5) & 3) * 4); }

// ---- the epilogue arithmetic of the reference path (CUTLASS GEMM epilogue, vLLM activation and quant) ----
// CUTLASS LinearCombination, beta = 0 and C cleared: fma(beta, C, alpha * acc), then cvt.rn to bf16
__device__ __forceinline__ float epi(float alpha, float acc) { return __fadd_rn(__fmul_rn(alpha, acc), 0.0f); }
// vLLM packed_gelu_tanh_kernel (float math, bf16 result)
__device__ __forceinline__ float gelu_tanh_f(float x) {
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = x * x * x;
  float inner = BETA * (x + KAPPA * x_cube);
  return 0.5f * x * (1.0f + ::tanhf(inner));
}
// vLLM packed_compute<act_first, no clamp>: bf16(float(bf16(gelu_tanh(gate))) * (float(up) + beta)), beta = 0
__device__ __forceinline__ __nv_bfloat162 gelu_tanh_mul2(__nv_bfloat162 g, __nv_bfloat162 u, float beta) {
  float2 gf = __bfloat1622float2(g);
  __nv_bfloat162 act = __floats2bfloat162_rn(gelu_tanh_f(gf.x), gelu_tanh_f(gf.y));
  float2 a = __bfloat1622float2(act);
  float2 uf = __bfloat1622float2(u);
  a.x = __fmul_rn(a.x, __fadd_rn(uf.x, beta));
  a.y = __fmul_rn(a.y, __fadd_rn(uf.y, beta));
  return __floats2bfloat162_rn(a.x, a.y);
}
__device__ __forceinline__ float rcp_approx_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}
__device__ __forceinline__ uint32_t e2m1x2(float lo, float hi) {
  uint32_t v;
  asm volatile(
      "{\n\t.reg .b8 b;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\n\t"
      "cvt.u32.u8 %0, b;\n\t}"
      : "=r"(v)
      : "f"(lo), "f"(hi));
  return v;
}

template <int BM_, int BN_, int BK_, int STAGES_, int WM_, int WN_, int MODE_ = 0, int POLB_ = 1, int POLA_ = 2,
          int EPI_ = 0, int SCHED_ = 0, int PACK_ = 0, int PROBE_ = 0, int OPOL_ = 0, int ZS_ = 0>
struct Cfg {
  static constexpr int BM = BM_, BN = BN_, BK = BK_, STAGES = STAGES_, WM = WM_, WN = WN_, MODE = MODE_;
  static constexpr int POLB = POLB_, POLA = POLA_;
  static constexpr int EPI = EPI_;               // MODE 0 stores: 0 = 4-byte (bf16x2) stores, 1 = 16-byte stores
  static constexpr int SCHED = SCHED_;           // 0: static round-robin units; 1: dynamic (global atomic counter)
  static constexpr int PACK = PACK_;             // 1: B / SFB from the packed records (E69)
  static constexpr int PROBE = PROBE_;           // 0 full, 1 producer only, 2 consumer only, 3 no stores (E69 probes)
  static constexpr int OPOL = OPOL_;             // 1: MODE 0 output stores with L2 evict_last
  static constexpr int ZS = ZS_;                 // 1: skip the MMAs of all-zero chunks past K (exact, see header)
  static constexpr int NCW = WM * WN;            // consumer warps
  static constexpr int THREADS = (NCW + 1) * 32; // + 1 producer warp
  static constexpr int RB = BK / 2;              // bytes per tile row (64 or 128)
  static constexpr int CPS = BK / 64;            // 64-wide K chunks per stage
  static constexpr int WTM = BM / WM, WTN = BN / WN;
  static constexpr int MT = WTM / 16, NT = WTN / 8;
  static constexpr int A_BYTES = BM * RB, B_BYTES = BN * RB;
  static constexpr int SFA_BYTES = CPS * 512;             // one 128-row atom per chunk
  static constexpr int SFB_BLOCKS = PACK ? 1 : (MODE == 1 ? 2 : BN / 128);
  static constexpr int SFB_BYTES = SFB_BLOCKS * CPS * 512;
  static constexpr int STAGE_BYTES = A_BYTES + B_BYTES + SFA_BYTES + SFB_BYTES;
  static constexpr int TABLE_BYTES = (4 * kMaxE + 1) * 4;
  static constexpr int BAR_BYTES = 2 * STAGES * 8;
  static constexpr int SMEM = 1024 /*align slack*/ + STAGES * STAGE_BYTES + BAR_BYTES + TABLE_BYTES;
  static constexpr int UNIT_N = MODE == 1 ? BN / 2 : BN; // output columns (MODE 1: intermediate columns) per unit
  static constexpr int PK_CHUNK = BN * 32;               // PACK: weight bytes of one 64-wide K chunk of a unit tile
  static_assert(BN % 128 == 0, "BN must be a multiple of 128 (scale atoms)");
  static_assert(MODE == 0 || (BN == 128 && (WN == 2 || WN == 4)),
                "MODE 1 needs a 64 gate + 64 up tile over two or four warp columns");
  static_assert(BM == 64 || BM == 128, "BM");
  static_assert(BK == 128 || BK == 256, "BK");
  static_assert(!PACK || BN == 128, "PACK: one 128-row scale atom per chunk");
  static_assert(NT % 4 == 0 || MODE == 0, "MODE 1 tiles");
  static_assert(NT % 2 == 0, "NT must be even");
  static_assert(STAGE_BYTES % 1024 == 0, "stage alignment");
  static_assert(EPI == 0 || NT % 4 == 0, "16-byte stores need 4 n8 tiles per quad transpose");
  static_assert(PROBE != 3 || MODE == 0, "PROBE 3 is MODE 0 only");
  static_assert(PROBE <= 7, "PROBE 0..7");
  static_assert(OPOL == 0 || MODE == 0, "OPOL: MODE 0");
};

// 4x4 transpose of 32-bit words inside a quad (lanes 4g..4g+3): in: a[k] = this lane's 2 columns of n8 tile k;
// out: b[k] = lane k's 2 columns of n8 tile tg (so this lane holds the 8 columns of tile tg)
__device__ __forceinline__ void quad_transpose(const uint32_t (&a)[4], uint32_t (&b)[4], int lane) {
  const int tg = lane & 3;
  uint32_t res[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) res[k] = (k == tg) ? a[k] : 0u;
#pragma unroll
  for (int rr = 1; rr < 4; ++rr) {
    const int ks = (tg + rr) & 3;
    uint32_t send = ks == 0 ? a[0] : ks == 1 ? a[1] : ks == 2 ? a[2] : a[3];
    const int src = (tg - rr) & 3;
    uint32_t recv = __shfl_sync(0xffffffffu, send, (lane & ~3) | src);
#pragma unroll
    for (int k = 0; k < 4; ++k) res[k] = (k == src) ? recv : res[k];
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) b[k] = res[k];
}

struct Unit {
  int e, m0, n0, rows;
};

// dynamic unit counter (SCHED = 1): [0] next unit, [1] CTAs done; the last CTA of a launch resets both, so every
// launch (CUDA-graph replays included) starts from 0. Kernels using it must not run concurrently with each other.
__device__ int g_bmoe69_sched[2];

template <class C>
__device__ __forceinline__ Unit decode(int u, const int* s_prefix, const int* s_m, int E) {
  int lo = 0, hi = E;
  while (hi - lo > 1) {
    int mid = (lo + hi) >> 1;
    if (s_prefix[mid] <= u) lo = mid; else hi = mid;
  }
  int e = lo, local = u - s_prefix[e], M = s_m[e];
  int mtn = (M + C::BM - 1) / C::BM;
  int mt = local % mtn, nt = local / mtn;
  Unit r;
  r.e = e;
  r.m0 = mt * C::BM;
  r.n0 = nt * C::UNIT_N;
  r.rows = min(C::BM, M - r.m0);
  return r;
}

// first smem B row of n8 tile j of warp column wn
template <class C>
__device__ __forceinline__ int b_row_of_tile(int wn, int j) {
  if constexpr (C::MODE == 1) {
    return (j < C::NT / 2 ? 0 : C::BN / 2) + wn * (C::WTN / 2) + 8 * (j % (C::NT / 2));
  } else {
    return wn * C::WTN + 8 * j;
  }
}

template <class C>
__global__ void __launch_bounds__(C::THREADS, 1) bmoe_kernel(const __grid_constant__ Params p) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* stages = smem;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + C::STAGES * C::STAGE_BYTES);
  int* s_prefix = reinterpret_cast<int*>(smem + C::STAGES * C::STAGE_BYTES + C::BAR_BYTES);
  int* s_m = s_prefix + (kMaxE + 1);
  int* s_off = s_m + kMaxE;
  int* s_sfo = s_off + kMaxE;
  __shared__ int warp_tot[16];
  __shared__ int s_unit[2];
  __shared__ __align__(8) uint64_t sched_bar[4];   // [0,1] unit published (1 arrival), [2,3] unit read (NCW)

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int E = p.E;
  const int ntn = (C::MODE == 1 ? p.I : p.N) / C::UNIT_N;

  // ---- unit table: units of expert e = ceil(m_e / BM) * ntn; exclusive prefix in s_prefix ----
  {
    int units = 0;
    if (tid < E) {
      int m = __ldg(p.problem_sizes + 3 * tid);
      s_m[tid] = m;
      s_off[tid] = __ldg(p.expert_offsets + tid);
      s_sfo[tid] = __ldg(p.sf_offsets + tid);
      units = ((m + C::BM - 1) / C::BM) * ntn;
    }
    int v = units;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      int t = __shfl_up_sync(0xffffffffu, v, o);
      if (lane >= o) v += t;
    }
    if (lane == 31) warp_tot[warp] = v;
    if (tid == 0) {
      for (int s = 0; s < C::STAGES; ++s) {
        mbar_init(smem_u32(&bars[s]), 1);
        mbar_init(smem_u32(&bars[C::STAGES + s]), C::NCW);
      }
      for (int s = 0; s < 2; ++s) {
        mbar_init(smem_u32(&sched_bar[s]), 1);
        mbar_init(smem_u32(&sched_bar[2 + s]), C::NCW);
      }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    int base = 0;
    for (int w = 0; w < warp; ++w) base += warp_tot[w];
    if (tid < E) s_prefix[tid] = base + v - units;
    if (tid == 0) {
      int tot = 0;
      for (int w = 0; w < (E + 31) / 32; ++w) tot += warp_tot[w];
      s_prefix[E] = tot;
    }
    __syncthreads();
  }
  const int total = s_prefix[E];
  const int steps = p.n_chunks / C::CPS;

  if (warp == C::NCW) {
    // ================= producer warp =================
    if (lane == 0) {
      const uint64_t pol_b = make_policy<C::POLB>();
      const uint64_t pol_a = make_policy<C::POLA>();
      const int K4 = p.K / 64;               // scale atoms per 128-row block
      const size_t b_sf_e = (size_t)p.N * (p.K / 16);
      int stage = 0;
      uint32_t phase = 0;
      int u, slot = 0;
      uint32_t sph = 0;
      int issued = 0;                        // PROBE 2: stages loaded for real so far
      if constexpr (C::SCHED == 1) u = atomicAdd(&g_bmoe69_sched[0], 1); else u = blockIdx.x;
      while (true) {
        if constexpr (C::SCHED == 1) {
          mbar_wait(smem_u32(&sched_bar[2 + slot]), sph ^ 1);
          s_unit[slot] = u;
          mbar_arrive(smem_u32(&sched_bar[slot]));
          if (++slot == 2) { slot = 0; sph ^= 1; }
        }
        if (u >= total) break;
        int u_next;   // fetched now, used after this unit's loads (hides the atomic's latency)
        if constexpr (C::SCHED == 1) u_next = atomicAdd(&g_bmoe69_sched[0], 1); else u_next = u + gridDim.x;
        Unit t = decode<C>(u, s_prefix, s_m, E);
        const int hbox = (t.rows + 15) >> 4;  // 1..8
        const int arow = s_off[t.e] + t.m0;
        const int arb = (s_sfo[t.e] + t.m0) >> 7;  // A scale row-block
        const uint8_t* sfa_base = p.a_sf + (size_t)arb * K4 * 512;
        int brow0 = 0, brow1 = 0;              // MODE 1: gate rows, up rows (TMA path)
        const uint8_t *sfb0 = nullptr, *sfb1 = nullptr;
        const uint8_t *pkb = nullptr, *pks = nullptr;   // PACK: this unit tile's records
        if constexpr (C::PACK) {
          const size_t rec = ((size_t)t.e * ntn + t.n0 / C::UNIT_N) * (size_t)p.n_real;
          pkb = p.b_pk + rec * C::PK_CHUNK;
          pks = p.b_sf + rec * 512;
        } else if constexpr (C::MODE == 1) {
          brow0 = t.e * p.N + t.n0;
          brow1 = t.e * p.N + p.I + t.n0;
          sfb0 = p.b_sf + (size_t)t.e * b_sf_e + (size_t)(t.n0 >> 7) * K4 * 512;
          sfb1 = p.b_sf + (size_t)t.e * b_sf_e + (size_t)((p.I + t.n0) >> 7) * K4 * 512;
        } else {
          brow0 = t.e * p.N + t.n0;
          sfb0 = p.b_sf + (size_t)t.e * b_sf_e + (size_t)(t.n0 >> 7) * K4 * 512;
        }
        for (int s = 0; s < steps; ++s) {
          mbar_wait(smem_u32(&bars[C::STAGES + stage]), phase ^ 1);
          uint8_t* st = stages + stage * C::STAGE_BYTES;
          const uint32_t full = smem_u32(&bars[stage]);
          if ((C::PROBE == 2 || C::PROBE >= 4) && issued >= C::STAGES) {   // consumer-only family
            mbar_arrive(full);                 // consumer-only probe: the stage keeps its earlier contents
          } else {
            ++issued;
            const int c0 = s * C::CPS;
            const int nreal = max(0, min(C::CPS, p.n_real - c0));
            const uint32_t sA = smem_u32(st), sB = smem_u32(st + C::A_BYTES);
            const uint32_t sSFA = smem_u32(st + C::A_BYTES + C::B_BYTES), sSFB = sSFA + C::SFA_BYTES;
            if constexpr (C::PACK) {
              const uint32_t bytes = hbox * 16 * C::RB + nreal * (C::PK_CHUNK + 512 + 512);
              mbar_expect_tx(full, bytes);
              tma_load_2d(sA, &p.tmA[hbox - 1], full, s * C::RB, arow, pol_a);
              if (nreal > 0) {
                bulk_load(sB, pkb + (size_t)c0 * C::PK_CHUNK, nreal * C::PK_CHUNK, full, pol_b);
                bulk_load(sSFA, sfa_base + (size_t)c0 * 512, nreal * 512, full, pol_a);
                bulk_load(sSFB, pks + (size_t)c0 * 512, nreal * 512, full, pol_b);
              }
            } else {
              const uint32_t bytes = hbox * 16 * C::RB + C::B_BYTES + nreal * 512 * (1 + C::SFB_BLOCKS);
              mbar_expect_tx(full, bytes);
              tma_load_2d(sA, &p.tmA[hbox - 1], full, s * C::RB, arow, pol_a);
              if constexpr (C::MODE == 1) {
                tma_load_2d(sB, &p.tmB, full, s * C::RB, brow0, pol_b);
                tma_load_2d(sB + C::B_BYTES / 2, &p.tmB, full, s * C::RB, brow1, pol_b);
              } else {
                tma_load_2d(sB, &p.tmB, full, s * C::RB, brow0, pol_b);
              }
              if (nreal > 0) {
                bulk_load(sSFA, sfa_base + (size_t)c0 * 512, nreal * 512, full, pol_a);
                if constexpr (C::MODE == 1) {
                  bulk_load(sSFB, sfb0 + (size_t)c0 * 512, nreal * 512, full, pol_b);
                  bulk_load(sSFB + C::CPS * 512, sfb1 + (size_t)c0 * 512, nreal * 512, full, pol_b);
                } else {
#pragma unroll
                  for (int nb = 0; nb < C::SFB_BLOCKS; ++nb)
                    bulk_load(sSFB + nb * C::CPS * 512, sfb0 + (size_t)nb * K4 * 512 + (size_t)c0 * 512,
                              nreal * 512, full, pol_b);
                }
              }
            }
          }
          if (++stage == C::STAGES) { stage = 0; phase ^= 1; }
        }
        u = u_next;
      }
      if constexpr (C::SCHED == 1) {
        if (atomicAdd(&g_bmoe69_sched[1], 1) == (int)gridDim.x - 1) {
          atomicExch(&g_bmoe69_sched[0], 0);
          atomicExch(&g_bmoe69_sched[1], 0);
        }
      }
    }
    return;
  }

  // ================= consumer warps =================
  const int wm = warp / C::WN, wn = warp % C::WN;
  const int g = lane >> 2, tg = lane & 3;
  const int lq = lane >> 3, li = lane & 7;
  int stage = 0;
  uint32_t phase = 0;
  int cslot = 0;
  uint32_t csph = 0;
  for (int u = blockIdx.x;; u += gridDim.x) {
    if constexpr (C::SCHED == 1) {
      mbar_wait(smem_u32(&sched_bar[cslot]), csph);
      u = *reinterpret_cast<volatile int*>(&s_unit[cslot]);
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_u32(&sched_bar[2 + cslot]));
      if (++cslot == 2) { cslot = 0; csph ^= 1; }
    }
    if (u >= total) break;
    Unit t = decode<C>(u, s_prefix, s_m, E);
    const int mbase = wm * C::WTM;           // first row of this warp in the unit
    const int nvalid = t.rows - mbase;      // rows of this warp's band that are real
    if constexpr (C::PROBE == 1) {
      for (int s = 0; s < steps; ++s) {
        mbar_wait(smem_u32(&bars[stage]), phase);
        __syncwarp();
        if (lane == 0) mbar_arrive(smem_u32(&bars[C::STAGES + stage]));
        if (++stage == C::STAGES) { stage = 0; phase ^= 1; }
      }
      continue;
    }
    // B scale rows: smem B row n lives in scale block blk(n) at row-in-atom rin(n)
    int sfb_rin0, sfb_rin1;
    if constexpr (C::MODE == 1 && !C::PACK) {
      sfb_rin0 = t.n0 & 127;
      sfb_rin1 = ((p.I + t.n0) & 127) - C::BN / 2;
    } else {
      sfb_rin0 = 0;
      sfb_rin1 = 0;
    }
    float acc[C::MT][C::NT][4];
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
      for (int j = 0; j < C::NT; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.0f;
    const int a_rowblk_off = t.m0 & 127;     // row offset of this unit inside its 128-row scale atom
    uint32_t af[C::MT][4], bfr[C::NT][2], sfa[C::MT], sfb[C::NT];
    bool first = true;                       // PROBE 4 / 6: operands loaded once per unit
    for (int s = 0; s < steps; ++s) {
      mbar_wait(smem_u32(&bars[stage]), phase);
      const uint32_t st = smem_u32(stages + stage * C::STAGE_BYTES);
      const uint32_t sA = st, sB = st + C::A_BYTES, sSFA = st + C::A_BYTES + C::B_BYTES,
                     sSFB = sSFA + C::SFA_BYTES;
      if (nvalid > 0) {
#pragma unroll
        for (int cc = 0; cc < C::CPS; ++cc) {
          const int chunk = s * C::CPS + cc;
          if (C::ZS && chunk >= p.n_real) continue;   // ZS: the all-zero chunk changes nothing that reaches the output
          const bool ld_frag = !(C::PROBE == 4 && !first);
          const bool ld_sf = !((C::PROBE == 4 || C::PROBE == 6) && !first);
          first = false;
          if (chunk < p.n_real) {
#pragma unroll
            for (int i = 0; i < C::MT; ++i) {
              const int r = mbase + i * 16 + (lq & 1) * 8 + li;
              if (ld_frag) ldsm_x4(sA + swz<C::RB>(r, cc * 2 + (lq >> 1)), af[i][0], af[i][1], af[i][2], af[i][3]);
              const int ra = a_rowblk_off + mbase + i * 16 + g + 8 * (lane & 1);
              if (ld_sf) sfa[i] = lds32(sSFA + cc * 512 + (ra & 31) * 16 + ((ra >> 5) & 3) * 4);
            }
#pragma unroll
            for (int jp = 0; jp < C::NT / 2; ++jp) {
              const int n = b_row_of_tile<C>(wn, 2 * jp) + (lq >> 1) * 8 + li;
              uint32_t baddr;
              if constexpr (C::PACK) baddr = sB + cc * C::PK_CHUNK + pk_off(n, lq & 1);
              else baddr = sB + swz<C::RB>(n, cc * 2 + (lq & 1));
              if (ld_frag) ldsm_x4(baddr, bfr[2 * jp][0], bfr[2 * jp][1], bfr[2 * jp + 1][0], bfr[2 * jp + 1][1]);
            }
#pragma unroll
            for (int j = 0; j < C::NT; ++j) {
              const int nb = b_row_of_tile<C>(wn, j) + g;  // smem B row of this thread's column
              if (!ld_sf) {
              } else if constexpr (C::PACK) {
                sfb[j] = lds32(sSFB + cc * 512 + atom_off(nb));
              } else {
                int blk, rin;
                if constexpr (C::MODE == 1) {
                  blk = nb >= C::BN / 2;
                  rin = blk ? sfb_rin1 + nb : sfb_rin0 + nb;
                } else {
                  blk = nb >> 7;
                  rin = nb & 127;
                }
                sfb[j] = lds32(sSFB + blk * C::CPS * 512 + cc * 512 + (rin & 31) * 16 + ((rin >> 5) & 3) * 4);
              }
            }
          } else {
#pragma unroll
            for (int i = 0; i < C::MT; ++i) { af[i][0] = af[i][1] = af[i][2] = af[i][3] = 0u; sfa[i] = 0u; }
#pragma unroll
            for (int j = 0; j < C::NT; ++j) { bfr[j][0] = bfr[j][1] = 0u; sfb[j] = 0u; }
          }
#pragma unroll
          for (int i = 0; i < C::MT; ++i) {
            if (i * 16 < nvalid) {
#pragma unroll
              for (int j = 0; j < C::NT; ++j) mma_nvf4(acc[i][j], af[i], bfr[j], sfa[i], sfb[j]);
            }
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_u32(&bars[C::STAGES + stage]));
      if (++stage == C::STAGES) { stage = 0; phase ^= 1; }
    }
    if (nvalid <= 0) continue;
    if constexpr (C::PROBE == 5) continue;   // consumer-only probe without the epilogue
    if constexpr (C::PROBE == 7) {           // checksum of every accumulator; an impossible value stores one word
      uint32_t x = 0;
#pragma unroll
      for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int j = 0; j < C::NT; ++j)
#pragma unroll
          for (int r = 0; r < 4; ++r) x += __float_as_uint(acc[i][j][r]);
      if (x == 0x7fbadbadu) {
        if constexpr (C::MODE == 0) p.out[0] = __float2bfloat16_rn(1.0f);
        else p.q_out[0] = 1;
      }
      continue;
    }
    const float alpha = __ldg(p.alphas + t.e);
    if constexpr (C::MODE == 0 && C::EPI == 1) {
      // ---- epilogue, 16-byte stores: same values, quad-transposed so each lane writes 8 consecutive columns ----
      __nv_bfloat16* obase = p.out + (size_t)(s_off[t.e] + t.m0) * p.ldo + t.n0 + wn * C::WTN + 8 * tg;
#pragma unroll
      for (int i = 0; i < C::MT; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = mbase + i * 16 + g + 8 * h;
#pragma unroll
          for (int jq = 0; jq < C::NT / 4; ++jq) {
            uint32_t a4[4], b4[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              __nv_bfloat162 v = __floats2bfloat162_rn(epi(alpha, acc[i][4 * jq + k][2 * h]),
                                                       epi(alpha, acc[i][4 * jq + k][2 * h + 1]));
              a4[k] = *reinterpret_cast<uint32_t*>(&v);
            }
            quad_transpose(a4, b4, lane);
            if (r < t.rows) {
              if constexpr (C::OPOL == 1)
                st_v4_hint(obase + (size_t)r * p.ldo + 32 * jq, make_uint4(b4[0], b4[1], b4[2], b4[3]), make_policy<2>());
              else
                *reinterpret_cast<uint4*>(obase + (size_t)r * p.ldo + 32 * jq) = make_uint4(b4[0], b4[1], b4[2], b4[3]);
            }
          }
        }
      }
    } else if constexpr (C::MODE == 0 && C::PROBE == 3) {
      continue;                                  // timing probe: no output stores
    } else if constexpr (C::MODE == 0) {
      // ---- epilogue: out = bf16_rn(fma(0, 0, alpha * acc)) for the real rows ----
      uint64_t opol = 0;
      if constexpr (C::OPOL == 1) opol = make_policy<2>();
      __nv_bfloat16* orow = p.out + (size_t)(s_off[t.e] + t.m0) * p.ldo + t.n0 + wn * C::WTN + 2 * tg;
#pragma unroll
      for (int i = 0; i < C::MT; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = mbase + i * 16 + g + 8 * h;
          if (r < t.rows) {
            __nv_bfloat16* o = orow + (size_t)r * p.ldo;
#pragma unroll
            for (int j = 0; j < C::NT; ++j) {
              __nv_bfloat162 v2 = __floats2bfloat162_rn(epi(alpha, acc[i][j][2 * h]), epi(alpha, acc[i][j][2 * h + 1]));
              if constexpr (C::OPOL == 1)
                st_b32_hint(o + j * 8, *reinterpret_cast<uint32_t*>(&v2), opol);
              else
                *reinterpret_cast<__nv_bfloat162*>(o + j * 8) = v2;
            }
          }
        }
      }
    } else {
      // ---- fused epilogue: c1 (bf16) -> gelu_tanh(gate) * up (bf16) -> NVFP4 (vLLM's experts quant) ----
      const float sfscale = __ldg(p.q_gscale + t.e);
      const float beta = 0.0f;
      const int K4i = (p.I + 63) / 64;
      const int qcol0 = t.n0 + wn * (C::WTN / 2);           // first intermediate column of this warp
#pragma unroll
      for (int i = 0; i < C::MT; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = mbase + i * 16 + g + 8 * h;            // row inside the unit
#pragma unroll
          for (int q = 0; q < C::NT / 4; ++q) {               // 16-column quant blocks of this warp
            __nv_bfloat162 v[2];
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
              const int j = 2 * q + jj;
              __nv_bfloat162 gb = __floats2bfloat162_rn(epi(alpha, acc[i][j][2 * h]), epi(alpha, acc[i][j][2 * h + 1]));
              __nv_bfloat162 ub = __floats2bfloat162_rn(epi(alpha, acc[i][j + C::NT / 2][2 * h]),
                                                        epi(alpha, acc[i][j + C::NT / 2][2 * h + 1]));
              v[jj] = gelu_tanh_mul2(gb, ub, beta);
            }
            // amax over the 16 values of the block (4 per thread, the quad holds the block)
            __nv_bfloat162 lmax = __hmax2(__habs2(v[0]), __habs2(v[1]));
            uint32_t lm = *reinterpret_cast<uint32_t*>(&lmax);
            uint32_t o1 = __shfl_xor_sync(0xffffffffu, lm, 1);
            lmax = __hmax2(*reinterpret_cast<__nv_bfloat162*>(&o1), lmax);
            lm = *reinterpret_cast<uint32_t*>(&lmax);
            uint32_t o2 = __shfl_xor_sync(0xffffffffu, lm, 2);
            lmax = __hmax2(*reinterpret_cast<__nv_bfloat162*>(&o2), lmax);
            const float vecMax = __bfloat162float(__hmax(lmax.x, lmax.y));
            float SFValue = sfscale * (vecMax * rcp_approx_ftz(6.0f));
            __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(SFValue);
            SFValue = float(sf8);
            const float oscale = SFValue != 0.0f ? rcp_approx_ftz(SFValue * rcp_approx_ftz(sfscale)) : 0.0f;
            float2 f0 = __bfloat1622float2(v[0]), f1 = __bfloat1622float2(v[1]);
            const uint32_t b0 = e2m1x2(f0.x * oscale, f0.y * oscale);  // block byte tg
            const uint32_t b1 = e2m1x2(f1.x * oscale, f1.y * oscale);  // block byte 4 + tg
            uint32_t y = b0 | (b1 << 16);
            y |= __shfl_xor_sync(0xffffffffu, y, 1) << 8;            // even tg: bytes (tg, tg+1) | (4+tg, 5+tg)
            const uint32_t y2 = __shfl_xor_sync(0xffffffffu, y, 2);
            if (tg == 0 && r < t.rows) {
              const int row = s_off[t.e] + t.m0 + r;
              const int qc = qcol0 + 16 * q;                      // first column of the block
              uint2 w;
              w.x = (y & 0xFFFFu) | ((y2 & 0xFFFFu) << 16);
              w.y = (y >> 16) | (y2 & 0xFFFF0000u);
              *reinterpret_cast<uint2*>(p.q_out + (size_t)row * (p.I / 2) + qc / 2) = w;
              const int srow = s_sfo[t.e] + t.m0 + r;
              const int kb = qc / 16;
              const size_t off = ((size_t)(srow >> 7) * K4i + (kb >> 2)) * 512 + (srow & 31) * 16 +
                                 ((srow >> 5) & 3) * 4 + (kb & 3);
              p.q_sf[off] = sf8.__x;
            }
          }
        }
      }
    }
  }
}

// ---- E69: drop the dirty L2 lines of a dead buffer (whole 128-byte lines only) ----
__global__ void discard_kernel(uint8_t* base, size_t lines) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < lines; i += (size_t)gridDim.x * blockDim.x)
    asm volatile("discard.global.L2 [%0], 128;" ::"l"(base + i * 128) : "memory");
}

// up to 4 dead buffers in one launch (C6: one launch after w13, one after the tail)
struct DiscardRanges {
  uint8_t* base[4];
  unsigned long long lines[4];
  int n;
};
__global__ void discard_multi_kernel(const __grid_constant__ DiscardRanges r) {
  const unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  for (int k = 0; k < r.n; ++k)
    for (unsigned long long i = tid; i < r.lines[k]; i += stride)
      asm volatile("discard.global.L2 [%0], 128;" ::"l"(r.base[k] + i * 128) : "memory");
}

// ---- E69: what does the all-zero k64 chunk (w2's 12th chunk) do to an accumulator? one warp; each lane's 4 inputs
// acc_in[4 lane + r] go through mma(acc, A = 0, B = 0, sfa = 0, sfb = 0) as bmoe's consumer issues it ----
__global__ void zero_mma_kernel(const float* acc_in, float* out) {
  float d[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) d[r] = acc_in[threadIdx.x * 4 + r];
  uint32_t a[4] = {0u, 0u, 0u, 0u}, b[2] = {0u, 0u};
  mma_nvf4(d, a, b, 0u, 0u);
#pragma unroll
  for (int r = 0; r < 4; ++r) out[threadIdx.x * 4 + r] = d[r];
}

// ---- E69: block-scaled MMA throughput (each warp: NACC independent accumulators per iteration, register operands;
// FRAG: 0 = one A / one B fragment for all, 1 = ruj's consumer pattern MT 2 x NT 8 with distinct fragments) ----
template <int NACC, int FRAG>
__global__ void mma_rate_kernel(long long* cyc, float* sink, int iters, uint32_t seed) {
  float acc[NACC][4];
#pragma unroll
  for (int k = 0; k < NACC; ++k)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[k][r] = 0.0f;
  const uint32_t l = threadIdx.x + seed;
  uint32_t af[2][4], bf[8][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int r = 0; r < 4; ++r) af[i][r] = (l * (2654435761u + 97u * (i * 4 + r))) & 0x77777777u;
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int r = 0; r < 2; ++r) bf[j][r] = (l * (40503u + 131u * (j * 2 + r))) & 0x77777777u;
  const uint32_t sf = 0x38383838u;   // e4m3 1.0 in every byte
  __syncwarp();
  const long long t0 = clock64();
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < NACC; ++k) {
      if constexpr (FRAG == 0) mma_nvf4(acc[k], af[0], bf[0], sf, sf);
      else mma_nvf4(acc[k], af[(k / 8) & 1], bf[k & 7], sf, sf);
    }
  }
  const long long t1 = clock64();
  float s = 0.0f;
#pragma unroll
  for (int k = 0; k < NACC; ++k)
#pragma unroll
    for (int r = 0; r < 4; ++r) s += acc[k][r];
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if ((threadIdx.x & 31) == 0) {
    cyc[w] = t1 - t0;
    sink[w] = s;
  }
}

// ---- E69: L2 isolation probe: read a buffer with an L2 evict_last hint (marks its lines persisting when this context
// has a persisting set-aside); the sum keeps the loads alive ----
__global__ void persist_fill_kernel(const uint4* p, size_t n16, int iters, unsigned* sink) {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  unsigned acc = 0;
  for (int it = 0; it < iters; ++it)
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
      uint4 v;
      asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
                   : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                   : "l"(p + i), "l"(pol));
      acc += v.x ^ v.w;
    }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

// ---- E69: weight re-pack (one CTA per record = (expert, unit tile, real 64-wide K chunk)) ----
// MODE 1 (w13 [E, N = 2I, K/2]): unit tile nt = gate rows nt*64 .. +63 then up rows I + nt*64 .. +63 (smem rows 0-127);
// MODE 0 (w2 [E, N, K/2]): unit tile nt = rows nt*128 .. +127. Record = 128 rows x 32 B of weights (pk_off order) and
// one 512-byte 128x4 scale atom (row n = smem row n). Source scales: vLLM's swizzled [E][N/128][K/64][512] layout.
__global__ void pack_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ sf, uint8_t* __restrict__ pw,
                            uint8_t* __restrict__ ps, int N, int K, int mode, int ntn, int n_real) {
  const int c = blockIdx.x % n_real;
  const int nt = (blockIdx.x / n_real) % ntn;
  const int e = blockIdx.x / (n_real * ntn);
  const int I = N / 2, KB = K / 2, K4 = K / 64;
  const size_t rec = blockIdx.x;
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {     // 128 rows x 2 halves of 16 bytes
    const int n = i >> 1, h = i & 1;
    const int row = mode == 1 ? (n < 64 ? nt * 64 + n : I + nt * 64 + (n - 64)) : nt * 128 + n;
    const uint4 v = *reinterpret_cast<const uint4*>(w + ((size_t)e * N + row) * KB + c * 32 + h * 16);
    *reinterpret_cast<uint4*>(pw + rec * 4096 + pk_off(n, h)) = v;
  }
  for (int i = threadIdx.x; i < 128; i += blockDim.x) {     // 128 rows x one 4-byte word (k-blocks 4c .. 4c+3)
    const int n = i;
    const int row = mode == 1 ? (n < 64 ? nt * 64 + n : I + nt * 64 + (n - 64)) : nt * 128 + n;
    const uint32_t v = *reinterpret_cast<const uint32_t*>(
        sf + (size_t)e * N * (K / 16) + ((size_t)(row >> 7) * K4 + c) * 512 + atom_off(row));
    *reinterpret_cast<uint32_t*>(ps + rec * 512 + atom_off(n)) = v;
  }
}

}  // namespace bmoe69

// ===== host side (ruj's bmoe_host.cuh, + PACK / PROBE dispatch and the pack entry point) =====
namespace bmoe69 {

using EncodeTiledFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn encode_fn() {
  static EncodeTiledFn fn = nullptr;
  static std::once_flag once;
  std::call_once(once, [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaError_t err = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
    TORCH_CHECK(err == cudaSuccess && q == cudaDriverEntryPointSuccess && p != nullptr,
                "bmoe69: cuTensorMapEncodeTiled not available");
    fn = reinterpret_cast<EncodeTiledFn>(p);
  });
  return fn;
}

// 2D u8 tensor [rows, cols] (cols contiguous), box [box_cols, box_rows]
static void make_tmap(CUtensorMap* tm, const void* base, uint64_t rows, uint64_t cols, uint32_t box_cols,
                      uint32_t box_rows, CUtensorMapSwizzle swz) {
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {cols};
  cuuint32_t box[2] = {box_cols, box_rows};
  cuuint32_t es[2] = {1, 1};
  CUresult r = encode_fn()(tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box, es,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "bmoe69: cuTensorMapEncodeTiled failed (", (int)r, ")");
}

static int sm_count_dev(int dev) {
  static std::unordered_map<int, int> cache;
  static std::mutex mu;
  std::lock_guard<std::mutex> g(mu);
  auto it = cache.find(dev);
  if (it != cache.end()) return it->second;
  int n = 0;
  cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  cache[dev] = n;
  return n;
}

static void check_common(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& a_sf,
                         const torch::Tensor& b_sf, const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
                         const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets) {
  TORCH_CHECK(a.scalar_type() == at::kByte && b.scalar_type() == at::kByte, "bmoe69: a/b must be uint8 (packed e2m1)");
  TORCH_CHECK(problem_sizes.scalar_type() == at::kInt && expert_offsets.scalar_type() == at::kInt &&
                  sf_offsets.scalar_type() == at::kInt && alphas.scalar_type() == at::kFloat,
              "bmoe69: index/alpha dtypes");
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && a_sf.is_cuda() && b_sf.is_cuda() && problem_sizes.is_cuda(),
              "bmoe69: cuda tensors");
  TORCH_CHECK(a.dim() == 2 && a.stride(1) == 1 && a.stride(0) == a.size(1), "bmoe69: a must be contiguous [rows, K/2]");
  TORCH_CHECK(b.is_contiguous() && a_sf.is_contiguous() && b_sf.is_contiguous(), "bmoe69: contiguity");
  TORCH_CHECK(problem_sizes.is_contiguous() && expert_offsets.is_contiguous() && sf_offsets.is_contiguous() &&
                  alphas.is_contiguous(),
              "bmoe69: contiguity of index tensors");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(a.data_ptr()) & 15) == 0 &&
                  (reinterpret_cast<uintptr_t>(b.data_ptr()) & 15) == 0 &&
                  (reinterpret_cast<uintptr_t>(a_sf.data_ptr()) & 15) == 0 &&
                  (reinterpret_cast<uintptr_t>(b_sf.data_ptr()) & 15) == 0,
              "bmoe69: 16-byte alignment");
}

template <class C>
static void launch(Params& p, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& a_sf,
                   const torch::Tensor& b_sf, const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
                   const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets, int64_t N, int sm_limit) {
  const int E = static_cast<int>(expert_offsets.size(0));
  const int64_t K = a.size(1) * 2, rows = a.size(0);
  TORCH_CHECK(E <= kMaxE, "bmoe69: too many experts");
  TORCH_CHECK(K % 64 == 0, "bmoe69: K must be a multiple of 64, got ", K);
  TORCH_CHECK(b.numel() == (int64_t)E * N * (K / 2), "bmoe69: b has unexpected size");
  TORCH_CHECK(b_sf.numel() == (int64_t)E * N * (K / 16), "bmoe69: b_sf has unexpected size");
  TORCH_CHECK(N % 128 == 0, "bmoe69: N must be a multiple of 128, got ", N);
  if (rows == 0) return;
  const CUtensorMapSwizzle swz = C::RB == 64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B;
  for (int i = 0; i < kNumABoxes; ++i)
    make_tmap(&p.tmA[i], a.data_ptr(), (uint64_t)rows, (uint64_t)(K / 2), C::RB, 16 * (i + 1), swz);
  if (!C::PACK)
    make_tmap(&p.tmB, b.data_ptr(), (uint64_t)E * N, (uint64_t)(K / 2), C::RB, C::MODE == 1 ? C::BN / 2 : C::BN, swz);
  p.a_sf = static_cast<const uint8_t*>(a_sf.data_ptr());
  p.b_sf = static_cast<const uint8_t*>(b_sf.data_ptr());
  p.b_pk = static_cast<const uint8_t*>(b.data_ptr());
  p.alphas = alphas.data_ptr<float>();
  p.problem_sizes = problem_sizes.data_ptr<int32_t>();
  p.expert_offsets = expert_offsets.data_ptr<int32_t>();
  p.sf_offsets = sf_offsets.data_ptr<int32_t>();
  p.E = E;
  p.N = (int)N;
  p.K = (int)K;
  p.n_chunks = (int)((K + 255) / 256) * 4;
  p.n_real = (int)(K / 64);
  p.I = (int)(N / 2);
  const int dev = a.get_device();
  int grid = sm_count_dev(dev);
  if (sm_limit > 0 && sm_limit < grid) grid = sm_limit;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(dev).stream();
  static bool attr_set = false;
  if (!attr_set) {
    TORCH_CHECK(cudaFuncSetAttribute(bmoe_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) ==
                    cudaSuccess,
                "bmoe69: smem attribute");
    attr_set = true;
  }
  bmoe_kernel<C><<<grid, C::THREADS, C::SMEM, stream>>>(p);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "bmoe69: launch failed: ", cudaGetErrorString(err));
}

template <class C>
static void run_mm(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& a_sf,
                   const torch::Tensor& b_sf, const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
                   const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets, int sm_limit) {
  static_assert(C::MODE == 0, "run_mm is MODE 0");
  check_common(a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets);
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.stride(1) == 1 && out.size(0) == a.size(0),
              "bmoe69: out must be bf16 [rows, N] with unit column stride");
  TORCH_CHECK((out.stride(0) % 2) == 0 && (reinterpret_cast<uintptr_t>(out.data_ptr()) & 3) == 0, "bmoe69: out align");
  Params p;
  memset(&p, 0, sizeof(p));
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ldo = (int)out.stride(0);
  TORCH_CHECK(out.size(1) % C::BN == 0, "bmoe69: N must be a multiple of BN");
  launch<C>(p, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, out.size(1), sm_limit);
}

template <class C>
static void run_w13_fused(torch::Tensor& q_out, torch::Tensor& q_sf, const torch::Tensor& a, const torch::Tensor& b,
                          const torch::Tensor& a_sf, const torch::Tensor& b_sf, const torch::Tensor& alphas,
                          const torch::Tensor& problem_sizes, const torch::Tensor& expert_offsets,
                          const torch::Tensor& sf_offsets, const torch::Tensor& q_gscale, int sm_limit) {
  static_assert(C::MODE == 1, "run_w13_fused is MODE 1");
  check_common(a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets);
  const int E = static_cast<int>(expert_offsets.size(0));
  const int64_t K = a.size(1) * 2;
  const int64_t N = b.numel() / ((int64_t)E * (K / 2));
  const int64_t I = N / 2;
  TORCH_CHECK(I % 64 == 0, "bmoe69: fused w13 needs I % 64 == 0, got ", I);
  TORCH_CHECK(q_out.scalar_type() == at::kByte && q_out.is_contiguous() && q_out.size(0) == a.size(0) &&
                  q_out.size(1) == I / 2,
              "bmoe69: q_out must be uint8 [rows, I/2]");
  TORCH_CHECK(q_sf.is_contiguous(), "bmoe69: q_sf");
  TORCH_CHECK(q_gscale.scalar_type() == at::kFloat && q_gscale.numel() >= E, "bmoe69: q_gscale");
  Params p;
  memset(&p, 0, sizeof(p));
  p.q_out = static_cast<uint8_t*>(q_out.data_ptr());
  p.q_sf = static_cast<uint8_t*>(q_sf.data_ptr());
  p.q_gscale = q_gscale.data_ptr<float>();
  launch<C>(p, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, N, sm_limit);
}

}  // namespace bmoe69

// fused w13 = ruj's cfg 3 (BM128 BN128 BK128 5 stages 4x2, evict_first B / evict_last A, dynamic scheduler) with
// pack (0: vLLM's layout via TMA, 1: packed records) and probe (0 full, 1 producer only, 2 consumer only)
#define B69_FARGS q_out, q_sf, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, q_gscale, (int)sm_limit
void w13_fused69(torch::Tensor q_out, torch::Tensor q_sf, torch::Tensor a, torch::Tensor b, torch::Tensor a_sf,
                 torch::Tensor b_sf, torch::Tensor alphas, torch::Tensor problem_sizes, torch::Tensor expert_offsets,
                 torch::Tensor sf_offsets, torch::Tensor q_gscale, int64_t pack, int64_t probe, int64_t sm_limit) {
  using namespace bmoe69;
  switch (pack * 10 + probe) {
    case 0: run_w13_fused<Cfg<128, 128, 128, 5, 4, 2, 1, 1, 2, 0, 1, 0, 0>>(B69_FARGS); break;
    case 1: run_w13_fused<Cfg<128, 128, 128, 5, 4, 2, 1, 1, 2, 0, 1, 0, 1>>(B69_FARGS); break;
    case 2: run_w13_fused<Cfg<128, 128, 128, 5, 4, 2, 1, 1, 2, 0, 1, 0, 2>>(B69_FARGS); break;
    case 10: run_w13_fused<Cfg<128, 128, 128, 5, 4, 2, 1, 1, 2, 0, 1, 1, 0>>(B69_FARGS); break;
    case 11: run_w13_fused<Cfg<128, 128, 128, 5, 4, 2, 1, 1, 2, 0, 1, 1, 1>>(B69_FARGS); break;
    case 12: run_w13_fused<Cfg<128, 128, 128, 5, 4, 2, 1, 1, 2, 0, 1, 1, 2>>(B69_FARGS); break;
    case 4: run_w13_fused<Cfg<128, 128, 128, 5, 4, 2, 1, 1, 2, 0, 1, 0, 4>>(B69_FARGS); break;
    case 5: run_w13_fused<Cfg<128, 128, 128, 5, 4, 2, 1, 1, 2, 0, 1, 0, 5>>(B69_FARGS); break;
    case 6: run_w13_fused<Cfg<128, 128, 128, 5, 4, 2, 1, 1, 2, 0, 1, 0, 6>>(B69_FARGS); break;
    case 7: run_w13_fused<Cfg<128, 128, 128, 5, 4, 2, 1, 1, 2, 0, 1, 0, 7>>(B69_FARGS); break;
    // C2 probe: 16 consumer warps (4 per SM sub-partition), 4 x 4 warps of 32 x 32 (pack argument 10 = this layout)
    case 100: run_w13_fused<Cfg<128, 128, 128, 5, 4, 4, 1, 1, 2, 0, 1, 0, 0>>(B69_FARGS); break;
    case 102: run_w13_fused<Cfg<128, 128, 128, 5, 4, 4, 1, 1, 2, 0, 1, 0, 2>>(B69_FARGS); break;
    case 107: run_w13_fused<Cfg<128, 128, 128, 5, 4, 4, 1, 1, 2, 0, 1, 0, 7>>(B69_FARGS); break;
    default: TORCH_CHECK(false, "bmoe69: unknown pack/probe ", pack, "/", probe);
  }
}

// w2 = ruj's variant 11 (BM128 BN128 BK128 5 stages 4x2, dynamic scheduler, 4-byte stores) with pack / probe
#define B69_ARGS out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, (int)sm_limit
void moe_mm69(torch::Tensor out, torch::Tensor a, torch::Tensor b, torch::Tensor a_sf, torch::Tensor b_sf,
              torch::Tensor alphas, torch::Tensor problem_sizes, torch::Tensor expert_offsets, torch::Tensor sf_offsets,
              int64_t pack, int64_t probe, int64_t sm_limit) {
  using namespace bmoe69;
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "bmoe69: out must be bf16");
  switch (pack * 10 + probe) {
    case 0: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 0, 1, 0, 0>>(B69_ARGS); break;
    case 1: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 0, 1, 0, 1>>(B69_ARGS); break;
    case 2: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 0, 1, 0, 2>>(B69_ARGS); break;
    case 10: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 0, 1, 1, 0>>(B69_ARGS); break;
    case 11: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 0, 1, 1, 1>>(B69_ARGS); break;
    case 12: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 0, 1, 1, 2>>(B69_ARGS); break;
    case 4: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 0, 1, 0, 4>>(B69_ARGS); break;
    case 5: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 0, 1, 0, 5>>(B69_ARGS); break;
    case 6: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 0, 1, 0, 6>>(B69_ARGS); break;
    case 7: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 0, 1, 0, 7>>(B69_ARGS); break;
    case 100: run_mm<Cfg<128, 128, 128, 5, 4, 4, 0, 1, 2, 0, 1, 0, 0>>(B69_ARGS); break;
    case 102: run_mm<Cfg<128, 128, 128, 5, 4, 4, 0, 1, 2, 0, 1, 0, 2>>(B69_ARGS); break;
    case 107: run_mm<Cfg<128, 128, 128, 5, 4, 4, 0, 1, 2, 0, 1, 0, 7>>(B69_ARGS); break;
    default: TORCH_CHECK(false, "bmoe69: unknown pack/probe ", pack, "/", probe);
  }
}

// w2 (ruj's variant 11 geometry) for the c3 probes: flags bit0 = OPOL (L2 evict_last c3 stores), bit1 = ZS (skip the
// all-zero 12th chunk), bit2 = 16-byte stores (EPI 1, ruj's variant 12 epilogue), bit3 = 16 consumer warps (4 x 4,
// C2-lite; with OPOL / ZS only); probe 0 (full) or 3 (no stores). w13_fused69 pack argument 10 = the 16-warp layout.
void moe_mm69x(torch::Tensor out, torch::Tensor a, torch::Tensor b, torch::Tensor a_sf, torch::Tensor b_sf,
               torch::Tensor alphas, torch::Tensor problem_sizes, torch::Tensor expert_offsets, torch::Tensor sf_offsets,
               int64_t flags, int64_t probe, int64_t sm_limit) {
  using namespace bmoe69;
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "bmoe69: out must be bf16");
  if (flags & 4)
    TORCH_CHECK((out.stride(0) % 8) == 0 && (reinterpret_cast<uintptr_t>(out.data_ptr()) & 15) == 0,
                "bmoe69: 16-byte stores need a 16-byte aligned out");
  switch (flags * 10 + probe) {
    case 0: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 0, 1, 0, 0, 0, 0>>(B69_ARGS); break;
    case 3: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 0, 1, 0, 3, 0, 0>>(B69_ARGS); break;
    case 10: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 0, 1, 0, 0, 1, 0>>(B69_ARGS); break;
    case 20: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 0, 1, 0, 0, 0, 1>>(B69_ARGS); break;
    case 30: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 0, 1, 0, 0, 1, 1>>(B69_ARGS); break;
    case 40: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 1, 1, 0, 0, 0, 0>>(B69_ARGS); break;
    case 50: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 1, 1, 0, 0, 1, 0>>(B69_ARGS); break;
    case 70: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 1, 1, 0, 0, 1, 1>>(B69_ARGS); break;
    // flags bit 8: 16 consumer warps (4 x 4 warps of 32 x 32; E69 C2-lite), with OPOL (1) / ZS (2)
    case 80: run_mm<Cfg<128, 128, 128, 5, 4, 4, 0, 1, 2, 0, 1, 0, 0, 0, 0>>(B69_ARGS); break;
    case 90: run_mm<Cfg<128, 128, 128, 5, 4, 4, 0, 1, 2, 0, 1, 0, 0, 1, 0>>(B69_ARGS); break;
    case 100: run_mm<Cfg<128, 128, 128, 5, 4, 4, 0, 1, 2, 0, 1, 0, 0, 0, 1>>(B69_ARGS); break;
    case 110: run_mm<Cfg<128, 128, 128, 5, 4, 4, 0, 1, 2, 0, 1, 0, 0, 1, 1>>(B69_ARGS); break;
    default: TORCH_CHECK(false, "bmoe69: unknown flags/probe ", flags, "/", probe);
  }
}

// the L2 set-aside for persisting (evict_last) lines of this context: request `bytes` (clamped to the device maximum),
// return {granted, device maximum, L2 size}. Application-level CUDA limit of this process only.
std::vector<int64_t> l2_persist69(int64_t bytes) {
  int dev = 0, mx = 0, l2 = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&mx, cudaDevAttrMaxPersistingL2CacheSize, dev);
  cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
  if (bytes >= 0) {
    size_t want = (size_t)std::min<int64_t>(bytes, mx);
    TORCH_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, want) == cudaSuccess, "l2_persist69: set limit");
  }
  size_t got = 0;
  cudaDeviceGetLimit(&got, cudaLimitPersistingL2CacheSize);
  return {(int64_t)got, (int64_t)mx, (int64_t)l2};
}

// block-scaled MMA throughput: returns per-warp cycles for iters x NACC MMAs; warps = 4 * per_smsp per CTA, one CTA per SM
torch::Tensor mma_rate69(int64_t per_smsp, int64_t nacc, int64_t frag, int64_t iters, int64_t ctas) {
  auto o64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  const int warps = (int)(4 * per_smsp);
  auto cyc = torch::zeros({ctas * warps}, o64);
  auto sink = torch::zeros({ctas * warps}, o64.dtype(torch::kFloat));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  dim3 g((unsigned)ctas), b((unsigned)(warps * 32));
  long long* c = reinterpret_cast<long long*>(cyc.data_ptr<int64_t>());
  float* s = sink.data_ptr<float>();
  const int it = (int)iters;
  if (frag == 0) {
    if (nacc == 4) bmoe69::mma_rate_kernel<4, 0><<<g, b, 0, stream>>>(c, s, it, 7u);
    else if (nacc == 8) bmoe69::mma_rate_kernel<8, 0><<<g, b, 0, stream>>>(c, s, it, 7u);
    else if (nacc == 16) bmoe69::mma_rate_kernel<16, 0><<<g, b, 0, stream>>>(c, s, it, 7u);
    else TORCH_CHECK(false, "mma_rate69: nacc 4/8/16");
  } else {
    TORCH_CHECK(nacc == 16, "mma_rate69: frag 1 needs nacc 16");
    bmoe69::mma_rate_kernel<16, 1><<<g, b, 0, stream>>>(c, s, it, 7u);
  }
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "mma_rate69: launch failed: ", cudaGetErrorString(err));
  return cyc;
}

// L2 isolation probe: read t (contiguous) iters times with an L2 evict_last hint, on the current stream
void persist_fill69(torch::Tensor t, int64_t iters) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && (t.numel() * t.element_size()) % 16 == 0, "persist_fill69: tensor");
  auto sink = torch::zeros({1}, t.options().dtype(torch::kInt32));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(t.get_device()).stream();
  const size_t n16 = (size_t)t.numel() * t.element_size() / 16;
  bmoe69::persist_fill_kernel<<<1024, 256, 0, stream>>>(reinterpret_cast<const uint4*>(t.data_ptr()), n16, (int)iters,
                                                       reinterpret_cast<unsigned*>(sink.data_ptr()));
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "persist_fill69: launch failed: ", cudaGetErrorString(err));
}

// discard.global.L2 over the whole 128-byte lines of t (contiguous, 128-byte aligned), on the current stream
void discard69(torch::Tensor t) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "discard69: contiguous cuda tensor");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr()) & 127) == 0, "discard69: 128-byte aligned base");
  const size_t lines = (size_t)t.numel() * t.element_size() / 128;
  if (lines == 0) return;
  const int blocks = (int)std::min<size_t>((lines + 255) / 256, 4096);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(t.get_device()).stream();
  bmoe69::discard_kernel<<<blocks, 256, 0, stream>>>(static_cast<uint8_t*>(t.data_ptr()), lines);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "discard69: launch failed: ", cudaGetErrorString(err));
}

// discard.global.L2 over the whole 128-byte lines of up to 4 dead buffers (contiguous, 128-byte aligned), one launch
void discard69n(std::vector<torch::Tensor> ts) {
  TORCH_CHECK(!ts.empty() && ts.size() <= 4, "discard69n: 1..4 tensors");
  bmoe69::DiscardRanges r;
  memset(&r, 0, sizeof(r));
  unsigned long long most = 0;
  int dev = -1;
  for (auto& t : ts) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "discard69n: contiguous cuda tensors");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr()) & 127) == 0, "discard69n: 128-byte aligned bases");
    dev = t.get_device();
    r.base[r.n] = static_cast<uint8_t*>(t.data_ptr());
    r.lines[r.n] = (unsigned long long)t.numel() * t.element_size() / 128;
    most = std::max(most, r.lines[r.n]);
    ++r.n;
  }
  if (most == 0) return;
  const int blocks = (int)std::min<unsigned long long>((most + 255) / 256, 2048);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(dev).stream();
  bmoe69::discard_multi_kernel<<<blocks, 256, 0, stream>>>(r);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "discard69n: launch failed: ", cudaGetErrorString(err));
}

// one all-zero k64 chunk on the 128 accumulators in acc_in (float32 [128]): returns the 128 results
torch::Tensor zero_mma69(torch::Tensor acc_in) {
  TORCH_CHECK(acc_in.is_cuda() && acc_in.scalar_type() == at::kFloat && acc_in.numel() == 128 && acc_in.is_contiguous(),
              "zero_mma69: float32 [128]");
  auto out = torch::empty_like(acc_in);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(acc_in.get_device()).stream();
  bmoe69::zero_mma_kernel<<<1, 32, 0, stream>>>(acc_in.data_ptr<float>(), out.data_ptr<float>());
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "zero_mma69: launch failed: ", cudaGetErrorString(err));
  return out;
}

// re-pack w [E, N, K/2] (uint8 e2m1 pairs) + sf (vLLM's swizzled e4m3 block scales, E*N*K/16 bytes) into pw / ps (same
// sizes): mode 1 = w13 (unit tiles of 64 gate + 64 up rows), mode 0 = w2 (128-row unit tiles). Out of place.
void pack69(torch::Tensor w, torch::Tensor sf, torch::Tensor pw, torch::Tensor ps, int64_t mode) {
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kByte && w.dim() == 3 && w.is_contiguous(), "pack69: w");
  TORCH_CHECK(sf.is_cuda() && sf.is_contiguous() && pw.is_contiguous() && ps.is_contiguous(), "pack69: contiguity");
  const int64_t E = w.size(0), N = w.size(1), K = w.size(2) * 2;
  TORCH_CHECK(K % 64 == 0 && N % 128 == 0 && (mode == 0 || mode == 1), "pack69: shape");
  TORCH_CHECK(sf.numel() * sf.element_size() == E * N * (K / 16), "pack69: sf size");
  TORCH_CHECK(pw.numel() * pw.element_size() == w.numel() && ps.numel() * ps.element_size() == E * N * (K / 16),
              "pack69: output sizes");
  TORCH_CHECK(pw.data_ptr() != w.data_ptr() && ps.data_ptr() != sf.data_ptr(), "pack69: out of place only");
  const int ntn = mode == 1 ? (int)(N / 2 / 64) : (int)(N / 128);
  const int n_real = (int)(K / 64);
  if (mode == 1) TORCH_CHECK((N / 2) % 64 == 0, "pack69: I % 64");
  const int64_t recs = E * ntn * n_real;
  TORCH_CHECK(recs * 4096 == w.numel(), "pack69: record count");
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(w.get_device()).stream();
  bmoe69::pack_kernel<<<(unsigned)recs, 256, 0, stream>>>(
      static_cast<const uint8_t*>(w.data_ptr()), static_cast<const uint8_t*>(sf.data_ptr()),
      static_cast<uint8_t*>(pw.data_ptr()), static_cast<uint8_t*>(ps.data_ptr()), (int)N, (int)K, (int)mode, ntn,
      n_real);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "pack69: launch failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w13_fused69", &w13_fused69, "bmoe (team ruj) fused w13 cfg 3 + E69 pack/probe knobs");
  m.def("moe_mm69", &moe_mm69, "bmoe (team ruj) w2 variant 11 + E69 pack/probe knobs");
  m.def("pack69", &pack69, "E69 weight + block-scale re-pack (out of place)");
  m.def("moe_mm69x", &moe_mm69x, "bmoe w2 v11 geometry with E69 flags (1 evict_last c3, 2 zero-chunk skip, 4 16-B stores)");
  m.def("persist_fill69", &persist_fill69, "read a buffer with L2 evict_last hints (L2 isolation probe)");
  m.def("mma_rate69", &mma_rate69, "block-scaled m16n8k64 MMA throughput microbenchmark (per-warp cycles)");
  m.def("l2_persist69", &l2_persist69, "set / get this context's persisting-L2 set-aside");
  m.def("discard69", &discard69, "discard.global.L2 over the 128-byte lines of a dead buffer");
  m.def("discard69n", &discard69n, "discard.global.L2 over the 128-byte lines of up to 4 dead buffers, one launch");
  m.def("zero_mma69", &zero_mma69, "one all-zero k64 mxf4nvf4 MMA chunk on 128 accumulators");
}
