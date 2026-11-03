// SPDX-License-Identifier: Apache-2.0
// nv4s.cu (kai lever N1, 10-28): rgh's NVFP4 gate|up GEMM (nv4g.cu: CUTLASS 4.5.0 sm120 block-scaled collective, mma.sync
// kind::mxf4nvf4.block_scale, e2m1 x e2m1, ue4m3 scale per 16 K, fp32 accumulation) with SwiGLU and the global amax of
// the next NVFP4 quantisation folded into the epilogue:
//
//   Y[m, j] = bf16( silu_mul( gu_round(alpha, acc[m, 2j]), gu_round(alpha, acc[m, 2j + 1]) ) ),  amax[0] = max |Y|
//
// for a weight whose rows are interleaved at start-up (n1.py: row 2j = gate row j, row 2j + 1 = up row j), so the gate
// and up value of one output column sit in the same thread: the sm120 epilogue visits the accumulators in fragments of 4
// (one m16n8 MMA's output per thread), elements (0, 1) and (2, 3) being columns (n, n + 1) of one row (checked per pair
// on the coordinate tensor: the kernel traps if not). No bf16 [M, 2I] gate|up tensor is written, and the separate
// _silu_amax kernel (actq.py) is not needed; the NVFP4 quant pass of the down projection (nv4q._quant_dyn_kernel) reads Y
// and amax as before.
//
// EXACT (bit-identical to nv4g on the original weight + K.silu_mul + nv4q's atomic amax):
//   1. mainloop: the collective, element types, layouts and alignments are nv4g.cu's struct Nv4 (tests compare the text);
//      every output element is ONE fp32 accumulator chain over the 64-wide k blocks in order from 0, independent of the
//      tile, the warp layout, the CTA that owns the tile and the pipeline stage count (the stage count differs from
//      nv4g's: StageCountAutoCarveout sees a smaller epilogue). Interleaving B's rows (with their block scales) moves an
//      output column, it does not change its chain: column 2j here is column j of nv4g, column 2j + 1 is column I + j.
//   2. epilogue arithmetic: gu_round is nv4g's LinearCombination (alpha * acc, + beta * C with beta 0 and C cleared, bf16
//      round to nearest); silu_mul is the compiled K.silu_mul's (nv4s_silu.cuh), checked at start on every bf16 g.
//   3. amax: max of |Y| over all valid elements: a per-thread max, a warp max, one integer atomicMax per warp and tile on
//      the fp32 bits (all values >= +0, so integer order = float order); order-independent, as nv4q / actq's atomics.
//   Data-parallel persistent scheduler only (no split-K / stream-K); raster order / swizzle are run-time arguments (they
//   only choose which CTA computes a tile), as in nv4t.cu.
// C ABI: nv4s_gemm(tile, raster, swizzle, A, B, SFA, SFB, alpha, Y, amax, M, N, K, ldy, workspace, ws_bytes, stream)
//        -> 0 ok; 1 unknown tile, 2 CUTLASS refused, 3 workspace too small, 4 init, 5 launch, 6 bad raster/swizzle,
//        7 N odd; N = 2 I (B rows interleaved), Y [M, I] bf16 with leading dimension ldy, amax fp32 (zeroed by the caller)
//        nv4s_ws(tile, raster, swizzle, M, N, K) -> workspace bytes
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "nv4s_silu.cuh"

using namespace cute;

#ifndef NV4S_TILE
#define NV4S_TILE 3
#endif

namespace n1 {

struct SwigluArgs {
  const float* alpha_ptr = nullptr;
  __nv_bfloat16* y = nullptr;
  int64_t ldy = 0;
  float* amax = nullptr;
};

// EVT root node: consumes the accumulator fragment (child Sm90AccFetch), writes Y and the tile's amax; no D tensor
struct Sm90SwigluStore {
  using ElementAux = cutlass::bfloat16_t;  // the epilogue's register / smem element type when D is void
  struct SharedStorage {};
  using Arguments = SwigluArgs;
  using Params = SwigluArgs;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& a, void*) { return a; }
  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const& a) {
    return a.alpha_ptr != nullptr && a.y != nullptr && a.amax != nullptr;
  }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }
  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const&, Arguments const&, void*, cudaStream_t,
                                              cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  CUTLASS_HOST_DEVICE Sm90SwigluStore() {}
  CUTLASS_HOST_DEVICE Sm90SwigluStore(Params const& p, SharedStorage const&) : params(p) {}
  Params params;

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(cutlass::epilogue::fusion::detail::ProducerLoadArgs<Args...> const&) {
    return cutlass::epilogue::fusion::detail::EmptyProducerLoadCallbacks{};
  }

  template <class CTensor, class ThrResidue>
  struct ConsumerStoreCallbacks : cutlass::epilogue::fusion::detail::EmptyConsumerStoreCallbacks {
    static constexpr int MAXP = 32;  // pairs of one epilogue subtile per thread (static_assert below)
    CTensor tCcD;          // (CPY,CPY_M,CPY_N,EPI_M,EPI_N) coordinates relative to this thread's first element
    ThrResidue residue;    // (M, N) - this thread's first element
    Params params;
    int m0, n0;            // this thread's first element (global)
    float alpha;
    float amax;
    bool quad_ok;          // the 4 lanes of this quad hold gate|up columns qb + 2t, + 1 of the same rows (vector stores)
    uint16_t yv[MAXP];     // this subtile's y values (bf16 bits), by pair index

    CUTLASS_DEVICE ConsumerStoreCallbacks(CTensor t, ThrResidue r, Params const& p, int m0_, int n0_)
        : tCcD(t), residue(r), params(p), m0(m0_), n0(n0_), alpha(*p.alpha_ptr), amax(0.0f) {
      int lane = int(threadIdx.x & 31), q = lane & 3;
      int nq = __shfl_sync(0xffffffffu, n0, lane & ~3);
      int mq = __shfl_sync(0xffffffffu, m0, lane & ~3);
      bool ok = nq == n0 - 2 * q && mq == m0 && (nq & 7) == 0 && (p.ldy & 3) == 0 &&
                (reinterpret_cast<uintptr_t>(p.y) & 7) == 0;
      quad_ok = __all_sync(0xffffffffu, ok);
    }

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<cutlass::bfloat16_t, FragmentSize>
    visit(cutlass::Array<ElementAccumulator, FragmentSize> const&, int epi_v, int epi_m, int epi_n,
          cutlass::Array<ElementInput, FragmentSize> const& acc) {
      static_assert(FragmentSize % 2 == 0, "N1: pairs of adjacent columns per fragment");
      Tensor tC = tCcD(_, _, _, epi_m, epi_n);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; i += 2) {
        auto c0 = tC(epi_v * FragmentSize + i);
        auto c1 = tC(epi_v * FragmentSize + i + 1);
        if (!(get<0>(c1) == get<0>(c0) && get<1>(c1) == get<1>(c0) + 1 && (get<1>(c0) & 1) == 0)) {
          __trap();  // the pair layout N1 relies on does not hold for this configuration (folds away when it does)
        }
        float g = gu_round(alpha, float(acc[i]));
        float u = gu_round(alpha, float(acc[i + 1]));
        __nv_bfloat16 yb = __float2bfloat16_rn(silu_mul(g, u));
        yv[(epi_v * FragmentSize + i) / 2] = __bfloat16_as_ushort(yb);
        if (elem_less(c0, residue)) {
          amax = fmaxf(amax, fabsf(__bfloat162float(yb)));
        }
      }
      return cutlass::Array<cutlass::bfloat16_t, FragmentSize>{};  // no D: the collective stores nothing
    }

    // after the subtile's visits: store its y. Fast path (static layout test, folded by the compiler): every pair this
    // thread holds is the (2q, 2q + 1) column pair of an 8-column MMA atom (relative column a multiple of 8), so for each
    // (row, atom) slot the quad holds its 4 consecutive y. Groups of 4 slots are transposed within the quad (two xor
    // shuffles): lane q then holds the 4 consecutive y of slot 4G + q -> one 8-byte store per lane and group. Otherwise
    // one 2-byte store per y. Values are only moved, never recomputed.
    CUTLASS_DEVICE void end_loop(int epi_m, int epi_n) {
      Tensor tC = tCcD(_, _, _, epi_m, epi_n);
      constexpr int NE = decltype(size(tC))::value;
      constexpr int NP = NE / 2;
      static_assert(NE % 2 == 0 && NP <= MAXP, "N1: subtile too large for the y buffer");
      bool fast = NP % 4 == 0;
      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < NP; ++p) {
        fast = fast && (int(get<1>(tC(2 * p))) % 8 == 0);
      }
      if (fast && quad_ok) {
        int q = int(threadIdx.x & 3);
        int jq = (n0 >> 1) - q;  // the quad's first y column
        int rows = int(get<0>(residue));
        CUTLASS_PRAGMA_UNROLL
        for (int G = 0; G < NP / 4; ++G) {
          uint32_t A[4];
          int dm = 0, dn = 0;
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < 4; ++i) {
            A[i] = yv[4 * G + i];
            auto c = tC(2 * (4 * G + i));
            if (q == i) {
              dm = int(get<0>(c));
              dn = int(get<1>(c));
            }
          }
          CUTLASS_PRAGMA_UNROLL
          for (int k = 1; k <= 2; k <<= 1) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) {
              if (i & k) continue;
              int j = i | k;
              uint32_t snd = (q & k) ? A[i] : A[j];
              uint32_t rcv = __shfl_xor_sync(0xffffffffu, snd, k);
              if (q & k) A[i] = rcv; else A[j] = rcv;
            }
          }
          if (dm < rows) {
            uint2 v;
            v.x = A[0] | (A[1] << 16);
            v.y = A[2] | (A[3] << 16);
            *reinterpret_cast<uint2*>(params.y + int64_t(m0 + dm) * params.ldy + jq + (dn >> 1)) = v;
          }
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int p = 0; p < NP; ++p) {
          auto c = tC(2 * p);
          if (elem_less(c, residue)) {
            int m = m0 + int(get<0>(c));
            int n = n0 + int(get<1>(c));
            reinterpret_cast<uint16_t*>(params.y)[int64_t(m) * params.ldy + (n >> 1)] = yv[p];
          }
        }
      }
    }

    CUTLASS_DEVICE void end() {
      float v = amax;
      CUTLASS_PRAGMA_UNROLL
      for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
      }
      if ((threadIdx.x & 31) == 0 && v > 0.0f) {
        atomicMax(reinterpret_cast<int*>(params.amax), __float_as_int(v));
      }
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      cutlass::epilogue::fusion::detail::ConsumerStoreArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    int m0 = int(M) - int(get<0>(args.residue_tCcD));
    int n0 = int(N) - int(get<1>(args.residue_tCcD));
    return ConsumerStoreCallbacks<decltype(args.tCcD), decltype(args.residue_tCcD)>(args.tCcD, args.residue_tCcD,
                                                                                     params, m0, n0);
  }
};

}  // namespace n1

// ---- the mainloop types are nv4g.cu's struct Nv4 (tests compare them); the epilogue is N1's (void C / D, EVT above)
template <class TileShape, class MainSchedule, class EpiSchedule>
struct Nv4S {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementAcc = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignA = 32, AlignB = 32, AlignC = 8;
  using ClusterShape = Shape<_1, _1, _1>;
  using Fusion = cutlass::epilogue::fusion::Sm90EVT<n1::Sm90SwigluStore, cutlass::epilogue::fusion::Sm90AccFetch>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementAcc, void, LayoutC, AlignC, void,
      LayoutC, AlignC, EpiSchedule, Fusion>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementA, LayoutA, AlignA, ElementB, LayoutB,
      AlignB, ElementAcc, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainSchedule>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                                          CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using Cfg = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  static bool sched_ok(int raster, int swz) {
    return raster >= 0 && raster <= 2 && (swz == 1 || swz == 2 || swz == 4 || swz == 8);
  }

  static typename Gemm::Arguments args(const void* A, const void* B, const void* SFA, const void* SFB,
                                       const float* alpha, void* Y, float* amax, int M, int N, int K, int ldy,
                                       int raster, int swz) {
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    auto lA = Cfg::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto lB = Cfg::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
    n1::SwigluArgs sw;
    sw.alpha_ptr = alpha;
    sw.y = static_cast<__nv_bfloat16*>(Y);
    sw.ldy = ldy;
    sw.amax = amax;
    typename Gemm::Arguments a{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {static_cast<const typename ElementA::DataType*>(A), sA, static_cast<const typename ElementB::DataType*>(B), sB,
         static_cast<const typename ElementA::ScaleFactorType*>(SFA), lA,
         static_cast<const typename ElementB::ScaleFactorType*>(SFB), lB},
        {{}, nullptr, sC, nullptr, sD}};
    a.epilogue.thread = {{}, sw};
    using RO = decltype(a.scheduler.raster_order);
    a.scheduler.raster_order = raster == 1 ? RO::AlongM : (raster == 2 ? RO::AlongN : RO::Heuristic);
    a.scheduler.max_swizzle_size = swz;
    return a;
  }

  static size_t ws(int M, int N, int K, int raster, int swz) {
    auto a = args(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, N / 2, raster, swz);
    return Gemm::get_workspace_size(a);
  }

  static int run(const void* A, const void* B, const void* SFA, const void* SFB, const float* alpha, void* Y,
                 float* amax, int M, int N, int K, int ldy, void* wsp, size_t wsb, cudaStream_t st, int raster,
                 int swz) {
    if (!sched_ok(raster, swz)) return 6;
    if (N & 1) return 7;
    auto a = args(A, B, SFA, SFB, alpha, Y, amax, M, N, K, ldy, raster, swz);
    Gemm g;
    if (g.can_implement(a) != cutlass::Status::kSuccess) return 2;
    if (Gemm::get_workspace_size(a) > wsb) return 3;
    if (g.initialize(a, wsp, st) != cutlass::Status::kSuccess) return 4;
    if (g.run(st) != cutlass::Status::kSuccess) return 5;
    return 0;
  }
};

using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using Ping = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using EpiAuto = cutlass::epilogue::collective::EpilogueScheduleAuto;

// tile numbers = nv4t.cu's for the same shape / schedule
#if NV4S_TILE == 1
using G = Nv4S<Shape<_128, _128, _256>, Coop, EpiAuto>;
#elif NV4S_TILE == 2
using G = Nv4S<Shape<_128, _256, _128>, Coop, EpiAuto>;
#elif NV4S_TILE == 3
using G = Nv4S<Shape<_128, _128, _128>, Ping, EpiAuto>;
#elif NV4S_TILE == 4
using G = Nv4S<Shape<_128, _128, _256>, Ping, EpiAuto>;
#elif NV4S_TILE == 8
using G = Nv4S<Shape<_256, _128, _128>, Coop, EpiAuto>;
#else
#error "NV4S_TILE: 1 2 3 4 8"
#endif

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
extern "C" int CAT(nv4s_run_, NV4S_TILE)(const void* A, const void* B, const void* SFA, const void* SFB,
                                         const float* alpha, void* Y, float* amax, int M, int N, int K, int ldy,
                                         void* wsp, size_t wsb, cudaStream_t st, int raster, int swz) {
  return G::run(A, B, SFA, SFB, alpha, Y, amax, M, N, K, ldy, wsp, wsb, st, raster, swz);
}
extern "C" size_t CAT(nv4s_ws_, NV4S_TILE)(int M, int N, int K, int raster, int swz) {
  return G::sched_ok(raster, swz) ? G::ws(M, N, K, raster, swz) : 0;
}
