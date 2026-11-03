// SPDX-License-Identifier: Apache-2.0
// nv4t.cu (kai lever B, 10-28): rgh's dense NVFP4 GEMM (nv4g.cu, CUTLASS 4.5.0 sm120 block-scaled collective, mma.sync
// kind::mxf4nvf4.block_scale, e2m1 x e2m1, ue4m3 scale per 16 K, fp32 accumulation, D bf16 = alpha * A . B^T) with more
// tile / schedule configurations and the persistent scheduler's raster order and swizzle as run-time arguments, for a
// measured per-(N, K, M range) table (nv4t.py, MAM_NV4TAB).
// EXACT (bit-identical to nv4g for every configuration here): struct Nv4 below is nv4g.cu's text unchanged (tests compare
// it). In this collective every output element is ONE fp32 accumulator chain over the 64-wide k blocks 0, 1, 2, ... in
// order, started from 0 (clear(accum), then cute::gemm per k_block, k tiles in order; tile K 128 / 256 = 2 / 4 k blocks
// of the same chain), one mxf4nvf4 mma per k block on the same operands whatever the tile M x N, the warp layout
// (cooperative 4x2 / ping-pong 2x2 warps) or the CTA that owns the tile; the epilogue is alpha * acc -> bf16 (beta 0,
// C never read). Raster order and swizzle only change which CTA computes which output tile and when. Data-parallel
// persistent scheduler only (no split-K / stream-K: those change the k order, see nv4k.cu).
// Tiles (NV4T_TILE, one translation unit each): 0-5 = nv4g variants 0-5 (128x128x128 auto, 128x128x256 coop,
// 128x256x128 coop, 128x128x128 ping-pong, 128x128x256 ping-pong, 128x64x256 ping-pong); new: 6 = 128x64x128 ping-pong,
// 7 = 128x64x128 coop, 8 = 256x128x128 coop, 9 = 128x64x256 coop, 10 = 256x64x128 coop.
// C ABI: nv4t_gemm(tile, raster, swizzle, A, B, SFA, SFB, alpha, D, M, N, K, ldd, workspace, ws_bytes, stream) -> 0 ok
//        (nv4g's codes: 1 unknown tile, 2 CUTLASS refused, 3 workspace too small, 4 init, 5 launch; 6 bad raster/swizzle)
//        raster 0 = CUTLASS heuristic (nv4g's default), 1 = along M, 2 = along N; swizzle 1 (nv4g's default), 2, 4, 8
//        nv4t_ws(tile, raster, swizzle, M, N, K) -> workspace bytes
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#ifndef NV4T_TILE
#define NV4T_TILE 0
#endif

// ---- nv4g.cu's struct Nv4, verbatim ----
template <class TileShape, class MainSchedule, class EpiSchedule>
struct Nv4 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementC = cutlass::bfloat16_t;
  using ElementD = cutlass::bfloat16_t;
  using ElementAcc = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignA = 32, AlignB = 32, AlignC = 8;
  using ClusterShape = Shape<_1, _1, _1>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementAcc, ElementC, LayoutC, AlignC, ElementD,
      LayoutC, AlignC, EpiSchedule>::CollectiveOp;
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
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using Cfg = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  static typename Gemm::Arguments args(const void* A, const void* B, const void* SFA, const void* SFB,
                                       const float* alpha, void* D, int M, int N, int K, int ldd) {
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, {M, ldd, 1});
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, {M, ldd, 1});
    LayoutSFA lA = Cfg::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    LayoutSFB lB = Cfg::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
    typename Gemm::Arguments a{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {static_cast<const typename ElementA::DataType*>(A), sA, static_cast<const typename ElementB::DataType*>(B), sB,
         static_cast<const typename ElementA::ScaleFactorType*>(SFA), lA,
         static_cast<const typename ElementB::ScaleFactorType*>(SFB), lB},
        {{1.0f, 0.0f}, static_cast<const ElementC*>(D), sC, static_cast<ElementD*>(D), sD}};
    a.epilogue.thread.alpha_ptr = alpha;
    a.epilogue.thread.beta = 0.0f;
    return a;
  }

  static size_t ws(int M, int N, int K) {
    auto a = args(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, N);
    return Gemm::get_workspace_size(a);
  }

  static int run(const void* A, const void* B, const void* SFA, const void* SFB, const float* alpha, void* D, int M,
                 int N, int K, int ldd, void* wsp, size_t wsb, cudaStream_t st) {
    auto a = args(A, B, SFA, SFB, alpha, D, M, N, K, ldd);
    Gemm g;
    if (g.can_implement(a) != cutlass::Status::kSuccess) return 2;
    if (Gemm::get_workspace_size(a) > wsb) return 3;
    if (g.initialize(a, wsp, st) != cutlass::Status::kSuccess) return 4;
    if (g.run(st) != cutlass::Status::kSuccess) return 5;
    return 0;
  }
};

// lever B: Nv4 with the persistent tile scheduler's raster order / swizzle set (nothing else differs from nv4g)
template <class TileShape, class MainSchedule, class EpiSchedule>
struct Nv4T : Nv4<TileShape, MainSchedule, EpiSchedule> {
  using Base = Nv4<TileShape, MainSchedule, EpiSchedule>;
  using Gemm = typename Base::Gemm;

  static bool sched_ok(int raster, int swz) {
    return raster >= 0 && raster <= 2 && (swz == 1 || swz == 2 || swz == 4 || swz == 8);
  }

  static typename Gemm::Arguments args(const void* A, const void* B, const void* SFA, const void* SFB,
                                       const float* alpha, void* D, int M, int N, int K, int ldd, int raster, int swz) {
    auto a = Base::args(A, B, SFA, SFB, alpha, D, M, N, K, ldd);
    using RO = decltype(a.scheduler.raster_order);
    a.scheduler.raster_order = raster == 1 ? RO::AlongM : (raster == 2 ? RO::AlongN : RO::Heuristic);
    a.scheduler.max_swizzle_size = swz;
    return a;
  }

  static size_t ws(int M, int N, int K, int raster, int swz) {
    auto a = args(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, N, raster, swz);
    return Gemm::get_workspace_size(a);
  }

  static int run(const void* A, const void* B, const void* SFA, const void* SFB, const float* alpha, void* D, int M,
                 int N, int K, int ldd, void* wsp, size_t wsb, cudaStream_t st, int raster, int swz) {
    if (!sched_ok(raster, swz)) return 6;
    auto a = args(A, B, SFA, SFB, alpha, D, M, N, K, ldd, raster, swz);
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
using MainAuto = cutlass::gemm::collective::KernelScheduleAuto;

#if NV4T_TILE == 0
using G = Nv4T<Shape<_128, _128, _128>, MainAuto, EpiAuto>;
#elif NV4T_TILE == 1
using G = Nv4T<Shape<_128, _128, _256>, Coop, EpiAuto>;
#elif NV4T_TILE == 2
using G = Nv4T<Shape<_128, _256, _128>, Coop, EpiAuto>;
#elif NV4T_TILE == 3
using G = Nv4T<Shape<_128, _128, _128>, Ping, EpiAuto>;
#elif NV4T_TILE == 4
using G = Nv4T<Shape<_128, _128, _256>, Ping, EpiAuto>;
#elif NV4T_TILE == 5
using G = Nv4T<Shape<_128, _64, _256>, Ping, EpiAuto>;
#elif NV4T_TILE == 6
using G = Nv4T<Shape<_128, _64, _128>, Ping, EpiAuto>;
#elif NV4T_TILE == 7
using G = Nv4T<Shape<_128, _64, _128>, Coop, EpiAuto>;
#elif NV4T_TILE == 8
using G = Nv4T<Shape<_256, _128, _128>, Coop, EpiAuto>;
#elif NV4T_TILE == 9
using G = Nv4T<Shape<_128, _64, _256>, Coop, EpiAuto>;
#elif NV4T_TILE == 10
using G = Nv4T<Shape<_256, _64, _128>, Coop, EpiAuto>;
#endif

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
extern "C" int CAT(nv4t_run_, NV4T_TILE)(const void* A, const void* B, const void* SFA, const void* SFB,
                                         const float* alpha, void* D, int M, int N, int K, int ldd, void* wsp,
                                         size_t wsb, cudaStream_t st, int raster, int swz) {
  return G::run(A, B, SFA, SFB, alpha, D, M, N, K, ldd, wsp, wsb, st, raster, swz);
}
extern "C" size_t CAT(nv4t_ws_, NV4T_TILE)(int M, int N, int K, int raster, int swz) {
  return G::sched_ok(raster, swz) ? G::ws(M, N, K, raster, swz) : 0;
}
