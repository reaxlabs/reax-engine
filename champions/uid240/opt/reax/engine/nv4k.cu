// SPDX-License-Identifier: Apache-2.0
// nv4k.cu (kai lever K, 10-28): stream-K variants of rgh's dense NVFP4 GEMM (nv4g.cu) for sm_120a (RTX PRO 6000
// Blackwell): the same CUTLASS 4.5.0 sm120 block-scaled collective (mma.sync kind::mxf4nvf4.block_scale, e2m1 x e2m1,
// ue4m3 scale per 16 K, fp32 accumulation, D bf16 = alpha * A . B^T, alpha from a device pointer), cooperative kernel,
// with CUTLASS's StreamKScheduler (sm120 -> PersistentTileSchedulerSm100StreamK) in Deterministic reduction mode instead
// of the default data-parallel persistent scheduler. Why: at the c8 step sizes the N = 5120 outputs (down K 17408, out /
// o K 6144) are 2-3 waves of 128x128 tiles on 188 SMs; stream-K evens the last wave.
// NOT bit-identical to nv4g: a tile whose k range is split is summed from fp32 partials (fixed order per M: deterministic
// run to run, but the split depends on M, so batch composition changes the rounding of the split tiles; NVFP4 activations
// with a per-call power-of-two global scale are batch-dependent already). Spends a little distance; measure.
// Variants (nv4k_gemm v): 6 = 128x128x128 coop stream-K, 7 = 128x256x128 coop stream-K, 8 = 128x128x256 coop stream-K
// (same tiles as nv4g variants 1 / 2 with K 128 or 256). One variant per translation unit (NV4K_VARIANT), like nv4g.cu.
// C ABI: nv4k_gemm(v, A, B, SFA, SFB, alpha, D, M, N, K, ldd, workspace, ws_bytes, stream) -> 0 ok (nv4g's codes)
//        nv4k_ws(v, M, N, K) -> workspace bytes (stream-K partials + tile counters; the caller zeroes nothing: CUTLASS
//        initialises its counters in initialize())
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#ifndef NV4K_VARIANT
#define NV4K_VARIANT 6
#endif

template <class TileShape>
struct Nv4K {
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
  using MainSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpiSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
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
                                                          CollectiveEpilogue, cutlass::gemm::StreamKScheduler>;
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
    using SKP = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams;
    using RM = SKP::ReductionMode;
    using DM = SKP::DecompositionMode;
    a.scheduler.reduction_mode = RM::Deterministic;
    a.scheduler.decomposition_mode = DM::StreamK;
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

#if NV4K_VARIANT == 6
using G = Nv4K<Shape<_128, _128, _128>>;
#elif NV4K_VARIANT == 7
using G = Nv4K<Shape<_128, _256, _128>>;
#elif NV4K_VARIANT == 8
using G = Nv4K<Shape<_128, _128, _256>>;
#endif

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
extern "C" int CAT(nv4k_run_, NV4K_VARIANT)(const void* A, const void* B, const void* SFA, const void* SFB,
                                            const float* alpha, void* D, int M, int N, int K, int ldd, void* wsp,
                                            size_t wsb, cudaStream_t st) {
  return G::run(A, B, SFA, SFB, alpha, D, M, N, K, ldd, wsp, wsb, st);
}
extern "C" size_t CAT(nv4k_ws_, NV4K_VARIANT)(int M, int N, int K) { return G::ws(M, N, K); }
