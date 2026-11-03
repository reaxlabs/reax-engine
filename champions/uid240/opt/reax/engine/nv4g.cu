// SPDX-License-Identifier: Apache-2.0
// nv4g.cu (rgh lane Q): dense NVFP4 GEMM for sm_120a (RTX PRO 6000 Blackwell), built on NVIDIA CUTLASS 4.x (BSD-3)
// sm120 block-scaled tensor-op collective (mma.sync kind::mxf4nvf4.block_scale, e2m1 x e2m1, ue4m3 scales per 16 K,
// fp32 accumulation).  D[M,N] (bf16, row-major, leading dim ldd) = alpha * sum_k A[m,k] * B[n,k]
//   A: [M, K] e2m1 packed 2 per byte, row-major (K contiguous)      SFA: ue4m3 [ceil(M/128)*128, K/16] in the 128x4 swizzle
//   B: [N, K] e2m1 packed 2 per byte, row-major (= column-major KxN) SFB: same for N
//   alpha: device fp32 pointer (product of the two per-tensor global scales)
// The swizzled scale layout is CUTLASS Sm1xxBlkScaledConfig = cuBLASLt VEC16_UE4M3 = torch SWIZZLE_32_4_4 (fp4.to_blocked).
// C ABI (ctypes): nv4_gemm(variant, A, B, SFA, SFB, alpha, D, M, N, K, ldd, workspace, ws_bytes, stream) -> 0 ok
//                 nv4_ws(variant, M, N, K) -> workspace bytes
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

#ifndef NV4_VARIANT
#define NV4_VARIANT 0
#endif

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

using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using Ping = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using EpiAuto = cutlass::epilogue::collective::EpilogueScheduleAuto;
using MainAuto = cutlass::gemm::collective::KernelScheduleAuto;

// one variant per translation unit (NV4_VARIANT), linked into one .so; nv4_dispatch.cu routes by number
#if NV4_VARIANT == 0
using G = Nv4<Shape<_128, _128, _128>, MainAuto, EpiAuto>;
#elif NV4_VARIANT == 1
using G = Nv4<Shape<_128, _128, _256>, Coop, EpiAuto>;
#elif NV4_VARIANT == 2
using G = Nv4<Shape<_128, _256, _128>, Coop, EpiAuto>;
#elif NV4_VARIANT == 3
using G = Nv4<Shape<_128, _128, _128>, Ping, EpiAuto>;
#elif NV4_VARIANT == 4
using G = Nv4<Shape<_128, _128, _256>, Ping, EpiAuto>;
#elif NV4_VARIANT == 5
using G = Nv4<Shape<_128, _64, _256>, Ping, EpiAuto>;
#endif

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
extern "C" int CAT(nv4_run_, NV4_VARIANT)(const void* A, const void* B, const void* SFA, const void* SFB,
                                          const float* alpha, void* D, int M, int N, int K, int ldd, void* wsp,
                                          size_t wsb, cudaStream_t st) {
  return G::run(A, B, SFA, SFB, alpha, D, M, N, K, ldd, wsp, wsb, st);
}
extern "C" size_t CAT(nv4_ws_, NV4_VARIANT)(int M, int N, int K) { return G::ws(M, N, K); }
