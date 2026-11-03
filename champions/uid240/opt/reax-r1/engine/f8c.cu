// SPDX-License-Identifier: Apache-2.0
// f8c.cu (kai lever R5-1, 10-28): FP8 e4m3 x e4m3 GEMMs for sm_120a (RTX PRO 6000 Blackwell) on the NVIDIA CUTLASS 4.5.0
// sm120 dense collective (TMA loads + warp-specialized cooperative / ping-pong mainloop, mma.sync kind::f8f6f4
// m16n8k32 e4m3 x e4m3 -> f32, cluster 1x1x1), with rgh's f8g.cu semantics:
//   kind 0 rowwise:  out[r, n] = bf16((acc[r, n] * sa[r]) * sw[n])   (EVT: Sm90ColBroadcast sa, Sm90RowBroadcast sw)
//   kind 1 unscaled: out[r, n] = bf16(acc[r, n])                     (LinearCombination alpha 1, beta 0, no C)
// A [M, K] e4m3 row-major (lda), B [N, K] e4m3 row-major (= Linear.w8.t() storage, ldb = K), D [M, N] bf16 (ldd).
// The SwiGLU gate/up GEMM = kind 1 into [M, 2H] (bf16(acc), exactly the value f8g's _smqs epilogue rounds first) followed
// by f8c_swq (f8c_dispatch.cu): rgh's smqs1 math + f8_rowquant in one pass per row.
// One (variant, kind) per translation unit (F8C_VARIANT, F8C_KIND); f8c_dispatch.cu routes by number.
// C ABI: f8c_run_<v>_<k>(A, lda, B, ldb, sa, sw, D, ldd, M, N, K, swizzle, raster, ws, ws_bytes, stream) -> 0 ok
//        f8c_ws_<v>_<k>(M, N, K) -> workspace bytes
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#ifndef F8C_VARIANT
#define F8C_VARIANT 0
#endif
#ifndef F8C_KIND
#define F8C_KIND 0
#endif

namespace f8c {
using EA = cutlass::float_e4m3_t;
using EB = cutlass::float_e4m3_t;
using ED = cutlass::bfloat16_t;
using Acc = float;
using LA = cutlass::layout::RowMajor;
using LB = cutlass::layout::ColumnMajor;
using LD = cutlass::layout::RowMajor;
constexpr int AlA = 16, AlB = 16, AlD = 8;
using Cl = Shape<_1, _1, _1>;
namespace fu = cutlass::epilogue::fusion;

// kind 0: bf16((acc * sa[m]) * sw[n]), fp32 multiplies left to right (f8g.cu tile8 MODE 0: acc * rs * w)
template <class TS>
using RowwiseEVT = fu::Sm90EVT<
    fu::Sm90Compute<cutlass::multiplies, ED, float, cutlass::FloatRoundStyle::round_to_nearest>,
    fu::Sm90EVT<fu::Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
                 fu::Sm90AccFetch, fu::Sm90ColBroadcast<0, TS, float, float, Stride<_1, _0, int64_t>>>,
    fu::Sm90RowBroadcast<0, TS, float, float, Stride<_0, _1, int64_t>>>;
// kind 1: bf16(1 * acc) (beta 0, no source)
using Unscaled = fu::LinearCombination<ED, float, void, float>;

template <class TS, class MainSched, class EpiSched, int KIND>
struct G8 {
  using Fusion = cute::conditional_t<KIND == 0, RowwiseEVT<TS>, Unscaled>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TS, Cl, cutlass::epilogue::collective::EpilogueTileAuto, Acc,
      float, void, LD, AlD, ED, LD, AlD, EpiSched, Fusion>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, EA, LA, AlA, EB, LB, AlB, Acc, TS, Cl,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainSched>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
                                                      void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using StrideD = typename Kernel::StrideD;

  static int nsm() {
    static int n = 0;
    if (n == 0) {
      int d = 0;
      cudaGetDevice(&d);
      cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d);
    }
    return n;
  }

  static typename Gemm::Arguments args(const void* A, int lda, const void* B, int ldb, const float* sa, const float* sw,
                                       void* D, int ldd, int M, int N, int K, int swz, int raster) {
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, {M, lda, 1});
    StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, {N, ldb, 1});
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, {M, ldd, 1});
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, {M, ldd, 1});
    cutlass::KernelHardwareInfo hw;
    hw.device_id = 0;
    hw.sm_count = nsm();
    typename Gemm::Arguments a{cutlass::gemm::GemmUniversalMode::kGemm,
                               {M, N, K, 1},
                               {static_cast<const EA*>(A), sA, static_cast<const EB*>(B), sB},
                               {{}, nullptr, sC, static_cast<ED*>(D), sD},
                               hw};
    if constexpr (KIND == 0) {
      a.epilogue.thread = {
          {{}, {sa, 0.f, {_1{}, _0{}, int64_t(0)}}, {}},   // acc * sa[m]
          {sw, 0.f, {_0{}, _1{}, int64_t(0)}},            // * sw[n]
          {}};
    } else {
      a.epilogue.thread.alpha = 1.0f;
      a.epilogue.thread.beta = 0.0f;
    }
    a.scheduler.max_swizzle_size = swz;
    using RO = cutlass::gemm::kernel::detail::RasterOrderOptions;
    a.scheduler.raster_order = raster == 1 ? RO::AlongM : raster == 2 ? RO::AlongN : RO::Heuristic;
    return a;
  }

  static size_t ws(int M, int N, int K) {
    auto a = args(nullptr, K, nullptr, K, nullptr, nullptr, nullptr, N, M, N, K, 1, 0);
    return Gemm::get_workspace_size(a);
  }

  static int run(const void* A, int lda, const void* B, int ldb, const float* sa, const float* sw, void* D, int ldd,
                 int M, int N, int K, int swz, int raster, void* wsp, size_t wsb, cudaStream_t st) {
    auto a = args(A, lda, B, ldb, sa, sw, D, ldd, M, N, K, swz, raster);
    Gemm g;
    if (g.can_implement(a) != cutlass::Status::kSuccess) return 2;
    if (Gemm::get_workspace_size(a) > wsb) return 3;
    if (g.initialize(a, wsp, st) != cutlass::Status::kSuccess) return 4;
    if (g.run(st) != cutlass::Status::kSuccess) return 5;
    return (int)cudaGetLastError() ? 6 : 0;
  }
};

using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using Ping = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using EpiAuto = cutlass::epilogue::collective::EpilogueScheduleAuto;
}  // namespace f8c

// variants (tile M x N x K bytes, mainloop schedule); stages = as many as fit next to the epilogue (StageCountAutoCarveout)
#if F8C_VARIANT == 0
using T = f8c::G8<Shape<_128, _128, _128>, f8c::Coop, f8c::EpiAuto, F8C_KIND>;
#elif F8C_VARIANT == 1
using T = f8c::G8<Shape<_128, _256, _64>, f8c::Coop, f8c::EpiAuto, F8C_KIND>;
#elif F8C_VARIANT == 2
using T = f8c::G8<Shape<_256, _128, _64>, f8c::Coop, f8c::EpiAuto, F8C_KIND>;
#elif F8C_VARIANT == 3
using T = f8c::G8<Shape<_128, _128, _128>, f8c::Ping, f8c::EpiAuto, F8C_KIND>;
#elif F8C_VARIANT == 4
using T = f8c::G8<Shape<_128, _128, _64>, f8c::Ping, f8c::EpiAuto, F8C_KIND>;
#elif F8C_VARIANT == 5
using T = f8c::G8<Shape<_64, _128, _128>, f8c::Ping, f8c::EpiAuto, F8C_KIND>;
#elif F8C_VARIANT == 6
using T = f8c::G8<Shape<_128, _64, _128>, f8c::Ping, f8c::EpiAuto, F8C_KIND>;
#elif F8C_VARIANT == 7
using T = f8c::G8<Shape<_128, _128, _64>, f8c::Coop, f8c::EpiAuto, F8C_KIND>;
#endif

#define CAT4(a, b, c, d) a##b##c##d
#define NAME(p, v, k) CAT4(p, v, _, k)
extern "C" int NAME(f8c_run_, F8C_VARIANT, F8C_KIND)(const void* A, int lda, const void* B, int ldb, const float* sa,
                                                    const float* sw, void* D, int ldd, int M, int N, int K, int swz,
                                                    int raster, void* wsp, size_t wsb, cudaStream_t st) {
  return T::run(A, lda, B, ldb, sa, sw, D, ldd, M, N, K, swz, raster, wsp, wsb, st);
}
extern "C" size_t NAME(f8c_ws_, F8C_VARIANT, F8C_KIND)(int M, int N, int K) { return T::ws(M, N, K); }
extern "C" int NAME(f8c_stages_, F8C_VARIANT, F8C_KIND)() { return T::CollectiveMainloop::DispatchPolicy::Stages; }
