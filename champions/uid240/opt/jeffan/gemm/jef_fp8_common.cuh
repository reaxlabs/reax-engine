// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 team ruj (Jeffan engine lever G, on pjs's E62 Jeffan engine)
//
// jef_fp8gemm: FP8 x FP8 -> BF16 GEMM for the online-FP8 dense layers of Jeffan-31b (Gemma-4 31B) on SM120
// (RTX PRO 6000 Blackwell), built on NVIDIA CUTLASS 4.x (BSD-3-Clause) headers.
//
//   out[M,N] = bf16_rn( scale_a[m] * ( scale_b * sum_k a[m,k] * b[k,n] ) )
//
// a: e4m3 [M,K] row-major (per-token dynamic activation quantisation), b: e4m3 [K,N] column-major (the transposed
// [N,K] weight), scale_a: fp32 [M] (per token), scale_b: fp32 [1] (per tensor). This is exactly the arithmetic of
// vLLM's cutlass_scaled_mm on SM120 (c3x ScaledEpilogue: Compute1(ScaleA, Compute0(ScaleB, Acc)), fp32 multiplies,
// round-to-nearest bf16; mainloop MainloopSm120TmaWarpSpecialized with the SM120 16x8x32 e4m3 MMA, fp32
// accumulation over K in increasing order, no split-K). Only the CTA tile shape and the kernel schedule differ per
// configuration: that changes which CTA computes an output element, not how it is computed, so every configuration
// gives the same bits as vLLM's kernel (checked on the GPU against ops.cutlass_scaled_mm, see test_fp8gemm.py).
#pragma once
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

namespace jef {
using namespace cute;

template <class TileShape, class KernelSchedule>
struct Fp8Gemm {
  using ElementAB = cutlass::float_e4m3_t;
  using ElementD = cutlass::bfloat16_t;
  using ElementAcc = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using ClusterShape = Shape<_1, _1, _1>;

  // vLLM c3x ScaledEpilogue, written with the stock CUTLASS visitor nodes
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, float, float, Stride<_1, _0, int64_t>>;
  using ScaleB = cutlass::epilogue::fusion::Sm90ScalarBroadcast<float, Stride<_0, _0, int64_t>>;
  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, float, float,
                                                          cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;
  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementD, float,
                                                          cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT = cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVT0>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, float, void, LayoutD, AlignD, ElementD, LayoutD,
      AlignD, cutlass::epilogue::collective::EpilogueScheduleAuto, EVT>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementAB, LayoutA, AlignAB, ElementAB, LayoutB, AlignAB,
      ElementAcc, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                                          CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <class G>
void run(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sa,
         const torch::Tensor& sb) {
  using Gemm = typename G::Gemm;
  using GemmKernel = typename G::GemmKernel;
  int m = static_cast<int>(a.size(0)), k = static_cast<int>(a.size(1)), n = static_cast<int>(b.size(1));
  using StrideA = Stride<int64_t, Int<1>, int64_t>;
  using StrideB = Stride<int64_t, Int<1>, int64_t>;
  using StrideD = typename GemmKernel::StrideD;
  StrideA a_stride{a.stride(0), Int<1>{}, 0};
  StrideB b_stride{b.stride(1), Int<1>{}, 0};
  StrideD d_stride{out.stride(0), Int<1>{}, Int<0>{}};
  typename GemmKernel::ProblemShape prob{m, n, k, 1};
  typename GemmKernel::MainloopArguments mainloop{static_cast<typename G::ElementAB const*>(a.data_ptr()), a_stride,
                                                  static_cast<typename G::ElementAB const*>(b.data_ptr()), b_stride};
  typename G::EVT::Arguments evt{
      {static_cast<float const*>(sa.data_ptr()), 0.f, {}},                     // ScaleA: per-token column
      {{{0.f}, {static_cast<float const*>(sb.data_ptr())}, {}}, {}, {}},      // Compute0(ScaleB scalar, Acc)
      {}};                                                                      // Compute1
  auto* d_ptr = static_cast<typename G::ElementD*>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue{evt, nullptr, d_stride, d_ptr, d_stride};
  typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, prob, mainloop, epilogue};
  Gemm op;
  TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess, "jef_fp8gemm: problem not supported by this config");
  size_t ws = Gemm::get_workspace_size(args);
  torch::Tensor workspace;
  void* ws_ptr = nullptr;
  if (ws > 0) {
    workspace = torch::empty({static_cast<int64_t>(ws)}, torch::TensorOptions().dtype(torch::kUInt8).device(a.device()));
    ws_ptr = workspace.data_ptr();
  }
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  cutlass::Status st = op.run(args, ws_ptr, stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "jef_fp8gemm: kernel launch failed");
}
}  // namespace jef

#define JEF_FP8_CFG(NAME, TM, TN, TK, SCHED)                                                                       \
  void NAME(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sa,            \
            const torch::Tensor& sb) {                                                                             \
    jef::run<jef::Fp8Gemm<cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>, SCHED>>(out, a, b, sa, sb);     \
  }
