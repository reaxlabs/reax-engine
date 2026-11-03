// SPDX-License-Identifier: Apache-2.0
// blink_gemm: bf16 GEMM C[M,N] = A[M,K] * W[N,K]^T with fp32 accumulation, built on CUTLASS 2.x (BSD-3) device GEMM
// for SM80-class tensor cores (mma.sync m16n8k16), no split-k and no sliced-k: every output element accumulates its
// K products in 16-wide MMA steps in increasing k order from zero, the same chain as Triton's tl.dot loop, so the
// result can be bit-identical to vLLM's batch-invariant matmul (checked by the caller before use).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>

template <int TBM, int TBN, int TBK, int WM, int WN, int STAGES>
struct Cfg {
  using Epi = cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, float, float,
                                                           cutlass::epilogue::thread::ScaleType::Nothing>;
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::bfloat16_t, cutlass::layout::RowMajor,
      cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
      cutlass::bfloat16_t, cutlass::layout::RowMajor,
      float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TBM, TBN, TBK>, cutlass::gemm::GemmShape<WM, WN, TBK>,
      cutlass::gemm::GemmShape<16, 8, 16>, Epi,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, STAGES>;
};

template <typename G>
static void run(const torch::Tensor& a, const torch::Tensor& w, torch::Tensor& c) {
  int M = a.size(0), K = a.size(1), N = w.size(0);
  typename G::Arguments args({M, N, K},
                             {reinterpret_cast<cutlass::bfloat16_t*>(a.data_ptr()), (int64_t)a.stride(0)},
                             {reinterpret_cast<cutlass::bfloat16_t*>(w.data_ptr()), (int64_t)w.stride(0)},
                             {reinterpret_cast<cutlass::bfloat16_t*>(c.data_ptr()), (int64_t)c.stride(0)},
                             {reinterpret_cast<cutlass::bfloat16_t*>(c.data_ptr()), (int64_t)c.stride(0)},
                             {});
  G gemm;
  auto st = gemm.can_implement(args);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "blink_gemm: config cannot implement this problem");
  st = gemm.initialize(args, nullptr, c10::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(st == cutlass::Status::kSuccess, "blink_gemm: initialize failed");
  st = gemm(c10::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(st == cutlass::Status::kSuccess, "blink_gemm: launch failed");
}

torch::Tensor mm(torch::Tensor a, torch::Tensor w, int64_t cfg) {
  TORCH_CHECK(a.is_cuda() && w.is_cuda() && a.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16);
  TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && a.size(1) == w.size(1) && a.stride(1) == 1 && w.stride(1) == 1);
  auto c = torch::empty({a.size(0), w.size(0)}, a.options());
  switch (cfg) {
    case 0: run<Cfg<128, 128, 32, 64, 64, 4>::Gemm>(a, w, c); break;
    case 1: run<Cfg<128, 128, 64, 64, 64, 3>::Gemm>(a, w, c); break;
    case 2: run<Cfg<128, 256, 32, 64, 64, 3>::Gemm>(a, w, c); break;
    case 3: run<Cfg<256, 128, 32, 64, 64, 3>::Gemm>(a, w, c); break;
    case 4: run<Cfg<64, 128, 32, 32, 64, 5>::Gemm>(a, w, c); break;
    case 5: run<Cfg<128, 64, 32, 64, 32, 5>::Gemm>(a, w, c); break;
    case 6: run<Cfg<64, 64, 64, 32, 32, 5>::Gemm>(a, w, c); break;
    case 7: run<Cfg<128, 128, 32, 64, 64, 5>::Gemm>(a, w, c); break;
    case 8: run<Cfg<64, 64, 32, 32, 32, 6>::Gemm>(a, w, c); break;
    case 9: run<Cfg<32, 64, 64, 32, 32, 4>::Gemm>(a, w, c); break;
    case 10: run<Cfg<64, 128, 64, 32, 64, 4>::Gemm>(a, w, c); break;
    case 11: run<Cfg<128, 64, 64, 64, 32, 4>::Gemm>(a, w, c); break;
    case 12: run<Cfg<64, 64, 64, 32, 32, 3>::Gemm>(a, w, c); break;
    case 13: run<Cfg<32, 128, 64, 32, 64, 3>::Gemm>(a, w, c); break;
    default: TORCH_CHECK(false, "blink_gemm: unknown config");
  }
  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mm", &mm, "bf16 GEMM a @ w.T (CUTLASS sm80 tensor ops)"); }
