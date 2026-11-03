// SPDX-License-Identifier: Apache-2.0
// blink_rgemm: bf16 x bf16 -> fp32 GEMM C[M,N] = A[M,K] * W[N,K]^T for Blink's MoE router (N = 128 experts), built on
// CUTLASS 2.x (BSD-3) device GEMM for SM80-class tensor cores (mma.sync m16n8k16), no split-K and no sliced-K: every
// output element accumulates its K products in 16-wide MMA steps in increasing k order from zero in fp32 and is stored
// as fp32 - the same chain as cuBLAS's tensor-op kernels without split-K (checked bit for bit before use).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>

template <int TBM, int TBN, int TBK, int WM, int WN, int STAGES>
struct RCfg {
  using Epi = cutlass::epilogue::thread::LinearCombination<float, 4, float, float,
                                                           cutlass::epilogue::thread::ScaleType::Nothing>;
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::bfloat16_t, cutlass::layout::RowMajor,
      cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
      float, cutlass::layout::RowMajor,
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
                             {c.data_ptr<float>(), (int64_t)c.stride(0)},
                             {c.data_ptr<float>(), (int64_t)c.stride(0)},
                             {});
  G gemm;
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "blink_rgemm: config cannot implement this problem");
  cudaStream_t st = c10::cuda::getCurrentCUDAStream().stream();
  TORCH_CHECK(gemm.initialize(args, nullptr, st) == cutlass::Status::kSuccess, "blink_rgemm: initialize failed");
  TORCH_CHECK(gemm(st) == cutlass::Status::kSuccess, "blink_rgemm: launch failed");
}

// out = a @ w.T in fp32 (out: [M, N] fp32, row stride N)
void mm_f32_out(torch::Tensor a, torch::Tensor w, torch::Tensor out, int64_t cfg) {
  TORCH_CHECK(a.is_cuda() && w.is_cuda() && a.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16);
  TORCH_CHECK(out.scalar_type() == at::kFloat && out.dim() == 2 && out.stride(1) == 1);
  TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && a.size(1) == w.size(1) && a.stride(1) == 1 && w.stride(1) == 1);
  switch (cfg) {
    case 0: run<RCfg<32, 64, 64, 32, 32, 4>::Gemm>(a, w, out); break;
    case 1: run<RCfg<64, 64, 64, 32, 32, 4>::Gemm>(a, w, out); break;
    case 2: run<RCfg<32, 128, 64, 32, 64, 3>::Gemm>(a, w, out); break;
    case 3: run<RCfg<16, 64, 64, 16, 32, 4>::Gemm>(a, w, out); break;
    case 4: run<RCfg<64, 128, 32, 32, 64, 4>::Gemm>(a, w, out); break;
    case 5: run<RCfg<32, 32, 64, 16, 16, 4>::Gemm>(a, w, out); break;
    default: TORCH_CHECK(false, "blink_rgemm: unknown config");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mm_f32_out", &mm_f32_out, "bf16 x bf16 -> fp32 GEMM out = a @ w.T (CUTLASS sm80 tensor ops)");
}
