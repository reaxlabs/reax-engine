// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 team ruj (Jeffan engine lever G, on pjs's E62 Jeffan engine)
// Torch ops of jef_fp8gemm (loaded with torch.ops.load_library):
//   jef::fp8_mm(Tensor! out, Tensor a, Tensor b, Tensor scale_a, Tensor scale_b) -> ()
//       same contract as vLLM's _C::cutlass_scaled_mm(out, a, b, scale_a, scale_b, None) for e4m3 inputs, bf16 out,
//       per-token scale_a [M] and per-tensor scale_b [1]; the configuration comes from the table (N, K, M) below
//   jef::fp8_mm_cfg(Tensor! out, Tensor a, Tensor b, Tensor scale_a, Tensor scale_b, int cfg) -> ()   (tests)
//   jef::fp8_set_table(int[] table) -> ()   flattened (N, K, M_max, cfg) rows, first match wins; no match = cfg 0
//   jef::fp8_cfg_names() -> str
#include "jef_fp8_common.cuh"
#include <torch/library.h>
#include <mutex>
#include <vector>

void jef_fp8_run_cfg0(torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
void jef_fp8_run_cfg1(torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
void jef_fp8_run_cfg2(torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
void jef_fp8_run_cfg3(torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
void jef_fp8_run_cfg4(torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
void jef_fp8_run_cfg5(torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
void jef_fp8_run_cfg6(torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
void jef_fp8_run_cfg7(torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
void jef_fp8_run_cfg8(torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
void jef_fp8_run_cfg9(torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
void jef_fp8_run_cfg10(torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
void jef_fp8_run_cfg11(torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
void jef_fp8_run_cfg12(torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
void jef_fp8_run_cfg13(torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);

namespace {
using RunFn = void (*)(torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
constexpr RunFn kCfgs[] = {
    jef_fp8_run_cfg0,
    jef_fp8_run_cfg1,
    jef_fp8_run_cfg2,
    jef_fp8_run_cfg3,
    jef_fp8_run_cfg4,
    jef_fp8_run_cfg5,
    jef_fp8_run_cfg6,
    jef_fp8_run_cfg7,
    jef_fp8_run_cfg8,
    jef_fp8_run_cfg9,
    jef_fp8_run_cfg10,
    jef_fp8_run_cfg11,
    jef_fp8_run_cfg12,
    jef_fp8_run_cfg13};
constexpr int kNumCfgs = sizeof(kCfgs) / sizeof(kCfgs[0]);
const char* kNames = "64x64x128p, 128x64x128p, 64x128x128p, 128x128x64p, 128x128x128c, 128x128x64c, 256x128x64c, 128x256x64c, 64x128x64p, 128x64x64p, 128x192x64c, 256x192x64c, 64x192x128p, 64x256x64p";

struct Row { int64_t n, k, m_max, cfg; };
std::vector<Row> g_table;
std::mutex g_mu;

void check(const torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sa,
           const torch::Tensor& sb) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && out.dim() == 2, "jef_fp8gemm: 2-D tensors only");
  TORCH_CHECK(a.scalar_type() == at::ScalarType::Float8_e4m3fn && b.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "jef_fp8gemm: e4m3 inputs");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16, "jef_fp8gemm: bf16 output");
  TORCH_CHECK(a.size(1) == b.size(0) && out.size(0) == a.size(0) && out.size(1) == b.size(1), "jef_fp8gemm: shapes");
  TORCH_CHECK(a.stride(1) == 1 && b.stride(0) == 1 && out.stride(1) == 1, "jef_fp8gemm: a row-major, b column-major");
  TORCH_CHECK(sa.scalar_type() == at::ScalarType::Float && sb.scalar_type() == at::ScalarType::Float, "jef_fp8gemm: fp32 scales");
  TORCH_CHECK(sa.numel() == a.size(0) && sa.is_contiguous(), "jef_fp8gemm: per-token scale_a");
  TORCH_CHECK(sb.numel() == 1, "jef_fp8gemm: per-tensor scale_b");
}

void fp8_mm_cfg(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sa,
                const torch::Tensor& sb, int64_t cfg) {
  check(out, a, b, sa, sb);
  TORCH_CHECK(cfg >= 0 && cfg < kNumCfgs, "jef_fp8gemm: bad cfg");
  if (a.size(0) == 0) return;
  const c10::cuda::CUDAGuard guard(a.device());
  kCfgs[cfg](out, a, b, sa, sb);
}

void fp8_mm(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sa,
            const torch::Tensor& sb) {
  int64_t m = a.size(0), k = a.size(1), n = b.size(1), cfg = 0;
  for (const auto& r : g_table) {
    if (r.n == n && r.k == k && m <= r.m_max) { cfg = r.cfg; break; }
  }
  fp8_mm_cfg(out, a, b, sa, sb, cfg);
}

void fp8_set_table(c10::IntArrayRef t) {
  TORCH_CHECK(t.size() % 4 == 0, "jef_fp8gemm: table rows are (N, K, M_max, cfg)");
  std::vector<Row> rows;
  for (size_t i = 0; i < t.size(); i += 4) {
    TORCH_CHECK(t[i + 3] >= 0 && t[i + 3] < kNumCfgs, "jef_fp8gemm: bad cfg in table");
    rows.push_back({t[i], t[i + 1], t[i + 2], t[i + 3]});
  }
  std::lock_guard<std::mutex> lk(g_mu);
  g_table = rows;
}

std::string fp8_cfg_names() { return kNames; }
}  // namespace

TORCH_LIBRARY(jef, m) {
  m.def("fp8_mm(Tensor! out, Tensor a, Tensor b, Tensor scale_a, Tensor scale_b) -> ()");
  m.def("fp8_mm_cfg(Tensor! out, Tensor a, Tensor b, Tensor scale_a, Tensor scale_b, int cfg) -> ()");
  m.def("fp8_set_table(int[] table) -> ()", &fp8_set_table);
  m.def("fp8_cfg_names() -> str", &fp8_cfg_names);
}
TORCH_LIBRARY_IMPL(jef, CUDA, m) {
  m.impl("fp8_mm", &fp8_mm);
  m.impl("fp8_mm_cfg", &fp8_mm_cfg);
}
