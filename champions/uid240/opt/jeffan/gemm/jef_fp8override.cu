// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 team ruj (Jeffan engine lever G, on pjs's E62 Jeffan engine)
// jef_fp8override (jef_fp8override.so): routes vLLM's _C::cutlass_scaled_mm (CUDA) to jef::fp8_mm (jef_fp8gemm.so, loaded first).
// The compiled model graph is unchanged (it still calls _C::cutlass_scaled_mm); only the kernel behind the op is ours.
// Loaded by the Jeffan engine core only when JEF_GEMM=1. jef::fp8_mm accepts exactly the engine's case (e4m3 x e4m3,
// per-token fp32 scale_a, per-tensor fp32 scale_b, bf16 out) and refuses anything else, as does a bias here.
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace {
void jef_cutlass_scaled_mm(at::Tensor& out, const at::Tensor& a, const at::Tensor& b, const at::Tensor& a_scales,
                           const at::Tensor& b_scales, const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(!bias.has_value() || !bias->defined(), "jef_fp8override: bias is not supported");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("jef::fp8_mm", "")
                       .typed<void(at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                   const at::Tensor&)>();
  op.call(out, a, b, a_scales, b_scales);
}
}  // namespace

TORCH_LIBRARY_IMPL(_C, CUDA, m) { m.impl("cutlass_scaled_mm", TORCH_FN(jef_cutlass_scaled_mm)); }
