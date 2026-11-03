// SPDX-License-Identifier: Apache-2.0
// nv4s_silu.cuh (kai lever N1, 10-28): the per-element numerics of N1's fused gate|up epilogue (nv4s.cu), shared with the
// start-up probe kernel (nv4s_dispatch.cu: nv4s_silu_probe) so the check exercises the very function the epilogue runs.
//
//   gu_round(alpha, acc) = bf16_rn(fp32_rn(alpha * acc) + 0.0f)
//       = rgh's nv4g epilogue: CUTLASS LinearCombination D = bf16(beta * C + alpha * acc) with beta = 0 and C cleared to 0
//         (Sm90TreeVisitor's homogeneous_multiply_add specialisation: cute::clear(tCrC) when C is not loaded), i.e. the
//         fp32 product, plus +0 (turns a -0 product into +0), rounded to bf16 by NumericArrayConverter round_to_nearest.
//   silu_mul(g, u) = fp32 g / (1 + exp(-g)) * u (one bf16 rounding by the caller): the compiled K.silu_mul
//         (Inductor's silu decomposition), checked bit-for-bit against K.silu_mul at start for every bf16 g.
//
// N1_SILU = 1 (the shipped build) is the compiled K.silu_mul's PTX instruction sequence written out as inline PTX, so
// neither nvcc nor ptxas can contract or reorder it differently; N1_SILU = 0 (nvcc's expf and IEEE add / divide /
// multiply) is kept as the counter-example the start-up probe rejects (it differs from K.silu_mul: Inductor's code
// contracts 2^k * ex2(r) + 1 into one fma and divides with div.full.f32). gu_round uses explicitly rounded intrinsics
// (_rn) so nvcc cannot contract its multiply into the add.
#pragma once
#include <cuda_bf16.h>

#ifndef N1_SILU
#define N1_SILU 1
#endif

namespace n1 {

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float gu_round(float alpha, float acc) {
  return bf16r(__fadd_rn(__fmul_rn(alpha, acc), 0.0f));
}

#if N1_SILU == 0
// libdevice __nv_expf (nvcc's expf without -ftz / fast math), then IEEE add / divide / multiply
__device__ __forceinline__ float silu_mul(float g, float u) {
  float e = expf(-g);
  return __fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, e)), u);
}
#elif N1_SILU == 1
// the compiled K.silu_mul's own instruction sequence (Inductor's triton_poi_fused_mul_silu_slice_0 in the engine image,
// torch 2.14 / Triton 3.8 PTX, dumped by tools/n1_ptxdump.py): libdevice __nv_expf as Triton links it (nvvm-reflect
// __CUDA_FTZ = 1: ftz forms), LLVM's contraction of the expf's final scale multiply with the "+ 1" into one fma, the
// division as div.full.f32 (Triton's default fp32 division), the product as mul.f32 (round to nearest)
__device__ __forceinline__ float silu_mul(float g, float u) {
  float y;
  asm("{\n\t"
      ".reg .f32 n, a, b, c, d, e, f, h, x, s, den, q;\n\t"
      ".reg .b32 cb, sb;\n\t"
      "neg.f32 n, %1;\n\t"
      "fma.rn.ftz.f32 a, n, 0f3BBB989D, 0f3F000000;\n\t"
      "cvt.ftz.sat.f32.f32 b, a;\n\t"
      "fma.rm.ftz.f32 c, b, 0f437C0000, 0f4B400001;\n\t"
      "add.f32 d, c, 0fCB40007F;\n\t"
      "neg.f32 e, d;\n\t"
      "fma.rn.ftz.f32 f, n, 0f3FB8AA3B, e;\n\t"
      "fma.rn.ftz.f32 h, n, 0f32A57060, f;\n\t"
      "mov.b32 cb, c;\n\t"
      "shl.b32 sb, cb, 23;\n\t"
      "mov.b32 s, sb;\n\t"
      "ex2.approx.ftz.f32 x, h;\n\t"
      "fma.rn.f32 den, x, s, 0f3F800000;\n\t"
      "div.full.f32 q, %1, den;\n\t"
      "mul.f32 %0, q, %2;\n\t"
      "}"
      : "=f"(y) : "f"(g), "f"(u));
  return y;
}
#else
#error "N1_SILU: 0 (nvcc expf, IEEE ops) or 1 (the compiled K.silu_mul's sequence)"
#endif

}  // namespace n1
