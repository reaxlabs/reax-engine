// SPDX-License-Identifier: Apache-2.0
// blink_actq: the MoE "mid" step of Blink's NVFP4 experts in one kernel: GELU(tanh)(gate) * up followed by the NVFP4
// experts quantisation of the result, i.e. what vLLM 0.30.0 runs as two kernels in run_cutlass_moe_fp4
//   torch.ops._C.gelu_tanh_and_mul(c2, c1)                       (csrc/libtorch_stable/activation_kernels.cu,
//                                                                 act_and_mul_kernel, packed bf16x2 path)
//   torch.ops._C.scaled_fp4_experts_quant(fp4, sf, c2, gscale, expert_offsets, blockscale_offsets)
//                                                                (csrc/libtorch_stable/quantization/fp4/
//                                                                 nvfp4_experts_quant.cu, cvt_fp16_to_fp4)
// without writing and re-reading the bf16 activation c2. The per-element arithmetic is vLLM's own device code,
// copied below (vLLM, Apache-2.0, copyright the vLLM contributors and NVIDIA CORPORATION; see NOTICE): the activation
// of a bf16 pair is computed in fp32 and rounded to bf16 (packed_gelu_tanh_kernel), converted back, multiplied by
// (up + beta) with beta = 0 passed at run time, and rounded to bf16 (packed_compute); each 16-element block is
// quantised exactly as cvt_warp_fp16_to_fp4 (absolute max, SF = gscale * max * rcp.approx(6) -> e4m3,
// output scale = rcp.approx(SF * rcp.approx(gscale)), cvt.rn.satfinite.e2m1x2) and its scale byte is written at the
// swizzled position of cvt_quant_to_fp4_get_sf_out_offset in its expert's block-scale region. One thread handles one
// 16-element block (vLLM: two threads of 8); the max over 16 values is order-free, so the bytes are identical.
// Built without --use_fast_math, like vLLM's kernels (tanhf, no flush-to-zero outside the explicit .ftz rcp).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdint>

namespace {

// ---- vLLM 0.30.0 csrc/libtorch_stable/activation_kernels.cu: packed_gelu_tanh_kernel (verbatim math) ----
__device__ __forceinline__ __nv_bfloat162 packed_gelu_tanh_kernel(const __nv_bfloat162& val) {
  float2 fval = __bfloat1622float2(val);
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;

  float x_cube = fval.x * fval.x * fval.x;
  float inner = BETA * (fval.x + KAPPA * x_cube);
  fval.x = 0.5f * fval.x * (1.0f + ::tanhf(inner));

  x_cube = fval.y * fval.y * fval.y;
  inner = BETA * (fval.y + KAPPA * x_cube);
  fval.y = 0.5f * fval.y * (1.0f + ::tanhf(inner));
  return __float22bfloat162_rn(fval);
}

// ---- vLLM packed_compute<__nv_bfloat162, packed_gelu_tanh_kernel, act_first=true, HAS_CLAMP=false> ----
__device__ __forceinline__ __nv_bfloat162 gelu_mul_pair(const __nv_bfloat162& gate, const __nv_bfloat162& up,
                                                        const float beta) {
  float2 u = __bfloat1622float2(up);
  float2 activated = __bfloat1622float2(packed_gelu_tanh_kernel(gate));
  activated.x *= u.x + beta;
  activated.y *= u.y + beta;
  return __float22bfloat162_rn(activated);
}

// ---- vLLM 0.30.0 csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh ----
__device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}

__device__ __forceinline__ uint32_t fp32_vec8_to_e2m1(float2 (&array)[4]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}\n"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
}

// cvt_quant_to_fp4_get_sf_out_offset (byte offset inside one expert's region; kIdx = 16-element block index)
__device__ __forceinline__ int64_t sf_offset(int mIdx, int kIdx, int numKTiles) {
  int32_t mTileIdx = mIdx >> 7;         // mIdx / 128
  int32_t outerMIdx = mIdx & 31;        // mIdx % 32
  int32_t innerMIdx = (mIdx >> 5) & 3;  // (mIdx / 32) % 4
  int32_t kTileIdx = kIdx >> 2;         // kIdx / 4
  int32_t innerKIdx = kIdx & 3;         // kIdx % 4
  return (static_cast<int64_t>(mTileIdx) * numKTiles + kTileIdx) << 9 | (outerMIdx << 4) | (innerMIdx << 2) |
         innerKIdx;
}

// one thread = one row x one 16-element block of the activation output
__global__ void __launch_bounds__(256) gelu_quant_kernel(
    const __nv_bfloat16* __restrict__ in,  // [rows, 2 * d]: gate | up
    int rows, int d, const float* __restrict__ gscale, const int32_t* __restrict__ eoff,
    const int32_t* __restrict__ boff, int n_experts, uint8_t* __restrict__ out,  // [rows, d / 2]
    uint8_t* __restrict__ sf, const float beta) {
  extern __shared__ int32_t s_off[];  // n_experts + 1
  for (int i = threadIdx.x; i <= n_experts; i += blockDim.x) s_off[i] = eoff[i];
  __syncthreads();
  const int nsf = d / 16;
  const int numKTiles = (d + 63) / 64;
  const int valid_rows = min(rows, s_off[n_experts]);
  const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= static_cast<int64_t>(valid_rows) * nsf) return;
  const int row = static_cast<int>(gid / nsf);
  const int j = static_cast<int>(gid % nsf);
  // the expert whose rows [eoff[e], eoff[e + 1]) hold this row (vLLM: the same search over the same offsets)
  int lo = 0, hi = n_experts - 1, e = 0;
  while (lo <= hi) {
    int mid = (lo + hi) / 2;
    if (row >= s_off[mid] && row < s_off[mid + 1]) {
      e = mid;
      break;
    } else if (row < s_off[mid]) {
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }
  const int row_in_expert = row - s_off[e];
  // 16 gate + 16 up values (two 16-byte loads each)
  const uint4* g4 = reinterpret_cast<const uint4*>(in + static_cast<int64_t>(row) * 2 * d + j * 16);
  const uint4* u4 = reinterpret_cast<const uint4*>(in + static_cast<int64_t>(row) * 2 * d + d + j * 16);
  uint4 gv[2] = {g4[0], g4[1]};
  uint4 uv[2] = {u4[0], u4[1]};
  const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(gv);
  const __nv_bfloat162* u2 = reinterpret_cast<const __nv_bfloat162*>(uv);
  __nv_bfloat162 act[8];
#pragma unroll
  for (int i = 0; i < 8; i++) act[i] = gelu_mul_pair(g2[i], u2[i], beta);
  // ---- cvt_warp_fp16_to_fp4 (NVFP4 scale factor, UE8M0 off) on the 16 bf16 values ----
  __nv_bfloat162 localMax = __habs2(act[0]);
#pragma unroll
  for (int i = 1; i < 8; i++) localMax = __hmax2(localMax, __habs2(act[i]));
  const float vecMax = __bfloat162float(__hmax(localMax.x, localMax.y));
  const float SFScaleVal = gscale == nullptr ? 1.0f : gscale[e];
  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
  uint8_t fp8SFVal;
  reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
  SFValue = float(tmp);
  sf[static_cast<int64_t>(boff[e]) * numKTiles * 4 + sf_offset(row_in_expert, j, numKTiles)] = fp8SFVal;
  const float outputScale =
      SFValue != 0.0f ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;
  float2 f0[4], f1[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    f0[i] = __bfloat1622float2(act[i]);
    f0[i].x *= outputScale;
    f0[i].y *= outputScale;
    f1[i] = __bfloat1622float2(act[4 + i]);
    f1[i].x *= outputScale;
    f1[i].y *= outputScale;
  }
  uint32_t* o = reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(row) * (d / 2) + j * 8);
  o[0] = fp32_vec8_to_e2m1(f0);
  o[1] = fp32_vec8_to_e2m1(f1);
}

// debug / check entry: the bf16 activation only (same device function), [rows, d] from [rows, 2 * d]
__global__ void gelu_mul_kernel(const __nv_bfloat16* __restrict__ in, int rows, int d,
                                __nv_bfloat16* __restrict__ out, const float beta) {
  const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pairs = d / 2;
  if (gid >= static_cast<int64_t>(rows) * pairs) return;
  const int row = static_cast<int>(gid / pairs);
  const int p = static_cast<int>(gid % pairs);
  const __nv_bfloat162* g = reinterpret_cast<const __nv_bfloat162*>(in + static_cast<int64_t>(row) * 2 * d);
  const __nv_bfloat162* u = reinterpret_cast<const __nv_bfloat162*>(in + static_cast<int64_t>(row) * 2 * d + d);
  reinterpret_cast<__nv_bfloat162*>(out + static_cast<int64_t>(row) * d)[p] = gelu_mul_pair(g[p], u[p], beta);
}

}  // namespace

void gelu_quant(torch::Tensor in, torch::Tensor gscale, torch::Tensor eoff, torch::Tensor boff, torch::Tensor out,
                torch::Tensor sf) {
  TORCH_CHECK(in.is_cuda() && in.scalar_type() == at::kBFloat16 && in.dim() == 2 && in.is_contiguous());
  TORCH_CHECK(gscale.scalar_type() == at::kFloat && gscale.is_contiguous());
  TORCH_CHECK(eoff.scalar_type() == at::kInt && boff.scalar_type() == at::kInt && eoff.numel() == boff.numel());
  TORCH_CHECK(out.scalar_type() == at::kByte && out.is_contiguous() && sf.is_contiguous());
  const int rows = in.size(0), d = in.size(1) / 2;
  TORCH_CHECK(d % 16 == 0 && out.size(0) == rows && out.size(1) == d / 2);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(in.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0);
  const int n_experts = eoff.numel() - 1;
  TORCH_CHECK(gscale.numel() >= n_experts);
  if (rows == 0) return;
  const int64_t work = static_cast<int64_t>(rows) * (d / 16);
  const int threads = 256;
  const int64_t blocks = (work + threads - 1) / threads;
  gelu_quant_kernel<<<blocks, threads, (n_experts + 1) * sizeof(int32_t), c10::cuda::getCurrentCUDAStream().stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(in.data_ptr()), rows, d, gscale.data_ptr<float>(), eoff.data_ptr<int32_t>(),
      boff.data_ptr<int32_t>(), n_experts, out.data_ptr<uint8_t>(), reinterpret_cast<uint8_t*>(sf.data_ptr()), 0.0f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gelu_mul(torch::Tensor in, torch::Tensor out) {
  TORCH_CHECK(in.is_cuda() && in.scalar_type() == at::kBFloat16 && in.dim() == 2 && in.is_contiguous());
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous());
  const int rows = in.size(0), d = in.size(1) / 2;
  TORCH_CHECK(d % 2 == 0 && out.size(0) == rows && out.size(1) == d);
  if (rows == 0) return;
  const int64_t work = static_cast<int64_t>(rows) * (d / 2);
  gelu_mul_kernel<<<(work + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(in.data_ptr()), rows, d,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), 0.0f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gelu_quant", &gelu_quant, "GELU(tanh)(gate) * up + NVFP4 experts quantisation (vLLM numerics, one kernel)");
  m.def("gelu_mul", &gelu_mul, "GELU(tanh)(gate) * up in bf16 (vLLM gelu_tanh_and_mul numerics; check entry)");
}
