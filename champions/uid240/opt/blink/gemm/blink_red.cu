// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 pjs team (K2).
//
// blink_red: one-pass CUDA replicas of the two Inductor reductions at the tail of every Blink decoder layer,
//   red0 = triton_red_fused_add_bi_mm_mul_rms_norm_0  (Inductor source c3wwo3b2e4upplnm2ju3ef4m4mi5mxufiqpgzw7nhgjptheg4xhi)
//   red2 = triton_red_fused_add_bi_mm_mul_rms_norm_2  (Inductor source cbpkosjpa5brpjsapkulaacetht4nwifxjojih4hildoqq5svlk6)
// bit-identical to their Triton 3.7.1 sm_120 builds (single config XBLOCK 1, R0_BLOCK 1024, num_warps 8, 2816 columns).
//
// What is copied from the compiled Triton kernels (PTX + SASS of the J4a cache, cubins NJU3V2CP... / OLRXHUX2...):
//  * thread t of the 256 owns columns 1024 c + 4 t + j (j = 0..3) of chunk c = 0, 1, 2; chunk 2 only for t < 192;
//  * per column lane: acc = fma(v, v, acc) chained over the chunks in order, starting from +0;
//  * tl.sum: thread sum ((a0 + a1) + a2) + a3, warp butterfly xor 16, 8, 4, 2, 1 (own + partner), then the eight warp
//    sums as Triton's second butterfly (xor 4, 2, 1 over lanes 0..7) leaves them in lane 0:
//    ((W0 + W4) + (W2 + W6)) + ((W1 + W5) + (W3 + W7));
//  * rsqrt(S / 2816 + 1e-6): ptxas turns Triton's div.full.f32 + add.f32 into ONE FFMA(S, 0x39ba2e8c, 1e-6) followed by
//    MUFU.RSQ (rsqrt.approx.ftz.f32) - written out explicitly here, S / 2816.f + 1e-6f would round differently;
//  * every elementwise multiply-add exactly as LLVM contracted it (fma.rn), every other op as a separately rounded
//    mul / add (__fmul_rn / __fadd_rn cannot be re-contracted by ptxas); bf16 -> f32 by a 16-bit shift (as Triton's
//    SASS), f32 -> bf16 by cvt.rn.bf16x2.f32.
// What is different: each row is read once and kept in registers (Triton reads x three times and the residual twice),
// one barrier per reduction instead of two, and red2's fp32 scratch output (out_ptr2, deleted by the Inductor wrapper
// right after the kernel) is written only when the caller asks for it.
#include <cuda_bf16.h>
#include <stdint.h>
#ifndef BLINKRED_KERNELS_ONLY  // -DBLINKRED_KERNELS_ONLY: device code alone (SASS inspection, raw cubin tests)
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#endif

namespace blinkred {

constexpr int H = 2816;   // r0_numel
constexpr int NT = 256;   // Triton num_warps 8
constexpr int T2 = 192;   // threads that own chunk-2 columns: 4 t + 2048 < 2816

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ void unpack4(const uint2 u, float (&f)[4]) {
  f[0] = __uint_as_float(u.x << 16);
  f[1] = __uint_as_float(u.x & 0xffff0000u);
  f[2] = __uint_as_float(u.y << 16);
  f[3] = __uint_as_float(u.y & 0xffff0000u);
}

__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

__device__ __forceinline__ uint2 ldg8(const bf16* p) { return __ldg(reinterpret_cast<const uint2*>(p)); }
__device__ __forceinline__ uint2 ld8(const bf16* p) { return *reinterpret_cast<const uint2*>(p); }

__device__ __forceinline__ float bf16f(const bf16* p) {
  const unsigned short u = __ldg(reinterpret_cast<const unsigned short*>(p));
  return __uint_as_float(static_cast<uint32_t>(u) << 16);
}

// 1 / sqrt(S / 2816 + 1e-6) exactly as the Triton build computes it (see the header)
__device__ __forceinline__ float inv_rms(float s) {
  const float t = __fmaf_rn(s, __uint_as_float(0x39ba2e8cu), __uint_as_float(0x358637bdu));
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(t));
  return r;
}

// per-lane sum of squares over the chunks: fma(v, v, +0) for chunk 0, then fma(v, v, acc)
__device__ __forceinline__ void sq_acc(const float (&v)[3][4], bool has2, float (&a)[4]) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float s = __fmaf_rn(v[0][j], v[0][j], 0.0f);
    s = __fmaf_rn(v[1][j], v[1][j], s);
    if (has2) s = __fmaf_rn(v[2][j], v[2][j], s);
    a[j] = s;
  }
}

__device__ __forceinline__ float warp_part(const float (&a)[4]) {
  float s = __fadd_rn(__fadd_rn(__fadd_rn(a[0], a[1]), a[2]), a[3]);
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
  return s;
}

__device__ __forceinline__ float cross8(const float* w) {
  return __fadd_rn(__fadd_rn(__fadd_rn(w[0], w[4]), __fadd_rn(w[2], w[6])),
                   __fadd_rn(__fadd_rn(w[1], w[5]), __fadd_rn(w[3], w[7])));
}

// red0: S1 = sum x^2 (written to out0), h = fma(r1 x, w1, res), S2 = sum h^2, y = r2 h,
//       out4 = y w3, out5 = (y s) w5, out6 = y w6 (bf16)
__global__ void __launch_bounds__(NT) red0_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w1,
                                                  const bf16* __restrict__ res, const bf16* __restrict__ w3,
                                                  const bf16* __restrict__ sp, const bf16* __restrict__ w5,
                                                  const bf16* __restrict__ w6, float* __restrict__ s1out,
                                                  bf16* __restrict__ o4, bf16* __restrict__ o5, bf16* __restrict__ o6,
                                                  int m) {
  __shared__ float sm[2][8];
  const int row = blockIdx.x;
  if (row >= m) return;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const bool has2 = t < T2;
  const size_t rb = static_cast<size_t>(row) * H;

  float xv[3][4], rv[3][4], wv[3][4];
  uint2 q3[3], q5[3], q6[3];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const int col = 1024 * c + 4 * t;
    if (c < 2 || has2) {
      unpack4(ldg8(x + rb + col), xv[c]);
      unpack4(ldg8(res + rb + col), rv[c]);
      unpack4(ldg8(w1 + col), wv[c]);
      q3[c] = ldg8(w3 + col);
      q5[c] = ldg8(w5 + col);
      q6[c] = ldg8(w6 + col);
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) xv[c][j] = rv[c][j] = wv[c][j] = 0.0f;
      q3[c] = q5[c] = q6[c] = make_uint2(0u, 0u);
    }
  }
  const float sc = bf16f(sp);

  float a[4];
  sq_acc(xv, has2, a);
  float s = warp_part(a);
  if (lane == 0) sm[0][warp] = s;
  __syncthreads();
  const float s1 = cross8(sm[0]);
  if (t == 0) s1out[row] = s1;
  const float r1 = inv_rms(s1);

  float h[3][4];
#pragma unroll
  for (int c = 0; c < 3; ++c)
#pragma unroll
    for (int j = 0; j < 4; ++j) h[c][j] = __fmaf_rn(__fmul_rn(r1, xv[c][j]), wv[c][j], rv[c][j]);
  sq_acc(h, has2, a);
  s = warp_part(a);
  if (lane == 0) sm[1][warp] = s;
  __syncthreads();
  const float r2 = inv_rms(cross8(sm[1]));

#pragma unroll
  for (int c = 0; c < 3; ++c) {
    if (c == 2 && !has2) break;
    const int col = 1024 * c + 4 * t;
    float f3[4], f5[4], f6[4], p4[4], p5[4], p6[4];
    unpack4(q3[c], f3);
    unpack4(q5[c], f5);
    unpack4(q6[c], f6);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float y = __fmul_rn(r2, h[c][j]);
      p4[j] = __fmul_rn(y, f3[j]);
      p5[j] = __fmul_rn(__fmul_rn(y, sc), f5[j]);
      p6[j] = __fmul_rn(y, f6[j]);
    }
    *reinterpret_cast<uint2*>(o4 + rb + col) = make_uint2(pack2(p4[0], p4[1]), pack2(p4[2], p4[3]));
    *reinterpret_cast<uint2*>(o5 + rb + col) = make_uint2(pack2(p5[0], p5[1]), pack2(p5[2], p5[3]));
    *reinterpret_cast<uint2*>(o6 + rb + col) = make_uint2(pack2(p6[0], p6[1]), pack2(p6[2], p6[3]));
  }
}

// red2: Sa = sum a^2, Sb = sum b^2, t = fma(ra a, w2, (rb b) w3) (fp32, out2 when asked), Sc = sum t^2,
//       u = fma(rc t, w4, fma(r1 x, w1, res)) s with r1 from S1 (red0's out0); u -> xio (bf16, in place),
//       S4 = sum u^2 (fp32 u), out5 = (r4 bf16(u)) w9
__global__ void __launch_bounds__(NT) red2_kernel(bf16* xio, const bf16* __restrict__ av_, const bf16* __restrict__ bv_,
                                                  const bf16* __restrict__ w2, const bf16* __restrict__ w3,
                                                  const bf16* __restrict__ w4, const float* __restrict__ s1in,
                                                  const bf16* __restrict__ w1, const bf16* __restrict__ res,
                                                  const bf16* __restrict__ sp, const bf16* __restrict__ w9,
                                                  float* __restrict__ tout, bf16* __restrict__ out, int m,
                                                  int write_t) {
  __shared__ float sm[4][8];
  const int row = blockIdx.x;
  if (row >= m) return;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const bool has2 = t < T2;
  const size_t rb = static_cast<size_t>(row) * H;

  float av[3][4], bv[3][4];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const int col = 1024 * c + 4 * t;
    if (c < 2 || has2) {
      unpack4(ldg8(av_ + rb + col), av[c]);
      unpack4(ldg8(bv_ + rb + col), bv[c]);
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) av[c][j] = bv[c][j] = 0.0f;
    }
  }
  float aa[4], ab[4];
  sq_acc(av, has2, aa);
  sq_acc(bv, has2, ab);
  const float sa = warp_part(aa), sb = warp_part(ab);
  if (lane == 0) {
    sm[0][warp] = sa;
    sm[1][warp] = sb;
  }
  // operands of the later phases, loaded while the first reduction completes
  uint2 q1[3], q2[3], q3[3], q4[3], q9[3], qx[3], qr[3];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const int col = 1024 * c + 4 * t;
    if (c < 2 || has2) {
      q2[c] = ldg8(w2 + col);
      q3[c] = ldg8(w3 + col);
      q4[c] = ldg8(w4 + col);
      q1[c] = ldg8(w1 + col);
      q9[c] = ldg8(w9 + col);
      qx[c] = ld8(xio + rb + col);
      qr[c] = ldg8(res + rb + col);
    } else {
      q1[c] = q2[c] = q3[c] = q4[c] = q9[c] = qx[c] = qr[c] = make_uint2(0u, 0u);
    }
  }
  const float s1 = __ldg(s1in + row);
  const float sc = bf16f(sp);
  __syncthreads();
  const float ra = inv_rms(cross8(sm[0])), rbn = inv_rms(cross8(sm[1]));

  float tv[3][4];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    float f2[4], f3[4];
    unpack4(q2[c], f2);
    unpack4(q3[c], f3);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      tv[c][j] = __fmaf_rn(__fmul_rn(ra, av[c][j]), f2[j], __fmul_rn(__fmul_rn(rbn, bv[c][j]), f3[j]));
  }
  float at[4];
  sq_acc(tv, has2, at);
  const float st = warp_part(at);
  if (lane == 0) sm[2][warp] = st;
  if (write_t) {
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      if (c == 2 && !has2) break;
      *reinterpret_cast<float4*>(tout + rb + 1024 * c + 4 * t) = make_float4(tv[c][0], tv[c][1], tv[c][2], tv[c][3]);
    }
  }
  const float r1 = inv_rms(s1);
  float hv[3][4];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    float fx[4], f1[4], fr[4];
    unpack4(qx[c], fx);
    unpack4(q1[c], f1);
    unpack4(qr[c], fr);
#pragma unroll
    for (int j = 0; j < 4; ++j) hv[c][j] = __fmaf_rn(__fmul_rn(r1, fx[j]), f1[j], fr[j]);
  }
  __syncthreads();
  const float rc = inv_rms(cross8(sm[2]));

  float uv[3][4];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    float f4[4];
    unpack4(q4[c], f4);
#pragma unroll
    for (int j = 0; j < 4; ++j) uv[c][j] = __fmul_rn(__fmaf_rn(__fmul_rn(rc, tv[c][j]), f4[j], hv[c][j]), sc);
  }
  float au[4];
  sq_acc(uv, has2, au);
  const float su = warp_part(au);
  if (lane == 0) sm[3][warp] = su;
  uint32_t ub[3][2];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    ub[c][0] = pack2(uv[c][0], uv[c][1]);
    ub[c][1] = pack2(uv[c][2], uv[c][3]);
    if (c < 2 || has2) *reinterpret_cast<uint2*>(xio + rb + 1024 * c + 4 * t) = make_uint2(ub[c][0], ub[c][1]);
  }
  __syncthreads();
  const float r4 = inv_rms(cross8(sm[3]));
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    if (c == 2 && !has2) break;
    float fu[4], f9[4], p[4];
    unpack4(make_uint2(ub[c][0], ub[c][1]), fu);
    unpack4(q9[c], f9);
#pragma unroll
    for (int j = 0; j < 4; ++j) p[j] = __fmul_rn(__fmul_rn(r4, fu[j]), f9[j]);
    *reinterpret_cast<uint2*>(out + rb + 1024 * c + 4 * t) = make_uint2(pack2(p[0], p[1]), pack2(p[2], p[3]));
  }
}

#ifndef BLINKRED_KERNELS_ONLY
// ---------------------------------------------------------------------------------------------------------------------
// host side: argument lists in the Triton kernels' order + the raw stream handle the Inductor wrapper passes to
// CachingAutotuner.run (raw_stream0 = the current stream); return false (caller runs the Triton kernel) when an operand
// is not what the kernel was built for. No allocation, no synchronisation.

static bool rows_ok(const torch::Tensor& t, int64_t m, at::ScalarType dt, int64_t width, int align) {
  if (!t.is_cuda() || t.scalar_type() != dt) return false;
  if (t.numel() < m * width) return false;
  if (width > 1 && !t.is_contiguous()) return false;
  return reinterpret_cast<uintptr_t>(t.data_ptr()) % align == 0;
}

static bool vec_ok(const torch::Tensor& t, int64_t n) {
  return t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.numel() >= n && t.is_contiguous() &&
         reinterpret_cast<uintptr_t>(t.data_ptr()) % 8 == 0;
}

// S1 output / input: Inductor's buf2 is (m, 1) with strides (1, m): element r at offset r
static bool s1_ok(const torch::Tensor& t, int64_t m) {
  return t.is_cuda() && t.scalar_type() == at::kFloat && t.numel() >= m && t.dim() >= 1 && t.stride(0) == 1 &&
         reinterpret_cast<uintptr_t>(t.data_ptr()) % 4 == 0;
}

bool red0(torch::Tensor in0, torch::Tensor in1, torch::Tensor in2, torch::Tensor in3, torch::Tensor in4,
          torch::Tensor in5, torch::Tensor in6, torch::Tensor out0, torch::Tensor out4, torch::Tensor out5,
          torch::Tensor out6, int64_t xnumel, int64_t stream) {
  const int64_t m = xnumel;
  if (m < 0 || m > (int64_t(1) << 31) - 1) return false;
  const auto B = at::kBFloat16;
  if (!rows_ok(in0, m, B, H, 8) || !rows_ok(in2, m, B, H, 8) || !rows_ok(out4, m, B, H, 8) ||
      !rows_ok(out5, m, B, H, 8) || !rows_ok(out6, m, B, H, 8))
    return false;
  if (!vec_ok(in1, H) || !vec_ok(in3, H) || !vec_ok(in5, H) || !vec_ok(in6, H)) return false;
  if (!in4.is_cuda() || in4.scalar_type() != B || in4.numel() < 1) return false;
  if (!s1_ok(out0, m)) return false;
  const int dev = in0.get_device();
  for (const auto* p : {&in1, &in2, &in3, &in4, &in5, &in6, &out0, &out4, &out5, &out6})
    if (p->get_device() != dev) return false;
  if (m == 0) return true;
  const c10::cuda::CUDAGuard guard(dev);
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);  // the stream Triton's launcher would get (raw_stream0)
  auto P = [](const torch::Tensor& t) { return reinterpret_cast<const bf16*>(t.data_ptr()); };
  auto Q = [](torch::Tensor& t) { return reinterpret_cast<bf16*>(t.data_ptr()); };
  red0_kernel<<<static_cast<unsigned>(m), NT, 0, st>>>(P(in0), P(in1), P(in2), P(in3), P(in4), P(in5), P(in6),
                                                      out0.data_ptr<float>(), Q(out4), Q(out5), Q(out6),
                                                      static_cast<int>(m));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

bool red2(torch::Tensor io0, torch::Tensor in0, torch::Tensor in1, torch::Tensor in2, torch::Tensor in3,
          torch::Tensor in4, torch::Tensor in5, torch::Tensor in6, torch::Tensor in7, torch::Tensor in8,
          torch::Tensor in9, torch::Tensor out2, torch::Tensor out5, int64_t xnumel, bool write_out2,
          int64_t stream) {
  const int64_t m = xnumel;
  if (m < 0 || m > (int64_t(1) << 31) - 1) return false;
  const auto B = at::kBFloat16;
  if (!rows_ok(io0, m, B, H, 8) || !rows_ok(in0, m, B, H, 8) || !rows_ok(in1, m, B, H, 8) ||
      !rows_ok(in7, m, B, H, 8) || !rows_ok(out5, m, B, H, 8))
    return false;
  if (!vec_ok(in2, H) || !vec_ok(in3, H) || !vec_ok(in4, H) || !vec_ok(in6, H) || !vec_ok(in9, H)) return false;
  if (!in8.is_cuda() || in8.scalar_type() != B || in8.numel() < 1) return false;
  if (!s1_ok(in5, m)) return false;
  if (write_out2 && !rows_ok(out2, m, at::kFloat, H, 16)) return false;
  const int dev = io0.get_device();
  for (const auto* p : {&in0, &in1, &in2, &in3, &in4, &in5, &in6, &in7, &in8, &in9, &out5})
    if (p->get_device() != dev) return false;
  if (write_out2 && out2.get_device() != dev) return false;
  if (m == 0) return true;
  const c10::cuda::CUDAGuard guard(dev);
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);  // the stream Triton's launcher would get (raw_stream0)
  auto P = [](const torch::Tensor& t) { return reinterpret_cast<const bf16*>(t.data_ptr()); };
  red2_kernel<<<static_cast<unsigned>(m), NT, 0, st>>>(
      reinterpret_cast<bf16*>(io0.data_ptr()), P(in0), P(in1), P(in2), P(in3), P(in4), in5.data_ptr<float>(), P(in6),
      P(in7), P(in8), P(in9), write_out2 ? out2.data_ptr<float>() : nullptr, reinterpret_cast<bf16*>(out5.data_ptr()),
      static_cast<int>(m), write_out2 ? 1 : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

#endif  // BLINKRED_KERNELS_ONLY
}  // namespace blinkred

#ifndef BLINKRED_KERNELS_ONLY
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("red0", &blinkred::red0,
        "exact one-pass replica of triton_red_fused_add_bi_mm_mul_rms_norm_0 (Triton argument order, + raw stream); "
        "False = not launched, run the Triton kernel");
  m.def("red2", &blinkred::red2,
        "exact one-pass replica of triton_red_fused_add_bi_mm_mul_rms_norm_2 (Triton argument order, + write_out2, "
        "raw stream); False = not launched, run the Triton kernel");
}
#endif  // BLINKRED_KERNELS_ONLY
