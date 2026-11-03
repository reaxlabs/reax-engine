// SPDX-License-Identifier: Apache-2.0
// f8c_dispatch.cu (kai lever R5-1, 10-28): C ABI of libf8c.so.
//   f8c_gemm(variant, kind, A, lda, B, ldb, sa, sw, D, ldd, M, N, K, swizzle, raster, ws, ws_bytes, stream)
//       routes to the CUTLASS sm120 instantiation f8c_run_<variant>_<kind> (f8c.cu); -1 = no such instantiation
//   f8c_ws(variant, kind, M, N, K), f8c_stages(variant, kind)
//   f8c_swq_launch(gu, ldg, M, H, xs, sw, y, q, s, stream): the SwiGLU tail of f8g's gate/up GEMM, from gu = bf16(acc)
//       [M, 2H] (kind 1 output; gate columns [0, H), up [H, 2H)): per row y = rgh's smqs1 (f8g.cu, included below,
//       unchanged) -> bf16, then f8_rowquant's amax / scale / e4m3 quant of that bf16 row, one CTA per row with the row
//       held in shared memory. y != null: also write y [M, H]; q == null: y only (nv4's swiglu_y).
#include <cuda_runtime.h>
#include <cstddef>

#include "f8g.cu"  // rgh's f8::smqs1, f8::div_full (and f8g's kernels, unused here)

#define F8C_DECL(v, k)                                                                                                  \
  extern "C" int f8c_run_##v##_##k(const void*, int, const void*, int, const float*, const float*, void*, int, int, int, \
                                   int, int, int, void*, size_t, cudaStream_t);                                         \
  extern "C" size_t f8c_ws_##v##_##k(int, int, int);                                                                    \
  extern "C" int f8c_stages_##v##_##k();
#define F8C_ALL(X) X(0, 0) X(1, 0) X(2, 0) X(3, 0) X(4, 0) X(5, 0) X(6, 0) X(7, 0) \
                   X(0, 1) X(1, 1) X(2, 1) X(3, 1) X(4, 1) X(5, 1) X(6, 1) X(7, 1)
F8C_ALL(F8C_DECL)

extern "C" int f8c_gemm(int v, int k, const void* A, int lda, const void* B, int ldb, const float* sa, const float* sw,
                        void* D, int ldd, int M, int N, int K, int swz, int raster, void* ws, size_t wsb, void* stream) {
  if (M == 0) return 0;
#define F8C_RUN(vv, kk) \
  if (v == vv && k == kk) return f8c_run_##vv##_##kk(A, lda, B, ldb, sa, sw, D, ldd, M, N, K, swz, raster, ws, wsb, (cudaStream_t)stream);
  F8C_ALL(F8C_RUN)
  return -1;
}

extern "C" size_t f8c_ws(int v, int k, int M, int N, int K) {
#define F8C_WS(vv, kk) \
  if (v == vv && k == kk) return f8c_ws_##vv##_##kk(M, N, K);
  F8C_ALL(F8C_WS)
  return (size_t)-1;
}

extern "C" int f8c_stages(int v, int k) {
#define F8C_ST(vv, kk) \
  if (v == vv && k == kk) return f8c_stages_##vv##_##kk();
  F8C_ALL(F8C_ST)
  return -1;
}

// one CTA (512 threads) per row; H % 8 == 0, ldg % 8 == 0; dynamic smem H bf16
extern "C" __global__ void __launch_bounds__(512) f8c_swq(const __nv_bfloat16* __restrict__ gu, int ldg, int H,
                                                          const float* __restrict__ xs, const float* __restrict__ sw,
                                                          __nv_bfloat16* __restrict__ yo, uint8_t* __restrict__ q,
                                                          float* __restrict__ s) {
  extern __shared__ __align__(16) unsigned char f8c_smem[];
  uint4* ys = reinterpret_cast<uint4*>(f8c_smem);
  const size_t r = blockIdx.x;
  const uint4* gr = reinterpret_cast<const uint4*>(gu + r * (size_t)ldg);
  const uint4* ur = reinterpret_cast<const uint4*>(gu + r * (size_t)ldg + H);
  const float x = xs[r];
  const int nc = H / 8;
  float m = 0.f;
  for (int c = threadIdx.x; c < nc; c += blockDim.x) {
    const uint4 gv = gr[c], uv = ur[c];
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gv);
    const __nv_bfloat162* u2 = reinterpret_cast<const __nv_bfloat162*>(&uv);
    const float4 wg0 = reinterpret_cast<const float4*>(sw)[2 * c], wg1 = reinterpret_cast<const float4*>(sw)[2 * c + 1];
    const float4 wu0 = reinterpret_cast<const float4*>(sw + H)[2 * c], wu1 = reinterpret_cast<const float4*>(sw + H)[2 * c + 1];
    const float wg[8] = {wg0.x, wg0.y, wg0.z, wg0.w, wg1.x, wg1.y, wg1.z, wg1.w};
    const float wu[8] = {wu0.x, wu0.y, wu0.z, wu0.w, wu1.x, wu1.y, wu1.z, wu1.w};
    uint4 out;
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 g = __bfloat1622float2(g2[e]), u = __bfloat1622float2(u2[e]);
      const float y0 = f8::smqs1(g.x, u.x, x, wg[2 * e], wu[2 * e]);
      const float y1 = f8::smqs1(g.y, u.y, x, wg[2 * e + 1], wu[2 * e + 1]);
      o2[e] = __floats2bfloat162_rn(y0, y1);
      const float2 f = __bfloat1622float2(o2[e]);
      m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
    ys[c] = out;
    if (yo) reinterpret_cast<uint4*>(yo + r * (size_t)H)[c] = out;
  }
  if (!q) return;
  // f8_rowquant (f8g.cu) from here on, reading the row from shared memory instead of global
  __shared__ float red[16];
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
#pragma unroll
    for (int o = 16; o; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
    if (threadIdx.x == 0) red[0] = t;
  }
  __syncthreads();
  const float sc = f8::div_full(fmaxf(red[0], 1e-12f), 448.0f);
  if (threadIdx.x == 0) s[r] = sc;
  uint2* qr = reinterpret_cast<uint2*>(q + r * (size_t)H);
  for (int c = threadIdx.x; c < nc; c += blockDim.x) {
    const uint4 v = ys[c];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
    uint32_t w[2];
#pragma unroll
    for (int e = 0; e < 4; e += 2) {
      const float2 f0 = __bfloat1622float2(p[e]), f1 = __bfloat1622float2(p[e + 1]);
      const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(
          make_float2(fminf(fmaxf(f8::div_full(f0.x, sc), -448.f), 448.f), fminf(fmaxf(f8::div_full(f0.y, sc), -448.f), 448.f)),
          __NV_SATFINITE, __NV_E4M3);
      const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(
          make_float2(fminf(fmaxf(f8::div_full(f1.x, sc), -448.f), 448.f), fminf(fmaxf(f8::div_full(f1.y, sc), -448.f), 448.f)),
          __NV_SATFINITE, __NV_E4M3);
      w[e / 2] = (uint32_t)lo | ((uint32_t)hi << 16);
    }
    qr[c] = make_uint2(w[0], w[1]);
  }
}

extern "C" int f8c_swq_launch(const void* gu, int ldg, int M, int H, const float* xs, const float* sw, void* y, void* q,
                              float* s, void* stream) {
  if (M == 0) return 0;
  if (H % 8 || ldg % 8) return -2;
  const size_t sm = (size_t)H * 2;
  if (sm > 48 * 1024) {
    static bool set = false;
    if (!set) {
      cudaFuncSetAttribute(f8c_swq, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      set = true;
    }
  }
  f8c_swq<<<M, 512, sm, (cudaStream_t)stream>>>((const __nv_bfloat16*)gu, ldg, H, xs, sw, (__nv_bfloat16*)y,
                                                (uint8_t*)q, s);
  return (int)cudaGetLastError();
}
