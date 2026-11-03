// SPDX-License-Identifier: Apache-2.0
// nv4s_dispatch.cu (kai lever N1, 10-28): C ABI of libnv4s.so. Routes a tile number to the nv4s.cu translation unit
// compiled with -DNV4S_TILE=<t>, and holds the start-up probe of N1's epilogue numerics (nv4s_silu_probe: the same
// n1::gu_round / n1::silu_mul device functions as the epilogue, on caller-given inputs).
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstddef>

#include "nv4s_silu.cuh"

#define NV4S_TILES(X) X(1) X(2) X(3) X(4) X(8)
#define DECL(t) \
  extern "C" int nv4s_run_##t(const void*, const void*, const void*, const void*, const float*, void*, float*, int, int, \
                              int, int, void*, size_t, cudaStream_t, int, int); \
  extern "C" size_t nv4s_ws_##t(int, int, int, int, int);
NV4S_TILES(DECL)

extern "C" int nv4s_has(int t) {
  switch (t) {
#define HAS(k) case k: return 1;
    NV4S_TILES(HAS)
    default: return 0;
  }
}

extern "C" int nv4s_silu_mode() { return N1_SILU; }

extern "C" int nv4s_gemm(int t, int raster, int swz, const void* A, const void* B, const void* SFA, const void* SFB,
                         const float* alpha, void* Y, float* amax, int M, int N, int K, int ldy, void* wsp, size_t wsb,
                         cudaStream_t st) {
  switch (t) {
#define CASE(k) case k: return nv4s_run_##k(A, B, SFA, SFB, alpha, Y, amax, M, N, K, ldy, wsp, wsb, st, raster, swz);
    NV4S_TILES(CASE)
    default: return 1;
  }
}

extern "C" size_t nv4s_ws(int t, int raster, int swz, int M, int N, int K) {
  switch (t) {
#define WS(k) case k: return nv4s_ws_##k(M, N, K, raster, swz);
    NV4S_TILES(WS)
    default: return 0;
  }
}

// probe: y[i] = bf16(silu_mul(g[i], u[i])) for bf16 g, u (as the epilogue after gu_round), and
// r[i] = gu_round(alpha, a[i]) for fp32 a (when a != nullptr)
__global__ void n1_probe_kernel(const __nv_bfloat16* g, const __nv_bfloat16* u, __nv_bfloat16* y, const float* a,
                                const float* alpha, __nv_bfloat16* r, long n) {
  long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] = __float2bfloat16_rn(n1::silu_mul(__bfloat162float(g[i]), __bfloat162float(u[i])));
  if (a != nullptr) r[i] = __float2bfloat16_rn(n1::gu_round(*alpha, a[i]));
}

extern "C" int nv4s_silu_probe(const void* g, const void* u, void* y, const float* a, const float* alpha, void* r,
                               long n, cudaStream_t st) {
  if (n <= 0) return 0;
  int th = 256;
  long bl = (n + th - 1) / th;
  n1_probe_kernel<<<(unsigned)bl, th, 0, st>>>(static_cast<const __nv_bfloat16*>(g),
                                               static_cast<const __nv_bfloat16*>(u), static_cast<__nv_bfloat16*>(y), a,
                                               alpha, static_cast<__nv_bfloat16*>(r), n);
  return cudaGetLastError() == cudaSuccess ? 0 : 5;
}
