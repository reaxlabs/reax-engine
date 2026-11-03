// SPDX-License-Identifier: Apache-2.0
// nv4k_dispatch.cu (kai lever K, 10-28): C ABI entry of libnv4k.so; routes a variant number (6, 7, 8) to the nv4k.cu
// translation unit compiled with -DNV4K_VARIANT=<v> (one CUTLASS stream-K tile configuration each, see nv4k.cu).
// Same argument list and return codes as rgh's nv4_gemm / nv4_ws (nv4_dispatch.cu).
#include <cuda_runtime.h>
#include <cstddef>
#define DECL(v) \
  extern "C" int nv4k_run_##v(const void*, const void*, const void*, const void*, const float*, void*, int, int, int, \
                              int, void*, size_t, cudaStream_t); \
  extern "C" size_t nv4k_ws_##v(int, int, int);
DECL(6) DECL(7) DECL(8)
extern "C" int nv4k_gemm(int v, const void* A, const void* B, const void* SFA, const void* SFB, const float* alpha,
                         void* D, int M, int N, int K, int ldd, void* wsp, size_t wsb, cudaStream_t st) {
  switch (v) {
#define CASE(k) case k: return nv4k_run_##k(A, B, SFA, SFB, alpha, D, M, N, K, ldd, wsp, wsb, st);
    CASE(6) CASE(7) CASE(8)
    default: return 1;
  }
}
extern "C" size_t nv4k_ws(int v, int M, int N, int K) {
  switch (v) {
#define WS(k) case k: return nv4k_ws_##k(M, N, K);
    WS(6) WS(7) WS(8)
    default: return 0;
  }
}
