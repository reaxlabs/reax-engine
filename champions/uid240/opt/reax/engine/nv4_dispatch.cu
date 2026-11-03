// SPDX-License-Identifier: Apache-2.0
// nv4_dispatch.cu (rgh lane Q): C ABI entry of libnv4g.so; routes a variant number to the nv4g.cu translation unit
// compiled with -DNV4_VARIANT=<v> (one CUTLASS tile/schedule configuration each, see nv4g.cu).
#include <cuda_runtime.h>
#include <cstddef>
#define DECL(v) \
  extern "C" int nv4_run_##v(const void*, const void*, const void*, const void*, const float*, void*, int, int, int, int, \
                             void*, size_t, cudaStream_t); \
  extern "C" size_t nv4_ws_##v(int, int, int);
DECL(0) DECL(1) DECL(2) DECL(3) DECL(4) DECL(5)
extern "C" int nv4_nvariants() { return 6; }
extern "C" int nv4_gemm(int v, const void* A, const void* B, const void* SFA, const void* SFB, const float* alpha, void* D,
                        int M, int N, int K, int ldd, void* wsp, size_t wsb, cudaStream_t st) {
  switch (v) {
#define CASE(k) case k: return nv4_run_##k(A, B, SFA, SFB, alpha, D, M, N, K, ldd, wsp, wsb, st);
    CASE(0) CASE(1) CASE(2) CASE(3) CASE(4) CASE(5)
    default: return 1;
  }
}
extern "C" size_t nv4_ws(int v, int M, int N, int K) {
  switch (v) {
#define WS(k) case k: return nv4_ws_##k(M, N, K);
    WS(0) WS(1) WS(2) WS(3) WS(4) WS(5)
    default: return 0;
  }
}
