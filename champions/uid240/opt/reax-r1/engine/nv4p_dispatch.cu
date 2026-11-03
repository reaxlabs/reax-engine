// SPDX-License-Identifier: Apache-2.0
// nv4p_dispatch.cu (kai lever N2): C ABI entry of libnv4p.so; routes a tile number to the nv4p.cu translation unit compiled
// with -DNV4T_TILE=<t> (same tiles as nv4t_dispatch.cu).
#include <cuda_runtime.h>
#include <cstddef>
#define DECL(t) \
  extern "C" int nv4p_run_##t(const void*, const void*, const void*, const void*, const float*, void*, int, int, int, int, \
                              void*, size_t, cudaStream_t, int, int); \
  extern "C" size_t nv4t_ws_##t(int, int, int, int, int);
#define NV4P_TILES(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10)
NV4P_TILES(DECL)
extern "C" int nv4p_has(int t) {
  switch (t) {
#define HAS(k) case k: return 1;
    NV4P_TILES(HAS)
    default: return 0;
  }
}
extern "C" int nv4p_gemm(int t, int raster, int swz, const void* A, const void* B, const void* SFA, const void* SFB,
                         const float* alpha, void* D, int M, int N, int K, int ldd, void* wsp, size_t wsb,
                         cudaStream_t st) {
  switch (t) {
#define CASE(k) case k: return nv4p_run_##k(A, B, SFA, SFB, alpha, D, M, N, K, ldd, wsp, wsb, st, raster, swz);
    NV4P_TILES(CASE)
    default: return 1;
  }
}
extern "C" size_t nv4p_ws(int t, int raster, int swz, int M, int N, int K) {
  switch (t) {
#define WS(k) case k: return nv4t_ws_##k(M, N, K, raster, swz);
    NV4P_TILES(WS)
    default: return 0;
  }
}
