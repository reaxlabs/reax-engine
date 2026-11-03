// f8p.cu (kai lever N2, 10-28, KAI_PDL): programmatic dependent launch (PDL) twins of rgh's f8g FP8 kernels, sm_120a.
//
// Each f8g kernel below is f8g.cu's kernel (the same template instantiation f8::hetero8<CB, CS, MODE>, the same row-quant
// code) with two instructions put first: griddepcontrol.wait (no global memory access before the previous kernel of the
// stream has completed and its writes are visible) and griddepcontrol.launch_dependents (the next kernel may be
// scheduled once every CTA of this one has started). They are launched with cudaLaunchKernelEx and the attribute
// cudaLaunchAttributeProgrammaticStreamSerialization, so the launch of each one overlaps the previous kernel's tail.
// EXACT: no arithmetic instruction is added or changed (the PDL instructions only order the grid against its neighbours);
// the grid / smem / tile split of each launch is f8g.cu's launch() text (tests/test_kai_pdl.py compares it).
// Build: nvcc -O3 -arch=sm_120a -shared -Xcompiler -fPIC -cudart static -Xptxas -v -o libf8p.so f8p.cu (= f8g's build
// line; engine/f8p_build.sh). libf8p.so also carries rgh's f8_gemm / f8_rowquant_launch from the included f8g.cu.
#include "f8g.cu"

namespace f8p {
__device__ __forceinline__ void gdc() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
}  // namespace f8p

#define F8P_KERNEL(NAME, CB, CS, MODE)                                                                             \
    extern "C" __global__ void __launch_bounds__(CB::THREADS, 1) NAME##_pdl(f8::Args8 a) {                         \
        f8p::gdc();                                                                                                 \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        f8::hetero8<CB, CS, MODE>(a, reinterpret_cast<q8::U*>(smem_raw));                                          \
    }

F8P_KERNEL(f8r_b128x256, P_B128x256, P_S64x256, 0)
F8P_KERNEL(f8r_b128x128, P_B128x128, P_S64x128, 0)
F8P_KERNEL(f8r_b256x128, P_B256x128, P_S64x128, 0)
F8P_KERNEL(f8u_b128x256, P_B128x256, P_S64x256, 1)
F8P_KERNEL(f8u_b128x128, P_B128x128, P_S64x128, 1)
F8P_KERNEL(f8w_b128x256, W_B128x256, W_S64x256, 2)
F8P_KERNEL(f8w_b128x128, W_B128x128, W_S64x128, 2)
F8P_KERNEL(f8r_b128x256k, P_B128x256k, P_S64x256k, 0)
F8P_KERNEL(f8r_b256x128k, P_B256x128k, P_S64x128k, 0)
F8P_KERNEL(f8r_b128x128k, P_B128x128k, P_S64x128k, 0)
F8P_KERNEL(f8u_b128x256k, P_B128x256k, P_S64x256k, 1)
F8P_KERNEL(f8u_b256x128k, P_B256x128k, P_S64x128k, 1)
F8P_KERNEL(f8w_b128x256k, W_B128x256k, W_S64x256k, 2)
F8P_KERNEL(f8w_b256x128k, W_B256x128k, W_S64x128k, 2)

// f8g.cu's f8_rowquant with the griddepcontrol pair first (body verbatim; tests/test_kai_pdl.py compares it)
extern "C" __global__ void __launch_bounds__(512) f8_rowquant_pdl(const __nv_bfloat16* __restrict__ y, int H,
                                                                  uint8_t* __restrict__ q, float* __restrict__ s) {
    f8p::gdc();
    const size_t r = blockIdx.x;
    const uint4* yr = reinterpret_cast<const uint4*>(y + r * H);
    const int nc = H / 8;
    __shared__ float red[16];
    float m = 0.f;
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        const uint4 v = yr[c];
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 f = __bfloat1622float2(p[e]);
            m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
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
    uint2* qr = reinterpret_cast<uint2*>(q + r * H);
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        const uint4 v = yr[c];
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

// ------------------------------------------------------------------------------------------------ host side
namespace {
template <class K, class... Args>
int launch_pdl(K k, dim3 grid, dim3 block, int smem, cudaStream_t st, Args... args) {
    cudaLaunchConfig_t c = {};
    c.gridDim = grid;
    c.blockDim = block;
    c.dynamicSmemBytes = smem;
    c.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    c.attrs = at;
    c.numAttrs = 1;
    return (int)cudaLaunchKernelEx(&c, k, args...);
}

// f8g.cu's launch() with the launch line replaced by launch_pdl (the text above it is compared by the tests)
template <class CB, class CS>
int launch_p(void (*k)(f8::Args8), f8::Args8 a, cudaStream_t st, int mbig) {
    static_assert(CB::SMEM >= CS::SMEM, "smem");
    const int ncols = CB::EPI == 1 ? a.N / 2 : a.N;
    if (ncols % (CB::EPI == 1 ? CB::BN / 2 : CB::BN)) return -2;
    if (a.K % (2 * CB::BK * 2)) return -3;
    // rows [0, mb) big tiles, the rest small tiles (mbig: -1 = as many full big tiles as fit)
    int mb = (a.M / CB::BM) * CB::BM;
    if (mbig >= 0 && mbig < mb) mb = (mbig / CB::BM) * CB::BM;
    a.mb = mb;
    const int ntb = ncols / (CB::EPI == 1 ? CB::BN / 2 : CB::BN);
    const int nts = ncols / (CS::EPI == 1 ? CS::BN / 2 : CS::BN);
    const int tb = (mb / CB::BM) * ntb;
    const int ts = ((a.M - mb + CS::BM - 1) / CS::BM) * nts;
    if (ntb != nts) return -4;
    static void* done[64];
    bool seen = false;
    for (int i = 0; i < 64 && done[i]; ++i) seen |= done[i] == (void*)k;
    if (!seen) {
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, CB::SMEM);
        for (int i = 0; i < 64; ++i) if (!done[i]) { done[i] = (void*)k; break; }
    }
    if (tb + ts == 0) return 0;
    return launch_pdl(k, dim3(tb + ts), dim3(CB::THREADS), CB::SMEM, st, a);
}
}  // namespace

// f8_gemm's ABI and routing (f8g.cu), PDL launches of the twins
extern "C" int f8p_gemm(int kind, int cfg, const void* A, int lda, const void* B, int ldb, const float* sa, const float* sw,
                        void* out, int ldo, int M, int N, int K, int mbig, void* stream, int raster) {
    f8::Args8 a{(const uint8_t*)A, (const uint8_t*)B, sa, sw, (__nv_bfloat16*)out, M, N, K, lda, ldb, ldo, 0, raster};
    cudaStream_t st = (cudaStream_t)stream;
    if (kind == 0) {
        if (cfg == 0) return launch_p<P_B128x256, P_S64x256>(f8r_b128x256_pdl, a, st, mbig);
        if (cfg == 1) return launch_p<P_B128x128, P_S64x128>(f8r_b128x128_pdl, a, st, mbig);
        if (cfg == 2) return launch_p<P_B256x128, P_S64x128>(f8r_b256x128_pdl, a, st, mbig);
        if (cfg == 3) return launch_p<P_B128x256k, P_S64x256k>(f8r_b128x256k_pdl, a, st, mbig);
        if (cfg == 4) return launch_p<P_B256x128k, P_S64x128k>(f8r_b256x128k_pdl, a, st, mbig);
        if (cfg == 5) return launch_p<P_B128x128k, P_S64x128k>(f8r_b128x128k_pdl, a, st, mbig);
    } else if (kind == 1) {
        if (cfg == 0) return launch_p<P_B128x256, P_S64x256>(f8u_b128x256_pdl, a, st, mbig);
        if (cfg == 1) return launch_p<P_B128x128, P_S64x128>(f8u_b128x128_pdl, a, st, mbig);
        if (cfg == 3) return launch_p<P_B128x256k, P_S64x256k>(f8u_b128x256k_pdl, a, st, mbig);
        if (cfg == 4) return launch_p<P_B256x128k, P_S64x128k>(f8u_b256x128k_pdl, a, st, mbig);
    } else if (kind == 2) {
        if (cfg == 0) return launch_p<W_B128x256, W_S64x256>(f8w_b128x256_pdl, a, st, mbig);
        if (cfg == 1) return launch_p<W_B128x128, W_S64x128>(f8w_b128x128_pdl, a, st, mbig);
        if (cfg == 3) return launch_p<W_B128x256k, W_S64x256k>(f8w_b128x256k_pdl, a, st, mbig);
        if (cfg == 4) return launch_p<W_B256x128k, W_S64x128k>(f8w_b256x128k_pdl, a, st, mbig);
    }
    return -1;
}

extern "C" int f8p_rowquant_launch(const void* y, int M, int H, void* q, float* s, void* stream) {
    if (M == 0) return 0;
    return launch_pdl(f8_rowquant_pdl, dim3(M), dim3(512), 0, (cudaStream_t)stream, (const __nv_bfloat16*)y, H,
                      (uint8_t*)q, s);
}
