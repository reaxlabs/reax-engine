// f8g.cu (mam lane, 10-27): FP8 e4m3 W8A8 GEMMs for sm_120 (RTX PRO 6000 Blackwell) with the PplxEngine semantics
//   rowwise:  out[r, n] = bf16(acc[r, n] * sa[r] * sw[n])            (== torch._scaled_mm rowwise, fp32 accumulate)
//   unscaled: out[r, n] = bf16(acc[r, n])                            (== torch._scaled_mm tensor-wise scales 1)
//   swiglu:   gu GEMM + the reference's _smqs math in the epilogue:  g = bf16(acc_g) * sa * sw_g, u = bf16(acc_u) * sa * sw_u,
//             y[r, c] = bf16(bf16(silu(g)) * bf16(u))  -> y [M, H] bf16 (then f8_rowquant: per-row amax / 448 e4m3)
// A [M, K] e4m3 row-major (lda), B [N, K] e4m3 row-major (= Linear.w8.t() storage), fp32 accumulate (mma.sync m16n8k32).
// Mainloop = q8.cu's (NG9B cp.async ring, XOR-swizzled smem, ldmatrix, double-buffered fragments) with the e4m3 mma.
// Build: nvcc -O3 -arch=sm_120a -shared -Xcompiler -fPIC -cudart static -o libf8g.so f8g.cu
#include "f8common.cuh"
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace f8 {
using namespace q8;

DEVI void mma_e4(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <class C>
DEVI void mainloop8(const U* __restrict__ A, const U* __restrict__ B, int M, int Ku, int lda, int ldb, int m0, int n0,
                    int n1, U* smem, float (&acc)[C::MT][C::NT][4]) {
    constexpr int KS = C::KS, BK = C::BK, S = C::STAGES, MT = C::MT, NT = C::NT;
    const uint32_t sA0 = smem_u32(smem);
    const uint32_t sB0 = sA0 + S * C::A_STAGE * 2;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int KT = Ku / BK;
    auto sa = [&](int s) { return sA0 + s * C::A_STAGE * 2; };
    auto sb = [&](int s) { return sB0 + s * C::B_STAGE * 2; };
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
    Loader<C> ld;
    ld.init(A, B, m0, n0, n1, M, lda, ldb);
#pragma unroll
    for (int t = 0; t < S - 1; ++t) {
        if (t < KT) ld.issue_all(sa(t), sb(t), t * BK);
        cp_commit();
    }
    Frag<C> fr;
    fr.init(warp / C::WN, warp % C::WN, lane);
    cp_wait<S - 2>();
    __syncthreads();
    fr.load(0, sa(0), sb(0), 0);
    int s = 0;
    for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            if (ks == KS - 1) {
                cp_wait<S - 2>();
                __syncthreads();
            }
            if (ks == 0) {
                const int tt = kt + S - 1;
                if (tt < KT) ld.issue_all(sa(tt % S), sb(tt % S), tt * BK);
            }
            if (ks < KS - 1 || kt + 1 < KT) {
                const int ns = ks < KS - 1 ? s : (s + 1 == S ? 0 : s + 1);
                fr.load((ks + 1) & 1, sa(ns), sb(ns), ks < KS - 1 ? ks + 1 : 0);
            }
            const int cb = ks & 1;
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int ii = 0; ii < MT; ++ii) {
                    const int i = (j & 1) ? MT - 1 - ii : ii;
                    mma_e4(acc[i][j], fr.af[cb][i], fr.bf[cb][j][0], fr.bf[cb][j][1]);
                }
            if (ks == 0) cp_commit();
        }
        if (++s == S) s = 0;
    }
    cp_wait<0>();
    __syncthreads();
}

struct Args8 {
    const uint8_t* A;     // [M, lda] e4m3
    const uint8_t* B;     // [N, ldb] e4m3
    const float* sa;      // [M] (null: 1)
    const float* sw;      // [N] (null: 1)
    __nv_bfloat16* out;   // [M, ldo]
    int M, N, K, lda, ldb, ldo, mb, raster;
};

// the instructions Triton 3.8 emits for the reference _smqs kernel (pplx_engine.py): exp(x) = ex2.approx.f32(x * log2e)
// (no ftz), fp32 '/' = div.full.f32 (2-ulp full-range approximation, NOT IEEE div.rn)
DEVI float ex2_approx(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
DEVI float div_full(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}
// reference _smqs element math: g = bf16(acc_g) * xs * ws_g (left to right), silu = div.full(g, 1 + ex2(g * -log2e)),
// y = bf16(bf16(silu) * bf16(u))
DEVI float smqs1(float accg, float accu, float xs, float wg, float wu) {
    const float g = __fmul_rn(__fmul_rn(__bfloat162float(__float2bfloat16_rn(accg)), xs), wg);
    const float u = __fmul_rn(__fmul_rn(__bfloat162float(__float2bfloat16_rn(accu)), xs), wu);
    const float e = ex2_approx(__fmul_rn(g, -1.44269502162933349609375f));   // 0fBFB8AA3B
    const float a = __bfloat162float(__float2bfloat16_rn(div_full(g, __fadd_rn(1.0f, e))));
    return __fmul_rn(a, __bfloat162float(__float2bfloat16_rn(u)));
}

template <class C>
DEVI void store_swiglu8(const float (&v)[C::MT][C::NT][4], const Args8& a, int m0, int o0, int H) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int g = lane / 4, t4 = lane % 4;
    constexpr int NH = C::NT / 2;
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = m0 + wm * C::WTM + i * 16 + g + h * 8;
            if (r >= a.M) continue;
            const float xs = a.sa[r];
            const int cb = o0 + wn * (C::WTN / 2) + t4 * 2;
            __nv_bfloat16* row = a.out + (size_t)r * a.ldo + cb;
#pragma unroll
            for (int j = 0; j < NH; ++j) {
                const float2 wg = *reinterpret_cast<const float2*>(a.sw + cb + j * 8);
                const float2 wu = *reinterpret_cast<const float2*>(a.sw + H + cb + j * 8);
                const float y0 = smqs1(v[i][j][2 * h], v[i][j + NH][2 * h], xs, wg.x, wu.x);
                const float y1 = smqs1(v[i][j][2 * h + 1], v[i][j + NH][2 * h + 1], xs, wg.y, wu.y);
                *reinterpret_cast<__nv_bfloat162*>(row + j * 8) = __floats2bfloat162_rn(y0, y1);
            }
        }
}

// MODE 0 = rowwise (sa, sw), 1 = unscaled, 2 = swiglu (EPI 1 config)
template <class C, int MODE>
DEVI void tile8(const Args8& a, int m0, int nt, U* smem) {
    const int H = C::EPI == 1 ? a.N / 2 : 0;
    const int nbase = C::EPI == 1 ? nt * (C::BN / 2) : nt * C::BN;
    const int n1 = C::EPI == 1 ? H + nbase : -1;
    float acc[C::MT][C::NT][4];
    mainloop8<C>(reinterpret_cast<const U*>(a.A), reinterpret_cast<const U*>(a.B), a.M, a.K / 2, a.lda / 2, a.ldb / 2, m0,
                 nbase, n1, smem, acc);
    if constexpr (MODE == 2) {
        store_swiglu8<C>(acc, a, m0, nbase, H);
    } else {
        if constexpr (MODE == 0) {
            float rs[C::MT][2];
#pragma unroll
            for (int i = 0; i < C::MT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = row_of<C>(i, h, m0);
                    rs[i][h] = r < a.M ? a.sa[r] : 0.f;
                }
#pragma unroll
            for (int j = 0; j < C::NT; ++j) {
                const float2 w = *reinterpret_cast<const float2*>(a.sw + col_of<C>(j, 0, nbase, 0));
#pragma unroll
                for (int i = 0; i < C::MT; ++i)
#pragma unroll
                    for (int q = 0; q < 4; ++q) acc[i][j][q] = acc[i][j][q] * rs[i][q >> 1] * ((q & 1) ? w.y : w.x);
            }
        }
        store_tile<C>(acc, a.out, a.M, a.ldo, m0, nbase, smem);
    }
}

template <class CB, class CS, int MODE>
DEVI void hetero8(const Args8& a, U* smem) {
    static_assert(CB::THREADS == CS::THREADS, "threads");
    // CTA order: n-tile major, m fastest, the small tiles of an n-tile right after its big tiles (F8_RASTER 1), so every
    // weight tile is read from DRAM once while it is L2-resident (the tail tiles after ALL big tiles re-streamed the whole
    // weight matrix when it is larger than L2: gate/up 178 MB)
    // (raster 0 = all big tiles first, m fastest, then the small tiles: better tail balance when the weights fit in L2)
    const int mtb = a.mb / CB::BM;
    const int mts = (a.M - a.mb + CS::BM - 1) / CS::BM;
    const int b = blockIdx.x;
    if (a.raster) {
        const int per = mtb + mts;
        const int nt = b / per, r = b - nt * per;
        if (r < mtb) {
            tile8<CB, MODE>(a, r * CB::BM, nt, smem);
        } else {
            tile8<CS, MODE>(a, a.mb + (r - mtb) * CS::BM, nt, smem);
        }
    } else {
        const int ncols = CB::EPI == 1 ? a.N / 2 : a.N;
        const int tb = mtb * (ncols / (CB::EPI == 1 ? CB::BN / 2 : CB::BN));
        if (b < tb) {
            tile8<CB, MODE>(a, (b % mtb) * CB::BM, b / mtb, smem);
        } else {
            const int bs = b - tb;
            tile8<CS, MODE>(a, a.mb + (bs % mts) * CS::BM, bs / mts, smem);
        }
    }
}
}  // namespace f8

#define F8_KERNEL(NAME, CB, CS, MODE)                                                                              \
    extern "C" __global__ void __launch_bounds__(CB::THREADS, 1) NAME(f8::Args8 a) {                               \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        f8::hetero8<CB, CS, MODE>(a, reinterpret_cast<q8::U*>(smem_raw));                                          \
    }

// (BM, BN, BK units, warps M, warps N, stages, epilogue)
using P_B128x256 = q8::Cfg<128, 256, 64, 2, 4, 2, 2>;
using P_S64x256 = q8::Cfg<64, 256, 64, 2, 4, 2, 2>;
using P_B128x128 = q8::Cfg<128, 128, 64, 2, 4, 3, 2>;
using P_S64x128 = q8::Cfg<64, 128, 64, 2, 4, 3, 2>;
using P_B256x128 = q8::Cfg<256, 128, 64, 4, 2, 2, 2>;
using W_B128x256 = q8::Cfg<128, 256, 64, 2, 4, 2, 1>;
using W_S64x256 = q8::Cfg<64, 256, 64, 2, 4, 2, 1>;
using W_B128x128 = q8::Cfg<128, 128, 64, 2, 4, 3, 1>;
using W_S64x128 = q8::Cfg<64, 128, 64, 2, 4, 3, 1>;
// BK 32 units (64 fp8 columns) deeper rings (cuBLAS's sm89 e4m3 kernel: 256x128x64, 4 stages)
using P_B128x256k = q8::Cfg<128, 256, 32, 2, 4, 4, 2>;
using P_S64x256k = q8::Cfg<64, 256, 32, 2, 4, 4, 2>;
using P_B256x128k = q8::Cfg<256, 128, 32, 4, 2, 4, 2>;
using P_S64x128k = q8::Cfg<64, 128, 32, 2, 4, 4, 2>;
using P_B128x128k = q8::Cfg<128, 128, 32, 2, 4, 6, 2>;
using W_B128x256k = q8::Cfg<128, 256, 32, 2, 4, 4, 1>;
using W_S64x256k = q8::Cfg<64, 256, 32, 2, 4, 4, 1>;
using W_B256x128k = q8::Cfg<256, 128, 32, 4, 2, 4, 1>;
using W_S64x128k = q8::Cfg<64, 128, 32, 2, 4, 4, 1>;

F8_KERNEL(f8r_b128x256, P_B128x256, P_S64x256, 0)
F8_KERNEL(f8r_b128x128, P_B128x128, P_S64x128, 0)
F8_KERNEL(f8r_b256x128, P_B256x128, P_S64x128, 0)
F8_KERNEL(f8u_b128x256, P_B128x256, P_S64x256, 1)
F8_KERNEL(f8u_b128x128, P_B128x128, P_S64x128, 1)
F8_KERNEL(f8w_b128x256, W_B128x256, W_S64x256, 2)
F8_KERNEL(f8w_b128x128, W_B128x128, W_S64x128, 2)
F8_KERNEL(f8r_b128x256k, P_B128x256k, P_S64x256k, 0)
F8_KERNEL(f8r_b256x128k, P_B256x128k, P_S64x128k, 0)
F8_KERNEL(f8r_b128x128k, P_B128x128k, P_S64x128k, 0)
F8_KERNEL(f8u_b128x256k, P_B128x256k, P_S64x256k, 1)
F8_KERNEL(f8u_b256x128k, P_B256x128k, P_S64x128k, 1)
F8_KERNEL(f8w_b128x256k, W_B128x256k, W_S64x256k, 2)
F8_KERNEL(f8w_b256x128k, W_B256x128k, W_S64x128k, 2)

// per-row e4m3 quant of a bf16 [M, H] matrix (the reference's _q / _smqs tail): s = max(amax, 1e-12) / 448,
// q = sat_e4m3(y / s); one CTA per row, H % 8 == 0
extern "C" __global__ void __launch_bounds__(512) f8_rowquant(const __nv_bfloat16* __restrict__ y, int H,
                                                              uint8_t* __restrict__ q, float* __restrict__ s) {
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
template <class CB, class CS>
int launch(void (*k)(f8::Args8), f8::Args8 a, cudaStream_t st, int mbig) {
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
    k<<<tb + ts, CB::THREADS, CB::SMEM, st>>>(a);
    return (int)cudaGetLastError();
}
}  // namespace

// kind: 0 rowwise, 1 unscaled, 2 swiglu; cfg: 0 = 128x256 (+64x256), 1 = 128x128 (+64x128), 2 = 256x128 (rowwise only)
extern "C" int f8_gemm(int kind, int cfg, const void* A, int lda, const void* B, int ldb, const float* sa, const float* sw,
                       void* out, int ldo, int M, int N, int K, int mbig, void* stream, int raster) {
    f8::Args8 a{(const uint8_t*)A, (const uint8_t*)B, sa, sw, (__nv_bfloat16*)out, M, N, K, lda, ldb, ldo, 0, raster};
    cudaStream_t st = (cudaStream_t)stream;
    if (kind == 0) {
        if (cfg == 0) return launch<P_B128x256, P_S64x256>(f8r_b128x256, a, st, mbig);
        if (cfg == 1) return launch<P_B128x128, P_S64x128>(f8r_b128x128, a, st, mbig);
        if (cfg == 2) return launch<P_B256x128, P_S64x128>(f8r_b256x128, a, st, mbig);
        if (cfg == 3) return launch<P_B128x256k, P_S64x256k>(f8r_b128x256k, a, st, mbig);
        if (cfg == 4) return launch<P_B256x128k, P_S64x128k>(f8r_b256x128k, a, st, mbig);
        if (cfg == 5) return launch<P_B128x128k, P_S64x128k>(f8r_b128x128k, a, st, mbig);
    } else if (kind == 1) {
        if (cfg == 0) return launch<P_B128x256, P_S64x256>(f8u_b128x256, a, st, mbig);
        if (cfg == 1) return launch<P_B128x128, P_S64x128>(f8u_b128x128, a, st, mbig);
        if (cfg == 3) return launch<P_B128x256k, P_S64x256k>(f8u_b128x256k, a, st, mbig);
        if (cfg == 4) return launch<P_B256x128k, P_S64x128k>(f8u_b256x128k, a, st, mbig);
    } else if (kind == 2) {
        if (cfg == 0) return launch<W_B128x256, W_S64x256>(f8w_b128x256, a, st, mbig);
        if (cfg == 1) return launch<W_B128x128, W_S64x128>(f8w_b128x128, a, st, mbig);
        if (cfg == 3) return launch<W_B128x256k, W_S64x256k>(f8w_b128x256k, a, st, mbig);
        if (cfg == 4) return launch<W_B256x128k, W_S64x128k>(f8w_b256x128k, a, st, mbig);
    }
    return -1;
}

extern "C" int f8_rowquant_launch(const void* y, int M, int H, void* q, float* s, void* stream) {
    if (M == 0) return 0;
    f8_rowquant<<<M, 512, 0, (cudaStream_t)stream>>>((const __nv_bfloat16*)y, H, (uint8_t*)q, s);
    return (int)cudaGetLastError();
}
