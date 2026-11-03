// Q8 (rgh, native rewrite 10-24): W8A8 int8 projections for sm_120 (RTX PRO 6000).
//   out[r, n] = sa[r] * sw[n] * (Aq[r, :] . Bq[n, :]) + bias[n] + AH[r, :] . BH[n, :]
// Aq [M, K] / Bq [N, K] int8 (per-row activation scales sa, per-output-column weight scales sw, an fp32 bias from the
// calibrated channel centres), AH [M, KH] / BH [N, KH] bf16 = the protected (outlier) channels, exact bf16 products.
// Mainloop = NG9B's (ngemm9b.cu: cp.async ring, XOR-swizzled smem + ldmatrix, double-buffered fragments, one barrier
// per k-tile, serpentine mma order) with the int8 operands addressed as 2-byte units: a 64-unit k-tile = 128 int8
// columns, each 16-unit k-step one mma.sync m16n8k32 s8 (A / B / C fragments have the m16n8k16 bf16 register layout
// byte for byte, so the ldmatrix addressing and the accumulator map are unchanged). The int32 sums are exact; after
// the int8 loop they are dequantised in registers and the protected channels run the same mainloop as bf16 m16n8k16
// into those fp32 accumulators.
// Kernels (all: M rows, rows >= M never loaded / stored; N % BN == 0; K % (2 BK) == 0; KH % BK == 0):
//   q8h_<big>_<small>: bf16 out [M, N] (ldo), rows [0, mb) big tiles (CTAs 0 .. Tb-1, m fastest), the rest small
//   q8w_<big>_<small>: N = 2H stacked [Wg; Wu], out [M, H] = SwiGLU(g, u) (gate / up rows interleaved per warp column)
//   q8p_<big>_<small>: grouped split-K, blockIdx.y = group p over int8 columns [p G, (p + 1) G) of Aq / Bq (K = G):
//                      fp32 part[p][M][N] = sa[p M + r] * sw[p N + n] * acc + (p == 0 ? bias[n] : 0) (+ AH . BH, p 0)
//   q8r_sum: out[M, N] bf16 = sum_p part[p] (p ascending, fp32), N % 4 == 0
//   q8x_prep: activation prepare, grid (M, P): x[r, lo[c]] -> v = (x - mu[c]) * is[c]; group p of G columns: amax,
//             sa[p M + r] = amax / 127, xq[r, c] = rn(v * 127 / amax) (int8); hi: xh[r, j] = x[r, hi[j]] (group 0 CTAs)
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>

#define DEVI __device__ __forceinline__

namespace q8 {

using U = uint16_t;   // the 2-byte addressing unit of both operand kinds

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
DEVI void cp_async16(uint32_t dst, const void* src, bool pred) {
    int sz = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
DEVI void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
DEVI void mma_s8(int* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI void mma_s8z(int* d, const uint32_t* a, uint32_t b0, uint32_t b1) {   // d = a . b (C = 0)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%10,%10,%10,%10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
DEVI void mma_bf(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <class T>
DEVI void mma(T* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    if constexpr (std::is_same<T, int>::value) mma_s8(d, a, b0, b1);
    else mma_bf(d, a, b0, b1);
}
DEVI void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
DEVI uint32_t pack_bf2(float a, float b) {
    __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}
DEVI float silu_f32(float x) { return __fdiv_rn(x, 1.0f + expf(-x)); }
DEVI __nv_bfloat16 swiglu1(__nv_bfloat16 g, __nv_bfloat16 u) {
    float s = __bfloat162float(__float2bfloat16_rn(silu_f32(__bfloat162float(g))));
    return __float2bfloat16_rn(s * __bfloat162float(u));
}

// ROT (rgh 10-24): randomised 128-point Hadamard rotation of a fine group, v -> H_128 (d o v) / sqrt(128) with d_i = -1
// where bit i of ROT_NEG is set: the same R = diag(d) H / sqrt(128) as the rotated weights of q8_gptq_rot.pt (the
// 1 / sqrt(128) is folded into the stored activation scale)
__device__ __constant__ uint32_t ROT_NEG[4] = {0x09bab9abu, 0xfc955699u, 0xfc9121c0u, 0xafc1071cu};
constexpr float ROT_RS = 0.08838834764831845f;
DEVI float rot_sign(int i, float v) { return ((ROT_NEG[i >> 5] >> (i & 31)) & 1u) ? -v : v; }
// butterflies over the index bits held in one thread (N consecutive elements) ...
template <int N>
DEVI void fwht_reg(float (&v)[N]) {
#pragma unroll
    for (int st = 1; st < N; st <<= 1)
#pragma unroll
        for (int e = 0; e < N; ++e)
            if ((e & st) == 0) {
                const float x = v[e], y = v[e + st];
                v[e] = x + y;
                v[e + st] = x - y;
            }
}
// ... then over the higher index bits held by lanes (lane masks 1 .. MMAX)
template <int N, int MMAX>
DEVI void fwht_lanes(float (&v)[N], int lane) {
#pragma unroll
    for (int m = 1; m <= MMAX; m <<= 1)
#pragma unroll
        for (int e = 0; e < N; ++e) {
            const float o = __shfl_xor_sync(0xffffffffu, v[e], m);
            v[e] = (lane & m) ? o - v[e] : v[e] + o;
        }
}

template <int BK>
DEVI int swz_f(int r) {
    if constexpr (BK == 32) return (r >> 1) & 3;
    else return r & 7;
}

// EPI: 2 = plain (bf16 stmatrix epilogue or fp32 partials), 1 = SwiGLU (interleaved B rows); BK in 2-byte units
template <int BM_, int BN_, int BK_, int WARPS_M, int WARPS_N, int STAGES_, int EPI_>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, STAGES = STAGES_, WM = WARPS_M, WN = WARPS_N, EPI = EPI_;
    static constexpr int CW = WARPS_M * WARPS_N;
    static constexpr int THREADS = CW * 32;
    static constexpr int WTM = BM / WARPS_M, WTN = BN / WARPS_N;
    static constexpr int MT = WTM / 16, NT = WTN / 8, KS = BK / 16, CH = BK / 8;
    static constexpr int A_STAGE = BM * BK, B_STAGE = BN * BK;
    static constexpr int SMEM_PIPE = STAGES * (A_STAGE + B_STAGE) * 2;
    static constexpr int CPAD = 8;
    static constexpr int SMEM_EPI = BM * (BN + CPAD) * 2;
    static constexpr int SMEM = SMEM_PIPE > SMEM_EPI ? SMEM_PIPE : SMEM_EPI;
    static_assert(MT >= 1 && NT % 2 == 0 && KS % 2 == 0 && STAGES >= 2, "tile");
};

// per-thread chunk loader (NG9B's): chunk q < NA = A rows r0 + q RS, q >= NA = B rows r0 + (q - NA) RS; offsets in units
template <class C>
struct Loader {
    static constexpr int RS = C::THREADS / C::CH;
    static constexpr int NA = (C::BM + RS - 1) / RS;
    static constexpr int NB = (C::BN + RS - 1) / RS;
    static constexpr int NQ = NA + NB;
    const U* pa;
    const U* pb;
    uint32_t offa[NA], offb[NB];
    uint32_t dst0;
    uint32_t amask;
    int r0;
    // K, lda, ldb in units; n1 >= 0: SwiGLU interleave (gate rows from n0, up rows from n1)
    DEVI void init(const U* A, const U* B, int m0, int n0, int n1, int M, int lda, int ldb) {
        const int tid = threadIdx.x;
        r0 = tid / C::CH;
        const int c = tid % C::CH;
        dst0 = (r0 * C::BK + ((c ^ swz_f<C::BK>(r0)) * 8)) * 2;
        pa = A;
        pb = B;
        amask = 0;
#pragma unroll
        for (int q = 0; q < NA; ++q) {
            const int r = r0 + q * RS;
            const int gm = m0 + r;
            const bool ok = gm < M;
            offa[q] = (uint32_t)((ok ? gm : 0) * lda + c * 8);
            amask |= (ok ? 1u : 0u) << q;
        }
#pragma unroll
        for (int q = 0; q < NB; ++q) {
            const int r = r0 + q * RS;
            int wr;
            if (C::EPI == 1 && n1 >= 0) {
                const int wn = r / C::WTN, h = (r % C::WTN) >= C::WTN / 2, cc = r % (C::WTN / 2);
                wr = (h ? n1 : n0) + wn * (C::WTN / 2) + cc;
            } else {
                wr = n0 + r;
            }
            offb[q] = (uint32_t)(wr * ldb + c * 8);
        }
    }
    DEVI void issue_all(uint32_t sA, uint32_t sB, int k0) {
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            if (q < NA) {
                if (C::BM % RS == 0 || r0 + q * RS < C::BM)
                    cp_async16(sA + dst0 + q * RS * C::BK * 2, pa + offa[q] + k0, (amask >> q) & 1);
            } else {
                const int qb = q - NA;
                if (C::BN % RS == 0 || r0 + qb * RS < C::BN)
                    cp_async16(sB + dst0 + qb * RS * C::BK * 2, pb + offb[qb] + k0, true);
            }
        }
    }
};

template <class C>
struct Frag {
    uint32_t a_off[C::KS], b_off[C::KS];
    uint32_t af[2][C::MT][4];
    uint32_t bf[2][C::NT][2];
    DEVI void init(int wm, int wn, int lane) {
        constexpr int BK = C::BK;
        const int a_r = wm * C::WTM + (lane % 16);
        const int a_f = swz_f<BK>(lane % 16);
        const int a_kc = lane / 16;
        const int b_r = wn * C::WTN + (lane / 16) * 8 + (lane % 8);
        const int b_f = swz_f<BK>((lane / 16) * 8 + (lane % 8));
        const int b_kc = (lane / 8) % 2;
#pragma unroll
        for (int ks = 0; ks < C::KS; ++ks) {
            a_off[ks] = (a_r * BK + (((ks * 2 + a_kc) ^ a_f) * 8)) * 2;
            b_off[ks] = (b_r * BK + (((ks * 2 + b_kc) ^ b_f) * 8)) * 2;
        }
    }
    DEVI void load(int buf, uint32_t sA, uint32_t sB, int ks) {
        const uint32_t a = sA + a_off[ks];
        const uint32_t b = sB + b_off[ks];
#pragma unroll
        for (int i = 0; i < C::MT; ++i)
            ldsm_x4(a + i * 16 * C::BK * 2, af[buf][i][0], af[buf][i][1], af[buf][i][2], af[buf][i][3]);
#pragma unroll
        for (int j = 0; j < C::NT; j += 2)
            ldsm_x4(b + j * 8 * C::BK * 2, bf[buf][j][0], bf[buf][j][1], bf[buf][j + 1][0], bf[buf][j + 1][1]);
    }
};

// acc (+)= A[m0:m0+BM, :Ku] . B-tile[:, :Ku] (Ku units); T = int (int8 data, m16n8k32 s8) or float (bf16, m16n8k16)
template <class C, class T, bool ZERO>
DEVI void mainloop(const U* __restrict__ A, const U* __restrict__ B, int M, int Ku, int lda, int ldb, int m0, int n0,
                   int n1, U* smem, T (&acc)[C::MT][C::NT][4]) {
    constexpr int KS = C::KS, BK = C::BK, S = C::STAGES, MT = C::MT, NT = C::NT;
    const uint32_t sA0 = smem_u32(smem);
    const uint32_t sB0 = sA0 + S * C::A_STAGE * 2;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int KT = Ku / BK;
    auto sa = [&](int s) { return sA0 + s * C::A_STAGE * 2; };
    auto sb = [&](int s) { return sB0 + s * C::B_STAGE * 2; };
    if constexpr (ZERO) {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][j][q] = T(0);
    }
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
                    mma<T>(acc[i][j], fr.af[cb][i], fr.bf[cb][j][0], fr.bf[cb][j][1]);
                }
            if (ks == 0) cp_commit();
        }
        if (++s == S) s = 0;
    }
    cp_wait<0>();
    __syncthreads();
}

// the global column of accumulator (j, e) of this thread: plain tiles n0 + wn WTN + j 8 + 2 t4 + e; SwiGLU tiles
// (o0 = the tile's first output column, H = N / 2): gate (j < NT / 2) o0 + .., up H + o0 + ..
template <class C>
DEVI int col_of(int j, int e, int nbase, int H) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wn = warp % C::WN, t4 = lane % 4;
    if constexpr (C::EPI == 1) {
        constexpr int NH = C::NT / 2;
        const int jj = j < NH ? j : j - NH;
        return (j < NH ? 0 : H) + nbase + wn * (C::WTN / 2) + jj * 8 + t4 * 2 + e;
    } else {
        return nbase + wn * C::WTN + j * 8 + t4 * 2 + e;
    }
}
template <class C>
DEVI int row_of(int i, int h, int m0) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN;
    return m0 + wm * C::WTM + i * 16 + lane / 4 + h * 8;
}

// int32 sums -> fp32: sa[r] sw[n] acc + bias[n] (bias may be null)
template <class C>
DEVI void dequant(const int (&ia)[C::MT][C::NT][4], float (&fa)[C::MT][C::NT][4], const float* __restrict__ sa,
                  const float* __restrict__ sw, const float* __restrict__ bias, int M, int m0, int nbase, int H) {
    float rs[C::MT][2];
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = row_of<C>(i, h, m0);
            rs[i][h] = r < M ? sa[r] : 0.f;
        }
#pragma unroll
    for (int j = 0; j < C::NT; ++j) {
        const int c0 = col_of<C>(j, 0, nbase, H);
        const float2 w = *reinterpret_cast<const float2*>(sw + c0);
        const float2 b = bias ? *reinterpret_cast<const float2*>(bias + c0) : make_float2(0.f, 0.f);
#pragma unroll
        for (int i = 0; i < C::MT; ++i)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float s = rs[i][q >> 1] * ((q & 1) ? w.y : w.x);
                fa[i][j][q] = fmaf((float)ia[i][j][q], s, (q & 1) ? b.y : b.x);
            }
    }
}

// plain bf16 epilogue: stmatrix.x4 -> padded smem tile -> 16-byte stores of rows < M (NG9B's store_tile)
template <class C>
DEVI void store_tile(const float (&v)[C::MT][C::NT][4], __nv_bfloat16* __restrict__ Cm, int M, int ldc, int m0, int n0,
                     U* smem) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    constexpr int LD = C::BN + C::CPAD;
    const int mi = lane / 8, rr = lane % 8;
    const uint32_t base = smem_u32(smem) +
        (uint32_t)(((wm * C::WTM + (mi & 1) * 8 + rr) * LD + wn * C::WTN + (mi >> 1) * 8) * 2);
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int j = 0; j < C::NT; j += 2) {
            const uint32_t a = base + (uint32_t)((i * 16 * LD + j * 8) * 2);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(a),
                         "r"(pack_bf2(v[i][j][0], v[i][j][1])), "r"(pack_bf2(v[i][j][2], v[i][j][3])),
                         "r"(pack_bf2(v[i][j + 1][0], v[i][j + 1][1])), "r"(pack_bf2(v[i][j + 1][2], v[i][j + 1][3]))
                         : "memory");
        }
    named_sync(1, C::THREADS);
    constexpr int CPR = C::BN / 8;
    const __nv_bfloat16* sm = reinterpret_cast<const __nv_bfloat16*>(smem);
    for (int i = threadIdx.x; i < C::BM * CPR; i += C::THREADS) {
        int r = i / CPR, c = i % CPR;
        int gm = m0 + r;
        if (gm < M)
            *reinterpret_cast<uint4*>(Cm + (size_t)gm * ldc + n0 + c * 8) = *reinterpret_cast<const uint4*>(sm + r * LD + c * 8);
    }
}

// SwiGLU register epilogue (NG9B's store_swiglu): n-tile j < NT/2 = gate columns, j + NT/2 the matching up columns
template <class C>
DEVI void store_swiglu(const float (&v)[C::MT][C::NT][4], __nv_bfloat16* __restrict__ Out, int M, int ldo, int m0,
                       int o0) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int g = lane / 4, t4 = lane % 4;
    constexpr int NH = C::NT / 2;
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = m0 + wm * C::WTM + i * 16 + g + h * 8;
            if (r >= M) continue;
            __nv_bfloat16* row = Out + (size_t)r * ldo + o0 + wn * (C::WTN / 2) + t4 * 2;
#pragma unroll
            for (int j = 0; j < NH; ++j) {
                const __nv_bfloat162 gg = __floats2bfloat162_rn(v[i][j][2 * h], v[i][j][2 * h + 1]);
                const __nv_bfloat162 uu = __floats2bfloat162_rn(v[i][j + NH][2 * h], v[i][j + NH][2 * h + 1]);
                __nv_bfloat162 o;
                o.x = swiglu1(gg.x, uu.x);
                o.y = swiglu1(gg.y, uu.y);
                *reinterpret_cast<__nv_bfloat162*>(row + j * 8) = o;
            }
        }
}

// fp32 partial epilogue: float2 stores of rows < M
template <class C>
DEVI void store_part(const float (&v)[C::MT][C::NT][4], float* __restrict__ P, int M, int N, int m0, int n0) {
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = row_of<C>(i, h, m0);
            if (r >= M) continue;
            float* row = P + (size_t)r * N;
#pragma unroll
            for (int j = 0; j < C::NT; ++j)
                *reinterpret_cast<float2*>(row + col_of<C>(j, 0, n0, 0)) = make_float2(v[i][j][2 * h], v[i][j][2 * h + 1]);
        }
}
}  // namespace q8
