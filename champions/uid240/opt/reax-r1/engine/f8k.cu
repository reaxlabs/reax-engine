// f8k.cu (kai lever K, 10-28): exact "chained" stream-K for rgh's f8g FP8 e4m3 GEMMs, sm_120a (RTX PRO 6000 Blackwell).
//
// Why: at the c8 step sizes (M ~1,150-1,750) the N = 5120 GEMMs (MLP down K 17408, GDN out_proj / attention o_proj
// K 6144) are 2-5 waves of f8g's 1-CTA-per-SM tiles; the last wave is partly empty (wave efficiency 0.77-0.97, mean ~0.87
// at the profiled M). Stream-K gives every SM the same amount of k-loop work instead.
//
// Exactness (bit-identical to f8g's f8_gemm for every config): f8g accumulates each output element as one fp32 chain over
// the 32-wide k chunks 0, 1, 2, ... from 0.f (one m16n8k32 e4m3 mma per chunk; f8g.cu header / engine/KERNELS.md). Here a
// tile whose k range is split between two CTAs is NOT summed from two partial sums (that would change the rounding):
// CTA g runs the HEAD [0, ks) from 0.f and stores its fp32 accumulators unchanged (st.global.cg, 16 B per thread per
// fragment); CTA g + 1 loads them as its starting accumulators and runs the TAIL [ks, KT): the same mma instructions on
// the same chunks in the same order, and fp32 store / load is exact. The head is the FIRST work of CTA g and the tail the
// LAST work of CTA g + 1, so the consumer almost never waits. If it still finds no published head after spin_ns (e.g. the
// CTAs were not all co-resident), it claims the tile (flag 0 -> 2) and computes the head itself: again the same chain,
// same bits; deadlock-free. Flags are 0 at every kernel end (the consumer resets 1 -> 0 after reading; the producer resets
// 2 -> 0 when the consumer took over), so the kernel is CUDA-graph safe without a memset. One workspace per stream.
// Mainloop = f8g's mainloop8 with the k range as a parameter (mainloop8_seg; tests/test_lever_k.py checks the text);
// epilogues = f8g's tile8 rowwise epilogue + store_tile and store_swiglu8 (unchanged); no new arithmetic instruction.
//
// Schedule: engine/f8k_sched.cuh (tiles = f8g's hetero big/small tiles, n-tile major; equal unit ranges per CTA).
// Build: nvcc -O3 -arch=sm_120a -shared -Xcompiler -fPIC -cudart static -Xptxas -v -o libf8k.so f8k.cu
// (libf8k.so also carries rgh's f8_gemm / f8_rowquant_launch from the included f8g.cu, for the microbenchmark.)
#include "f8g.cu"
#include "f8k_sched.cuh"

namespace f8k {
using namespace q8;
using f8::Args8;
using f8::mma_e4;
using f8::store_swiglu8;

// floats per CTA slot of the partial workspace: THREADS * MT * NT * 4 of the largest config (256 threads x 128)
constexpr int PART = 256 * 128;

struct SKA {
    float* part;              // [G][PART] fp32 accumulators of the head of the tile split after CTA g (slot g)
    int* flag;                // [G] 0 idle, 1 published, 2 taken over by the consumer
    unsigned long long spin;  // consumer wait budget (ns) before it computes the head itself
    int redo;                 // test mode (spin_ns < 0 in f8k_gemm): the consumer waits for the head, ignores it and
                              // recomputes [0, ks) itself, so the cold take-over path can be checked for bit-equality
};

// f8g's mainloop8 with the k-tile range [kt0, kt1) as a parameter (KT = kt1 - kt0, the A / B base pointers advanced by kt0
// k-tiles) and the accumulators NOT zeroed (the caller starts them at 0.f or at a published head); the rest is mainloop8's
// text (tests/test_lever_k.py)
template <class C>
DEVI void mainloop8_seg(const U* __restrict__ A, const U* __restrict__ B, int M, int lda, int ldb, int m0, int n0, int n1,
                        U* smem, float (&acc)[C::MT][C::NT][4], int kt0, int kt1) {
    constexpr int KS = C::KS, BK = C::BK, S = C::STAGES, MT = C::MT, NT = C::NT;
    const uint32_t sA0 = smem_u32(smem);
    const uint32_t sB0 = sA0 + S * C::A_STAGE * 2;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int KT = kt1 - kt0;
    auto sa = [&](int s) { return sA0 + s * C::A_STAGE * 2; };
    auto sb = [&](int s) { return sB0 + s * C::B_STAGE * 2; };
    Loader<C> ld;
    ld.init(A + kt0 * BK, B + kt0 * BK, m0, n0, n1, M, lda, ldb);
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

DEVI int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}
DEVI unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
    return t;
}

// fp32 accumulators <-> slot, fragment-major so each 16-byte access is coalesced across the CTA; no conversion
template <class C>
DEVI void save_part(const float (&acc)[C::MT][C::NT][4], float* p) {
    float4* p4 = reinterpret_cast<float4*>(p);
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int j = 0; j < C::NT; ++j)
            __stcg(p4 + (i * C::NT + j) * C::THREADS + threadIdx.x,
                   make_float4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]));
}
// acc = the published head (ld) or 0.f, per element by a predicated load: no control-flow merge of two initialisations
// in front of the k loop (that merge made nvcc 13.4 spill the 128x256 tiles' mainloop)
template <class C>
DEVI void load_or_zero(float (&acc)[C::MT][C::NT][4], const float* p, bool ld) {
    const float4* p4 = reinterpret_cast<const float4*>(p);
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int j = 0; j < C::NT; ++j) {
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (ld) v = __ldcg(p4 + (i * C::NT + j) * C::THREADS + threadIdx.x);
            acc[i][j][0] = v.x;
            acc[i][j][1] = v.y;
            acc[i][j][2] = v.z;
            acc[i][j][3] = v.w;
        }
}

// The schedule values (tile, k range, role) are computed by every thread from the kernel parameters and blockIdx only, so
// the compiler keeps them in uniform registers (no vector-register cost across the k loop; uniform loop control as in
// f8g's mainloop). The one data-dependent decision (did the producer publish the head?) only selects the starting
// accumulators (predicated load) and whether the head is computed here first (cold path).
template <class C, int MODE>
DEVI void seg(const Args8& a, const SKA& s, const Geo& g, int t, int kb, int ke, int role, U* smem, int* took) {
    static_assert(C::THREADS * C::MT * C::NT * 4 <= PART, "slot");
    float acc[C::MT][C::NT][4];
    const int c = blockIdx.x;
    __syncthreads();  // the previous segment's epilogue has finished reading smem
    bool ld = false;
    if (role == 2) {
        if (threadIdx.x == 0) {
            int* f = s.flag + (c - 1);
            int v = ld_acquire(f);
            if (v != 1) {
                const unsigned long long t0 = gtimer();
                while (true) {
                    __nanosleep(200);
                    v = ld_acquire(f);
                    if (v == 1) break;
                    if (gtimer() - t0 > s.spin) {
                        v = atomicCAS(f, 0, 2);   // claim: the producer will find 2 and reset it
                        if (v != 1) v = 2;
                        break;
                    }
                }
            }
            *took = v;
        }
        __syncthreads();
        ld = *took == 1 && !s.redo;
    }
    load_or_zero<C>(acc, s.part + (size_t)(c > 0 ? c - 1 : 0) * PART, ld);
    int m0, nt;
    bool big;
    tile_rc(g, t, &m0, &nt, &big);
    const int H = C::EPI == 1 ? a.N / 2 : 0;
    const int nbase = C::EPI == 1 ? nt * (C::BN / 2) : nt * C::BN;
    const int n1 = C::EPI == 1 ? H + nbase : -1;
    const U* A = reinterpret_cast<const U*>(a.A);
    const U* B = reinterpret_cast<const U*>(a.B);
    // the head was never published: compute it here first from 0.f (same chunks, same order, same bits)
    if (role == 2 && !ld) mainloop8_seg<C>(A, B, a.M, a.lda / 2, a.ldb / 2, m0, nbase, n1, smem, acc, 0, kb);
    mainloop8_seg<C>(A, B, a.M, a.lda / 2, a.ldb / 2, m0, nbase, n1, smem, acc, kb, ke);
    if (role == 1) {
        save_part<C>(acc, s.part + (size_t)c * PART);
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            const int old = atomicCAS(s.flag + c, 0, 1);
            if (old == 2) atomicExch(s.flag + c, 0);  // the consumer computed the head itself: nobody reads the slot
        }
        return;
    }
    if (role == 2 && threadIdx.x == 0 && *took == 1) atomicExch(s.flag + (c - 1), 0);  // slot consumed: back to 0
    // epilogue = f8g tile8's
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

// persistent CTA c of G: [head] [full tiles] [tail], one loop so each tile config is instantiated once
template <class CB, class CS, int MODE>
DEVI void persist(const Args8& a, const SKA& s, const Geo& g, U* smem) {
    static_assert(CB::THREADS == CS::THREADS && CB::BN == CS::BN, "pair");
    __shared__ int took;
    const CtaRange r = cta_range(g, gridDim.x, blockIdx.x);
    const int hd = r.head_t >= 0 ? 1 : 0;
    const int nf = r.last_full >= r.first_full ? r.last_full - r.first_full + 1 : 0;
    const int n = hd + nf + (r.tail_t >= 0 ? 1 : 0);
#pragma unroll 1
    for (int w = 0; w < n; ++w) {
        const int role = w < hd ? 1 : (w < hd + nf ? 0 : 2);
        const int t = role == 1 ? r.head_t : (role == 0 ? r.first_full + (w - hd) : r.tail_t);
        const int kb = role == 2 ? r.tail_ks : 0;
        const int ke = role == 1 ? r.head_ks : g.KT;
        int m0, nt;
        bool big;
        tile_rc(g, t, &m0, &nt, &big);
        if (big) {
            seg<CB, MODE>(a, s, g, t, kb, ke, role, smem, &took);
        } else {
            seg<CS, MODE>(a, s, g, t, kb, ke, role, smem, &took);
        }
    }
}
}  // namespace f8k

#define F8K_KERNEL(NAME, CB, CS, MODE)                                                                             \
    extern "C" __global__ void __launch_bounds__(CB::THREADS, 1) NAME(f8::Args8 a, f8k::SKA s, f8k::Geo g) {      \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        f8k::persist<CB, CS, MODE>(a, s, g, reinterpret_cast<q8::U*>(smem_raw));                                   \
    }

F8K_KERNEL(f8ks_r_b128x256, P_B128x256, P_S64x256, 0)
F8K_KERNEL(f8ks_r_b128x128, P_B128x128, P_S64x128, 0)
F8K_KERNEL(f8ks_r_b256x128, P_B256x128, P_S64x128, 0)
F8K_KERNEL(f8ks_r_b128x256k, P_B128x256k, P_S64x256k, 0)
F8K_KERNEL(f8ks_r_b128x128k, P_B128x128k, P_S64x128k, 0)
F8K_KERNEL(f8ks_w_b128x256, W_B128x256, W_S64x256, 2)
F8K_KERNEL(f8ks_w_b128x128, W_B128x128, W_S64x128, 2)
F8K_KERNEL(f8ks_w_b128x256k, W_B128x256k, W_S64x256k, 2)

namespace {
template <class CB, class CS>
int launch_sk(void (*k)(f8::Args8, f8k::SKA, f8k::Geo), const f8::Args8& a, const f8k::SKA& s, const f8k::Geo& g, int G,
              cudaStream_t st) {
    static_assert(CB::SMEM >= CS::SMEM, "smem");
    static void* done[16];
    bool seen = false;
    for (int i = 0; i < 16 && done[i]; ++i) seen |= done[i] == (void*)k;
    if (!seen) {
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, CB::SMEM);
        for (int i = 0; i < 16; ++i) if (!done[i]) { done[i] = (void*)k; break; }
    }
    k<<<G, CB::THREADS, CB::SMEM, st>>>(a, s, g);
    return (int)cudaGetLastError();
}
}  // namespace

// kind 0 rowwise (cfg 0 1 2 3 5), 2 SwiGLU (cfg 0 1 3): the f8_gemm arguments + G persistent CTAs (<= SMs), the workspace
// (part: G * f8k_part_floats() fp32, flag: G int32 zeroed once at allocation) and the consumer wait budget in ns
// (negative = test mode: same budget, and every consumer recomputes its head itself; see SKA::redo).
// -2 shape not covered by cfg, -5 too few tiles (fewer than one big tile per CTA): run f8_gemm instead.
extern "C" int f8k_gemm(int kind, int cfg, const void* A, int lda, const void* B, int ldb, const float* sa, const float* sw,
                        void* out, int ldo, int M, int N, int K, int mbig, void* stream, int G, void* part, void* flag,
                        long long spin_ns) {
    f8k::Geo g;
    if (!f8k::geo_for(kind, cfg, M, N, K, mbig, &g)) return -2;
    if (!f8k::sk_ok(g, G)) return -5;
    f8::Args8 a{(const uint8_t*)A, (const uint8_t*)B, sa, sw, (__nv_bfloat16*)out, M, N, K, lda, ldb, ldo, g.mb, 1};
    f8k::SKA s{(float*)part, (int*)flag, (unsigned long long)(spin_ns < 0 ? -spin_ns : spin_ns), spin_ns < 0 ? 1 : 0};
    cudaStream_t st = (cudaStream_t)stream;
    if (kind == 0) {
        if (cfg == 0) return launch_sk<P_B128x256, P_S64x256>(f8ks_r_b128x256, a, s, g, G, st);
        if (cfg == 1) return launch_sk<P_B128x128, P_S64x128>(f8ks_r_b128x128, a, s, g, G, st);
        if (cfg == 2) return launch_sk<P_B256x128, P_S64x128>(f8ks_r_b256x128, a, s, g, G, st);
        if (cfg == 3) return launch_sk<P_B128x256k, P_S64x256k>(f8ks_r_b128x256k, a, s, g, G, st);
        if (cfg == 5) return launch_sk<P_B128x128k, P_S64x128k>(f8ks_r_b128x128k, a, s, g, G, st);
    } else if (kind == 2) {
        if (cfg == 0) return launch_sk<W_B128x256, W_S64x256>(f8ks_w_b128x256, a, s, g, G, st);
        if (cfg == 1) return launch_sk<W_B128x128, W_S64x128>(f8ks_w_b128x128, a, s, g, G, st);
        if (cfg == 3) return launch_sk<W_B128x256k, W_S64x256k>(f8ks_w_b128x256k, a, s, g, G, st);
    }
    return -1;
}

extern "C" int f8k_part_floats() { return f8k::PART; }
