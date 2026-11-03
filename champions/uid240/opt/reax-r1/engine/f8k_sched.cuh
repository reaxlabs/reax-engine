// f8k_sched.cuh (kai lever K, 10-28): work schedule of the exact chained stream-K FP8 GEMMs (f8k.cu). Host + device.
// Same arithmetic as engine/f8k_sched.py (its docstring has the full description; tests/test_lever_k.py compiles this
// header for the host and compares the two):
//   tiles = f8g's hetero tiles (rows [0, mb) in BM-row big tiles, the rest in 64-row small tiles), n-tile major, big then
//   small, m fastest; a big tile's k-tile = R = BM / 64 units, a small one 1 unit; I units cut into G equal CTA ranges;
//   a cut inside a tile splits it at ks = ceil(offset / w): HEAD [0, ks) = CTA g (run first, publishes fp32 partials),
//   TAIL [ks, KT) = CTA g + 1 (run last, continues from them). Accepted only when I / G >= R * KT.
#pragma once

#if defined(__CUDACC__)
#define F8K_HD __host__ __device__ __forceinline__
#else
#define F8K_HD inline
#endif

namespace f8k {

struct Geo {
    int M, kind, cfg, BM, BMs, BN, BK, R, KT, mb, mtb, mts, nt, T;
    long long Wc, I;
};

// kind 0 (rowwise) cfg 0 1 2 3 5, kind 2 (SwiGLU) cfg 0 1 3: (BM big, BM small, BN = B rows, BK units), as f8g.cu
F8K_HD bool cfg_dims(int kind, int cfg, int* BM, int* BMs, int* BN, int* BK) {
    *BMs = 64;
    if (kind != 0 && kind != 2) return false;
    switch (cfg) {
        case 0: *BM = 128; *BN = 256; *BK = 64; return true;
        case 1: *BM = 128; *BN = 128; *BK = 64; return true;
        case 2: *BM = 256; *BN = 128; *BK = 64; return kind == 0;
        case 3: *BM = 128; *BN = 256; *BK = 32; return true;
        case 5: *BM = 128; *BN = 128; *BK = 32; return kind == 0;
        default: return false;
    }
}

F8K_HD bool geo_for(int kind, int cfg, int M, int N, int K, int mbig, Geo* g) {
    int BM, BMs, BNb, BK;
    if (!cfg_dims(kind, cfg, &BM, &BMs, &BNb, &BK)) return false;
    const int ncols = kind == 2 ? N / 2 : N;
    const int BN = kind == 2 ? BNb / 2 : BNb;
    if (M <= 0 || ncols % BN || K % (4 * BK)) return false;
    g->M = M; g->kind = kind; g->cfg = cfg;
    g->BM = BM; g->BMs = BMs; g->BN = BN; g->BK = BK; g->R = BM / BMs;
    g->KT = K / (2 * BK);
    int mb = (M / BM) * BM;
    if (mbig >= 0 && mbig < mb) mb = (mbig / BM) * BM;
    g->mb = mb;
    g->mtb = mb / BM;
    g->mts = (M - mb + BMs - 1) / BMs;
    g->nt = ncols / BN;
    g->T = g->nt * (g->mtb + g->mts);
    g->Wc = (long long)(g->R * g->mtb + g->mts) * g->KT;
    g->I = (long long)g->nt * g->Wc;
    return true;
}

F8K_HD bool sk_ok(const Geo& g, int G) { return G >= 2 && g.I / G >= (long long)g.R * g.KT; }

// tile index, first unit and unit weight of the tile holding unit u (< I)
F8K_HD void locate(const Geo& g, long long u, int* t, long long* U, int* w) {
    const long long col = u / g.Wc;
    const long long r = u - col * g.Wc;
    const long long bu = (long long)g.mtb * g.R * g.KT;
    if (r < bu) {
        const long long j = r / ((long long)g.R * g.KT);
        *t = (int)(col * (g.mtb + g.mts) + j);
        *U = col * g.Wc + j * g.R * g.KT;
        *w = g.R;
    } else {
        const long long j = (r - bu) / g.KT;
        *t = (int)(col * (g.mtb + g.mts) + g.mtb + j);
        *U = col * g.Wc + bu + j * g.KT;
        *w = 1;
    }
}

F8K_HD void split_at(const Geo& g, long long u, int* t, int* ks) {
    long long U;
    int w;
    locate(g, u, t, &U, &w);
    *ks = (int)((u - U + w - 1) / w);
}

struct CtaRange {
    int tail_t, tail_ks, first_full, last_full, head_t, head_ks;
};

F8K_HD CtaRange cta_range(const Geo& g, int G, int c) {
    const long long b = (long long)c * g.I / G, e = (long long)(c + 1) * g.I / G;
    CtaRange r{-1, 0, 0, -1, -1, 0};
    int t, ks;
    split_at(g, b, &t, &ks);
    if (ks == 0) {
        r.first_full = t;
    } else if (ks == g.KT) {
        r.first_full = t + 1;
    } else {
        r.tail_t = t; r.tail_ks = ks; r.first_full = t + 1;
    }
    if (e >= g.I) {
        r.last_full = g.T - 1;
    } else {
        split_at(g, e, &t, &ks);
        if (ks == 0) {
            r.last_full = t - 1;
        } else if (ks == g.KT) {
            r.last_full = t;
        } else {
            r.head_t = t; r.head_ks = ks; r.last_full = t - 1;
        }
    }
    return r;
}

// first row, n-tile and big? of tile t
F8K_HD void tile_rc(const Geo& g, int t, int* m0, int* ntile, bool* big) {
    const int per = g.mtb + g.mts;
    const int col = t / per, j = t - col * per;
    *ntile = col;
    if (j < g.mtb) {
        *m0 = j * g.BM;
        *big = true;
    } else {
        *m0 = g.mb + (j - g.mtb) * g.BMs;
        *big = false;
    }
}

}  // namespace f8k
