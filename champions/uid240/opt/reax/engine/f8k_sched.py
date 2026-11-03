"""f8k_sched.py (kai lever K, 10-28): the work schedule of the exact chained stream-K FP8 GEMMs (f8k.cu), in Python.

engine/f8k_sched.cuh is the same arithmetic for the kernel (tests/test_lever_k.py compiles it for the host and compares).

Tiles = f8g's hetero tiles: rows [0, mb) in BM-row "big" tiles, the rest in 64-row "small" tiles, n-tile major (all big
then all small tiles of one n-tile column, m fastest = f8g raster 1). A big tile's k-tile costs R = BM / 64 units, a small
one 1 unit. The I = nt * (R * mtb + mts) * KT units are cut into G equal ranges, one per persistent CTA (G = SM count).
A cut that falls inside a tile splits it at the k-tile ks = ceil(offset / w): CTA g owns the HEAD [0, ks) and runs it
FIRST (publishes the fp32 accumulators + a flag), CTA g + 1 owns the TAIL [ks, KT) and runs it LAST (loads them as its
starting accumulators, continues the k loop, writes the epilogue). Every output element then gets the k chunks
0, 1, ..., KT * KS - 1 in order from 0.f, exactly as in f8g's one-CTA tile: the answers are bit-identical.
Accepted only when every CTA owns at least one big tile's worth of units (I // G >= R * KT): then a tile is split
between at most two CTAs and the tail's producer is always CTA g - 1.
"""

# kind -> cfg -> (BM big, BM small, BN = B rows per tile, BK in 2-byte units); the f8g.cu Cfg pairs of the same cfg number
CFGS = {
    0: {0: (128, 64, 256, 64), 1: (128, 64, 128, 64), 2: (256, 64, 128, 64), 3: (128, 64, 256, 32), 5: (128, 64, 128, 32)},
    2: {0: (128, 64, 256, 64), 1: (128, 64, 128, 64), 3: (128, 64, 256, 32)},
}
NAMES = {"down": (0, 5120, 17408), "out": (0, 5120, 6144), "o": (0, 5120, 6144), "qkvz": (0, 16384, 5120),
         "qkv": (0, 14336, 5120), "gu": (2, 34816, 5120)}
ALL = {"down", "out", "qkvz", "qkv", "gu"}
DEFAULT_SHAPES = {NAMES["down"], NAMES["out"]}
# (kind, N, K) -> cfg; overridden per (N, K) by MAM_F8SK_CFG (tools/gemmbench_k.py measures the best one)
DEFAULT_CFG = {(0, 5120, 17408): 1, (0, 5120, 6144): 1, (0, 16384, 5120): 0, (0, 14336, 5120): 0, (2, 34816, 5120): 0}


class Geo:
    __slots__ = ("M", "N", "K", "kind", "cfg", "BM", "BMs", "BN", "BK", "R", "KT", "mb", "mtb", "mts", "nt", "T", "Wc",
                 "I")

    def __repr__(self):
        return "Geo(" + ", ".join(f"{k}={getattr(self, k)}" for k in self.__slots__) + ")"


def kind_of(N, K):
    return 2 if (N, K) == NAMES["gu"][1:] else 0


def geometry(M, N, K, kind, cfg, mbig=-1):
    if kind not in CFGS or cfg not in CFGS[kind]:
        raise ValueError(f"no stream-K kernel for kind {kind} cfg {cfg}")
    BM, BMs, BNb, BK = CFGS[kind][cfg]
    g = Geo()
    g.M, g.N, g.K, g.kind, g.cfg = M, N, K, kind, cfg
    g.BM, g.BMs, g.BK, g.R = BM, BMs, BK, BM // BMs
    ncols = N // 2 if kind == 2 else N
    g.BN = BNb // 2 if kind == 2 else BNb
    if M <= 0 or ncols % g.BN or K % (4 * BK):
        raise ValueError(f"shape M {M} N {N} K {K} not covered by cfg {cfg}")
    g.KT = K // (2 * BK)
    mb = (M // BM) * BM
    if 0 <= mbig < mb:
        mb = (mbig // BM) * BM
    g.mb = mb
    g.mtb, g.mts, g.nt = mb // BM, -(-(M - mb) // BMs), ncols // g.BN
    g.T = g.nt * (g.mtb + g.mts)
    g.Wc = (g.R * g.mtb + g.mts) * g.KT
    g.I = g.nt * g.Wc
    return g


def ok(M, N, K, kind, cfg, G, mbig=-1):
    try:
        g = geometry(M, N, K, kind, cfg, mbig)
    except ValueError:
        return False
    return G >= 2 and g.I // G >= g.R * g.KT


def _locate(g, u):
    """tile index, first unit and unit weight per k-tile of the tile that holds unit u (< I)"""
    col, r = divmod(u, g.Wc)
    bu = g.mtb * g.R * g.KT
    if r < bu:
        j = r // (g.R * g.KT)
        return col * (g.mtb + g.mts) + j, col * g.Wc + j * g.R * g.KT, g.R
    j = (r - bu) // g.KT
    return col * (g.mtb + g.mts) + g.mtb + j, col * g.Wc + bu + j * g.KT, 1


def _split(g, u):
    t, U, w = _locate(g, u)
    return t, (u - U + w - 1) // w


def cta_range(g, G, c):
    """CTA c's work: tail (tail_t, [tail_ks, KT)) or tail_t -1; full tiles first_full..last_full (inclusive, empty when
    last < first); head (head_t, [0, head_ks)) or head_t -1"""
    b, e = c * g.I // G, (c + 1) * g.I // G
    r = {"tail_t": -1, "tail_ks": 0, "first_full": 0, "last_full": -1, "head_t": -1, "head_ks": 0}
    t, ks = _split(g, b)
    if ks == 0:
        r["first_full"] = t
    elif ks == g.KT:
        r["first_full"] = t + 1
    else:
        r["tail_t"], r["tail_ks"], r["first_full"] = t, ks, t + 1
    if e >= g.I:
        r["last_full"] = g.T - 1
    else:
        t, ks = _split(g, e)
        if ks == 0:
            r["last_full"] = t - 1
        elif ks == g.KT:
            r["last_full"] = t
        else:
            r["head_t"], r["head_ks"], r["last_full"] = t, ks, t - 1
    return r


def plan(M, N, K, kind, cfg, G, mbig=-1):
    """per CTA, in execution order: [(tile, k-tile begin, k-tile end, role)], role head / full / tail"""
    g = geometry(M, N, K, kind, cfg, mbig)
    out = []
    for c in range(G):
        r = cta_range(g, G, c)
        segs = []
        if r["head_t"] >= 0:
            segs.append((r["head_t"], 0, r["head_ks"], "head"))
        segs += [(t, 0, g.KT, "full") for t in range(r["first_full"], r["last_full"] + 1)]
        if r["tail_t"] >= 0:
            segs.append((r["tail_t"], r["tail_ks"], g.KT, "tail"))
        out.append(segs)
    return out


def tile_rc(g, t):
    """(first row, n-tile, big?) of tile t"""
    col, j = divmod(t, g.mtb + g.mts)
    if j < g.mtb:
        return j * g.BM, col, True
    return g.mb + (j - g.mtb) * g.BMs, col, False


def parse_shapes(s):
    s = (s or "").strip()
    if not s:
        return set(DEFAULT_SHAPES)
    names = ALL if s == "all" else [x.strip() for x in s.split(",") if x.strip()]
    bad = [n for n in names if n not in NAMES]
    if bad:
        raise ValueError(f"MAM_F8SK_SHAPES: unknown {bad} (known: {sorted(NAMES)} or all)")
    return {NAMES[n] for n in names}


def parse_cfg(s):
    out = {}
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            nk, c = part.split("=")
            N, K = (int(x) for x in nk.split(":"))
            c = int(c)
        except ValueError:
            raise ValueError(f"MAM_F8SK_CFG: bad entry {part!r} (want N:K=cfg)") from None
        if c not in CFGS[kind_of(N, K)]:
            raise ValueError(f"MAM_F8SK_CFG: cfg {c} has no stream-K kernel for N {N} K {K}")
        out[(N, K)] = c
    return out
