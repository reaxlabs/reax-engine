# Kernels in our mammoth engine that are not in the reference (full-source record)

Every compiled part shipped in `/opt/reax/engine` is listed here with its source and exact build line. Everything not
listed is the reference's (reax-engine 5467286 / ref-mammoth@sha256:3080a593) or plain Python / Triton compiled at start.
Teammates' code is adopted verbatim (team code, free to adopt) and credited.

## Native: `/opt/reax/engine/libf8g.so` (from rgh's u4 image, unchanged)

Origin: rgh, team image `86.38.238.34/rgh/engine@sha256:22e5182d7968fcf07a1ef66d16ae49054c3384f9f5a14d7e03c82bc13928c085`,
`/opt/reax-m/engine/` (rgh's KERNELS.md, section "Native", reproduced below). Loaded with ctypes by `f8g.py` only when
`MAM_F8=1` or `MAM_F8_GU=1`; with both unset the library is never opened.

FP8 (e4m3) rowwise GEMM and fused gate/up GEMM + SwiGLU/quant epilogue for sm_120a (RTX PRO 6000 Blackwell).

| file | md5 | size (bytes) |
|---|---|---|
| `/opt/reax/engine/libf8g.so` | `c9dc7b0be051f93a9e5ea60289d96358` | 2997968 |
| `/opt/reax/engine/f8g.cu` | `921936f10efc8dd37375fdde2f3e1039` | 17199 |
| `/opt/reax/engine/f8common.cuh` (included by f8g.cu) | `951eca4724bab5c8fe168ea75cbeb06b` | 18254 |

Exact build (rgh's record: the build that produced the shipped binary, 2026-10-27 06:46Z, Ubuntu 26.04.1, host compiler
g++ 15.2.0, NVIDIA nvcc 13.4 V13.4.92 `cuda_13.4.r13.4/compiler.38855100_0`):

```
cd <directory holding f8g.cu and f8common.cuh>
/usr/local/cuda-13.4/bin/nvcc -O3 -arch=sm_120a -shared -Xcompiler -fPIC -cudart static -Xptxas -v -o libf8g_r3.so f8g.cu
# installed as libf8g.so (renamed only)
```

Rebuild check (rgh, 2026-10-27): nvcc writes a per-compilation anonymous-namespace hash into the local symbol table, so
an unstripped rebuild's md5 differs; the `strip --strip-all` md5 (`cbd2aed22902754d74f98f010e2b83ba`), every loadable
section, both sm_120a cubins (`ba8788ebc0e6d8231fab02e39970c6e3`, `546fedbed6e630fb5ab2792e0a2eddb4`) and the PTX
(`93876e1fc6af853e788944514e229b78`) are identical to the shipped binary. Keep `-Xptxas -v` (it changes the stripped bytes).

## c8a FP8 tile table (`f8g.py` `_TAB`, `MAM_F8TAB=1`): bit-identical by construction

From rgh's c8a image (`86.38.238.34/rgh/engine@sha256:224acedd3141607eb99297bec4633631f3c419b1b93f418bba689aa816d8a65a`,
dashboard #345, 1.179 exact): for the four FP8 GEMM shapes (N, K) = (5120, 6144) GDN out_proj / attention o_proj,
(5120, 17408) down, (14336, 5120) attention q|k|v, (16384, 5120) GDN in_proj_qkv|z, and the step sizes of the c8 / c16 closed
loops (M 961..3328, 64-row buckets), a measured (cfg, mbig, raster) replaces the wave model's pick. Same libf8g.so (md5
above), no new compiled code. Why the answers cannot change (checked on the source by `tests/test_rgh_c8a_q48m.py`; GPU
identity is checked by the screen):

1. f8g.cu has one mma (`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`), used by `mainloop8` for every config.
2. A k-tile of BK units (64 or 32) is KS = BK/16 k-steps of one mma each (16 units = 32 e4m3), walked kt = 0..KT-1,
   ks = 0..KS-1: every fp32 accumulator gets the 32-wide k chunks 0, 1, 2, ... in increasing order from 0.f, for any BK.
3. Tile shape, warp layout, `mbig` (which rows go to 128-row vs 64-row tiles) and `raster` (CTA order) only choose which
   thread computes an element, never its arithmetic.
4. The rowwise epilogue is per element `acc * sa[r] * sw[n]` then bf16, the same for every config.
5. A config that cannot cover K or N exactly returns an error (launch -2 / -3 / -4), never a partial sum; every table
   entry's K is a multiple of 4·BK and N of BN (test).

## Native: `/opt/reax/engine/libnv4g.so` (from rgh's q48m image, unchanged)

Origin: rgh, team image `86.38.238.34/rgh/engine@sha256:859001e190e4cfd0aecb028626146db9ef43c5b7a03c12aa0eeda757428f1533`
(dashboard #347 q48m), `/opt/reax/engine/`. Loaded with ctypes by `nv4.py` only when `MAM_NV4` resolves to a plan
(`nv4_plan.resolve`); with `MAM_NV4` unset neither `nv4.py` nor the library is imported.

Dense NVFP4 GEMM, D bf16 = alpha · A_fp4 · B_fp4ᵀ (block-scaled `mma.sync ... kind::mxf4nvf4.block_scale`, e2m1 x e2m1,
ue4m3 scale per 16 K, fp32 accumulation, alpha = g_x · g_w from a device pointer) for sm_120a, on the NVIDIA CUTLASS 4.5.0
sm120 block-scaled collective (headers only, BSD-3-Clause; the copy bundled with flashinfer 0.6.18.post1,
`flashinfer/data/cutlass/{include,tools/util/include}`). Six tile/schedule variants, one translation unit each:
0 = 128x128x128 auto, 1 = 128x128x256 cooperative, 2 = 128x256x128 cooperative, 3 = 128x128x128 ping-pong (default,
`MAM_NV4_VAR`), 4 = 128x128x256 ping-pong, 5 = 128x64x256 ping-pong; `nv4_tab.json` picks a variant per (256-row M bucket,
N, K) (two entries, measured by rgh; a tile-config table, no tensors).

| file | md5 | size (bytes) |
|---|---|---|
| `/opt/reax/engine/libnv4g.so` | `02be9eea787b1576c5e4a2ea28daac5b` | 2321328 |
| `/opt/reax/engine/nv4g.cu` | `f349b385645b05fcae1653145427728c` | 6818 |
| `/opt/reax/engine/nv4_dispatch.cu` | `69a122384039a45420793942d63e7b4f` | 1242 |
| `/opt/reax/engine/nv4g_build.sh` (the build script) | `bda1ee3e582c902a91944ed2430f2e54` | 979 |
| `/opt/reax/engine/nv4_tab.json` (variant table) | `e02fd53882b7b4e6dde2735b81390468` | 46 |

Exact build (rgh's record; nvcc 13.4 V13.4.92, `CUT` = the CUTLASS 4.5.0 tree above), i.e. `nv4g_build.sh`:

```
cd <directory holding nv4g.cu and nv4_dispatch.cu>
FL="-O3 -std=c++17 -arch=sm_120a --expt-relaxed-constexpr -Xcompiler -fPIC -DNDEBUG -I$CUT/include -I$CUT/tools/util/include"
for v in 0 1 2 3 4 5; do /usr/local/cuda-13.4/bin/nvcc $FL -DNV4_VARIANT=$v -c nv4g.cu -o nv4g_$v.o; done
/usr/local/cuda-13.4/bin/nvcc $FL -c nv4_dispatch.cu -o nv4_dispatch.o
/usr/local/cuda-13.4/bin/nvcc -shared -arch=sm_120a -cudart static -o libnv4g.so nv4g_0.o nv4g_1.o nv4g_2.o nv4g_3.o nv4g_4.o nv4g_5.o nv4_dispatch.o
```

Rebuild check (kai, 2026-10-28, CPU box, nvcc 13.0 instead of 13.4, CUTLASS 4.5.0 from flashinfer 0.6.18.post1): the
build line above compiles the shipped sources; the rebuilt library has the same 8 sm_120a cubins and the same 6 kernels
(the sorted list of mangled `cuobjdump -sass` function names is identical: md5 `8c1bc344d0494ab52914f7344385ec05`), so
the binary is these sources' 6 CUTLASS instantiations. Rebuild with the recorded toolchain (kai, 2026-10-28, GPU host:
`nv4g_build.sh` with nvcc 13.4 V13.4.92 and `CUT` = flashinfer 0.6.18.post1's `flashinfer/data/cutlass`, CUTLASS 4.5.0):
`cuobjdump -sass` of the rebuilt and the shipped library is identical (34,564 lines, md5
`49f2b63890dd2c19f35dd64245a5f06e`); the library file bytes (rebuilt md5 `bb8a60d8ed552b9005e0f784ac6c1c96`) and the
extracted ELF cubins differ only outside the machine code (symbol / section metadata).

What it computes (`nv4.py`, `MAM_NV4`): at start, after the engine is built, each planned projection's bf16 weight is read
from `/model` (safetensors) and quantised to NVFP4: g_w = amax / 2688 (fp32, per tensor), one e4m3 scale per 16 K
elements; `mse` = per 16-block the best of 8 scale candidates (amax / (c · g_w), c in 8, 7, 6.5, 6, 5.5, 5, 4.5, 4) by
squared error, `rtn` = c 6. The FP8 copy is released. Activations: per call, g_x = 2^ceil(log2(amax / 2688)) (power of
two: a block's e4m3 rounding then does not depend on which rows share the batch), block scales by RTN (`nv4q.py`, the
owner's `_quant16`). An FP8-quantised input that a fused producer hands over is dequantised to bf16 first. Projection
types: `gu` (MLP gate|up), `down` (MLP down), `qkvz` (GatedDeltaNet in_proj_qkv|z), `out` (GatedDeltaNet out_proj),
`qkv` (full attention q|k|v), `o` (full attention o_proj); the last layer's qkv / o are never replaced. rgh's q48m plan
`gu+down@48-62:mse,qkvz+out@48-63:mse,qkv+o@48-62:mse` = 60 projections (MLP 15 layers x 2, GDN 12 x 2, attention 3 x 2).

## Native: `/opt/reax/engine/libf8k.so` (kai lever K, exact chained stream-K FP8 GEMMs)

Origin: kai, 2026-10-28 (own code on rgh's f8g: f8k.cu `#include "f8g.cu"` and reuses its configs, mainloop text,
epilogues). Loaded with ctypes by `f8k.py` only when `MAM_F8SK=1` (and `MAM_F8` / `MAM_F8_GU` loaded f8g); unset = the
library is never opened. Python parts: `f8k.py` (front-end, wraps `f8g.rowwise` / `f8g.swiglu_q`), `f8k_sched.py` (the
schedule in Python, mirrored by `f8k_sched.cuh`; tests compile the header for the host and compare).

| file | md5 | size (bytes) |
|---|---|---|
| `/opt/reax/engine/libf8k.so` | `d0f897dfef897def7582ef1b1e65bbb1` | 3972560 |
| `/opt/reax/engine/f8k.cu` | `51faf714acd3f2d84db3c0144cb60e30` | 15331 |
| `/opt/reax/engine/f8k_sched.cuh` (included by f8k.cu) | `a2214433021158ea9f11b484cf260a04` | 4534 |

f8k.cu also includes `f8g.cu` and `f8common.cuh` (md5s in the libf8g.so section, unchanged), so libf8k.so carries rgh's
`f8_gemm` / `f8_rowquant_launch` too (used only by the microbenchmark; the engine keeps calling them in libf8g.so).

Exact build (kai, 2026-10-28T08:47Z, GPU host 31.22.104.157 compile only, Ubuntu 26.04.1, g++ 15.2.0, NVIDIA nvcc 13.4
V13.4.92 `cuda_13.4.r13.4/compiler.38855100_0`):

```
cd <directory holding f8k.cu, f8k_sched.cuh, f8g.cu, f8common.cuh>
/usr/local/cuda-13.4/bin/nvcc -O3 -arch=sm_120a -shared -Xcompiler -fPIC -cudart static -Xptxas -v -o libf8k.so f8k.cu
```

Rebuild check (kai review, 2026-10-28T09:0xZ, same host / toolchain / line): the rebuilt file has the same size and
`cuobjdump -sass` output identical to the shipped libf8k.so (149,939 SASS lines); only host-side bytes differ, as for
libnv4g.so.

ptxas (13.4): every kernel 0 bytes spill; f8ks_r_b128x256 228 registers (f8g's f8r_b128x256: 254). Two source forms were
rejected because 13.4 spilled the 128x256 mainloop (13.0 did not): a branch that either loads or zeroes the accumulators
before the k loop (now one predicated load per fragment, `load_or_zero`), and schedule state read from shared memory (now
computed from the kernel parameters and blockIdx, i.e. uniform registers, uniform k-loop control as in f8g).

What it computes: `f8k_gemm(kind, cfg, <f8_gemm's arguments>, G, part, flag, spin_ns)`, kind 0 rowwise (cfg 0 1 2 3 5 =
f8g's 128x256, 128x128, 256x128, 128x256 BK32, 128x128 BK32 with their 64-row small tiles) or 2 SwiGLU (cfg 0 1 3), on G
persistent CTAs (G = SM count). Tiles = f8g's hetero tiles; the k-loop work (a big tile's k-tile = BM / 64 units) is cut
into G equal ranges. A tile cut by a range boundary is split at a k-tile: its HEAD [0, ks) is the FIRST work of CTA g,
which stores the fp32 accumulators unchanged into slot g and sets flag[g] = 1; its TAIL [ks, KT) is the LAST work of CTA
g + 1, which loads them as its starting accumulators, continues the same k loop and writes f8g's epilogue. If no head is
published within spin_ns (`MAM_F8SK_SPIN_US`, default 2000), the consumer claims the tile (flag 0 -> 2) and computes the
head itself: deadlock-free when the CTAs are not all co-resident. Flags are 0 again at every kernel end (graph-safe, no
memset). Refused (-5: the caller runs f8_gemm) when a CTA would own less than one big tile (I / G < R * KT; e.g. c1).

Why the answers cannot change (bit-identical to f8_gemm for every cfg; checked on the source by `tests/test_lever_k.py`,
on the GPU by `tools/gemmbench_k.py`: torch.equal vs f8_gemm with the c8a plan, also in the test mode spin_ns < 0 where
every consumer recomputes its head = the take-over path):

1. f8g accumulates every output element as one fp32 chain over the 32-wide k chunks 0, 1, 2, ... from 0.f, one
   m16n8k32 e4m3 mma per chunk, for every tile config (section above).
2. f8k's `mainloop8_seg` is f8g's `mainloop8` text with only the k-tile range made a parameter (KT = kt1 - kt0, A / B
   base pointers advanced by kt0 k-tiles) and the zero-init moved to the caller (test compares the text).
3. A split tile is NOT summed from two partial sums: the tail CTA starts from the head's exact fp32 accumulators (fp32
   store / load is exact) and runs the remaining chunks in order, so each element sees the same mma sequence on the same
   operands as in one CTA. The take-over path computes [0, ks) from 0.f and then [ks, KT): the same chain.
4. f8k.cu has no arithmetic instruction of its own (no mma; the rowwise epilogue `acc * rs * w` + `store_tile` and
   `store_swiglu8` are f8g's); partials are copied as float4 without conversion.

## Native: `/opt/reax/engine/libnv4k.so` (kai lever K, CUTLASS stream-K NVFP4 variants; LOSSY ORDER)

Origin: kai, 2026-10-28; rgh's nv4g.cu GEMM (CUTLASS 4.5.0 sm120 block-scaled collective, same element / layout / alpha
code) with `cutlass::gemm::StreamKScheduler` (sm120 -> PersistentTileSchedulerSm100StreamK over the sm90 cooperative
kernel), reduction mode Deterministic, decomposition StreamK. Variants 6 = 128x128x128, 7 = 128x256x128, 8 = 128x128x256
(cooperative). Loaded by `nv4k.py` only when `MAM_NV4K` is set (after `MAM_NV4` installed NVFP4 projections); `nv4k.Router`
replaces nv4's ctypes handle and sends the listed (N, K) to `nv4k_gemm`, everything else (and rc 2 / 3 = CUTLASS refused /
workspace too small) to rgh's `nv4_gemm` unchanged.

| file | md5 | size (bytes) |
|---|---|---|
| `/opt/reax/engine/libnv4k.so` | `1e9ede6c4a9cfb6673cb2bf1de3b2f73` | 2080936 |
| `/opt/reax/engine/nv4k.cu` | `136702a3da60a7714cd52977ea197078` | 7288 |
| `/opt/reax/engine/nv4k_dispatch.cu` | `dbc25e1ebc7d77cf1e742697b1d4bc48` | 1244 |
| `/opt/reax/engine/nv4k_build.sh` (the build script) | `9e5a3d88fd21877ab49d38f1be39033c` | 1222 |

Exact build (kai, 2026-10-28, GPU host compile only, nvcc 13.4 V13.4.92, `CUT` = flashinfer 0.6.18.post1's
`flashinfer/data/cutlass`, CUTLASS 4.5.0 (the copy in the reference image layer on the host)), i.e. `nv4k_build.sh`:

```
cd <directory holding nv4k.cu and nv4k_dispatch.cu>
FL="-O3 -std=c++17 -arch=sm_120a --expt-relaxed-constexpr -Xcompiler -fPIC -DNDEBUG -I$CUT/include -I$CUT/tools/util/include"
for v in 6 7 8; do nice -n 19 /usr/local/cuda-13.4/bin/nvcc $FL -DNV4K_VARIANT=$v -c nv4k.cu -o nv4k_$v.o; done
nice -n 19 /usr/local/cuda-13.4/bin/nvcc $FL -c nv4k_dispatch.cu -o nv4k_dispatch.o
nice -n 19 /usr/local/cuda-13.4/bin/nvcc -shared -arch=sm_120a -cudart static -o libnv4k.so nv4k_6.o nv4k_7.o nv4k_8.o nv4k_dispatch.o
```

Not bit-identical to nv4g: a split tile is reduced from fp32 partials (fixed order for a given M, so run-to-run
deterministic, but the split moves with M; q48m's NVFP4 activations are batch-dependent already through the per-call
global scale). Spends distance budget: screen it.

## Native: `/opt/reax/engine/libnv4t.so` (kai lever B, exact NVFP4 tile table)

Origin: kai, 2026-10-28; rgh's nv4g.cu GEMM (its `struct Nv4` is copied verbatim into nv4t.cu; a test compares the text)
with more CUTLASS 4.5.0 sm120 tile / schedule configurations and the persistent scheduler's raster order and swizzle as
run-time arguments. Loaded by `nv4t.py` only when `MAM_NV4TAB` is set (`1` = the shipped `nv4t_tab.json`, or a table
path) after `MAM_NV4` installed NVFP4 projections; `nv4t.Router` replaces nv4's ctypes handle and sends the calls whose
(N, K, M) hit the table to `nv4t_gemm(tile, raster, swizzle, ...)`, everything else (and rc 1 / 2 / 3 / 6 = unknown
tile / CUTLASS refused / workspace too small / bad scheduler argument) to rgh's `nv4_gemm` with the variant nv4 picked.
Tiles: 0-5 = nv4g variants 0-5 (same template arguments), 6 = 128x64x128 ping-pong, 7 = 128x64x128 cooperative,
8 = 256x128x128 cooperative, 9 = 128x64x256 cooperative, 10 = 256x64x128 cooperative; raster 0 heuristic (nv4g's),
1 along M, 2 along N; swizzle 1 (nv4g's), 2, 4, 8.

| file | md5 | size (bytes) |
|---|---|---|
| `/opt/reax/engine/libnv4t.so` | `01ed1d89009f3c07ec4735089f738074` | 3206808 |
| `/opt/reax/engine/nv4t.cu` | `167d40f08ca9a4e3450c5457122cc9c4` | 10080 |
| `/opt/reax/engine/nv4t_dispatch.cu` | `24a9cea9dfdf48a8b843e3a1819212d4` | 1514 |
| `/opt/reax/engine/nv4t_build.sh` (the build script) | `5dca154f8defbaa9345096443ad777f2` | 1253 |
| `/opt/reax/engine/nv4t_tab.json` (tile table: tile / raster / swizzle per (N, K, M range), no tensors) | `1a3326660f8c15268585e2660122723b` | 1974 |

Exact build (kai, 2026-10-28, GPU host compile only, nvcc 13.4 V13.4.92, `CUT` = flashinfer 0.6.18.post1's
`flashinfer/data/cutlass`, CUTLASS 4.5.0 (the copy in the reference image layer on the host)), i.e. `nv4t_build.sh`:

```
cd <directory holding nv4t.cu and nv4t_dispatch.cu>
FL="-O3 -std=c++17 -arch=sm_120a --expt-relaxed-constexpr -Xcompiler -fPIC -DNDEBUG -I$CUT/include -I$CUT/tools/util/include"
for t in 0 1 2 3 4 5 6 7 8 9 10; do nice -n 19 /usr/local/cuda-13.4/bin/nvcc $FL -DNV4T_TILE=$t -c nv4t.cu -o nv4t_$t.o; done
nice -n 19 /usr/local/cuda-13.4/bin/nvcc $FL -c nv4t_dispatch.cu -o nv4t_dispatch.o
nice -n 19 /usr/local/cuda-13.4/bin/nvcc -shared -arch=sm_120a -cudart static -o libnv4t.so nv4t_0.o nv4t_1.o nv4t_2.o nv4t_3.o nv4t_4.o nv4t_5.o nv4t_6.o nv4t_7.o nv4t_8.o nv4t_9.o nv4t_10.o nv4t_dispatch.o
```

Why the answers cannot change (bit-identical to nv4g for every tile / raster / swizzle): in the sm120 block-scaled
collective (`sm120_blockscaled_mma_tma.hpp`) every output element is one fp32 accumulator chain: `clear(accum)`, then one
`mma.sync ... kind::mxf4nvf4.block_scale` (k 64, scales applied inside the instruction) per 64-wide k block, k tiles in
order, k blocks in order within a tile, so a 128- or 256-wide tile K is the same chain; tile M x N, the warp layout
(cooperative 4x2, ping-pong 2x2) and which CTA owns a tile do not change any element's operands or order. Data-parallel
persistent scheduler only (no split-K / stream-K: those reduce partial sums, see libnv4k.so). The epilogue is nv4g's
(`alpha * acc` -> bf16, beta 0). Checked on the GPU: `tools/nv4tab_bench.py` (torch.equal vs the current nv4g kernel for
every configuration and M, random e2m1 data with heavy-tailed block scales, plus mse-quantised Student-t weights x
`nv4q.quant_dyn` Student-t activations for the picks; runs b3 + b4 on 10-28: 0 mismatches over all 11 tiles x raster x swizzle at 16 M buckets x 5 shapes, 106 real-data checks equal, and the engine path `nv4.gemm` through `nv4t.Router` torch.equal at 50 (shape, M) points) and at every start (`nv4t.check`: each table entry at its lowest
and highest M vs `nv4_gemm` variant 3; an entry that differs or is refused is dropped).

## Native: `/opt/reax/engine/libf8c.so` (kai lever R5-1, CUTLASS sm120 FP8 GEMMs with f8g's math; EXACT)

Origin: kai, 2026-10-28. Dense FP8 GEMMs on the NVIDIA CUTLASS 4.5.0 sm120 collective (headers only, BSD-3-Clause; the copy
bundled with flashinfer 0.6.18.post1, `flashinfer/data/cutlass`, as for libnv4g.so): TMA loads, warp-specialized cooperative
/ ping-pong mainloop (`mma.sync ... kind::f8f6f4 m16n8k32` e4m3 x e4m3 -> f32), cluster 1x1x1, stages = as many as fit next
to the epilogue (`StageCountAutoCarveout`), persistent tile scheduler (raster / swizzle are run-time arguments). Eight
tile / schedule variants x two epilogues, one translation unit each (`f8c.cu`, `-DF8C_VARIANT -DF8C_KIND`): 0 128x128x128
coop, 1 128x256x64 coop, 2 256x128x64 coop, 3 128x128x128 ping-pong, 4 128x128x64 ping, 5 64x128x128 ping, 6 128x64x128
ping, 7 128x128x64 coop; kind 0 = f8g's rowwise epilogue as an EVT, `bf16((acc * sa[r]) * sw[n])` (Sm90ColBroadcast,
Sm90RowBroadcast, fp32 multiplies in f8g's order), kind 1 = `bf16(acc)` (LinearCombination alpha 1, beta 0, no C).
`f8c_dispatch.cu` = the C ABI + `f8c_swq` (gate/up tail: rgh's `smqs1` from `f8g.cu`, included unchanged, then
`f8_rowquant`'s quant text on the row held in shared memory). Loaded with ctypes by `f8c.py` only when `MAM_F8CUT=1` (and
`MAM_F8` / `MAM_F8_GU` loaded f8g); unset = the library is never opened.

| file | md5 | size (bytes) |
|---|---|---|
| `/opt/reax/engine/libf8c.so` | `6867f5e260aeb4dc4e408d9564bdd2f1` | 4983736 |
| `/opt/reax/engine/f8c.cu` | `e705a7e5360f78cb0292174e181722bf` | 8402 |
| `/opt/reax/engine/f8c_dispatch.cu` | `8faba54c7b5270feabd9e1123af1ffa3` | 6912 |
| `/opt/reax/engine/f8c_build.sh` (the build script) | `49bdbb4ccfb50296e0ed13ef1669fab5` | 1422 |

`f8c_dispatch.cu` also includes `f8g.cu` and `f8common.cuh` (md5s in the libf8g.so section, unchanged), so libf8c.so
carries rgh's f8g kernels too (unused: the engine keeps calling them in libf8g.so).

Exact build (kai, 2026-10-28T14:42Z, GPU host 31.22.104.157 compile only, Ubuntu 26.04.1, g++ 15.2.0, NVIDIA nvcc 13.4
V13.4.92 `cuda_13.4.r13.4/compiler.38855100_0`, `CUT` = flashinfer 0.6.18.post1's `flashinfer/data/cutlass`, CUTLASS 4.5.0),
i.e. `f8c_build.sh`:

```
cd <directory holding f8c.cu, f8c_dispatch.cu, f8g.cu, f8common.cuh>
FL="-O3 -std=c++17 -arch=sm_120a --expt-relaxed-constexpr -Xcompiler -fPIC -DNDEBUG -I$CUT/include -I$CUT/tools/util/include"
for v in 0 1 2 3 4 5 6 7; do for k in 0 1; do
  nice -n 19 /usr/local/cuda-13.4/bin/nvcc $FL -DF8C_VARIANT=$v -DF8C_KIND=$k -c f8c.cu -o f8c_${v}_${k}.o; done; done
nice -n 19 /usr/local/cuda-13.4/bin/nvcc $FL -Xptxas -v -c f8c_dispatch.cu -o f8c_dispatch.o
nice -n 19 /usr/local/cuda-13.4/bin/nvcc -shared -arch=sm_120a -cudart static -o libf8c.so f8c_?_?.o f8c_dispatch.o
```

Rebuild check (kai, 2026-10-28T15:41Z, same host / toolchain / script in a fresh directory): `cuobjdump -sass` of the rebuilt
and the shipped library is identical (151,064 lines, md5 `5acd02e951fcfd01f9ed0b074c4e9436`); the file bytes differ only
outside the machine code (rebuilt md5 `0ba1b249aeafa5f02d418dc562b2eb7b`), as for libnv4g.so. ptxas: `f8c_swq` 40
registers, 0 spill.

Why the answers do not change (GPU, `tools/gemmbench_c.py`, two runs 28 Oct, results in
`/home/me/engfast-phase1/v2/runs/r5-f8cut/`): every variant x raster x swizzle candidate (5,280 + 4,224 + 96 outputs: five
shapes, M 8..3360) was `torch.equal` to f8g's c8a-plan output (M >= 256) or `torch._scaled_mm` (M < 256), and `f8c_swq` on
f8g's own bf16(acc) equals f8g's SwiGLU + `f8_rowquant` bit for bit at every M. Reason: both mainloops give each fp32
accumulator the 32-wide k chunks 0, 1, 2, ... in order from 0 with one m16n8k32 e4m3 mma each (CUTLASS sm120 tiles K by
the 32-wide MMA atom and walks k-blocks in order; no split-K / stream-K schedule is compiled), and the epilogue arithmetic
is f8g's. Switch: `MAM_F8CUT=1` routes only the (shape, 64-row M bucket) plans of `f8c._TAB` (measured, > 3 % faster than
f8g in both runs) or `MAM_F8CUT_TAB`; everything else runs f8g unchanged.
## Native: `/opt/reax/engine/libf8p.so` and `/opt/reax/engine/libnv4p.so` (kai lever N2, PDL twins)

Origin: kai, 2026-10-28. Programmatic dependent launch (PDL) twins of the native GEMMs, loaded by `pdl.py` only when
`KAI_PDL=1` (unset = off: neither `pdl.py` nor these libraries is ever loaded). While the switch is on, `f8g._lib` and
nv4's ctypes handle (and lever B's `nv4t.Router`) are routed per call to the twins; the same kernels in the same order.

- `libf8p.so` (`f8p.cu`, which includes rgh's `f8g.cu` + `f8common.cuh` unchanged): each of f8g's 14 GEMM kernels is a
  `*_pdl` twin = the same template instantiation `f8::hetero8<CB, CS, MODE>` with `griddepcontrol.wait` and
  `griddepcontrol.launch_dependents` as its first two instructions; `f8_rowquant_pdl` = f8g's row-quant body verbatim after
  the same pair; launched with `cudaLaunchKernelEx` + `cudaLaunchAttributeProgrammaticStreamSerialization`, with f8g's
  `launch()` grid / smem / tile split text (tests compare it). SASS check (cuobjdump, 10-28): every GEMM twin's opcode
  sequence equals f8g's except `ACQBULK` (wait) + `PREEXIT` (launch_dependents) and `NOP`s; the row quant has the same
  opcode multiset (scheduling order differs; elementwise per row).
- `libnv4p.so` (`nv4p.cu`, which includes `nv4t.cu` unchanged, + `nv4p_dispatch.cu`): nv4t's 11 tiles (0-5 = rgh's nv4g
  variants 0-5) compiled with CUTLASS's own grid dependency control (`-DCUTLASS_ENABLE_GDC_FOR_SM100=1`; for sm_120a
  CUTLASS emits `griddepcontrol.wait` in its producer warps) and launched by `run(stream, nullptr, launch_with_pdl =
  true)`. Review fix (10-28): CUTLASS's consumer warps read `*alpha_ptr` (epilogue scalar broadcast) before any wait, and
  alpha is written by the quant kernel right before the GEMM; so the launched kernel is `WaitFirst<GemmKernel>` (nv4p.cu:
  the same `GemmUniversal`, `cutlass::arch::wait_on_dependent_grids()` as the first statement of `operator()` in every
  thread). SASS check (cuobjdump, tests/test_kai_pdl.py): in each of the 11 `WaitFirst` kernels `ACQBULK` precedes every
  global memory instruction, and the ordered sequence of floating-point / mma opcodes equals that of exactly one of
  libnv4t.so's kernels. (These CUTLASS sm120 kernels issue no `launch_dependents`: the next kernel is released when they
  complete.) libnv4p.so also carries nv4t.cu's unused `nv4t_run_<t>` entries (the included file).

| file | md5 | size (bytes) |
|---|---|---|
| `/opt/reax/engine/libf8p.so` | `625304ca8d60d1b3d7614a209bedd252` | 4362144 |
| `/opt/reax/engine/f8p.cu` | `18098a51a6e2ed8e4c85a6fef4660887` | 9287 |
| `/opt/reax/engine/f8p_build.sh` (the build script) | `3b507234da671d090642828a921e73c3` | 533 |
| `/opt/reax/engine/libnv4p.so` | `0215fdea168ecf5335990c4179911297` | 4628504 |
| `/opt/reax/engine/nv4p.cu` | `01b3828d2e16768b5ff1668ca3a15c24` | 2662 |
| `/opt/reax/engine/nv4p_dispatch.cu` | `62c77f09dfb89cb971f7bd6a73977d98` | 1414 |
| `/opt/reax/engine/nv4p_build.sh` (the build script) | `66cc4df0d6d346a1f1772f87ef92eadc` | 1414 |

Exact build (kai, 2026-10-28, GPU host compile only, nvcc 13.4 V13.4.92, g++ 15.2.0; `CUT` = flashinfer 0.6.18.post1's
`flashinfer/data/cutlass`, CUTLASS 4.5.0, the copy in the reference image layer on the host), i.e. `f8p_build.sh` and
`nv4p_build.sh` (f8g.cu, f8common.cuh and nv4t.cu: the md5s above in this file):

```
cd <directory holding f8p.cu, f8g.cu, f8common.cuh>
nice -n 19 /usr/local/cuda-13.4/bin/nvcc -O3 -arch=sm_120a -shared -Xcompiler -fPIC -cudart static -Xptxas -v -o libf8p.so f8p.cu
cd <directory holding nv4p.cu, nv4t.cu, nv4p_dispatch.cu>
FL="-O3 -std=c++17 -arch=sm_120a --expt-relaxed-constexpr -Xcompiler -fPIC -DNDEBUG -DCUTLASS_ENABLE_GDC_FOR_SM100=1 -I$CUT/include -I$CUT/tools/util/include"
for t in 0 1 2 3 4 5 6 7 8 9 10; do nice -n 19 /usr/local/cuda-13.4/bin/nvcc $FL -DNV4T_TILE=$t -c nv4p.cu -o nv4p_$t.o; done
nice -n 19 /usr/local/cuda-13.4/bin/nvcc $FL -c nv4p_dispatch.cu -o nv4p_dispatch.o
nice -n 19 /usr/local/cuda-13.4/bin/nvcc -shared -arch=sm_120a -cudart static -o libnv4p.so nv4p_0.o nv4p_1.o nv4p_2.o nv4p_3.o nv4p_4.o nv4p_5.o nv4p_6.o nv4p_7.o nv4p_8.o nv4p_9.o nv4p_10.o nv4p_dispatch.o
```

Why the answers cannot change: no arithmetic instruction is added, removed or changed (SASS checks above); the PDL pair
only lets a kernel be scheduled while its predecessor's last wave runs, and every twin waits for the predecessor's
completion (and memory flush) before its first global memory access. The Triton side (`pdl.py`: twins of our and fla's
pinned kernels with `tl.extra.cuda.gdc_wait()` / `gdc_launch_dependents()` first, launched with `launch_pdl=True`) and
Inductor's own PDL code generation (`triton.enable_pdl`, torch 2.14) for the compiled `pplx_engine.K` table follow the
same rule. GPU check: `tools/leverbench.py` config `PDL=1` (bitwise_diff 0 vs the same config without it).

## Native: `/opt/reax/engine/libnv4s.so` (kai lever N1, SwiGLU + amax in the NVFP4 gate|up epilogue)

Origin: kai, 2026-10-28; rgh's nv4g.cu GEMM (CUTLASS 4.5.0 sm120 block-scaled collective; the mainloop types and builder
call are nv4g.cu's struct Nv4 text, a test compares them) with an own epilogue visitor (EVT root `n1::Sm90SwigluStore` over
`Sm90AccFetch`, C and D void). Loaded with ctypes by `n1.py` only when `KAI_N1` is set (after `MAM_NV4` installed NVFP4
projections); unset = neither n1.py nor the library is imported. Tiles (NV4S_TILE, one translation unit each, nv4t.cu's
numbers): 1 = 128x128x256 coop, 2 = 128x256x128 coop, 3 = 128x128x128 ping-pong, 4 = 128x128x256 ping-pong,
8 = 256x128x128 coop; raster / swizzle as run-time arguments (as libnv4t.so).

| file | md5 | size (bytes) |
|---|---|---|
| `/opt/reax/engine/libnv4s.so` | `c8495b1e6a63a93836ca970545240a0c` | 2186208 |
| `/opt/reax/engine/nv4s.cu` | `b1304633e3699b07d82541ce97d6671b` | 17297 |
| `/opt/reax/engine/nv4s_silu.cuh` (included by nv4s.cu and nv4s_dispatch.cu) | `64f14875404bc364d0d4b2a58c96bcb2` | 3457 |
| `/opt/reax/engine/nv4s_dispatch.cu` | `b878b71bea03e05e12f20b7060094252` | 2853 |
| `/opt/reax/engine/nv4s_build.sh` (the build script) | `0dd8ea402e8c65acc180cc1b63287032` | 1423 |
| `/opt/reax/engine/n1_tab.json` (fused configuration per M range: tile / raster / swizzle, no tensors) | `07d1e0ef32ed787bf1365195eb9ab59f` | 699 |

Exact build (kai, 2026-10-28T17:04Z, GPU host compile only, Ubuntu 26.04.1, NVIDIA nvcc 13.4 V13.4.92
`cuda_13.4.r13.4/compiler.38855100_0`, `CUT` = flashinfer 0.6.18.post1's `flashinfer/data/cutlass`, CUTLASS 4.5.0), i.e.
`N1_SILU=1 nv4s_build.sh` (N1_SILU=1 is also the source default):

```
cd <directory holding nv4s.cu, nv4s_silu.cuh and nv4s_dispatch.cu>
FL="-O3 -std=c++17 -arch=sm_120a --expt-relaxed-constexpr -Xcompiler -fPIC -DNDEBUG -DN1_SILU=1 -I$CUT/include -I$CUT/tools/util/include"
for t in 1 2 3 4 8; do nice -n 19 /usr/local/cuda-13.4/bin/nvcc $FL -DNV4S_TILE=$t -c nv4s.cu -o nv4s_$t.o; done
nice -n 19 /usr/local/cuda-13.4/bin/nvcc $FL -c nv4s_dispatch.cu -o nv4s_dispatch.o
nice -n 19 /usr/local/cuda-13.4/bin/nvcc -shared -arch=sm_120a -cudart static -o libnv4s.so nv4s_1.o nv4s_2.o nv4s_3.o nv4s_4.o nv4s_8.o nv4s_dispatch.o
```

What it computes (`n1.py`, `KAI_N1=1`): at install every MLP gate|up projection whose down projection is NVFP4 too gets its
rows interleaved in place (row 2j = gate row j, row 2j + 1 = up row j; packed e2m1 rows and their ue4m3 block scales moved
together, the per-tensor global scale unchanged: identical to quantising the interleaved bf16 weight, test). The fused
GEMM writes y[m, j] = bf16(silu_mul(gu_round(acc[m, 2j]), gu_round(acc[m, 2j + 1]))) and atomically maxes |y| into the
down projection's activation buffer; nv4q's quant pass (`_quant_dyn_kernel`) and the down GEMM follow unchanged. Gone per
layer: the [M, 2I] bf16 gate|up tensor and actq's `_silu_amax_kernel` (or K.silu_mul + nv4q's amax pass).

Why the answers cannot change (bit-identical to nv4g on the original rows + K.silu_mul + nv4q's amax): nv4s.cu's header.
In short: (1) same mainloop collective and per-element fp32 k chain as nv4g (tile, warp layout, CTA, raster, swizzle and
pipeline stage count do not enter the chain; interleaving B's rows moves a column, not its chain); (2) gu_round is nv4g's
LinearCombination arithmetic (fp32 alpha * acc, + beta * C with beta 0 and C cleared, bf16 RN), silu_mul the compiled
K.silu_mul's (`nv4s_silu.cuh`, only explicitly rounded intrinsics, no contraction); (3) the amax is an order-free max (warp
max + integer atomicMax on non-negative fp32 bits). Checked on the GPU by `tools/n1bench.py` (run b4 on 10-28: the silu probe 0 differences over 65280 g x 64 u; 480 fused runs = 2 real layers (42, 62) x M 37 / 611 / 1503 / 2944 x every tile / raster / swizzle, 0 differing y / amax / packed / scale / buf elements; the same check finds the 1 differing y element of an N1_SILU=0 build (nvcc expf + IEEE ops) at layer 62, M 2944, which the probe alone does not) and at every start
by `n1.check` (after the warm-up): every finite bf16 g x 64 u values through the epilogue's own device functions vs
K.silu_mul, then the whole chain on two real layers' weights (original rows rebuilt) at M 37 / 611 / 1503 / 2944: y, amax,
packed fp4, scales and buf; any difference restores the original rows (exact inverse permutation) and the base chain.

## Run-time compiled (Triton, source = the shipped Python files)

| file | kernels | switch (default off) | origin |
|---|---|---|---|
| `pplx_engine.py` | `_sconv_kernel`, `_sconv3_kernel` (lag-table causal conv, fla `causal_conv1d_fwd_kernel` math), `_gnq2_kernel` (two-source gated-norm quant) | `MAM_SCONV=1` (+ `MAM_SCONV3`) | rgh f8m |
| `pplx_engine.py` | fla 0.5.2 autotuners pinned to one config each (`_FLA_PINS`) | `MAM_FLA_PIN=1` | rgh f8m |
| `gdnchain.py` | `gdn_fwd_h_chain_kernel` (prefix -> suffix DeltaNet state recurrence in one launch; fla 0.5.2 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` body, MIT) | `MAM_SCONV=1 MAM_FLA_PIN=1` (+ `MAM_CHAIN`), state-once steps only | rgh f8m (md5 `e93610fc47acf715db2c8313c95564ea`) |
| `gdnps.py` | `gdnps_pre_kernel` (pass P: u, w, gated masked q k^T per chunk), `gdnps_seq_kernel` + `_ps_chunk` (pass S: the prefix -> suffix state chain with chunk_fwd_o's output fused in, FF-PEEL); fla's pinned `chunk_gated_delta_rule_fwd_kkt_solve_kernel` into an unfilled A | `KAI_GDNPS=1` (+ `KAI_GDNPS_PEEL`, default 1; `KAI_GDNPS_CFG`), only where `MAM_CHAIN` acts; start-up bit-equality check (`gdnps.check`) or the reference path | kai lever D: port of rgh's v1 FF ps mode (team image rgh-g9r8k07, `s1eng/fla_fused.py` `_ps_pre_kernel` / `_ps_seq_kernel`) + kai's FF-PEEL (`_ps_seq_x_kernel`) to the varlen chain layout, H=16 / HV=48; fla 0.5.2 expressions (MIT) |
| `f8g.py` | ctypes front-end of libf8g.so; c8a measured tile table `_TAB` / `_plan` (`MAM_F8TAB`); `f8_gemm2` / `f8_rowquant2` hooks (`MAM_F8RQ2`, inert: the shipped libf8g.so has no `f8_gemm2`) | `MAM_F8=1`, `MAM_F8_GU=1`; table `MAM_F8TAB=1` (rgh default 1, here 0) | rgh c8a (rgh md5 `4ca67eada06593627a17f3a9253a249f`; ours `f734ba2779a37181416695d62ebc1d2a` = rgh's with the two defaults `MAM_F8TAB` / `MAM_F8RQ2` set to "0") |
| `nv4q.py` | `_amax_kernel`, `_quant_dyn_kernel` (NVFP4 activation quant, dynamic power-of-two global scale, the owner's `fp4_triton._quant16` block quantiser); rgh mx adds `_quant_dyn46_kernel` (only for `+x46` / `+perm` methods) | `MAM_NV4` | rgh mx (md5 `dfd10cee1716f58b081408abc115b8eb`; q48m's 6d5b8302 + lane QX) |
| `nv4.py` | start-up NVFP4 weight derivation (`quant_w`, rtn / mse; rgh mx: `gptq_quant_w`, calibration `_HCap` / `calibrate` / `calib_encs`, eager torch ops replayed as a CUDA graph per 128-column block), `NV4Linear`, `_mlp` patch; ctypes front-end of libnv4g.so (or `F.scaled_mm` with `MAM_NV4_GEMM=cublaslt`) | `MAM_NV4` / `REAX1_NV4` | rgh mx (md5 `7f0a6fbb516f9a051a315e2cdf6ca09f`; q48m's 3169510f + lane QX, q48m functions unchanged) |
| `nv4seed.py`, `reax_seed/` | the PUBLIC seed set (seed 0x52454158) rendered as request bodies for the GPTQ Hessians; `reax_seed/` = reax-subnet f811bf6 `s1items.py`, `s1contract.py`, `__init__.py` note + MIT `LICENSE`, unmodified (no data file: items are generated) | gptq plans | rgh mx (md5 `cf98e852...`; generator md5s in tests/test_rgh_mx.py) |
| `mxa.py` | rgh lane MXA switches; this fork uses only `d1_ok` / `check_d1` / `plain` (lever D1 -> `gdnps.chain(nseg=1)`, Triton kernels of `gdnps.py`) | `MXA_D1` | rgh mx (md5 `f443ac955966a32d89a07edbdf506bd9`) |
| `actq.py` | `_amax_scales_kernel` (one program: global amax of an NV4 input from its FP8 row scales, data pass when that is not exact), `_quant_deq_kernel` (nv4q's quant pass on bf16(fp32(x8) * xs) computed in registers), `_silu_amax_kernel` (silu(g) * u with the compiled K.silu_mul numerics + atomic amax); the quant pass otherwise `nv4q._quant_dyn_kernel` | `KAI_ACTQ=<deq,xs,silu \| 1>` (only with `MAM_NV4`); start-up bit-equality check after the warm-up (`actq.check`) or nv4's path | kai lever Q |
| `nv4_plan.py` | none: the `MAM_NV4=q48` preset and its layer lists (`MAM_NV4_MLP`, `MAM_NV4_REST`, `MAM_NV4_METHOD`) | `MAM_NV4` | kai |
| `f8k.py`, `f8k_sched.py` | none (ctypes front-end of libf8k.so; the stream-K schedule in Python) | `MAM_F8SK=1` (+ `MAM_F8SK_SHAPES`, `MAM_F8SK_CFG`, `MAM_F8SK_MBIG`, `MAM_F8SK_SPIN_US`) | kai lever K |
| `nv4k.py` | none (ctypes front-end of libnv4k.so, variant router over nv4's handle) | `MAM_NV4K=<N:K=v or name=v,...>` | kai lever K |
| `f8c.py` | none (ctypes front-end of libf8c.so; per-(shape, M bucket) plan table `_TAB`, `MAM_F8CUT_TAB` override, `levers.py` lever `F8CUT`) | `MAM_F8CUT=1` (+ `MAM_F8CUT_SHAPES`, `MAM_F8CUT_TAB`, `MAM_F8CUT_MIN_M`) | kai lever R5-1 |
| `nv4t.py` | none (ctypes front-end of libnv4t.so, table router over nv4's handle, start-up bit-equality check) | `MAM_NV4TAB=1` (or a table path) | kai lever B |
| `pdl.py` | Triton twins (source + `gdc_wait()` / `gdc_launch_dependents()` first) of the kernels in `pdl.TRITON_KERNELS` (ours and fla's pinned ones), Inductor PDL twins of `pplx_engine.K` (`options={"triton.enable_pdl": True}`), ctypes routers to libf8p.so / libnv4p.so | `KAI_PDL=1` | kai lever N2 |
| `n1.py` | none (ctypes front-end of libnv4s.so: gate\|up row interleave, nv4.gemm hook + `_mlp` patch, start-up bit-equality check; the down projection's quant pass is nv4q's `_quant_dyn_kernel`) | `KAI_N1=1` (or a table path; `KAI_N1_TILE` forces one configuration) | kai lever N1 |
| `e63/e63_engine.py` | `_ba_kernel` (GatedDeltaNet in_proj_b\|a bf16 GEMM, fixed 32x128x128 tile and K order: batch-invariant); whole-step CUDA graphs per (token, sequence) bucket over the reference kernels; fla autotune replaced by `e63/fla_pins.json` (7 kernels) | `KAI_E63=1` (mammoth-1 only; `E63_*` knobs) | pjs E63 (team image 86.38.238.34/pjs/engine@sha256:4df738e8...; `e63/KERNELS.md` is pjs's own record). Lines changed by kai are marked `kai` (file-relative pins path, B5 compatibility, rgh's `MAM_F8_GU` hook in `forward_static`) |

Nothing in `engine/e63/` is precompiled. The FLA pins (rgh's `_FLA_PINS`, pjs's `fla_pins.json`) are Triton tile
configurations (block sizes, warps, stages) chosen by timing/bit-class sweeps on the published GPU class: no weights, scales
or activation statistics (rules: "small configuration formats are code"; ENGINE-COMPETITION.md, 25 Oct decision).
