# Engine source for SN92 UID 240 (REAX engine v2, dm-all champion)

This repository publishes, byte for byte as shipped, the engine source in the committed image of UID 240:

- image: `ghcr.io/masterdev777/dm240@sha256:7941f51a96086cc6d40b527d8062c2b424335875462114aa36eee7cb9d8244d5`
- committing hotkey: `5EHYsZmpb6ANjPnH6HiHZrPyw2xqGD3ijenfhYUjQWrwih77` (UID 240; label `ai.reax.engine.hotkey`)
- crowned in the dm-all slot of 2026-10-30 17:30 UTC

Every file here sits at the same path as in the image (paths relative to `/`). `SHA256SUMS` lists all of them; each hash
equals the file in the image. Compiled files (`*.so`, `*.pyc`) are not part of the repository; they are in the image.

## The 21 native libraries

### 12 E62 libraries (blink / vision / jeffan): source, build script and toolchain per `KERNELS.md`

| library in the image | sha256 (image) | source | build |
|---|---|---|---|
| `/opt/blink/gemm/blink_actq.so` | `0c0dba0273578a521a13b7a357252953b4778c96b8a9c5b751a408bb5a7dda04` | `opt/blink/gemm/blink_actq.cu` | see `opt/blink/gemm/KERNELS.md` (exact command per library) |
| `/opt/blink/gemm/blink_gemm.so` | `afe1f0d9bd0f7d23d88ede6dd07f63fe0de4f2de40298b6d136c4803eb10c651` | `opt/blink/gemm/blink_gemm.cu` | see `opt/blink/gemm/KERNELS.md` (exact command per library) |
| `/opt/blink/gemm/blink_moegemm.so` | `f3f30d7f221003c38398c9a6398d8a6f16724ed85819ad01627925849497b819` | `opt/blink/gemm/blink_moegemm.cu` | see `opt/blink/gemm/KERNELS.md` (exact command per library) |
| `/opt/blink/gemm/blink_red.so` | `559aa8b9a45379925154fe9e4845d1ffec3c865d5e0c95b1b3f38b50f5f39b41` | `opt/blink/gemm/blink_red.cu` | see `opt/blink/gemm/KERNELS.md` (exact command per library) |
| `/opt/blink/gemm/blink_rgemm.so` | `8fe860f1db701099ab1b87b3da1672ab32ef727a8bc1e4da7036547bf32871e0` | `opt/blink/gemm/blink_rgemm.cu` | see `opt/blink/gemm/KERNELS.md` (exact command per library) |
| `/opt/blink/gemm/bmoe69.so` | `4b2a273402e5e3865cbc60679eb5a1c8efd2e758acd3e82b2351449012aaf495` | `opt/blink/gemm/bmoe69.cu` | see `opt/blink/gemm/KERNELS.md` (exact command per library) |
| `/opt/blink/gemm/k1gemm.so` | `338b4d008889710c11e3b5c2187c87fba9280ecdea3d651fd7f7b61e03e76c2e` | `opt/blink/gemm/k1gemm.cu` | see `opt/blink/gemm/KERNELS.md` (exact command per library) |
| `/opt/jeffan/gemm/jef_fp8gemm.so` | `ed6f90eb60cde24b7c76b1231d5796d95789198ae2c3fe11ac18d339dbe706da` | `opt/jeffan/gemm/jef_fp8gemm.cu + jef_fp8_cfg0..13.cu + jef_fp8_common.cuh` | see `opt/jeffan/gemm/KERNELS.md` (exact command per library) |
| `/opt/jeffan/gemm/jef_fp8override.so` | `1043f71a7e194c46ae30db9575d10e9c5dba09e8dc63fc93fa263e5a49e62f13` | `opt/jeffan/gemm/jef_fp8override.cu` | see `opt/jeffan/gemm/KERNELS.md` (exact command per library) |
| `/opt/vision/gemm/blink_gemm.so` | `0d51e393beaa2dcb26940acec1b15166988c8af501db53dca5fae3260986bcdf` | `opt/vision/gemm/blink_gemm.cu` | see `opt/vision/gemm/KERNELS.md` (exact command per library) |
| `/opt/vision/gemm/vattn.so` | `50c96c63113bde31cc9ba215efe377efa44af064495fd3e37a7bdccac6d90d55` | `opt/vision/gemm/vattn.cu` | see `opt/vision/gemm/KERNELS.md` (exact command per library) |
| `/opt/vision/gemm/vis_ops.so` | `d2b0fe13d00a9257fd47e2efe06149ba6b71beb11dbd0468cb212d1749bba65c` | `opt/vision/gemm/vis_ops.cu` | see `opt/vision/gemm/KERNELS.md` (exact command per library) |

Toolchain (from the base image, see the KERNELS.md files): vLLM 0.30.0 image, torch 2.13.0+cu130, CUDA 13.0 (nvcc V13.0.88),
NVIDIA CUTLASS headers from `flashinfer/data/cutlass`, target `sm_120a`. The KERNELS.md files describe the post-step that makes
independent rebuilds byte-identical, and list the expected sha256 of each library.

### 9 reax libraries (mammoth-1 / reax-1 engine): source and build scripts in `opt/reax/engine`

| library in the image | sha256 (image) | source - build |
|---|---|---|
| `/opt/reax/engine/libf8c.so` | `b58fd1f24fb275213d2c2becf2bcf1c56a001ed605e5f12320f86d3030033ae6` | f8c.cu, f8c_dispatch.cu (includes f8g.cu, f8common.cuh) - f8c_build.sh |
| `/opt/reax/engine/libf8g.so` | `93525784bd101a71f0d35d413ddf36c347fa2ed528e92407b312493fd7d24ec5` | f8g.cu, f8common.cuh - nvcc line in KERNELS.md |
| `/opt/reax/engine/libf8k.so` | `e6be7eb5c40f5a04ff40342325fefb9b9c2b133176f0632a9221e4c4c1e3d5e9` | f8k.cu, f8k_sched.cuh (includes f8g.cu, f8common.cuh) - nvcc line in KERNELS.md |
| `/opt/reax/engine/libf8p.so` | `2a765f3162ed84d9af1beadf5e287d660ad81ffc5f0046360962104e727fba91` | f8p.cu - f8p_build.sh |
| `/opt/reax/engine/libnv4g.so` | `fd17090a7eaf6108a49e4546526b7c65101e6b2657c8e4f6408e87ded0afe494` | nv4g.cu, nv4_dispatch.cu - nv4g_build.sh |
| `/opt/reax/engine/libnv4k.so` | `d6c33a2238791632777729ff119b9d6cecb8d80b1b6e35d0c9d13ef00ecb10c7` | nv4k.cu, nv4k_dispatch.cu - nv4k_build.sh |
| `/opt/reax/engine/libnv4p.so` | `5136a5113cb47fae8493988c9083f6799cad7ce613635ea0f8c754e3cd0d8965` | nv4p.cu, nv4p_dispatch.cu - nv4p_build.sh |
| `/opt/reax/engine/libnv4s.so` | `802e9aa289af55eb13d01fbf3dfde2c48d8ce4dc347589585386bd19e981fccb` | nv4s.cu, nv4s_silu.cuh, nv4s_dispatch.cu - nv4s_build.sh |
| `/opt/reax/engine/libnv4t.so` | `0f4567a1ab925ec3cab13e618de53fc53cba3b69c7019f162db9cafec43da561` | nv4t.cu, nv4t_dispatch.cu - nv4t_build.sh |

The same nine files also sit at `/opt/reax-r1/engine/` (the reax-1 slot's tree); their sha256 are identical to the ones above.
Toolchain: `/usr/local/cuda-13.4/bin/nvcc`, CUTLASS 4.5.0 from flashinfer 0.6.18.post1 (`flashinfer/data/cutlass`), `-arch=sm_120a`,
`-cudart static`. Each library's section in `opt/reax/engine/KERNELS.md` gives its sources, md5 and the exact nvcc lines; the image
build is recorded in `opt/reax/BUILD/` (Dockerfile, build command, source.env).

**These nine are shipped as built, not stripped, in the committed image, so a rebuild matches in its device code (SASS) but not
byte for byte in its host-side bytes. Please check them at the compiled-code level against this source** (for example
`cuobjdump -sass` of a rebuild against the shipped library, as noted per library in KERNELS.md).

## Licences

- `opt/blink`, `opt/vision`, `opt/jeffan`, `opt/e62`: Apache-2.0 (`LICENSE`); third-party parts as listed in `opt/e62/NOTICE`,
  `opt/blink/NOTICE.bj` and `opt/vision/NOTICE` (vLLM: Apache-2.0; NVIDIA CUTLASS headers: BSD-3-Clause).
- `opt/reax`, `opt/reax-r1`: REAX Engine License 1.1 (`LICENSE-REAX-ENGINE-1.1` = `licenses/LICENSE`) and Apache-2.0 as stated in
  `licenses/NOTICE`; third-party licences in `licenses/LICENSES/` (CUTLASS BSD-3-Clause) and `licenses/mammoth/`.

## Credits

pjs (E62 engines for blink / vision / jeffan, their kernels and the dispatcher), kai (mammoth-1 and reax-1 engines, kernels,
GPTQ), rgh (f8g / nv4g kernels and levers), team ruj (the five-model merge, host levers, the jeffan package, BLINK_PM_PAR).
