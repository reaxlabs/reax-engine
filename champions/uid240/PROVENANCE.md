# UID 240 engine (REAX engine competition v2, dm-all champion): provenance

This directory is the engine source of SN92 UID 240, copied unchanged from the miner's public repository. Every file
except this one (`PROVENANCE.md`) is byte-identical to the source commit below; `SHA256SUMS` (the miner's file) lists them
(`sha256sum -c SHA256SUMS` in this directory passes for all 315 files). The miner's own `README.md` describes the engine,
its 21 native libraries and their build.

| | |
|---|---|
| Source repository | https://github.com/masterdev777/sn92-uid240-engine |
| Source commit | `42c543a35f108eb4296c7924789f6fd9e35efca3` |
| Committed image (miner) | `ghcr.io/masterdev777/dm240@sha256:7941f51a96086cc6d40b527d8062c2b424335875462114aa36eee7cb9d8244d5` |
| Mirror (same digest) | `ghcr.io/reaxlabs/reax-engine-dm240@sha256:7941f51a96086cc6d40b527d8062c2b424335875462114aa36eee7cb9d8244d5` (`mirrors/champions.txt`) |
| Base image of the mammoth-1 path | `ghcr.io/reaxlabs/reax-engine-ref-mammoth@sha256:3080a593060542bccbdaf7cf0c66593fc182d9753b3facd50206df86d93b8588` |
| Committing hotkey | `5EHYsZmpb6ANjPnH6HiHZrPyw2xqGD3ijenfhYUjQWrwih77` |
| Crowned | dm-all slot, 2026-10-30 17:30 UTC |
| Compiled files | `*.so` / `*.pyc` are not in the source repository; they are in the image |

## Independent rebuild

We rebuilt the engine from this source commit in our own CI (2 Nov 2026): the 12 E62 libraries are byte-identical to the
image, the 9 reax libraries (`opt/reax/engine`, `opt/reax-r1/engine`) are identical at the SASS level (they ship unstripped,
so host-side bytes differ), all built for `sm_120a`. The build steps per library are in the `KERNELS.md` files and in
`opt/reax/BUILD/`.

## Licences (not changed by this repository)

- `opt/blink`, `opt/vision`, `opt/jeffan`, `opt/e62`: Apache-2.0 (`LICENSE`); third-party parts as listed in
  `opt/e62/NOTICE`, `opt/blink/NOTICE.bj` and `opt/vision/NOTICE`.
- `opt/reax`, `opt/reax-r1`: built on the REAX Engine; REAX Engine License 1.1 (`LICENSE-REAX-ENGINE-1.1`,
  `licenses/LICENSE`) and Apache-2.0 as stated in `licenses/NOTICE`; third-party licences in `licenses/LICENSES/`
  (NVIDIA CUTLASS headers: BSD-3-Clause) and `licenses/mammoth/`.

The MIT License of the rest of this repository does not apply to this directory.

## Credits

The UID 240 team (credits in the miner's `README.md`): pjs (E62 engines for blink / vision / jeffan, their kernels and the
dispatcher), kai (mammoth-1 and reax-1 engines, kernels, GPTQ), rgh (f8g / nv4g kernels and levers), team ruj (the five-model
merge, host levers, the jeffan package, BLINK_PM_PAR).
