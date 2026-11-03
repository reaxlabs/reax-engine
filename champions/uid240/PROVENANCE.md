# UID 240 engine (REAX engine competition v2, dm-all champion): provenance

This directory is the engine source of SN92 UID 240, copied unchanged from the miner's public repository. Every source
file is byte-identical to the source commit below and listed in the miner's `SHA256SUMS`. Three files differ from the
source commit: this `PROVENANCE.md` (added), `LICENSE` (now the MIT License, see below) and the miner's two licence files,
moved unchanged to `upstream-licenses/` (`LICENSE` -> `upstream-licenses/LICENSE.Apache-2.0`, `LICENSE-REAX-ENGINE-1.1` ->
`upstream-licenses/LICENSE-REAX-ENGINE-1.1`; their sha256 equal the `SHA256SUMS` entries for `LICENSE` and
`LICENSE-REAX-ENGINE-1.1`). All other 313 entries pass `sha256sum -c SHA256SUMS` in this directory. The miner's own `README.md` describes the engine,
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

## Licence

This directory is licensed under the **MIT License** (`LICENSE`), with the authors' copyright and attribution kept.
Third-party code inside it keeps its own licence, as listed in `LICENSE` (scope), the file headers, `opt/e62/NOTICE`,
`opt/blink/NOTICE.bj`, `opt/vision/NOTICE`, the `e63/NOTICE` files and `licenses/` (vLLM: Apache-2.0; NVIDIA CUTLASS
headers: BSD-3-Clause; flash-linear-attention: MIT; Hugging Face transformers: Apache-2.0; model prompt contracts:
Apache-2.0 by their authors; mammoth-1 notes in `licenses/mammoth/`). File headers are left as published, so a header that
says Apache-2.0 or REAX Engine License 1.1 shows the licence the code was first published under.

How the authors first published it (texts kept in `upstream-licenses/`, notices kept in place):
- `opt/blink`, `opt/vision`, `opt/jeffan`, `opt/e62`: Apache-2.0.
- `opt/reax`, `opt/reax-r1` (built on the REAX Engine): REAX Engine License 1.1 and Apache-2.0 as stated in `licenses/NOTICE`.

Basis for the MIT licence here:
- The parts the authors published under Apache-2.0 may be redistributed as part of a larger work whose licence differs,
  provided the Apache-2.0 text and notices are kept (Apache-2.0 section 4); they are kept in `upstream-licenses/`, the
  NOTICE files and the file headers.
- `opt/reax` and `opt/reax-r1` are built on the REAX Engine, whose licensor (productivity-boost.com Betriebs UG
  (haftungsbeschränkt) & Co. KG) licenses its own code in this repository under MIT.
- Entries to the REAX subnet's competitions grant the REAX Engine's licensor a licence to their changes: the REAX Engine
  License 1.0 (tags `license-1.0` to `license-1.1` of this repository) grants a perpetual, sublicensable licence for entries
  based on it, and the engine competition v2 licence condition (reax-subnet `docs/ENGINE-COMPETITION-V2.md`) grants a
  licence to use, modify and merge a committed `dm-all` engine into the REAX Engine.

## Credits

The UID 240 team (credits in the miner's `README.md`): pjs (E62 engines for blink / vision / jeffan, their kernels and the
dispatcher), kai (mammoth-1 and reax-1 engines, kernels, GPTQ), rgh (f8g / nv4g kernels and levers), team ruj (the five-model
merge, host levers, the jeffan package, BLINK_PM_PAR).
