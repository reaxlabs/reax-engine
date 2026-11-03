# bj-all1: one dm-all image with the team dashboard's best engine for every model (team ruj, 29 Oct 2026)

Requested by the user ("combine all models which are best in our dashboard"). Nothing in any engine was changed: every
slot runs the files, Python stack, environment and start command of the image that set the team dashboard's best for it.

| slot (REAX_MODEL) | engine | source image (team registry 86.38.182.86) | dashboard |
|---|---|---|---|
| blink (blink-v0.3-26b-a4b, text) | pjs's E62 Blink engine, `/opt/blink` (v2c tree + v2d-m3 vision P-mode) | ruj #436 bj-v2 `ruj/engine@sha256:b60974c1...` = pjs v2d-m3 #435 `a8817620...` + `BLINK_PM_PAR=1`; text path = pjs v2c #423 `d74f9f7b...` (3.74x) + P-mode | #423 |
| vision (same REAX_MODEL, one picture per item) | the same `/opt/blink` engine, P-mode | ruj #436 bj-v2 (3.06x) | #436 |
| jeffan (jeffan-31b-1710) | pjs's E61/E62 Jeffan engine `/opt/jeffan` + team ruj's jeffan5 package (6 files: FT/LF/EM, FR/TF, SUB, host levers) | levers of ruj #424 bj-g8a (1.99x); package image bj-g8v = pjs v2b + the 6 files | #424 |
| mammoth (mammoth-1) | kai's engine `/opt/reax/engine` (ck7c-g46: GPTQ g46 + N1/N2/B11/GDNFUSE/NOXSTORE + MINM) on the ref-mammoth base's `/opt/venv` | kai #429 `kai/engine@sha256:9765662f...` (1.34x) | #429 |
| reax1 (reax-1) | kai's engine (ck5-rx1: rq44 GPTQ + D1 + N1 + N2 PDL + B11 + GC freeze) at `/opt/reax-r1/engine` | kai #417 `kai/reax1@sha256:157f1472...` (1.41x) | #417 |

Layout: layers 1-44 = bj-v2 unchanged (`/opt/blink`, `/opt/vision`, `/opt/jeffan`, `/opt/e62` on vLLM 0.30.0); then the
ref-mammoth base's own layers of kai's images (`/opt/venv` torch 2.14.0 stack, `/opt/reax`, `/licenses`, blobs reused byte
for byte), kai #429's eight own layers (blobs reused), one layer with kai #417's `/opt/reax/{engine,BUILD}` under
`/opt/reax-r1` (bytes, modes and mtimes unchanged) and one layer with team ruj's files (`/opt/e62/dispatch.py` = pjs's
dispatcher + kai's two slots, the six jeffan5 files, this note). The dispatcher starts mammoth / reax1 as kai's images do
(cwd /opt/reax or /opt/reax-r1, `python -m engine.serve` from /opt/venv, the source image's Env + the docker run -e values).

Credits: pjs (E59-E72 Blink / Jeffan / vision engines, kernels, dispatcher), kai (mammoth / REAX 1 engines, GPTQ, kernels),
rgh (levers kai ported: f8m / mx / u4 lines, credited in kai's sources), team ruj (Blink host levers, jeffan5 package,
BLINK_PM_PAR line, this merge). Licenses: each tree's own (Apache-2.0 AND LicenseRef-REAX-Engine-1.1; kai's /licenses).
Sources of every compiled part: /opt/e62/KERNELS.md, /opt/reax/engine/KERNELS.md, /opt/reax-r1/engine/KERNELS.md.
