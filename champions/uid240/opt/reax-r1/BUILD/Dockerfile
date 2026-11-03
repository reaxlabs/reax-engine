# Development image of OUR mammoth engine (engine track v2, slot "mammoth", REAX_MODEL=mammoth-1).
#
# Base = the official mammoth reference image, pinned by digest (reaxlabs/reax-engine 5467286, docker/ref-qwen35). Same
# CUDA / torch 2.14.0+cu130 / transformers 5.17.0 / flash-linear-attention 0.5.2 stack, same ENV (REAX_PROFILE=reference,
# CIE_LAZY_LOAD=1, REAX_MAX_WAIT_MS=4, PLUMB_MODEL_DIR=/model, HF offline, caches in /tmp), same USER s1m (uid 10002),
# same ENTRYPOINT `python -m engine.serve`, same HEALTHCHECK. Only the engine tree /opt/reax/engine is replaced by ours.
#
# Rules this file keeps (docs/ENGINE-COMPETITION.md image rules, engreview.check_image_config, Option A):
#   non-root USER, no EXPOSE, no VOLUME, no LD_PRELOAD / proxy ENV, no weight-like files (only source + .pyc are added),
#   full source of every compiled part: engine/libf8g.so (rgh's FP8 GEMM library, copied unchanged from his u4 image) ships
#   with its source (engine/f8g.cu, f8common.cuh) and exact nvcc build line + rebuild check in engine/KERNELS.md.
#
# Dev variant images (dev/build.sh OPTS='k=v;k=v' ENVS='PPLX_A=1 KAI_B=2'): build.sh validates them, passes them as the
# build args KAI_ENGINE_OPTS / KAI_ENV (recorded in the labels kai.engfast.opts / kai.engfast.env) and replaces the
# @KAI-ENV@ marker line below with `ENV KAI_ENGINE_OPTS=${KAI_ENGINE_OPTS}` and `ENV <ENVS>` in the STAGED copy of this file
# (that rendered copy is what is built and what lands in /opt/reax/BUILD/Dockerfile). With both empty (the default) nothing
# is rendered and the image ENV stays identical to the reference's. dev/check_image.py accepts exactly these additions.
# A variant image is for measuring only; a submission bakes its options into engine/registry.py instead.
#
# Build from the fork root (dev/build.sh does this on the GPU host):
#   docker build -f dev/Dockerfile --build-arg GIT_SHA=$(git rev-parse HEAD) -t 127.0.0.1:5000/kai/mammoth:<tag> .
ARG REF_IMAGE=ghcr.io/reaxlabs/reax-engine-ref-mammoth@sha256:3080a593060542bccbdaf7cf0c66593fc182d9753b3facd50206df86d93b8588
FROM ${REF_IMAGE}

# re-declared: an ARG from before FROM is not visible after it (used in the base label below)
ARG REF_IMAGE
ARG GIT_SHA=unknown
ARG GIT_TREE=unknown
ARG GIT_DIRTY=unknown
ARG BUILD_TAG=unknown
# Placeholder on purpose: a dev image is bound to no hotkey, so it can never be crowned for anyone by accident. A real
# submission sets --build-arg HOTKEY=<ss58 of the committing hotkey> (and the copy-line labels) in a reviewed build.
ARG HOTKEY=test-build-no-hotkey
ARG KAI_ENGINE_OPTS=""
ARG KAI_ENV=""
ARG MODELS=mammoth

USER root
# Replace, never merge: files we deleted from engine/ must not survive from the reference layer.
RUN rm -rf /opt/reax/engine
COPY engine /opt/reax/engine
COPY LICENSE NOTICE /licenses/
COPY LICENSES /licenses/LICENSES/
COPY upstream/mammoth/LICENSE upstream/mammoth/NOTICE upstream/mammoth/MODEL-LICENSE.txt /licenses/mammoth/
# Build provenance inside the image: this Dockerfile and build.sh's record of the source state (commit, tree, dirty diff).
COPY dev/Dockerfile /opt/reax/BUILD/Dockerfile
COPY BUILD-INFO /opt/reax/BUILD/
# Byte-compile now: the root filesystem is read-only at run time (same step as the reference image).
RUN /opt/venv/bin/python -m compileall -q /opt/reax/engine \
 && chmod -R a+rX /opt/reax /licenses
USER s1m
WORKDIR /opt/reax
ENV KAI_MODEL_ENV_REAX1=MAM_F8=1,MAM_F8_GU=1,MAM_SCONV=1,MAM_FLA_PIN=1,MAM_ENCTHREAD=1,R1_FASTENC=1,KAI_SCHED_CLOSE=expect,KAI_NV4_MINM=256,KAI_HCI=1,KAI_ACTQ=1,R1_C1GRAPH=1,R1_C1GRAPH_RANGE=64:320,MAM_F8TAB=1,KAI_RDASYNC=1,MAM_NV4TAB=1,MAM_NV4=rq44,MXA_D1=1,KAI_N1=1,KAI_PDL=1,KAI_B11=1,KAI_GC_FREEZE=1

LABEL org.opencontainers.image.title="kai-mammoth-dev" \
      org.opencontainers.image.description="Development build of our mammoth engine on the reax-engine-ref-mammoth base (no weights inside)" \
      org.opencontainers.image.source="local fork of https://github.com/reaxlabs/reax-engine (branch engine/mammoth-v2)" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.licenses="LicenseRef-REAX-Engine-1.1 AND Apache-2.0 AND MIT" \
      ai.reax.engine.models="${MODELS}" \
      ai.reax.engine.hotkey="${HOTKEY}" \
      ai.reax.engine.commit="${GIT_SHA}" \
      ai.reax.engine.base="${REF_IMAGE}" \
      kai.engfast.tree="${GIT_TREE}" \
      kai.engfast.dirty="${GIT_DIRTY}" \
      kai.engfast.tag="${BUILD_TAG}" \
      kai.engfast.opts="${KAI_ENGINE_OPTS}" \
      kai.engfast.env="${KAI_ENV}"
