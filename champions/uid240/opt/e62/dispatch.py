# SPDX-License-Identifier: Apache-2.0
"""E62 dm-all entrypoint: pick the engine for the model this container serves and exec it (it becomes PID 1).

The validator starts the image once per declared model with REAX_MODEL=<model id> (engcomposite._DEFAULT_SLOTS):
  blink-v0.3-26b-a4b -> /opt/blink  (Blink text engine, E59 v8)
  jeffan-31b-1710    -> /opt/jeffan (Jeffan engine: E61 v1 + team ruj's #348 levers G / ROWS / BF / H and #373
                        levers SB / one-group BF + E62 J6 / J3;
                        batch-invariant kernels, Inductor configs pinned to start variant J1)
The per-slot defaults below are the ENV of the validated single-model images (e59-blink:v8 / e61-vision:v5 /
e61-jeffan:v1, minus the vLLM base image's ENV) plus the changes marked with the image version that made them (blink:
one batching group (v1fB), the pin table pins_rms.json (v1m), E64's c1 / c2 / bm1 blink trees and levers (v1p / v1q /
v1t), E66's RMS-flip hedge (v1t); jeffan: the J1 pin table (v1hJ1), team ruj's levers
(v1m), J6 (v1n), ruj's #348 tree with batch formation and the re-read threshold hedge (v1r), ruj's #373 SB / one-group
BF and FULL graphs to 1,024 tokens (v1u); blink: E64's bm7b tree with B7 (v1u), the MoE GEMM grid 160 and the
lone-alt grace (v1u2)). A variable already set
in the container (docker run -e) wins, as with an image ENV. The vision engine (/opt/vision, E61 v5) is in the image but only
reachable with E62_SLOT=vision. v2d-m0: this image declares vision (label blink,jeffan,vision); the validator gives the
vision slot the same REAX_MODEL as blink, so picture requests go to /opt/blink, whose P-mode (V1, BLINK_VISION=1) serves
them. Run with python3 -I -S: no site, no sitecustomize, no PYTHONPATH for this process itself.
E62_COOLDOWN_S=N > 0 (image default 0): the engine binds a private socket; once it is ready this process waits N s with
the GPU idle and only then moves the socket to $RUNTIME_UDS (the validator sees the engine ready after the pause); this
process then stays PID 1 and supervises the engine. With 0 the engine is exec'd (it becomes PID 1).

bj-all1 (team ruj, 29 Oct 2026; the user asked for one image with the team dashboard's best engine for every model): this
is pjs's v2d-m3 dispatcher (#435) with team ruj's bj-v2 line (BLINK_PM_PAR, #436) and two more slots, teammate kai's
engines, each from its own image and unchanged: mammoth-1 -> kai #429 (ck7c-g46, kai/engine@sha256:9765662f..., its tree
at /opt/reax/engine as in that image) and reax-1 -> kai #417 (ck5-rx1, kai/reax1@sha256:157f1472..., the same files under
/opt/reax-r1). Both run on the ref-mammoth reference base's own Python stack (/opt/venv: the reference image's layer, byte
for byte). They start as kai's images start them: cwd = the image WorkingDir, argv = its Entrypoint (python -m
engine.serve, found through that image's PATH), environment = that image's Env with everything docker run -e / the runtime
set on top, in docker's order (_kai_env). The blink, vision and jeffan slots are unchanged.
"""
import os
import signal
import sys
import time

_COMMON = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "OPENAI_API_KEY": "",
    "HOME": "/tmp", "VLLM_CACHE_ROOT": "/tmp/vllm-cache", "TRITON_CACHE_DIR": "/tmp/triton",
    "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor", "XDG_CACHE_HOME": "/tmp/.cache", "NUMBA_CACHE_DIR": "/tmp/numba",
    "REAX_MODEL_DIR": "/model",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "RAYON_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
    "TORCHINDUCTOR_COMPILE_THREADS": "8",
}

SLOTS = {
    "blink": (["python3", "/opt/blink/blink_engine.py"], {
        **_COMMON, "PYTHONPATH": "/opt/blink",
        "BLINK_CG_MODE": "FULL", "BLINK_NOSTATS": "1", "BLINK_UVLOOP": "1", "BLINK_FAST_ENC": "1", "BLINK_DETOK": "0",
        "BLINK_FASTIN": "1", "BLINK_INLINE_ENC": "1", "BLINK_GATHER_MS": "0", "BLINK_MOE_TAIL": "1",
        "BLINK_MOE_HEAD": "1", "BLINK_MOE_GEMM": "1", "BLINK_PAR": "1", "BLINK_RTR": "1",
        "BLINK_LABEL_HEAD": "1", "BLINK_BI_TILES": "1", "BLINK_PIPE": "1",
        # E62 v1fB/v1h (board A/B, 28 Oct: +1.5 % blink dps, answers unchanged): one group of 16, an idle GPU waits for all
        # clients at most 10 ms; v1e and earlier: BLINK_PIPE_CAP=0.5 (two pipelined groups of 8)
        "BLINK_PIPE_CAP": "0", "BLINK_PIPE_FULL": "1", "BLINK_PIPE_IDLE_MS": "10",
        # v1m: E64's pin table for the start variant the board's Blink reference lands on (the RMSNorm autotune pick)
        "BLINK_PIN_TABLE": "/opt/blink/pins_rms.json",
        # v1p (E64 nb): /opt/blink/blink_core.py = team ruj's #343 overlay (defaults BLINK_PIPE_ADAPT=1, BLINK_PIPE_QUIET_MS=1,
        # BLINK_PIPE_PRESCHED=1, BLINK_PIPE_CWIN_S=60; /opt/blink/NOTICE.bj); MoE GEMM grid: 160 SMs in #343, 128 with
        # BLINK_PAR_PRIO (v1q, E64: the MoE aux stream at high priority, -1.09 ms per convoy step; answers = pinR)
        # v1p (E64 c1): exact speed levers B1 (FULL graphs every 32 tokens from 640 to 2304), B3 (label readout via the
        # sampler, blink_read.py), B5 (fused GELU-up + NVFP4 quant, blink_actq.so); answers = pinR, 8192/8192
        "BLINK_GRAPH_SIZES": "1,2,4,8,16,24,32,40,48,56,64,96,128,160,192,256,320,384,448,512,640,672,704,736,768,800,832,864,896,928,960,992,1024,1056,1088,1120,1152,1184,1216,1248,1280,1312,1344,1376,1408,1440,1472,1504,1536,1568,1600,1632,1664,1696,1728,1760,1792,1824,1856,1888,1920,1952,1984,2016,2048,2080,2112,2144,2176,2208,2240,2272,2304,2336,2368,2400,2432,2464,2496,2528,2560,2816,3072,3584,4096,4608,5120,6144,7168,8192",
        "BLINK_READ": "1",
        "BLINK_MOE_ACTQ": "1",
        "BLINK_PAR_PRIO": "1",
        # v1u2 (E64 on bm7b): MoE GEMM grid back to 160 SMs with PRIO on: -0.72 ms per step (90 % CI -0.81..-0.64),
        # 3072/3072 screen answers identical (exact by construction: only which CTA computes a tile changes)
        "BLINK_MOE_GEMM_SMS": "0",   # v1y (E71 T1): 0 = the per-size table rules (a non-zero value overrides every row)
        "BLINK_MOE_W16": "1",   # v1y (E71 T1): 16-consumer-warp bmoe69 instances
        "BLINK_MOE_GEMM_TABLE": "/opt/blink/moe_gemm_t1x.json",   # v1y (E71 T1X)
        # v1t (E64 bm1): team ruj's bmoe fused MoE kernels (#362, ported by E64 with credit): w13 GEMM + gelu_tanh_and_mul +
        # NVFP4 experts quant in one kernel (FUSE, cfg 3), the w2 variant 11; E64's routing plan (PLAN); graph sizes to 2560
        # every 32 tokens. Exact: 8192/8192 = pinR (E64 gate), batch-invariant; -2.0 ms per step vs c3.
        "BLINK_MOE_FUSE": "1",
        "BLINK_MOE_FUSE_CFG": "3",
        "BLINK_MOE_PLAN": "1",
        # v1u (E64 bm7b): B7: in the last layer only (picked by weight data_ptr) the Q columns of qkv, o_proj, the MLP and
        # the MoE run on each request's last row only (query_start_loc); exact: 8192/8192 = pinR, 3072/3072 = bm1 (E64
        # gate), -0.80 ms per step vs bm1.
        "BLINK_LAST": "1",
        # v1t (E66): RMS-flip hedge: a flagged item of the selected shapes (n4c3) is also computed as the other RMSNorm start
        # variant (an alt copy of the request) and answers the centred-log average of both (d/2 from either variant).
        # Exact otherwise; unset = the plain bm1 engine. E66 h3: main == pinR 8192/8192, alt == F1 oracle, cost -3.9..-7 %.
        # v1v: hedge off (E62 vtb-u2 / hab: its cost on the balanced convoy ~8 %, about EV-neutral)
        "BLINK_HEDGE": "0",
        "BLINK_HEDGE_SELECT": "n4c3",
        # v1u2 (E62 EX-1b): a lone alt copy (its main launched without it) waits up to 20 ms for a main to join
        "BLINK_HEDGE_LONE_MS": "20",
        "BLINK_FRONTENDS": "3",
        # v1u2 (E64 bm7c): team ruj's #376 frontend connection balancing: the parent accepts every connection and
        # hands it to the frontend with the fewest open ones (E64: full 16-request steps 64-85 % -> 95 %, c16 +4.8-5.7 %;
        # 8192/8192 = pinR). Whole connections move, so a hedged item's main and alt stay on one frontend.
        "BLINK_FE_BALANCE": "1",
        "BLINK_K1": "1",   # v2a (K1)
        "BLINK_RED": "1",   # v2b (K2): exact one-pass CUDA replicas of red_fused_add_bi_mm_mul_rms_norm_0/_2 (blink_red.py)
        "BLINK_LEAN_OUT": "1",   # v1y (E64 b1): no telemetry-only walk in the output update (blink_out.py)
        "BLINK_DEFER_FREE": "1",   # v1y (E64 b2): KV block frees after the step's outputs are queued
        "BLINK_E72_LEANCACHE": "1",   # v1y (E72 L1): lean prefix-cache registration
        "BLINK_E72_ISL": "1",   # v1y (E72 L4): skip the redundant full-sequence admission pre-check
        "BLINK_E72_GPUWAIT": "1",   # v1y (E72 W1): wait on the step's copy event near its expected end
        "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE": "1",   # v1y (E72 O2): FE output pass one request at a time (vLLM env)
        "BLINK_HOST_DSAMP": "1",    # v2a (team ruj bj-b7 #406, with credit): sampler adds + staged writes after the FULL-graph launch, before sample_tokens
        "BLINK_HOST_NOMETA": "1",   # v2a (team ruj bj-b7 #406, with credit): FULL replays build no attention-metadata objects (self-check at first real step)
        "BLINK_HOST_XDIET": "1",    # v2c (team ruj bj-b8 #415, with credit): NODRAFT (no zero_() of the empty draft-token rows) + FUSEW (request-state + block-table staged writes in one launch of vLLM's multi-group write kernel); only on vLLM 0.30.0 ced6857's states.py (sha-pinned)
        "BLINK_VISION": "1", "VIS_FUSE": "1", "VIS_ENC_GRAPH": "1",   # v2d-m0 (V1): picture path in /opt/blink (P-mode: blink_pmode.py, blink_vis.py) + E61's exact vision-tower kernels / encoder graphs (/opt/vision/vfuse.py)
        "VIS_ATTN": "1", "VIS_UNPAD": "1",   # v2d-m1b (V2 A1/A2): exact vision-tower attention (/opt/vision/gemm/vattn.so via vattn.py) on the real (unpadded) rows only; byte-identical tower output
        "VIS_E5": "1", "VIS_E3": "1", "VIS_E4": "1",   # v2d-m2 (V3 E34): whole-encoder graph (E5), exact fused vision GEMMs (E3) and norm chains / head norms + RoPE (E4) in /opt/vision/gemm/vis_ops.so; byte-identical tower output
        "BLINK_PM_K1": "1", "BLINK_PM_MOE": "1", "BLINK_PM_LAST": "1", "BLINK_PM_HEAD": "1", "BLINK_PM_GRAPHS": "1", "BLINK_PM_LOOP": "pipe",   # v2d-m3 (V1 snap_m1): picture steps with v2b's K1 / MoE / LAST / LABEL_HEAD levers + P-only K1 tiles (k1_tiles_p.json), FULL CUDA graphs for the picture-step LM forward (blink_pmg.py), pipelined picture groups
        "BLINK_PM_PAR": "1",   # team ruj bj-v2 (bj/vision R2, credit pjs: his L1 switch, his blink_par.py): picture steps also run the MoE branch on the aux stream next to the dense MLP (same kernels, same inputs; parallel branches in the P-mode FULL graphs); screen p5 vs ref-blink@97d06a91 bit-identical
        "REAX_VISION_ENCODER_BUDGET": "total10", "BLINK_BI_TABLE": "/opt/vision/bi_tiles.json",   # v2d-m0 (V1): E61 v5's encoder budget; tile table = blink's rows + the vision-tower shapes
        "BLINK_HOST_UNIFY": "1",    # v2c (team ruj bj-b9 #419, with credit): one KV-cache group (vLLM's disable_hybrid_kv_cache_manager, set in the engine core before the KV cache is built): 16-token blocks for all 30 layers, sliding-window layers still windowed in-kernel, prefix hits on 16-token boundaries; only on vLLM 0.30.0 ced6857's kv_cache_utils.py (sha-pinned)
        # v2a NOMETA hazard: a lever or vLLM bump that stages per-step data into graph-read buffers inside
        # TritonAttentionMetadataBuilder.build or DefaultModelState.prepare_attn would be skipped in FULL replays: turn
        # NOMETA off or re-prove it. DSAMP hazard: nothing may read sampler state (or blink_read's _e64_n) before
        # execute_model returns. BLINK_HOST_LEAN / BLINK_HOST_DFREE stay unset (E64 LEAN_OUT covers LEAN's main cost,
        # DEFER_FREE covers DFREE); #415's BLINK_HOST_LATEFREE and BLINK_HOST_SWITCH_US stay unset. XDIET hazard: nothing
        # else may stage writes into req_states tensors, replace RequestState.add_request / apply_staged_writes, or rebuild
        # RequestState / BlockTables after the first real step. UNIFY hazard: decided at engine start (no live switch). A lever that assumes six
        # KV-cache groups or 32-token full-attention blocks, frees sliding-window KV itself, or needs sliding-window KV
        # savings (decode, or chunked prefill below max_model_len) must turn UNIFY off or re-prove it. Never in the
        # jeffan, vision or mammoth slots.
        # v1v (E69 port of team ruj's #386, bj-m1): the fused MoE head (BLINK_MOE_FHEAD: 3 kernels instead of 9,
        # exact by construction) and E69's switches in blink_moe.py / gemm/bmoe69.so (hints / skipped zero MMAs only;
        # every value unchanged); E69 gate: 8192 answers = pinR
        "BLINK_MOE_FHEAD": "1",
        "BLINK_MOE_DISCARD": "1",
        "BLINK_MOE_W2ZS": "1",
        # BLINK_MOE_L2PERSIST stays unset: E69's D-prime arm showed it slower, and this tree has its early version
        # (no idle reset)
    }),
    "vision": (["python3", "/opt/vision/vision_engine.py"], {
        **_COMMON, "PYTHONPATH": "/opt/vision",
        "REAX_VISION_ENCODER_BUDGET": "total10", "VIS_IMG_GATE": "0", "VIS_FRONTENDS": "2", "VIS_FUSE": "1",
        "VIS_ENC_GRAPH": "1", "BLINK_BI_TILES": "1", "BLINK_GEMM_DIR": "/opt/vision/gemm",
        "BLINK_BI_TABLE": "/opt/vision/bi_tiles.json",
    }),
    # jeffan: E61's engine (batch-invariant kernels like the reference) with its Inductor autotune picks pinned (jef_pin.py):
    # E61's tables reproduce the reference's start variants bit for bit (pins.json = J2, alt/a1_cakg1.json = J1). v1hJ1
    # and later pin J1. alt/cr.json (v1o, E65) is the "close-race first" variant, shipped but not the default.
    "jeffan": (["python3", "/opt/jeffan/jeffan_engine.py"], {
        **_COMMON, "PYTHONPATH": "/opt/jeffan",
        # v1r: J3 (E62 merge edit merge_j3.py): FULL CUDA graphs (attention included; TRITON_ATTN supports them for any
        # batch) for steps up to 640 tokens (v1u: 1,024), so c1 reads and J6's joint steps replay one graph (E62 j3, 28 Oct: exact on
        # 6144 c8 + 328 c1 items, c1 p50 -7..-22 ms, c8 +4.6 %; KV cache 18,436 tokens >= 16,448)
        "JEF_CG_MODE": "FULL",
        # v1u: + 768 / 896 / 1024 (SB steps are 1,024 tokens; E62 u1)
        "JEF_GRAPH_SIZES": "1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,160,192,256,320,384,448,512,640,768,896,1024",
        "JEF_PIN_TABLE": "/opt/jeffan/alt/a1_cakg1.json",
        # v1m: /opt/jeffan is team ruj's tree; its sitecustomize turns on JEF_GEMM=1 (lever G) and JEF_ROWS=1 by default.
        # v1r: ruj's #348 tree; its sitecustomize also sets JEF_ASYNC=0 and JEF_BF_MS=6 (batch formation) by default.
        # vLLM's batch-invariant mode (the reference's mode; v1k-v1m ran without it): with lever G (ruj's FP8 GEMMs replace
        # vLLM's single batch-invariant GEMM tile) it costs nothing: E62 m1 (BI 0) 27.52 vs m2 (BI 1) 27.66 dps on the
        # same 32 x 256 items (paired ratio 0.996 +- 0.007), both 8192/8192 bit-identical to the J1 reference emulation.
        "VLLM_BATCH_INVARIANT": "1",
        # v1n: J6 (E62 merge edit merge_j6.py): a decision that is alone (the validator's c1 latency items) issues read 2
        # with read 1 in one engine step (ruj's JEF_SPEC2 path); concurrent decisions keep the sequential re-read.
        "JEF_PAR_REREAD": "1",
        # v1r: ruj's lever H (re-read threshold hedge) with E62's capped shift: a decision whose first read's top choice
        # lies within JEF_HEDGE of the 0.9 re-read threshold gets both reads and answers the point on the gates' (floored,
        # centred) log segment from the read it would return toward the other one, at distance min(d/2, JEF_HEDGE_CAP):
        # with no flip at the reference the error is <= cap + the ordinary divergence (<= 0.48 measured on window items),
        # with a flip <= d - cap. E62 trace study (28 Oct, 8192 items, J1 engine vs J1 / J2 / cr references): delta 0.01,
        # cap 0.5 keeps every item <= 0.875 (no hedge: 1.225 vs cr; ruj's midpoint with dmax 2.0: 1.312 vs J2); 3.6 % of
        # items hedged, 2.1 % extra reads. JEF_HEDGE_DMAX applies only with JEF_HEDGE_CAP=0 (ruj's midpoint rule).
        # v1w: window 0.03 (team ruj #391, after board #383; E62 trace emulation, 8192 items vs J1 / J2 / cr: max
        # 0.500 / 0.875 / 0.725, mean 0.026 / 0.046 / 0.052, 9.8 % hedged, 5.6 % extra reads); cap 0.5 kept
        "JEF_HEDGE": "0.03",
        "JEF_HEDGE_CAP": "0.5",
        "JEF_HEDGE_DMAX": "1.4",
        # v1u: team ruj's #373 levers (jef_levers.py of their board image #373, merge_sb.py): SB = vLLM's per-step token
        # budget lowered to 1,024 new tokens (a CTA-tile edge of the FP8 GEMMs; a read that does not fit is split by
        # vLLM's chunked prefill, its earlier chunk's K/V read back from the FP8 cache as an unsplit read reads them);
        # one-group batch formation (BF_ALL: the post-step wait of up to JEF_BF_MS also waits for the successors of the
        # reads already queued); the frontend gather window off (BF replaces it). As ruj's #373 defaults. E62 u1: answers
        # = v1t bit for bit.
        "JEF_STEP_TOK": "1024",
        "JEF_BF_ALL": "1",
        "JEF_BF_MS": "20",   # v2a: 10 -> 20 (E62 p7 BW: computed tokens/s +1.8 %, BF timeouts 66 % -> 0 %, answers unchanged)
        "JEF_GATHER_MS": "0",
        # v1u3: E62 lever KV16 (jef_kv16.py): full-attention pages padded instead of 32-token blocks, so every KV
        # cache group has 16-token blocks and prefix-cache hits land on 16 tokens (E62 kv1 / kv0, 28 Oct: 8224 + 360
        # answers = v1u bit for bit, computed tokens -7.3 %, dps +5.6 %, KV cache 26,174 tokens >= 16,448); the same
        # idea was found independently by team ruj (#382)
        "JEF_KV16": "1",
        # v1w: team ruj's lever FE (#392, merge_fe.py): the question is encoded in the event loop and each read enters
        # vLLM as the EngineInput vLLM's raw-prompt path builds (same EngineCoreRequest), on uvloop. E62 fe1: answers =
        # v1u3 bit for bit
        "JEF_FE": "1",
        "JEF_UVLOOP": "1",
        "JEF_GQ": "1",   # v1z
        "JEF_LLP": "1",   # v1z
        "JEF_LLP_ATTN": "1",   # v1z
        "JEF_STEP_ALIGN": "512,640,768,896,1024",   # v1x
    }),
}


# bj-all1: THIS image's own Env (its config's Env, = bj-v2's; docker put it into this process's environment) and the Env of
# kai's two source images, verbatim (#417: PYTHONPATH follows its tree to /opt/reax-r1).
_THIS_IMAGE_ENV = [
    "PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "NVARCH=x86_64",
    "NVIDIA_REQUIRE_CUDA=cuda>=13.0 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=550,driver<551 brand=grid,driver>=550,driver<551 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 brand=quadro,driver>=550,driver<551 brand=quadrortx,driver>=550,driver<551 brand=nvidiartx,driver>=550,driver<551 brand=vapps,driver>=550,driver<551 brand=vpc,driver>=550,driver<551 brand=vcs,driver>=550,driver<551 brand=vws,driver>=550,driver<551 brand=cloudgaming,driver>=550,driver<551 brand=unknown,driver>=565,driver<566 brand=grid,driver>=565,driver<566 brand=tesla,driver>=565,driver<566 brand=nvidia,driver>=565,driver<566 brand=quadro,driver>=565,driver<566 brand=quadrortx,driver>=565,driver<566 brand=nvidiartx,driver>=565,driver<566 brand=vapps,driver>=565,driver<566 brand=vpc,driver>=565,driver<566 brand=vcs,driver>=565,driver<566 brand=vws,driver>=565,driver<566 brand=cloudgaming,driver>=565,driver<566 brand=unknown,driver>=570,driver<571 brand=grid,driver>=570,driver<571 brand=tesla,driver>=570,driver<571 brand=nvidia,driver>=570,driver<571 brand=quadro,driver>=570,driver<571 brand=quadrortx,driver>=570,driver<571 brand=nvidiartx,driver>=570,driver<571 brand=vapps,driver>=570,driver<571 brand=vpc,driver>=570,driver<571 brand=vcs,driver>=570,driver<571 brand=vws,driver>=570,driver<571 brand=cloudgaming,driver>=570,driver<571 brand=unknown,driver>=575,driver<576 brand=grid,driver>=575,driver<576 brand=tesla,driver>=575,driver<576 brand=nvidia,driver>=575,driver<576 brand=quadro,driver>=575,driver<576 brand=quadrortx,driver>=575,driver<576 brand=nvidiartx,driver>=575,driver<576 brand=vapps,driver>=575,driver<576 brand=vpc,driver>=575,driver<576 brand=vcs,driver>=575,driver<576 brand=vws,driver>=575,driver<576 brand=cloudgaming,driver>=575,driver<576",
    "NV_CUDA_CUDART_VERSION=13.0.96-1",
    "CUDA_VERSION=13.0.2",
    "LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
    "NVIDIA_VISIBLE_DEVICES=all",
    "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
    "DEBIAN_FRONTEND=noninteractive",
    "CUDA_HOME=/usr/local/cuda",
    "UV_HTTP_TIMEOUT=500",
    "UV_INDEX_STRATEGY=unsafe-best-match",
    "UV_LINK_MODE=copy",
    "UV_PYTHON_INSTALL_DIR=/opt/uv/python",
    "UV_CACHE_DIR=/opt/uv/cache",
    "UV_OVERRIDE=/etc/uv-overrides.txt",
    "VLLM_ENABLE_CUDA_COMPATIBILITY=0",
    "TORCH_CUDA_ARCH_LIST=7.5 8.0 8.6 8.9 9.0 10.0 12.0",
    "VLLM_USAGE_SOURCE=production-docker-image",
    "VLLM_BUILD_COMMIT=ced6857afa0ea7b2e3f0846a62e1394e90f15607",
    "VLLM_BUILD_PIPELINE=019d130e-464e-4ff7-b84b-492992c0c06b",
    "VLLM_BUILD_URL=https://buildkite.com/vllm/release-v2/builds/7022",
    "VLLM_IMAGE_TAG=vllm/vllm-openai:v0.30.0",
    "HF_HUB_OFFLINE=1",
    "TRANSFORMERS_OFFLINE=1",
    "VLLM_NO_USAGE_STATS=1",
    "DO_NOT_TRACK=1",
    "HF_HUB_DISABLE_TELEMETRY=1",
    "PYTHONDONTWRITEBYTECODE=1",
    "PYTHONUNBUFFERED=1",
    "OPENAI_API_KEY=",
    "HOME=/tmp",
    "VLLM_CACHE_ROOT=/tmp/vllm-cache",
    "TRITON_CACHE_DIR=/tmp/triton",
    "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor",
    "XDG_CACHE_HOME=/tmp/.cache",
    "NUMBA_CACHE_DIR=/tmp/numba",
    "REAX_MODEL_DIR=/model",
    "E62_COOLDOWN_S=0",
    "OMP_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "RAYON_NUM_THREADS=2",
    "TOKENIZERS_PARALLELISM=false",
    "TORCHINDUCTOR_COMPILE_THREADS=8",
]

# kai #429 = 86.38.182.86/kai/engine@sha256:9765662f7724a91ead294022e786cd45349a1a7096865593b1247daa34bc161a (mammoth 1.34x, #429)
_KAI429_ENV = [
    "PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "NVARCH=x86_64",
    "NVIDIA_REQUIRE_CUDA=cuda>=13.0 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=550,driver<551 brand=grid,driver>=550,driver<551 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 brand=quadro,driver>=550,driver<551 brand=quadrortx,driver>=550,driver<551 brand=nvidiartx,driver>=550,driver<551 brand=vapps,driver>=550,driver<551 brand=vpc,driver>=550,driver<551 brand=vcs,driver>=550,driver<551 brand=vws,driver>=550,driver<551 brand=cloudgaming,driver>=550,driver<551 brand=unknown,driver>=565,driver<566 brand=grid,driver>=565,driver<566 brand=tesla,driver>=565,driver<566 brand=nvidia,driver>=565,driver<566 brand=quadro,driver>=565,driver<566 brand=quadrortx,driver>=565,driver<566 brand=nvidiartx,driver>=565,driver<566 brand=vapps,driver>=565,driver<566 brand=vpc,driver>=565,driver<566 brand=vcs,driver>=565,driver<566 brand=vws,driver>=565,driver<566 brand=cloudgaming,driver>=565,driver<566 brand=unknown,driver>=570,driver<571 brand=grid,driver>=570,driver<571 brand=tesla,driver>=570,driver<571 brand=nvidia,driver>=570,driver<571 brand=quadro,driver>=570,driver<571 brand=quadrortx,driver>=570,driver<571 brand=nvidiartx,driver>=570,driver<571 brand=vapps,driver>=570,driver<571 brand=vpc,driver>=570,driver<571 brand=vcs,driver>=570,driver<571 brand=vws,driver>=570,driver<571 brand=cloudgaming,driver>=570,driver<571 brand=unknown,driver>=575,driver<576 brand=grid,driver>=575,driver<576 brand=tesla,driver>=575,driver<576 brand=nvidia,driver>=575,driver<576 brand=quadro,driver>=575,driver<576 brand=quadrortx,driver>=575,driver<576 brand=nvidiartx,driver>=575,driver<576 brand=vapps,driver>=575,driver<576 brand=vpc,driver>=575,driver<576 brand=vcs,driver>=575,driver<576 brand=vws,driver>=575,driver<576 brand=cloudgaming,driver>=575,driver<576",
    "NV_CUDA_CUDART_VERSION=13.0.96-1",
    "CUDA_VERSION=13.0.2",
    "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
    "NVIDIA_VISIBLE_DEVICES=all",
    "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
    "NV_CUDA_LIB_VERSION=13.0.2-1",
    "NV_NVTX_VERSION=13.0.85-1",
    "NV_LIBNPP_VERSION=13.0.1.2-1",
    "NV_LIBNPP_PACKAGE=libnpp-13-0=13.0.1.2-1",
    "NV_LIBCUSPARSE_VERSION=12.6.3.3-1",
    "NV_LIBCUBLAS_PACKAGE_NAME=libcublas-13-0",
    "NV_LIBCUBLAS_VERSION=13.1.0.3-1",
    "NV_LIBCUBLAS_PACKAGE=libcublas-13-0=13.1.0.3-1",
    "NV_LIBNCCL_PACKAGE_NAME=libnccl2",
    "NV_LIBNCCL_PACKAGE_VERSION=2.28.3-1",
    "NCCL_VERSION=2.28.3-1",
    "NV_LIBNCCL_PACKAGE=libnccl2=2.28.3-1+cuda13.0",
    "NVIDIA_PRODUCT_NAME=CUDA",
    "DEBIAN_FRONTEND=noninteractive",
    "PIP_NO_CACHE_DIR=1",
    "PIP_DISABLE_PIP_VERSION_CHECK=1",
    "PYTHONDONTWRITEBYTECODE=1",
    "PYTHONUNBUFFERED=1",
    "PYTHONPATH=/opt/reax",
    "REAX_PROFILE=reference",
    "CIE_LAZY_LOAD=1",
    "REAX_MAX_WAIT_MS=4",
    "PLUMB_MODEL_DIR=/model",
    "HF_HUB_OFFLINE=1",
    "TRANSFORMERS_OFFLINE=1",
    "HF_HUB_DISABLE_TELEMETRY=1",
    "OPENAI_API_KEY=",
    "HOME=/tmp",
    "TRITON_CACHE_DIR=/tmp/triton",
    "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor",
    "XDG_CACHE_HOME=/tmp/.cache",
    "KAI_MODEL_ENV_MAMMOTH=MAM_F8=1,MAM_F8_GU=1,MAM_FLA_PIN=1,MAM_SCONV=1,MAM_H2D1=1,MAM_ADAPT=1,KAI_MAX_WAIT_MS=16,MAM_ENCTHREAD=1,MAM_FASTENC=1,MAM_F8TAB=1,KAI_ACTQ=1,KAI_GDNPS=1,KAI_ENC1=1,KAI_HCI=1,KAI_RDASYNC=1,KAI_EARLY=1,MAM_NV4TAB=1,KAI_N1=1,KAI_PDL=1,KAI_B11=1,MAM_NV4=g46,KAI_NV4_MINM=256,KAI_GC_FREEZE=1,KAI_GDNFUSE=1,KAI_NOXSTORE=1",
    "KAI_MODEL_ENV_REAX1=MAM_F8=1,MAM_F8_GU=1,MAM_SCONV=1,MAM_FLA_PIN=1,MAM_ENCTHREAD=1,R1_FASTENC=1,KAI_SCHED_CLOSE=expect,KAI_NV4_MINM=256,KAI_HCI=1,KAI_ACTQ=1,R1_C1GRAPH=1,R1_C1GRAPH_RANGE=64:320,MAM_F8TAB=1,KAI_RDASYNC=1,MAM_NV4TAB=1,MXA_D1=1,KAI_N1=1,KAI_PDL=1,KAI_B11=1,KAI_GC_FREEZE=1,MAM_NV4=q48,MAM_NV4_METHOD=gptq,MAM_NV4_MLP=48-62,MAM_NV4_REST=48-63,MAM_NV4_REST_TYPES=qkvz,KAI_NOXSTORE=1",
]

# kai #417 = 86.38.182.86/kai/reax1@sha256:157f1472d983c9d0d11d2c74c94fead2d481f0dcae6a940c2594e462aebe3278 (reax1 1.41x, #417)
_KAI417_ENV = [
    "PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "NVARCH=x86_64",
    "NVIDIA_REQUIRE_CUDA=cuda>=13.0 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=550,driver<551 brand=grid,driver>=550,driver<551 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 brand=quadro,driver>=550,driver<551 brand=quadrortx,driver>=550,driver<551 brand=nvidiartx,driver>=550,driver<551 brand=vapps,driver>=550,driver<551 brand=vpc,driver>=550,driver<551 brand=vcs,driver>=550,driver<551 brand=vws,driver>=550,driver<551 brand=cloudgaming,driver>=550,driver<551 brand=unknown,driver>=565,driver<566 brand=grid,driver>=565,driver<566 brand=tesla,driver>=565,driver<566 brand=nvidia,driver>=565,driver<566 brand=quadro,driver>=565,driver<566 brand=quadrortx,driver>=565,driver<566 brand=nvidiartx,driver>=565,driver<566 brand=vapps,driver>=565,driver<566 brand=vpc,driver>=565,driver<566 brand=vcs,driver>=565,driver<566 brand=vws,driver>=565,driver<566 brand=cloudgaming,driver>=565,driver<566 brand=unknown,driver>=570,driver<571 brand=grid,driver>=570,driver<571 brand=tesla,driver>=570,driver<571 brand=nvidia,driver>=570,driver<571 brand=quadro,driver>=570,driver<571 brand=quadrortx,driver>=570,driver<571 brand=nvidiartx,driver>=570,driver<571 brand=vapps,driver>=570,driver<571 brand=vpc,driver>=570,driver<571 brand=vcs,driver>=570,driver<571 brand=vws,driver>=570,driver<571 brand=cloudgaming,driver>=570,driver<571 brand=unknown,driver>=575,driver<576 brand=grid,driver>=575,driver<576 brand=tesla,driver>=575,driver<576 brand=nvidia,driver>=575,driver<576 brand=quadro,driver>=575,driver<576 brand=quadrortx,driver>=575,driver<576 brand=nvidiartx,driver>=575,driver<576 brand=vapps,driver>=575,driver<576 brand=vpc,driver>=575,driver<576 brand=vcs,driver>=575,driver<576 brand=vws,driver>=575,driver<576 brand=cloudgaming,driver>=575,driver<576",
    "NV_CUDA_CUDART_VERSION=13.0.96-1",
    "CUDA_VERSION=13.0.2",
    "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
    "NVIDIA_VISIBLE_DEVICES=all",
    "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
    "NV_CUDA_LIB_VERSION=13.0.2-1",
    "NV_NVTX_VERSION=13.0.85-1",
    "NV_LIBNPP_VERSION=13.0.1.2-1",
    "NV_LIBNPP_PACKAGE=libnpp-13-0=13.0.1.2-1",
    "NV_LIBCUSPARSE_VERSION=12.6.3.3-1",
    "NV_LIBCUBLAS_PACKAGE_NAME=libcublas-13-0",
    "NV_LIBCUBLAS_VERSION=13.1.0.3-1",
    "NV_LIBCUBLAS_PACKAGE=libcublas-13-0=13.1.0.3-1",
    "NV_LIBNCCL_PACKAGE_NAME=libnccl2",
    "NV_LIBNCCL_PACKAGE_VERSION=2.28.3-1",
    "NCCL_VERSION=2.28.3-1",
    "NV_LIBNCCL_PACKAGE=libnccl2=2.28.3-1+cuda13.0",
    "NVIDIA_PRODUCT_NAME=CUDA",
    "DEBIAN_FRONTEND=noninteractive",
    "PIP_NO_CACHE_DIR=1",
    "PIP_DISABLE_PIP_VERSION_CHECK=1",
    "PYTHONDONTWRITEBYTECODE=1",
    "PYTHONUNBUFFERED=1",
    "PYTHONPATH=/opt/reax-r1",
    "REAX_PROFILE=reference",
    "CIE_LAZY_LOAD=1",
    "REAX_MAX_WAIT_MS=4",
    "PLUMB_MODEL_DIR=/model",
    "HF_HUB_OFFLINE=1",
    "TRANSFORMERS_OFFLINE=1",
    "HF_HUB_DISABLE_TELEMETRY=1",
    "OPENAI_API_KEY=",
    "HOME=/tmp",
    "TRITON_CACHE_DIR=/tmp/triton",
    "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor",
    "XDG_CACHE_HOME=/tmp/.cache",
    "KAI_MODEL_ENV_REAX1=MAM_F8=1,MAM_F8_GU=1,MAM_SCONV=1,MAM_FLA_PIN=1,MAM_ENCTHREAD=1,R1_FASTENC=1,KAI_SCHED_CLOSE=expect,KAI_NV4_MINM=256,KAI_HCI=1,KAI_ACTQ=1,R1_C1GRAPH=1,R1_C1GRAPH_RANGE=64:320,MAM_F8TAB=1,KAI_RDASYNC=1,MAM_NV4TAB=1,MAM_NV4=rq44,MXA_D1=1,KAI_N1=1,KAI_PDL=1,KAI_B11=1,KAI_GC_FREEZE=1",
]

# slot: (WorkingDir, Entrypoint, image Env) of the source image
KAI_SLOTS = {
    "mammoth": ("/opt/reax", ["python", "-m", "engine.serve"], _KAI429_ENV),
    "reax1": ("/opt/reax-r1", ["python", "-m", "engine.serve"], _KAI417_ENV),
}


def _kai_env(image_env):
    """The environment docker gives kai's source image, rebuilt from this process's: docker's order is PATH, HOSTNAME,
    [TERM], the docker run -e entries, then the image Env entries no -e entry replaced (moby: runconfig merge() +
    CreateDaemonEnvironment(); the -e entries themselves come in no fixed order). The trailing run of THIS image's Env
    entries (in order, minus the replaced ones) is dropped and the source image's Env takes its place; PATH becomes the
    source image's; everything else (HOSTNAME, every -e entry) stays as it came."""
    items = list(os.environ.items())
    this = [tuple(e.split("=", 1)) for e in _THIS_IMAGE_ENV]
    # PEP 538: this interpreter (python3 -I -S, C locale) appended LC_CTYPE=C.UTF-8 to its own environment; docker never
    # puts LC_CTYPE last (the image Env entries come last and this image's Env has none): drop it, kai's engine
    # interpreter coerces its locale itself, exactly as in kai's image
    if items and items[-1][0] == "LC_CTYPE" and "LC_CTYPE" not in dict(this):
        items.pop()
    j = len(items)
    for kv in reversed([kv for kv in this if kv[0] != "PATH"]):
        if j > 0 and items[j - 1] == kv:
            j -= 1
    head = items[:j]
    src = [tuple(e.split("=", 1)) for e in image_env]
    this_path, src_path = dict(this).get("PATH"), dict(src).get("PATH")
    env = {}
    for k, v in head:
        env[k] = src_path if (k == "PATH" and v == this_path and src_path is not None) else v
    for k, v in src:
        if k not in env:
            env[k] = v
    return env

# REAX_MODEL values the validator sends (model ids), plus the slot names for local tests.
MODEL_TO_SLOT = {
    "blink-v0.3-26b-a4b": "blink", "blink": "blink",
    "jeffan-31b-1710": "jeffan", "jeffan": "jeffan",
    "mammoth-1": "mammoth", "mammoth": "mammoth",   # bj-all1: kai #429
    "reax-1": "reax1", "reax1": "reax1",            # bj-all1: kai #417
}


def main() -> None:
    model = os.environ.get("REAX_MODEL", "").strip()
    slot = os.environ.get("E62_SLOT", "").strip().lower() or MODEL_TO_SLOT.get(model.lower(), "")
    if slot in KAI_SLOTS:   # bj-all1: kai's image start (cwd, argv, environment of the source image)
        cwd, argv, image_env = KAI_SLOTS[slot]
        env = _kai_env(image_env)
        print(f"[e62] REAX_MODEL={model!r} -> {slot}: cd {cwd}; exec {' '.join(argv)} (kai's image start)",
              file=sys.stderr, flush=True)
        os.chdir(cwd)
        os.execvpe(argv[0], argv, env)
    if slot not in SLOTS:
        print(f"[e62] REAX_MODEL={model!r}: this image serves blink-v0.3-26b-a4b, jeffan-31b-1710, mammoth-1 and "
              "reax-1 only", file=sys.stderr, flush=True)
        sys.exit(2)
    argv, defaults = SLOTS[slot]
    env = dict(os.environ)
    for k, v in defaults.items():
        env.setdefault(k, v)
    cool = float(os.environ.get("E62_COOLDOWN_S", "0") or 0)
    public = env.get("RUNTIME_UDS", "")
    if cool <= 0 or not public:
        print(f"[e62] REAX_MODEL={model!r} -> {slot}: exec {' '.join(argv)}", file=sys.stderr, flush=True)
        os.execvpe(argv[0], argv, env)
    private = public + ".e62"
    env["RUNTIME_UDS"] = private
    for p in (public, private):
        if os.path.lexists(p):
            os.unlink(p)
    print(f"[e62] REAX_MODEL={model!r} -> {slot}: {' '.join(argv)} on {private}, cool-down {cool:g} s once ready",
          file=sys.stderr, flush=True)
    pid = os.posix_spawnp(argv[0], argv, env, setpgroup=0)

    def stop(*_):
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            pass
        sys.exit(143)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    t0 = time.time()
    while not os.path.exists(private):
        if os.waitpid(pid, os.WNOHANG)[0]:
            print("[e62] engine exited during start-up", file=sys.stderr, flush=True)
            sys.exit(1)
        time.sleep(0.5)
    print(f"[e62] engine ready after {time.time() - t0:.0f} s; GPU idle for {cool:g} s", file=sys.stderr, flush=True)
    end = time.time() + cool
    while time.time() < end:
        if os.waitpid(pid, os.WNOHANG)[0]:
            sys.exit(1)
        time.sleep(0.5)
    os.rename(private, public)
    print(f"[e62] socket {public} up after {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    while True:
        p, st = os.wait()
        if p == pid:
            print(f"[e62] engine exited ({st})", file=sys.stderr, flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
