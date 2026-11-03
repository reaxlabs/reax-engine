// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 team ruj. jef_fp8gemm configuration 12: CTA tile 64x192x128, Pingpong schedule (see jef_fp8_common.cuh).
#include "jef_fp8_common.cuh"
JEF_FP8_CFG(jef_fp8_run_cfg12, 64, 192, 128, cutlass::gemm::KernelTmaWarpSpecializedPingpong)
