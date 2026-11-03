// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 team ruj. jef_fp8gemm configuration 0: CTA tile 64x64x128, Pingpong schedule (see jef_fp8_common.cuh).
#include "jef_fp8_common.cuh"
JEF_FP8_CFG(jef_fp8_run_cfg0, 64, 64, 128, cutlass::gemm::KernelTmaWarpSpecializedPingpong)
