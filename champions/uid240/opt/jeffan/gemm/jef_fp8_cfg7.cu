// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 team ruj. jef_fp8gemm configuration 7: CTA tile 128x256x64, Cooperative schedule (see jef_fp8_common.cuh).
#include "jef_fp8_common.cuh"
JEF_FP8_CFG(jef_fp8_run_cfg7, 128, 256, 64, cutlass::gemm::KernelTmaWarpSpecializedCooperative)
