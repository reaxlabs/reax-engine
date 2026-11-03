// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 team ruj. jef_fp8gemm configuration 4: CTA tile 128x128x128, Cooperative schedule (see jef_fp8_common.cuh).
#include "jef_fp8_common.cuh"
JEF_FP8_CFG(jef_fp8_run_cfg4, 128, 128, 128, cutlass::gemm::KernelTmaWarpSpecializedCooperative)
