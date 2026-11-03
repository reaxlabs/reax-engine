// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 team ruj. jef_fp8gemm configuration 6: CTA tile 256x128x64, Cooperative schedule (see jef_fp8_common.cuh).
#include "jef_fp8_common.cuh"
JEF_FP8_CFG(jef_fp8_run_cfg6, 256, 128, 64, cutlass::gemm::KernelTmaWarpSpecializedCooperative)
