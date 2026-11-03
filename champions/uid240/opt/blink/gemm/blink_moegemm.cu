// bj/moe (team ruj, 2026-10-28): this file = pjs's blink_moegemm.cu (unchanged except: its moe_mm is now
// moe_mm_cutlass, the PYBIND11 block is at the end) + the bmoe kernels and dispatcher appended below.
// SPDX-License-Identifier: Apache-2.0
// blink_moegemm: NVFP4 grouped GEMM for the MoE experts on SM120 (RTX PRO 6000), built on CUTLASS 4.x (BSD-3).
// out[rows of expert e] = alpha[e] * (A_e (e2m1, ue4m3 block scales, vector 16) x B_e^T (same)) in bf16.
// Inputs exactly as vLLM's cutlass_fp4_moe_mm(out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets)
// takes them (our own implementation of that interface). The arithmetic is the same block-scaled mma.sync
// (16x8x64, fp32 accumulation over K in increasing order, no split-K) and the same LinearCombination epilogue; only the
// tile shape / kernel schedule differ (a per-variant switch), which changes who computes an output element, not how.
// sm_limit > 0 runs the persistent grid on that many SMs (leaves the rest to work on another stream).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

using namespace cute;

template <class TileShape, class MainSchedule, class EpiSchedule>
struct MoeGemm {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;
  using ElementType = cutlass::float_e2m1_t;
  using ElementSF = cutlass::float_ue4m3_t;
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementC = cutlass::bfloat16_t;
  using ElementD = cutlass::bfloat16_t;
  using ElementAcc = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignA = 32;
  static constexpr int AlignB = 32;
  static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
  using ClusterShape = Shape<_1, _1, _1>;
  using Fusion = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementAcc, ElementC, ElementAcc>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementAcc, ElementC, LayoutC*, AlignC, ElementD,
      LayoutC*, AlignC, EpiSchedule, Fusion>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementA, LayoutA*, AlignA, ElementB, LayoutB*,
      AlignB, ElementAcc, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainSchedule>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename GemmKernel::InternalStrideA;
  using StrideB = typename GemmKernel::InternalStrideB;
  using StrideC = typename GemmKernel::InternalStrideC;
  using LayoutSFA = typename GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using ScaleConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;
};

// per-expert pointers and scale-factor layouts (one thread per expert)
template <class G>
__global__ void blink_group_starts(const uint8_t* a, const uint8_t* b, cutlass::bfloat16_t* out, const uint8_t* a_sf,
                                   const uint8_t* b_sf, const float* alphas, const int32_t* expert_offsets,
                                   const int32_t* sf_offsets, const int32_t* problem_sizes, int64_t* a_ptrs,
                                   int64_t* b_ptrs, int64_t* out_ptrs, int64_t* a_sf_ptrs, int64_t* b_sf_ptrs,
                                   int64_t* alpha_ptrs, typename G::LayoutSFA* layout_sfa,
                                   typename G::LayoutSFB* layout_sfb, int num_experts) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= num_experts) return;
  int64_t m = problem_sizes[e * 3], n = problem_sizes[e * 3 + 1], k = problem_sizes[e * 3 + 2];
  int64_t row0 = expert_offsets[e], sf_row0 = sf_offsets[e];
  a_ptrs[e] = reinterpret_cast<int64_t>(a + row0 * (k / 2));                 // A [rows, K/2] bytes
  b_ptrs[e] = reinterpret_cast<int64_t>(b + e * n * (k / 2));                // B [E, N, K/2] bytes
  out_ptrs[e] = reinterpret_cast<int64_t>(out + row0 * n);                   // out [rows, N] bf16
  a_sf_ptrs[e] = reinterpret_cast<int64_t>(a_sf + sf_row0 * (k / 16));       // A scales [sum(padded rows), K/16]
  b_sf_ptrs[e] = reinterpret_cast<int64_t>(b_sf + e * n * (k / 16));         // B scales [E, N, K/16]
  alpha_ptrs[e] = reinterpret_cast<int64_t>(alphas + e);
  layout_sfa[e] = G::ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
  layout_sfb[e] = G::ScaleConfig::tile_atom_to_shape_SFB(
      cute::make_shape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
}

static int sm_count(int dev) {
  static std::unordered_map<int, int> cache;
  auto it = cache.find(dev);
  if (it != cache.end()) return it->second;
  int n = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
  cache[dev] = n;
  return n;
}

template <class G>
static void run(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& a_sf,
                const torch::Tensor& b_sf, const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
                const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets, int raster, int sm_limit) {
  using Gemm = typename G::Gemm;
  int num_experts = static_cast<int>(expert_offsets.size(0));
  int64_t N = out.size(1), K = a.size(1) * 2;
  auto o64 = torch::TensorOptions().dtype(torch::kInt64).device(a.device());
  auto o8 = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  // one int64 workspace for the 6 pointer arrays + the strides; the layouts in a byte buffer
  torch::Tensor ptrs = torch::empty({9, num_experts}, o64);
  torch::Tensor lay = torch::empty({num_experts * (int64_t)(sizeof(typename G::LayoutSFA) + sizeof(typename G::LayoutSFB))}, o8);
  auto* p = ptrs.data_ptr<int64_t>();
  int64_t *a_ptrs = p, *b_ptrs = p + num_experts, *out_ptrs = p + 2 * num_experts, *a_sf_ptrs = p + 3 * num_experts,
          *b_sf_ptrs = p + 4 * num_experts, *alpha_ptrs = p + 5 * num_experts;
  // strides in elements: A and B rows are K e2m1 values, out rows N bf16 values
  ptrs.select(0, 6).fill_(K);
  ptrs.select(0, 7).fill_(K);
  ptrs.select(0, 8).fill_(out.stride(0));
  auto* layout_sfa = reinterpret_cast<typename G::LayoutSFA*>(lay.data_ptr());
  auto* layout_sfb = reinterpret_cast<typename G::LayoutSFB*>(
      static_cast<uint8_t*>(lay.data_ptr()) + num_experts * sizeof(typename G::LayoutSFA));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  blink_group_starts<G><<<(num_experts + 127) / 128, 128, 0, stream>>>(
      static_cast<const uint8_t*>(a.data_ptr()), static_cast<const uint8_t*>(b.data_ptr()),
      reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr()), static_cast<const uint8_t*>(a_sf.data_ptr()),
      static_cast<const uint8_t*>(b_sf.data_ptr()), alphas.data_ptr<float>(), expert_offsets.data_ptr<int32_t>(),
      sf_offsets.data_ptr<int32_t>(), problem_sizes.data_ptr<int32_t>(), a_ptrs, b_ptrs, out_ptrs, a_sf_ptrs, b_sf_ptrs,
      alpha_ptrs, layout_sfa, layout_sfb, num_experts);

  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90GroupParams<
      typename G::UnderlyingProblemShape>::RasterOrderOptions;
  typename G::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = raster == 0 ? RasterOrderOptions::AlongM : RasterOrderOptions::AlongN;
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a.get_device();
  hw_info.sm_count = sm_count(hw_info.device_id);
  if (sm_limit > 0 && sm_limit < hw_info.sm_count) hw_info.sm_count = sm_limit;   // persistent grid on fewer SMs

  typename G::GemmKernel::MainloopArguments mainloop{
      reinterpret_cast<const typename G::ElementType**>(a_ptrs), reinterpret_cast<typename G::StrideA*>(p + 6 * num_experts),
      reinterpret_cast<const typename G::ElementType**>(b_ptrs), reinterpret_cast<typename G::StrideB*>(p + 7 * num_experts),
      reinterpret_cast<const typename G::ElementSF**>(a_sf_ptrs), layout_sfa,
      reinterpret_cast<const typename G::ElementSF**>(b_sf_ptrs), layout_sfb};
  typename G::GemmKernel::EpilogueArguments epilogue{
      {}, nullptr, reinterpret_cast<typename G::StrideC*>(p + 8 * num_experts),
      reinterpret_cast<typename G::ElementD**>(out_ptrs), reinterpret_cast<typename G::StrideC*>(p + 8 * num_experts)};
  epilogue.thread.alpha_ptr_array = reinterpret_cast<float**>(alpha_ptrs);
  epilogue.thread.dAlpha = {_0{}, _0{}, 1};
  typename G::GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, reinterpret_cast<typename G::UnderlyingProblemShape*>(problem_sizes.data_ptr()), nullptr},
      mainloop, epilogue, hw_info, scheduler};
  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  torch::Tensor workspace = torch::empty({(int64_t)ws + 1}, o8);
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "blink_moegemm: cannot implement");
  TORCH_CHECK(gemm.initialize(args, workspace.data_ptr(), stream) == cutlass::Status::kSuccess,
              "blink_moegemm: initialize failed");
  TORCH_CHECK(gemm.run(args, workspace.data_ptr(), stream) == cutlass::Status::kSuccess, "blink_moegemm: run failed");
}

using Coop = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using Ping = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
using EpiCoop = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
using EpiPing = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
using EpiAuto = cutlass::epilogue::collective::EpilogueScheduleAuto;
using MainAuto = cutlass::gemm::collective::KernelScheduleAuto;

// variant: 0 = vLLM's choice (128x128x128, auto schedules = cooperative), 1 = pingpong 128x128x128,
// 2 = cooperative 128x128x256, 3 = pingpong 128x128x256, 4 = cooperative 128x256x128, 5 = pingpong 128x64x128,
// 6 = pingpong 128x64x256
static void moe_mm_cutlass(torch::Tensor out, torch::Tensor a, torch::Tensor b, torch::Tensor a_sf, torch::Tensor b_sf,
            torch::Tensor alphas, torch::Tensor problem_sizes, torch::Tensor expert_offsets, torch::Tensor sf_offsets,
            int64_t variant, int64_t raster, int64_t sm_limit) {
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && a.scalar_type() == at::kByte && b.scalar_type() == at::kByte);
  TORCH_CHECK(problem_sizes.scalar_type() == at::kInt && expert_offsets.scalar_type() == at::kInt &&
              sf_offsets.scalar_type() == at::kInt && alphas.scalar_type() == at::kFloat);
  switch (variant) {
    case 0: run<MoeGemm<Shape<_128, _128, _128>, MainAuto, EpiAuto>>(out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, (int)raster, (int)sm_limit); break;
    case 1: run<MoeGemm<Shape<_128, _128, _128>, Ping, EpiPing>>(out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, (int)raster, (int)sm_limit); break;
    case 2: run<MoeGemm<Shape<_128, _128, _256>, Coop, EpiCoop>>(out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, (int)raster, (int)sm_limit); break;
    case 3: run<MoeGemm<Shape<_128, _128, _256>, Ping, EpiPing>>(out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, (int)raster, (int)sm_limit); break;
    case 4: run<MoeGemm<Shape<_128, _256, _128>, Coop, EpiCoop>>(out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, (int)raster, (int)sm_limit); break;
    case 5: run<MoeGemm<Shape<_128, _64, _128>, Ping, EpiPing>>(out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, (int)raster, (int)sm_limit); break;
    case 6: run<MoeGemm<Shape<_128, _64, _256>, Ping, EpiPing>>(out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, (int)raster, (int)sm_limit); break;
    default: TORCH_CHECK(false, "blink_moegemm: unknown variant");
  }
}

// ===== inlined from bmoe_kernel.cuh =====
// SPDX-License-Identifier: Apache-2.0
// bmoe_kernel.cuh - persistent NVFP4 grouped GEMM for the Blink MoE experts on SM120 (RTX PRO 6000 Blackwell).
// Team ruj (bj/moe track), 2026-10-28. A drop-in for pjs's blink_moegemm moe_mm (CUTLASS 128x128x256 ptr-array
// grouped GEMM): the same inputs, and per output element the same arithmetic:
//   acc = +0; for c in 0 .. ceil(K/256)*4-1:            (64-wide K chunks in increasing order; chunks past K are
//     acc = mma.sync.m16n8k64.kind::mxf4nvf4             all-zero A, B and scale factors, exactly what CUTLASS's
//           .block_scale.scale_vec::4X (A_c, B_c, acc)   zero-filled 256-wide tail k-tile feeds the tensor core)
//   out = bf16_rn(fma(0, 0, alpha_e * acc))              (CUTLASS LinearCombination with beta = 0, C cleared)
// Only who computes an element and how the bytes reach shared memory differ:
//   * one persistent CTA per SM walks a static list of (expert, m-tile, n-tile) units; the list is built in shared
//     memory from problem_sizes at kernel start (no host sync, no setup kernels, CUDA-graph safe);
//   * B (weights) and A (activations) come by TMA from ONE descriptor per tensor (2D views [E*N, K/2] and
//     [rows, K/2]); A uses a box height that covers only the unit's rows (16-row steps), so nothing is re-described
//     per expert (CUTLASS re-writes and fences 4-5 tensormaps per tile) and no unused activation rows are fetched;
//   * the block scales (already in the 128x4 swizzled atom layout) come by 1D bulk copies;
//   * a deep smem ring (BK=128: 5 stages) is shared by all units of a CTA, so loads of the next unit overlap the
//     epilogue of the current one.
// MODE 1 (w13 only) fuses vLLM's gelu_tanh_and_mul and scaled_fp4_experts_quant into the epilogue: a unit holds
// 64 gate rows and the matching 64 up rows of w13, every thread has gate and up of the same (row, column) in its
// registers, rounds both to bf16 exactly as the GEMM would store them, applies vLLM's packed gelu_tanh * up
// (bf16 round trips included) and vLLM's NVFP4 block quantisation (same amax, rcp.approx, e4m3 / e2m1 cvts), and
// writes only the FP4 activations and their swizzled scales (the bf16 [rows, 2I] intermediate never exists).
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace bmoe {

constexpr int kMaxE = 128;
constexpr int kNumABoxes = 8;   // A box heights 16, 32, ..., 128 rows

struct __align__(64) Params {
  CUtensorMap tmA[kNumABoxes];  // A  [rows, K/2] bytes, box (BK/2) x (16*(i+1)) rows
  CUtensorMap tmB;              // B  [E*N, K/2] bytes, box (BK/2) x BN rows (MODE 1: BN/2 rows)
  const uint8_t* a_sf;          // A scales, 128x4 atoms; expert e starts at row-block sf_offsets[e]/128
  const uint8_t* b_sf;          // B scales [E][N/128][K/64] atoms of 512 B
  __nv_bfloat16* out;           // MODE 0: out rows of expert e start at expert_offsets[e]
  const float* alphas;          // [E]
  const int* problem_sizes;     // [E][3] (m, n, k)
  const int* expert_offsets;    // [E]
  const int* sf_offsets;        // [E]
  uint8_t* q_out;               // MODE 1: FP4 activations [rows, I/2]
  uint8_t* q_sf;                // MODE 1: their scales, swizzled, expert e at row sf_offsets[e]
  const float* q_gscale;        // MODE 1: [E] global scale of the quantisation (a2_gscale)
  const uint8_t* a_src;         // GA: unpermuted FP4 activations [tokens, K/2] (vLLM quant of the m token rows)
  const uint8_t* a_src_sf;      // GA: their scales, swizzled for the token rows (one "expert", offset 0)
  const int* a_map;             // GA: expert-sorted row i <- token a_map[i] (vLLM input_permutation)
  int E, N, K, ldo;
  int n_chunks;                 // ceil(K/256)*4: the chunk count of CUTLASS's 256-wide k-tiles
  int n_real;                   // K/64: chunks that carry data
  int I;                        // MODE 1: N/2
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred P1;\n\t"
      "BMOE_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra BMOE_DONE;\n\t"
      "bra BMOE_WAIT;\n\t"
      "BMOE_DONE:\n\t}" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* tm, uint32_t bar, int c0, int c1,
                                            uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(tm)), "r"(bar), "r"(c0), "r"(c1), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar,
                                          uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(
          dst),
      "l"(src), "r"(bytes), "r"(bar), "l"(policy)
      : "memory");
}
template <int P>
__device__ __forceinline__ uint64_t make_policy() {
  uint64_t p;
  if constexpr (P == 0) asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p));
  else if constexpr (P == 1) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  else asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
// stores with an L2 policy (OPOL: keep the GEMM output in L2 for the kernel that reads it next)
__device__ __forceinline__ void st_b32_hint(void* ptr, uint32_t v, uint64_t pol) {
  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;" ::"l"(ptr), "r"(v), "l"(pol) : "memory");
}
__device__ __forceinline__ void st_v4_hint(void* ptr, uint4 v, uint64_t pol) {
  asm volatile("st.global.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;" ::"l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w), "l"(pol)
               : "memory");
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_arrive_noinc(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ uint32_t lds32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(addr));
  return v;
}
// D = A x B + D, NVFP4 (e2m1 data, ue4m3 scales per 16 K), fp32 accumulation; the instruction CUTLASS's
// SM120_16x8x64_TN_VS<e2m1, e2m1, float, ue4m3, 16> issues (byte-id 0, thread-id 0 for both scale operands).
__device__ __forceinline__ void mma_nvf4(float* d, const uint32_t* a, const uint32_t* b, uint32_t sfa,
                                         uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "h"((uint16_t)0),
        "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

// smem byte offset of 16-byte chunk c of row r in a TMA tile with rows of RB bytes (64B or 128B swizzle)
template <int RB>
__device__ __forceinline__ uint32_t swz(int r, int c) {
  if constexpr (RB == 64) {
    return (uint32_t)(r * 64 + ((c ^ ((r >> 1) & 3)) << 4));
  } else {
    return (uint32_t)(r * 128 + ((c ^ (r & 7)) << 4));
  }
}

// ---- the epilogue arithmetic of the reference path (CUTLASS GEMM epilogue, vLLM activation and quant) ----
// CUTLASS LinearCombination, beta = 0 and C cleared: fma(beta, C, alpha * acc), then cvt.rn to bf16
__device__ __forceinline__ float epi(float alpha, float acc) { return __fadd_rn(__fmul_rn(alpha, acc), 0.0f); }
// vLLM packed_gelu_tanh_kernel (float math, bf16 result)
__device__ __forceinline__ float gelu_tanh_f(float x) {
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = x * x * x;
  float inner = BETA * (x + KAPPA * x_cube);
  return 0.5f * x * (1.0f + ::tanhf(inner));
}
// vLLM packed_compute<act_first, no clamp>: bf16(float(bf16(gelu_tanh(gate))) * (float(up) + beta)), beta = 0
__device__ __forceinline__ __nv_bfloat162 gelu_tanh_mul2(__nv_bfloat162 g, __nv_bfloat162 u, float beta) {
  float2 gf = __bfloat1622float2(g);
  __nv_bfloat162 act = __floats2bfloat162_rn(gelu_tanh_f(gf.x), gelu_tanh_f(gf.y));
  float2 a = __bfloat1622float2(act);
  float2 uf = __bfloat1622float2(u);
  a.x = __fmul_rn(a.x, __fadd_rn(uf.x, beta));
  a.y = __fmul_rn(a.y, __fadd_rn(uf.y, beta));
  return __floats2bfloat162_rn(a.x, a.y);
}
__device__ __forceinline__ float rcp_approx_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}
__device__ __forceinline__ uint32_t e2m1x2(float lo, float hi) {
  uint32_t v;
  asm volatile(
      "{\n\t.reg .b8 b;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\n\t"
      "cvt.u32.u8 %0, b;\n\t}"
      : "=r"(v)
      : "f"(lo), "f"(hi));
  return v;
}

template <int BM_, int BN_, int BK_, int STAGES_, int WM_, int WN_, int MODE_ = 0, int POLB_ = 1, int POLA_ = 2,
          int EPI_ = 0, int SCHED_ = 0, int GA_ = 0, int OPOL_ = 0>
struct Cfg {
  static constexpr int BM = BM_, BN = BN_, BK = BK_, STAGES = STAGES_, WM = WM_, WN = WN_, MODE = MODE_;
  static constexpr int POLB = POLB_, POLA = POLA_;
  static constexpr int EPI = EPI_;               // MODE 0 stores: 0 = 4-byte (bf16x2) stores, 1 = 16-byte stores
  static constexpr int SCHED = SCHED_;           // 0: static round-robin units; 1: dynamic (global atomic counter)
  static constexpr int GA = GA_;                 // 1: A rows gathered from the unpermuted tokens (cp.async, a_map)
  static constexpr int OPOL = OPOL_;             // 1: output stores with L2 evict_last (MODE 0 only)
  static constexpr int FULL_COUNT = GA ? 33 : 1;  // full-barrier arrivals: lane 0 (+ tx) and, GA, 32 cp.async arrives
  static constexpr int NCW = WM * WN;            // consumer warps
  static constexpr int THREADS = (NCW + 1) * 32; // + 1 producer warp
  static constexpr int RB = BK / 2;              // bytes per tile row (64 or 128)
  static constexpr int CPS = BK / 64;            // 64-wide K chunks per stage
  static constexpr int WTM = BM / WM, WTN = BN / WN;
  static constexpr int MT = WTM / 16, NT = WTN / 8;
  static constexpr int A_BYTES = BM * RB, B_BYTES = BN * RB;
  static constexpr int SFA_BYTES = CPS * 512;             // one 128-row atom per chunk
  static constexpr int SFB_BLOCKS = MODE == 1 ? 2 : BN / 128;
  static constexpr int SFB_BYTES = SFB_BLOCKS * CPS * 512;
  static constexpr int STAGE_BYTES = A_BYTES + B_BYTES + SFA_BYTES + SFB_BYTES;
  static constexpr int TABLE_BYTES = (4 * kMaxE + 1) * 4;
  static constexpr int BAR_BYTES = 2 * STAGES * 8;
  static constexpr int SMEM = 1024 /*align slack*/ + STAGES * STAGE_BYTES + BAR_BYTES + TABLE_BYTES;
  static constexpr int UNIT_N = MODE == 1 ? BN / 2 : BN; // output columns (MODE 1: intermediate columns) per unit
  static_assert(BN % 128 == 0, "BN must be a multiple of 128 (scale atoms)");
  static_assert(MODE == 0 || (BN == 128 && (WN == 2 || WN == 4)),
                "MODE 1 needs a 64 gate + 64 up tile over two or four warp columns");
  static_assert(BM == 64 || BM == 128, "BM");
  static_assert(BK == 128 || BK == 256, "BK");
  static_assert(NT % 4 == 0 || MODE == 0, "MODE 1 tiles");
  static_assert(NT % 2 == 0, "NT must be even");
  static_assert(STAGE_BYTES % 1024 == 0, "stage alignment");
  static_assert(EPI == 0 || NT % 4 == 0, "16-byte stores need 4 n8 tiles per quad transpose");
};

// 4x4 transpose of 32-bit words inside a quad (lanes 4g..4g+3): in: a[k] = this lane's 2 columns of n8 tile k;
// out: b[k] = lane k's 2 columns of n8 tile tg (so this lane holds the 8 columns of tile tg)
__device__ __forceinline__ void quad_transpose(const uint32_t (&a)[4], uint32_t (&b)[4], int lane) {
  const int tg = lane & 3;
  uint32_t res[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) res[k] = (k == tg) ? a[k] : 0u;
#pragma unroll
  for (int rr = 1; rr < 4; ++rr) {
    const int ks = (tg + rr) & 3;              // word this lane sends: its slice of tile (tg + rr)
    uint32_t send = ks == 0 ? a[0] : ks == 1 ? a[1] : ks == 2 ? a[2] : a[3];
    const int src = (tg - rr) & 3;             // lane whose slice of tile tg arrives
    uint32_t recv = __shfl_sync(0xffffffffu, send, (lane & ~3) | src);
#pragma unroll
    for (int k = 0; k < 4; ++k) res[k] = (k == src) ? recv : res[k];
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) b[k] = res[k];
}

struct Unit {
  int e, m0, n0, rows;
};

// dynamic unit counter (SCHED = 1): [0] next unit, [1] CTAs done; the last CTA of a launch resets both, so every
// launch (CUDA-graph replays included) starts from 0. Kernels using it must not run concurrently with each other.
__device__ int g_bmoe_sched[2];

template <class C>
__device__ __forceinline__ Unit decode(int u, const int* s_prefix, const int* s_m, int E) {
  int lo = 0, hi = E;
  while (hi - lo > 1) {
    int mid = (lo + hi) >> 1;
    if (s_prefix[mid] <= u) lo = mid; else hi = mid;
  }
  int e = lo, local = u - s_prefix[e], M = s_m[e];
  int mtn = (M + C::BM - 1) / C::BM;
  int mt = local % mtn, nt = local / mtn;
  Unit r;
  r.e = e;
  r.m0 = mt * C::BM;
  r.n0 = nt * C::UNIT_N;
  r.rows = min(C::BM, M - r.m0);
  return r;
}

// first smem B row of n8 tile j of warp column wn
template <class C>
__device__ __forceinline__ int b_row_of_tile(int wn, int j) {
  if constexpr (C::MODE == 1) {
    return (j < C::NT / 2 ? 0 : C::BN / 2) + wn * (C::WTN / 2) + 8 * (j % (C::NT / 2));
  } else {
    return wn * C::WTN + 8 * j;
  }
}

template <class C>
__global__ void __launch_bounds__(C::THREADS, 1) bmoe_kernel(const __grid_constant__ Params p) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* stages = smem;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + C::STAGES * C::STAGE_BYTES);
  int* s_prefix = reinterpret_cast<int*>(smem + C::STAGES * C::STAGE_BYTES + C::BAR_BYTES);
  int* s_m = s_prefix + (kMaxE + 1);
  int* s_off = s_m + kMaxE;
  int* s_sfo = s_off + kMaxE;
  __shared__ int warp_tot[16];
  __shared__ int s_unit[2];
  __shared__ __align__(8) uint64_t sched_bar[4];   // [0,1] unit published (1 arrival), [2,3] unit read (NCW)

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int E = p.E;
  const int ntn = (C::MODE == 1 ? p.I : p.N) / C::UNIT_N;

  // ---- unit table: units of expert e = ceil(m_e / BM) * ntn; exclusive prefix in s_prefix ----
  {
    int units = 0;
    if (tid < E) {
      int m = __ldg(p.problem_sizes + 3 * tid);
      s_m[tid] = m;
      s_off[tid] = __ldg(p.expert_offsets + tid);
      s_sfo[tid] = __ldg(p.sf_offsets + tid);
      units = ((m + C::BM - 1) / C::BM) * ntn;
    }
    int v = units;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      int t = __shfl_up_sync(0xffffffffu, v, o);
      if (lane >= o) v += t;
    }
    if (lane == 31) warp_tot[warp] = v;
    if (tid == 0) {
      for (int s = 0; s < C::STAGES; ++s) {
        mbar_init(smem_u32(&bars[s]), C::FULL_COUNT);
        mbar_init(smem_u32(&bars[C::STAGES + s]), C::NCW);
      }
      for (int s = 0; s < 2; ++s) {
        mbar_init(smem_u32(&sched_bar[s]), 1);
        mbar_init(smem_u32(&sched_bar[2 + s]), C::NCW);
      }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    int base = 0;
    for (int w = 0; w < warp; ++w) base += warp_tot[w];
    if (tid < E) s_prefix[tid] = base + v - units;
    if (tid == 0) {
      int tot = 0;
      for (int w = 0; w < (E + 31) / 32; ++w) tot += warp_tot[w];
      s_prefix[E] = tot;
    }
    __syncthreads();
  }
  const int total = s_prefix[E];
  const int steps = p.n_chunks / C::CPS;

  if (C::GA == 1 && warp == C::NCW) {
    // ================= producer warp, A gathered: all 32 lanes =================
    const uint64_t pol_b = make_policy<C::POLB>();
    const int K4 = p.K / 64;
    const size_t b_sf_e = (size_t)p.N * (p.K / 16);
    const int KB = p.K / 2;
    int stage = 0;
    uint32_t phase = 0;
    int u, slot = 0;
    uint32_t sph = 0;
    if constexpr (C::SCHED == 1) {
      u = 0;
      if (lane == 0) u = atomicAdd(&g_bmoe_sched[0], 1);
      u = __shfl_sync(0xffffffffu, u, 0);
    } else {
      u = blockIdx.x;
    }
    while (true) {
      if constexpr (C::SCHED == 1) {
        if (lane == 0) {
          mbar_wait(smem_u32(&sched_bar[2 + slot]), sph ^ 1);
          s_unit[slot] = u;
          mbar_arrive(smem_u32(&sched_bar[slot]));
        }
        if (++slot == 2) { slot = 0; sph ^= 1; }
        __syncwarp();
      }
      if (u >= total) break;
      int u_next = 0;
      if constexpr (C::SCHED == 1) {
        if (lane == 0) u_next = atomicAdd(&g_bmoe_sched[0], 1);
      } else {
        u_next = u + gridDim.x;
      }
      Unit t = decode<C>(u, s_prefix, s_m, E);
      constexpr int RJ = C::BM / 32;
      int srow[RJ];
#pragma unroll
      for (int j = 0; j < RJ; ++j) {
        const int r = lane + 32 * j;
        srow[j] = r < t.rows ? __ldg(p.a_map + s_off[t.e] + t.m0 + r) : -1;
      }
      const int a_rowblk_off = t.m0 & 127;
      int brow0, brow1;
      const uint8_t *sfb0, *sfb1;
      if constexpr (C::MODE == 1) {
        brow0 = t.e * p.N + t.n0;
        brow1 = t.e * p.N + p.I + t.n0;
        sfb0 = p.b_sf + (size_t)t.e * b_sf_e + (size_t)(t.n0 >> 7) * K4 * 512;
        sfb1 = p.b_sf + (size_t)t.e * b_sf_e + (size_t)((p.I + t.n0) >> 7) * K4 * 512;
      } else {
        brow0 = t.e * p.N + t.n0;
        brow1 = 0;
        sfb0 = p.b_sf + (size_t)t.e * b_sf_e + (size_t)(t.n0 >> 7) * K4 * 512;
        sfb1 = nullptr;
      }
      for (int s = 0; s < steps; ++s) {
        mbar_wait(smem_u32(&bars[C::STAGES + stage]), phase ^ 1);
        uint8_t* st = stages + stage * C::STAGE_BYTES;
        const uint32_t full = smem_u32(&bars[stage]);
        const int c0 = s * C::CPS;
        const int nreal = max(0, min(C::CPS, p.n_real - c0));
        const uint32_t sA = smem_u32(st), sSFA = smem_u32(st + C::A_BYTES + C::B_BYTES);
        if (lane == 0) {
          mbar_expect_tx(full, C::B_BYTES + nreal * 512 * C::SFB_BLOCKS);
          if constexpr (C::MODE == 1) {
            tma_load_2d(smem_u32(st + C::A_BYTES), &p.tmB, full, s * C::RB, brow0, pol_b);
            tma_load_2d(smem_u32(st + C::A_BYTES + C::B_BYTES / 2), &p.tmB, full, s * C::RB, brow1, pol_b);
          } else {
            tma_load_2d(smem_u32(st + C::A_BYTES), &p.tmB, full, s * C::RB, brow0, pol_b);
          }
          if (nreal > 0) {
            const uint32_t sfb_dst = sSFA + C::SFA_BYTES;
            if constexpr (C::MODE == 1) {
              bulk_load(sfb_dst, sfb0 + (size_t)c0 * 512, nreal * 512, full, pol_b);
              bulk_load(sfb_dst + C::CPS * 512, sfb1 + (size_t)c0 * 512, nreal * 512, full, pol_b);
            } else {
#pragma unroll
              for (int nb = 0; nb < C::SFB_BLOCKS; ++nb)
                bulk_load(sfb_dst + nb * C::CPS * 512, sfb0 + (size_t)nb * K4 * 512 + (size_t)c0 * 512, nreal * 512,
                          full, pol_b);
            }
          }
        }
        // the unit's A rows (16-byte chunks) and their scale words, from the L2-resident token rows
#pragma unroll
        for (int j = 0; j < RJ; ++j) {
          if (srow[j] >= 0) {
            const int r = lane + 32 * j;
            const uint8_t* src = p.a_src + (size_t)srow[j] * KB + s * C::RB;
            if (s * C::RB < KB) {
#pragma unroll
              for (int c = 0; c < C::RB / 16; ++c)
                if (s * C::RB + c * 16 < KB) cp_async16(sA + swz<C::RB>(r, c), src + c * 16);
            }
            const int ra = a_rowblk_off + r;
            const uint8_t* sfs = p.a_src_sf + (size_t)(srow[j] >> 7) * K4 * 512 + (srow[j] & 31) * 16 +
                                 ((srow[j] >> 5) & 3) * 4;
            for (int cc = 0; cc < nreal; ++cc)
              cp_async4(sSFA + cc * 512 + (ra & 31) * 16 + ((ra >> 5) & 3) * 4, sfs + (size_t)(c0 + cc) * 512);
          }
        }
        cp_async_arrive_noinc(full);
        if (++stage == C::STAGES) { stage = 0; phase ^= 1; }
      }
      if constexpr (C::SCHED == 1) u_next = __shfl_sync(0xffffffffu, u_next, 0);
      u = u_next;
    }
    if constexpr (C::SCHED == 1) {
      if (lane == 0 && atomicAdd(&g_bmoe_sched[1], 1) == (int)gridDim.x - 1) {
        atomicExch(&g_bmoe_sched[0], 0);
        atomicExch(&g_bmoe_sched[1], 0);
      }
    }
    return;
  }
  if (warp == C::NCW) {
    // ================= producer warp =================
    if (lane == 0) {
      const uint64_t pol_b = make_policy<C::POLB>();
      const uint64_t pol_a = make_policy<C::POLA>();
      const int K4 = p.K / 64;               // scale atoms per 128-row block
      const size_t b_sf_e = (size_t)p.N * (p.K / 16);
      int stage = 0;
      uint32_t phase = 0;
      int u, slot = 0;
      uint32_t sph = 0;
      if constexpr (C::SCHED == 1) u = atomicAdd(&g_bmoe_sched[0], 1); else u = blockIdx.x;
      while (true) {
        if constexpr (C::SCHED == 1) {
          mbar_wait(smem_u32(&sched_bar[2 + slot]), sph ^ 1);
          s_unit[slot] = u;
          mbar_arrive(smem_u32(&sched_bar[slot]));
          if (++slot == 2) { slot = 0; sph ^= 1; }
        }
        if (u >= total) break;
        int u_next;   // fetched now, used after this unit's loads (hides the atomic's latency)
        if constexpr (C::SCHED == 1) u_next = atomicAdd(&g_bmoe_sched[0], 1); else u_next = u + gridDim.x;
        Unit t = decode<C>(u, s_prefix, s_m, E);
        const int hbox = (t.rows + 15) >> 4;  // 1..8
        const int arow = s_off[t.e] + t.m0;
        const int arb = (s_sfo[t.e] + t.m0) >> 7;  // A scale row-block
        const uint8_t* sfa_base = p.a_sf + (size_t)arb * K4 * 512;
        int brow0, brow1;                      // MODE 1: gate rows, up rows
        const uint8_t *sfb0, *sfb1;
        if constexpr (C::MODE == 1) {
          brow0 = t.e * p.N + t.n0;
          brow1 = t.e * p.N + p.I + t.n0;
          sfb0 = p.b_sf + (size_t)t.e * b_sf_e + (size_t)(t.n0 >> 7) * K4 * 512;
          sfb1 = p.b_sf + (size_t)t.e * b_sf_e + (size_t)((p.I + t.n0) >> 7) * K4 * 512;
        } else {
          brow0 = t.e * p.N + t.n0;
          brow1 = 0;
          sfb0 = p.b_sf + (size_t)t.e * b_sf_e + (size_t)(t.n0 >> 7) * K4 * 512;
          sfb1 = nullptr;
        }
        for (int s = 0; s < steps; ++s) {
          mbar_wait(smem_u32(&bars[C::STAGES + stage]), phase ^ 1);
          uint8_t* st = stages + stage * C::STAGE_BYTES;
          const uint32_t full = smem_u32(&bars[stage]);
          const int c0 = s * C::CPS;
          const int nreal = max(0, min(C::CPS, p.n_real - c0));
          const uint32_t bytes = hbox * 16 * C::RB + C::B_BYTES + nreal * 512 * (1 + C::SFB_BLOCKS);
          mbar_expect_tx(full, bytes);
          tma_load_2d(smem_u32(st), &p.tmA[hbox - 1], full, s * C::RB, arow, pol_a);
          if constexpr (C::MODE == 1) {
            tma_load_2d(smem_u32(st + C::A_BYTES), &p.tmB, full, s * C::RB, brow0, pol_b);
            tma_load_2d(smem_u32(st + C::A_BYTES + C::B_BYTES / 2), &p.tmB, full, s * C::RB, brow1, pol_b);
          } else {
            tma_load_2d(smem_u32(st + C::A_BYTES), &p.tmB, full, s * C::RB, brow0, pol_b);
          }
          if (nreal > 0) {
            const uint32_t sf_dst = smem_u32(st + C::A_BYTES + C::B_BYTES);
            bulk_load(sf_dst, sfa_base + (size_t)c0 * 512, nreal * 512, full, pol_a);
            if constexpr (C::MODE == 1) {
              bulk_load(sf_dst + C::SFA_BYTES, sfb0 + (size_t)c0 * 512, nreal * 512, full, pol_b);
              bulk_load(sf_dst + C::SFA_BYTES + C::CPS * 512, sfb1 + (size_t)c0 * 512, nreal * 512, full, pol_b);
            } else {
#pragma unroll
              for (int nb = 0; nb < C::SFB_BLOCKS; ++nb)
                bulk_load(sf_dst + C::SFA_BYTES + nb * C::CPS * 512, sfb0 + (size_t)nb * K4 * 512 + (size_t)c0 * 512,
                          nreal * 512, full, pol_b);
            }
          }
          if (++stage == C::STAGES) { stage = 0; phase ^= 1; }
        }
        u = u_next;
      }
      if constexpr (C::SCHED == 1) {
        if (atomicAdd(&g_bmoe_sched[1], 1) == (int)gridDim.x - 1) {
          atomicExch(&g_bmoe_sched[0], 0);
          atomicExch(&g_bmoe_sched[1], 0);
        }
      }
    }
    return;
  }

  // ================= consumer warps =================
  const int wm = warp / C::WN, wn = warp % C::WN;
  const int g = lane >> 2, tg = lane & 3;
  const int lq = lane >> 3, li = lane & 7;
  int stage = 0;
  uint32_t phase = 0;
  int cslot = 0;
  uint32_t csph = 0;
  for (int u = blockIdx.x;; u += gridDim.x) {
    if constexpr (C::SCHED == 1) {
      mbar_wait(smem_u32(&sched_bar[cslot]), csph);
      u = *reinterpret_cast<volatile int*>(&s_unit[cslot]);
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_u32(&sched_bar[2 + cslot]));
      if (++cslot == 2) { cslot = 0; csph ^= 1; }
    }
    if (u >= total) break;
    Unit t = decode<C>(u, s_prefix, s_m, E);
    const int mbase = wm * C::WTM;           // first row of this warp in the unit
    const int nvalid = t.rows - mbase;      // rows of this warp's band that are real
    // B scale rows: smem B row n lives in scale block blk(n) at row-in-atom rin(n)
    int sfb_rin0, sfb_rin1;
    if constexpr (C::MODE == 1) {
      sfb_rin0 = t.n0 & 127;
      sfb_rin1 = ((p.I + t.n0) & 127) - C::BN / 2;
    } else {
      sfb_rin0 = 0;
      sfb_rin1 = 0;
    }
    float acc[C::MT][C::NT][4];
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
      for (int j = 0; j < C::NT; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.0f;
    const int a_rowblk_off = t.m0 & 127;     // row offset of this unit inside its 128-row scale atom
    for (int s = 0; s < steps; ++s) {
      mbar_wait(smem_u32(&bars[stage]), phase);
      const uint32_t st = smem_u32(stages + stage * C::STAGE_BYTES);
      const uint32_t sA = st, sB = st + C::A_BYTES, sSFA = st + C::A_BYTES + C::B_BYTES,
                     sSFB = sSFA + C::SFA_BYTES;
      if (nvalid > 0) {
#pragma unroll
        for (int cc = 0; cc < C::CPS; ++cc) {
          const int chunk = s * C::CPS + cc;
          uint32_t af[C::MT][4], bfr[C::NT][2], sfa[C::MT], sfb[C::NT];
          if (chunk < p.n_real) {
#pragma unroll
            for (int i = 0; i < C::MT; ++i) {
              const int r = mbase + i * 16 + (lq & 1) * 8 + li;
              ldsm_x4(sA + swz<C::RB>(r, cc * 2 + (lq >> 1)), af[i][0], af[i][1], af[i][2], af[i][3]);
              const int ra = a_rowblk_off + mbase + i * 16 + g + 8 * (lane & 1);
              sfa[i] = lds32(sSFA + cc * 512 + (ra & 31) * 16 + ((ra >> 5) & 3) * 4);
            }
#pragma unroll
            for (int jp = 0; jp < C::NT / 2; ++jp) {
              const int n = b_row_of_tile<C>(wn, 2 * jp) + (lq >> 1) * 8 + li;
              ldsm_x4(sB + swz<C::RB>(n, cc * 2 + (lq & 1)), bfr[2 * jp][0], bfr[2 * jp][1], bfr[2 * jp + 1][0],
                      bfr[2 * jp + 1][1]);
            }
#pragma unroll
            for (int j = 0; j < C::NT; ++j) {
              const int nb = b_row_of_tile<C>(wn, j) + g;  // smem B row of this thread's column
              int blk, rin;
              if constexpr (C::MODE == 1) {
                blk = nb >= C::BN / 2;
                rin = blk ? sfb_rin1 + nb : sfb_rin0 + nb;
              } else {
                blk = nb >> 7;
                rin = nb & 127;
              }
              sfb[j] = lds32(sSFB + blk * C::CPS * 512 + cc * 512 + (rin & 31) * 16 + ((rin >> 5) & 3) * 4);
            }
          } else {
#pragma unroll
            for (int i = 0; i < C::MT; ++i) { af[i][0] = af[i][1] = af[i][2] = af[i][3] = 0u; sfa[i] = 0u; }
#pragma unroll
            for (int j = 0; j < C::NT; ++j) { bfr[j][0] = bfr[j][1] = 0u; sfb[j] = 0u; }
          }
#pragma unroll
          for (int i = 0; i < C::MT; ++i) {
            if (i * 16 < nvalid) {
#pragma unroll
              for (int j = 0; j < C::NT; ++j) mma_nvf4(acc[i][j], af[i], bfr[j], sfa[i], sfb[j]);
            }
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_u32(&bars[C::STAGES + stage]));
      if (++stage == C::STAGES) { stage = 0; phase ^= 1; }
    }
    if (nvalid <= 0) continue;
    const float alpha = __ldg(p.alphas + t.e);
    uint64_t opol = 0;
    if constexpr (C::OPOL == 1) opol = make_policy<2>();
    if constexpr (C::MODE == 0 && C::EPI == 1) {
      // ---- epilogue, 16-byte stores: same values, quad-transposed so each lane writes 8 consecutive columns ----
      __nv_bfloat16* obase = p.out + (size_t)(s_off[t.e] + t.m0) * p.ldo + t.n0 + wn * C::WTN + 8 * tg;
#pragma unroll
      for (int i = 0; i < C::MT; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = mbase + i * 16 + g + 8 * h;
#pragma unroll
          for (int jq = 0; jq < C::NT / 4; ++jq) {
            uint32_t a4[4], b4[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              __nv_bfloat162 v = __floats2bfloat162_rn(epi(alpha, acc[i][4 * jq + k][2 * h]),
                                                       epi(alpha, acc[i][4 * jq + k][2 * h + 1]));
              a4[k] = *reinterpret_cast<uint32_t*>(&v);
            }
            quad_transpose(a4, b4, lane);
            if (r < t.rows) {
              if constexpr (C::OPOL == 1)
                st_v4_hint(obase + (size_t)r * p.ldo + 32 * jq, make_uint4(b4[0], b4[1], b4[2], b4[3]), opol);
              else
                *reinterpret_cast<uint4*>(obase + (size_t)r * p.ldo + 32 * jq) = make_uint4(b4[0], b4[1], b4[2], b4[3]);
            }
          }
        }
      }
    } else if constexpr (C::MODE == 0) {
      // ---- epilogue: out = bf16_rn(fma(0, 0, alpha * acc)) for the real rows ----
      __nv_bfloat16* orow = p.out + (size_t)(s_off[t.e] + t.m0) * p.ldo + t.n0 + wn * C::WTN + 2 * tg;
#pragma unroll
      for (int i = 0; i < C::MT; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = mbase + i * 16 + g + 8 * h;
          if (r < t.rows) {
            __nv_bfloat16* o = orow + (size_t)r * p.ldo;
#pragma unroll
            for (int j = 0; j < C::NT; ++j) {
              __nv_bfloat162 v2 = __floats2bfloat162_rn(epi(alpha, acc[i][j][2 * h]), epi(alpha, acc[i][j][2 * h + 1]));
              if constexpr (C::OPOL == 1)
                st_b32_hint(o + j * 8, *reinterpret_cast<uint32_t*>(&v2), opol);
              else
                *reinterpret_cast<__nv_bfloat162*>(o + j * 8) = v2;
            }
          }
        }
      }
    } else {
      // ---- fused epilogue: c1 (bf16) -> gelu_tanh(gate) * up (bf16) -> NVFP4 (vLLM's experts quant) ----
      const float sfscale = __ldg(p.q_gscale + t.e);
      const float beta = 0.0f;
      const int K4i = (p.I + 63) / 64;
      const int qcol0 = t.n0 + wn * (C::WTN / 2);           // first intermediate column of this warp
#pragma unroll
      for (int i = 0; i < C::MT; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = mbase + i * 16 + g + 8 * h;            // row inside the unit
#pragma unroll
          for (int q = 0; q < C::NT / 4; ++q) {               // 16-column quant blocks of this warp
            __nv_bfloat162 v[2];
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
              const int j = 2 * q + jj;
              __nv_bfloat162 gb = __floats2bfloat162_rn(epi(alpha, acc[i][j][2 * h]), epi(alpha, acc[i][j][2 * h + 1]));
              __nv_bfloat162 ub = __floats2bfloat162_rn(epi(alpha, acc[i][j + C::NT / 2][2 * h]),
                                                        epi(alpha, acc[i][j + C::NT / 2][2 * h + 1]));
              v[jj] = gelu_tanh_mul2(gb, ub, beta);
            }
            // amax over the 16 values of the block (4 per thread, the quad holds the block)
            __nv_bfloat162 lmax = __hmax2(__habs2(v[0]), __habs2(v[1]));
            uint32_t lm = *reinterpret_cast<uint32_t*>(&lmax);
            uint32_t o1 = __shfl_xor_sync(0xffffffffu, lm, 1);
            lmax = __hmax2(*reinterpret_cast<__nv_bfloat162*>(&o1), lmax);
            lm = *reinterpret_cast<uint32_t*>(&lmax);
            uint32_t o2 = __shfl_xor_sync(0xffffffffu, lm, 2);
            lmax = __hmax2(*reinterpret_cast<__nv_bfloat162*>(&o2), lmax);
            const float vecMax = __bfloat162float(__hmax(lmax.x, lmax.y));
            float SFValue = sfscale * (vecMax * rcp_approx_ftz(6.0f));
            __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(SFValue);
            SFValue = float(sf8);
            const float oscale = SFValue != 0.0f ? rcp_approx_ftz(SFValue * rcp_approx_ftz(sfscale)) : 0.0f;
            float2 f0 = __bfloat1622float2(v[0]), f1 = __bfloat1622float2(v[1]);
            const uint32_t b0 = e2m1x2(f0.x * oscale, f0.y * oscale);  // block byte tg
            const uint32_t b1 = e2m1x2(f1.x * oscale, f1.y * oscale);  // block byte 4 + tg
            uint32_t y = b0 | (b1 << 16);
            y |= __shfl_xor_sync(0xffffffffu, y, 1) << 8;            // even tg: bytes (tg, tg+1) | (4+tg, 5+tg)
            const uint32_t y2 = __shfl_xor_sync(0xffffffffu, y, 2);
            if (tg == 0 && r < t.rows) {
              const int row = s_off[t.e] + t.m0 + r;
              const int qc = qcol0 + 16 * q;                      // first column of the block
              uint2 w;
              w.x = (y & 0xFFFFu) | ((y2 & 0xFFFFu) << 16);
              w.y = (y >> 16) | (y2 & 0xFFFF0000u);
              *reinterpret_cast<uint2*>(p.q_out + (size_t)row * (p.I / 2) + qc / 2) = w;
              const int srow = s_sfo[t.e] + t.m0 + r;
              const int kb = qc / 16;
              const size_t off = ((size_t)(srow >> 7) * K4i + (kb >> 2)) * 512 + (srow & 31) * 16 +
                                 ((srow >> 5) & 3) * 4 + (kb & 3);
              p.q_sf[off] = sf8.__x;
            }
          }
        }
      }
    }
  }
}

}  // namespace bmoe

// ===== inlined from bmoe_host.cuh =====
// SPDX-License-Identifier: Apache-2.0
// bmoe_host.cuh - host side of the bmoe NVFP4 grouped GEMM (team ruj, bj/moe track, 2026-10-28):
// TMA descriptor encoding (driver entry point, no -lcuda), config dispatch, launch on the current stream.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <mutex>
#include <unordered_map>


namespace bmoe {

// cuTensorMapEncodeTiled (CUDA 12.0 ABI), fetched through the runtime: no -lcuda at link time
using EncodeTiledFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn encode_fn() {
  static EncodeTiledFn fn = nullptr;
  static std::once_flag once;
  std::call_once(once, [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaError_t err = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
    TORCH_CHECK(err == cudaSuccess && q == cudaDriverEntryPointSuccess && p != nullptr,
                "bmoe: cuTensorMapEncodeTiled not available");
    fn = reinterpret_cast<EncodeTiledFn>(p);
  });
  return fn;
}

// 2D u8 tensor [rows, cols] (cols contiguous), box [box_cols, box_rows]
static void make_tmap(CUtensorMap* tm, const void* base, uint64_t rows, uint64_t cols, uint32_t box_cols,
                      uint32_t box_rows, CUtensorMapSwizzle swz) {
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {cols};
  cuuint32_t box[2] = {box_cols, box_rows};
  cuuint32_t es[2] = {1, 1};
  CUresult r = encode_fn()(tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box, es,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "bmoe: cuTensorMapEncodeTiled failed (", (int)r, ")");
}

static int sm_count_dev(int dev) {
  static std::unordered_map<int, int> cache;
  static std::mutex mu;
  std::lock_guard<std::mutex> g(mu);
  auto it = cache.find(dev);
  if (it != cache.end()) return it->second;
  int n = 0;
  cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  cache[dev] = n;
  return n;
}

static void check_common(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& a_sf,
                         const torch::Tensor& b_sf, const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
                         const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets) {
  TORCH_CHECK(a.scalar_type() == at::kByte && b.scalar_type() == at::kByte, "bmoe: a/b must be uint8 (packed e2m1)");
  TORCH_CHECK(problem_sizes.scalar_type() == at::kInt && expert_offsets.scalar_type() == at::kInt &&
                  sf_offsets.scalar_type() == at::kInt && alphas.scalar_type() == at::kFloat,
              "bmoe: index/alpha dtypes");
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && a_sf.is_cuda() && b_sf.is_cuda() && problem_sizes.is_cuda(),
              "bmoe: cuda tensors");
  TORCH_CHECK(a.dim() == 2 && a.stride(1) == 1 && a.stride(0) == a.size(1), "bmoe: a must be contiguous [rows, K/2]");
  TORCH_CHECK(b.is_contiguous() && a_sf.is_contiguous() && b_sf.is_contiguous(), "bmoe: contiguity");
  TORCH_CHECK(problem_sizes.is_contiguous() && expert_offsets.is_contiguous() && sf_offsets.is_contiguous() &&
                  alphas.is_contiguous(),
              "bmoe: contiguity of index tensors");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(a.data_ptr()) & 15) == 0 &&
                  (reinterpret_cast<uintptr_t>(b.data_ptr()) & 15) == 0 &&
                  (reinterpret_cast<uintptr_t>(a_sf.data_ptr()) & 15) == 0 &&
                  (reinterpret_cast<uintptr_t>(b_sf.data_ptr()) & 15) == 0,
              "bmoe: 16-byte alignment");
}

template <class C>
static void launch(Params& p, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& a_sf,
                   const torch::Tensor& b_sf, const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
                   const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets, int64_t N, int sm_limit,
                   int64_t rows_override = -1) {
  // GA: `a` is the unpermuted token tensor [tokens, K/2]; the GEMM's rows = rows_override (= a_map.size(0))
  const int E = static_cast<int>(expert_offsets.size(0));
  const int64_t K = a.size(1) * 2, rows = rows_override >= 0 ? rows_override : a.size(0);
  TORCH_CHECK(E <= kMaxE, "bmoe: too many experts");
  TORCH_CHECK(K % 64 == 0, "bmoe: K must be a multiple of 64, got ", K);
  TORCH_CHECK(b.numel() == (int64_t)E * N * (K / 2), "bmoe: b has unexpected size");
  TORCH_CHECK(N % 128 == 0, "bmoe: N must be a multiple of 128, got ", N);
  if (rows == 0) return;
  const CUtensorMapSwizzle swz = C::RB == 64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B;
  if (!C::GA)
    for (int i = 0; i < kNumABoxes; ++i)
      make_tmap(&p.tmA[i], a.data_ptr(), (uint64_t)rows, (uint64_t)(K / 2), C::RB, 16 * (i + 1), swz);
  make_tmap(&p.tmB, b.data_ptr(), (uint64_t)E * N, (uint64_t)(K / 2), C::RB, C::MODE == 1 ? C::BN / 2 : C::BN, swz);
  p.a_sf = static_cast<const uint8_t*>(a_sf.data_ptr());
  p.b_sf = static_cast<const uint8_t*>(b_sf.data_ptr());
  p.alphas = alphas.data_ptr<float>();
  p.problem_sizes = problem_sizes.data_ptr<int32_t>();
  p.expert_offsets = expert_offsets.data_ptr<int32_t>();
  p.sf_offsets = sf_offsets.data_ptr<int32_t>();
  p.E = E;
  p.N = (int)N;
  p.K = (int)K;
  p.n_chunks = (int)((K + 255) / 256) * 4;
  p.n_real = (int)(K / 64);
  p.I = (int)(N / 2);
  const int dev = a.get_device();
  int grid = sm_count_dev(dev);
  if (sm_limit > 0 && sm_limit < grid) grid = sm_limit;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(dev).stream();
  static bool attr_set = false;
  if (!attr_set) {
    TORCH_CHECK(cudaFuncSetAttribute(bmoe_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) ==
                    cudaSuccess,
                "bmoe: smem attribute");
    attr_set = true;
  }
  bmoe_kernel<C><<<grid, C::THREADS, C::SMEM, stream>>>(p);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "bmoe: launch failed: ", cudaGetErrorString(err));
}

// out[rows of expert e] = alpha_e * A_e x B_e^T, bf16 (the moe_mm contract of vLLM's cutlass_fp4_moe_mm)
template <class C>
static void run_mm(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& a_sf,
                   const torch::Tensor& b_sf, const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
                   const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets, int sm_limit) {
  static_assert(C::MODE == 0, "run_mm is MODE 0");
  check_common(a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets);
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.stride(1) == 1 && out.size(0) == a.size(0),
              "bmoe: out must be bf16 [rows, N] with unit column stride");
  TORCH_CHECK((out.stride(0) % 2) == 0 && (reinterpret_cast<uintptr_t>(out.data_ptr()) & 3) == 0, "bmoe: out align");
  if (C::EPI == 1)
    TORCH_CHECK((out.stride(0) % 8) == 0 && (reinterpret_cast<uintptr_t>(out.data_ptr()) & 15) == 0,
                "bmoe: 16-byte stores need a 16-byte aligned out");
  Params p;
  memset(&p, 0, sizeof(p));
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ldo = (int)out.stride(0);
  TORCH_CHECK(out.size(1) % C::BN == 0, "bmoe: N must be a multiple of BN");
  launch<C>(p, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, out.size(1), sm_limit);
}

// GA variants: A = the unpermuted token rows `a_tok` [tokens, K/2] + their swizzled scales `a_tok_sf` (vLLM's
// scaled_fp4_experts_quant of the m token rows with one expert), gathered per expert-sorted row through a_map
template <class C>
static void run_mm_ga(torch::Tensor& out, const torch::Tensor& a_tok, const torch::Tensor& a_tok_sf,
                      const torch::Tensor& a_map, const torch::Tensor& b, const torch::Tensor& b_sf,
                      const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
                      const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets, int sm_limit) {
  static_assert(C::MODE == 0 && C::GA == 1, "run_mm_ga");
  check_common(a_tok, b, a_tok_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets);
  TORCH_CHECK(a_map.scalar_type() == at::kInt && a_map.is_contiguous() && a_map.is_cuda(), "bmoe: a_map int32");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.stride(1) == 1 &&
                  out.size(0) == a_map.size(0), "bmoe: out must be bf16 [rows, N]");
  Params p;
  memset(&p, 0, sizeof(p));
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ldo = (int)out.stride(0);
  p.a_src = static_cast<const uint8_t*>(a_tok.data_ptr());
  p.a_src_sf = static_cast<const uint8_t*>(a_tok_sf.data_ptr());
  p.a_map = a_map.data_ptr<int32_t>();
  launch<C>(p, a_tok, b, a_tok_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, out.size(1), sm_limit,
            a_map.size(0));
}

// fused w13: q_out [rows, I/2] (e2m1) and q_sf (swizzled ue4m3, expert e at row sf_offsets[e]) of
// gelu_tanh(gate) * up, gate/up = the bf16 w13 GEMM output halves (vLLM: gelu_tanh_and_mul + scaled_fp4_experts_quant)
template <class C>
static void run_w13_fused(torch::Tensor& q_out, torch::Tensor& q_sf, const torch::Tensor& a, const torch::Tensor& b,
                          const torch::Tensor& a_sf, const torch::Tensor& b_sf, const torch::Tensor& alphas,
                          const torch::Tensor& problem_sizes, const torch::Tensor& expert_offsets,
                          const torch::Tensor& sf_offsets, const torch::Tensor& q_gscale, int sm_limit) {
  static_assert(C::MODE == 1, "run_w13_fused is MODE 1");
  check_common(a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets);
  const int E = static_cast<int>(expert_offsets.size(0));
  const int64_t K = a.size(1) * 2;
  const int64_t N = b.numel() / ((int64_t)E * (K / 2));
  const int64_t I = N / 2;
  TORCH_CHECK(I % 64 == 0, "bmoe: fused w13 needs I % 64 == 0, got ", I);
  TORCH_CHECK(q_out.scalar_type() == at::kByte && q_out.is_contiguous() && q_out.size(0) == a.size(0) &&
                  q_out.size(1) == I / 2,
              "bmoe: q_out must be uint8 [rows, I/2]");
  TORCH_CHECK(q_sf.is_contiguous() && q_sf.element_size() * q_sf.numel() >= 0, "bmoe: q_sf");
  TORCH_CHECK(q_gscale.scalar_type() == at::kFloat && q_gscale.numel() >= E, "bmoe: q_gscale");
  Params p;
  memset(&p, 0, sizeof(p));
  p.q_out = static_cast<uint8_t*>(q_out.data_ptr());
  p.q_sf = static_cast<uint8_t*>(q_sf.data_ptr());
  p.q_gscale = q_gscale.data_ptr<float>();
  launch<C>(p, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, N, sm_limit);
}

// fused w13 with A gathered from the unpermuted token rows (GA)
template <class C>
static void run_w13_fused_ga(torch::Tensor& q_out, torch::Tensor& q_sf, const torch::Tensor& a_tok,
                             const torch::Tensor& a_tok_sf, const torch::Tensor& a_map, const torch::Tensor& b,
                             const torch::Tensor& b_sf, const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
                             const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets,
                             const torch::Tensor& q_gscale, int sm_limit) {
  static_assert(C::MODE == 1 && C::GA == 1, "run_w13_fused_ga");
  check_common(a_tok, b, a_tok_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets);
  TORCH_CHECK(a_map.scalar_type() == at::kInt && a_map.is_contiguous() && a_map.is_cuda(), "bmoe: a_map int32");
  const int E = static_cast<int>(expert_offsets.size(0));
  const int64_t K = a_tok.size(1) * 2;
  const int64_t N = b.numel() / ((int64_t)E * (K / 2));
  const int64_t I = N / 2;
  TORCH_CHECK(I % 64 == 0, "bmoe: fused w13 needs I % 64 == 0, got ", I);
  TORCH_CHECK(q_out.scalar_type() == at::kByte && q_out.is_contiguous() && q_out.size(0) == a_map.size(0) &&
                  q_out.size(1) == I / 2, "bmoe: q_out must be uint8 [rows, I/2]");
  TORCH_CHECK(q_gscale.scalar_type() == at::kFloat && q_gscale.numel() >= E, "bmoe: q_gscale");
  Params p;
  memset(&p, 0, sizeof(p));
  p.q_out = static_cast<uint8_t*>(q_out.data_ptr());
  p.q_sf = static_cast<uint8_t*>(q_sf.data_ptr());
  p.q_gscale = q_gscale.data_ptr<float>();
  p.a_src = static_cast<const uint8_t*>(a_tok.data_ptr());
  p.a_src_sf = static_cast<const uint8_t*>(a_tok_sf.data_ptr());
  p.a_map = a_map.data_ptr<int32_t>();
  launch<C>(p, a_tok, b, a_tok_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, N, sm_limit,
            a_map.size(0));
}

}  // namespace bmoe


// ===== bj/moe dispatcher (team ruj, Apache-2.0) =====
// variant 0-6: pjs's CUTLASS kernels (above, unchanged); variant >= 10: bmoe (bmoe_kernel.cuh), raster ignored.
// All bmoe variants: 1 producer warp (TMA) + consumer warps, persistent CTAs (sm_limit or all SMs), BK 128 unless
// noted, L2 evict_first for weights / evict_last for activations; S = dynamic unit scheduler, E16 = 16-byte stores.
//   10: BM128 BN128 5 stages 4x2          11: = 10 + S          12: = 10 + S + E16
//   13: BM64  BN128 6 stages 2x4          14: = 13 + S + E16    15: BM128 BN128 BK256 2 stages 4x2
//   16: BM128 BN128 5 stages 2x4          17: = 16 + S + E16
//   18: = 10 with L2 evict_normal for weights and activations   19: = 18 + S   20: = 13 + S   21: = 18 + S, A evict_normal only
#define BMOE_ARGS out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, (int)sm_limit
void moe_mm(torch::Tensor out, torch::Tensor a, torch::Tensor b, torch::Tensor a_sf, torch::Tensor b_sf,
            torch::Tensor alphas, torch::Tensor problem_sizes, torch::Tensor expert_offsets, torch::Tensor sf_offsets,
            int64_t variant, int64_t raster, int64_t sm_limit) {
  if (variant < 10) {
    moe_mm_cutlass(out, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, variant, raster, sm_limit);
    return;
  }
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "blink_moegemm: out must be bf16");
  using namespace bmoe;
  switch (variant) {
    case 10: run_mm<Cfg<128, 128, 128, 5, 4, 2>>(BMOE_ARGS); break;
    case 11: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 0, 1>>(BMOE_ARGS); break;
    case 12: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 1, 1>>(BMOE_ARGS); break;
    case 13: run_mm<Cfg<64, 128, 128, 6, 2, 4>>(BMOE_ARGS); break;
    case 14: run_mm<Cfg<64, 128, 128, 6, 2, 4, 0, 1, 2, 1, 1>>(BMOE_ARGS); break;
    case 15: run_mm<Cfg<128, 128, 256, 2, 4, 2>>(BMOE_ARGS); break;
    case 16: run_mm<Cfg<128, 128, 128, 5, 2, 4>>(BMOE_ARGS); break;
    case 17: run_mm<Cfg<128, 128, 128, 5, 2, 4, 0, 1, 2, 1, 1>>(BMOE_ARGS); break;
    case 18: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 0, 0>>(BMOE_ARGS); break;
    case 19: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 0, 0, 0, 1>>(BMOE_ARGS); break;
    case 20: run_mm<Cfg<64, 128, 128, 6, 2, 4, 0, 1, 2, 0, 1>>(BMOE_ARGS); break;
    case 21: run_mm<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 0, 0, 1>>(BMOE_ARGS); break;
    default: TORCH_CHECK(false, "blink_moegemm: unknown variant ", variant);
  }
}

// fused w13 GEMM + vLLM gelu_tanh_and_mul + vLLM scaled_fp4_experts_quant (bmoe MODE 1): 64 gate + 64 up rows
//   cfg 1: BM128 5 stages 4x2      cfg 2: BM64 5 stages 2x2      cfg 3: = 1 + dynamic unit scheduler
//   cfg 4: BM64 6 stages 2x4       cfg 5: = 4 + dynamic unit scheduler    cfg 6: = 3 with L2 evict_normal
#define BMOE_FARGS q_out, q_sf, a, b, a_sf, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, q_gscale, (int)sm_limit
void w13_fused(torch::Tensor q_out, torch::Tensor q_sf, torch::Tensor a, torch::Tensor b, torch::Tensor a_sf,
               torch::Tensor b_sf, torch::Tensor alphas, torch::Tensor problem_sizes, torch::Tensor expert_offsets,
               torch::Tensor sf_offsets, torch::Tensor q_gscale, int64_t cfg, int64_t sm_limit) {
  using namespace bmoe;
  switch (cfg) {
    case 1: run_w13_fused<Cfg<128, 128, 128, 5, 4, 2, 1>>(BMOE_FARGS); break;
    case 2: run_w13_fused<Cfg<64, 128, 128, 5, 2, 2, 1>>(BMOE_FARGS); break;
    case 3: run_w13_fused<Cfg<128, 128, 128, 5, 4, 2, 1, 1, 2, 0, 1>>(BMOE_FARGS); break;
    case 4: run_w13_fused<Cfg<64, 128, 128, 6, 2, 4, 1>>(BMOE_FARGS); break;
    case 5: run_w13_fused<Cfg<64, 128, 128, 6, 2, 4, 1, 1, 2, 0, 1>>(BMOE_FARGS); break;
    case 6: run_w13_fused<Cfg<128, 128, 128, 5, 4, 2, 1, 0, 0, 0, 1>>(BMOE_FARGS); break;
    default: TORCH_CHECK(false, "blink_moegemm: unknown fused cfg ", cfg);
  }
}

// A gathered from the unpermuted token rows through a_map (blink_moe.py BLINK_MOE_GA=1): variants 10-12, cfg 1/3
#define BMOE_GARGS out, a_tok, a_tok_sf, a_map, b, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, (int)sm_limit
void moe_mm_ga(torch::Tensor out, torch::Tensor a_tok, torch::Tensor a_tok_sf, torch::Tensor a_map, torch::Tensor b,
               torch::Tensor b_sf, torch::Tensor alphas, torch::Tensor problem_sizes, torch::Tensor expert_offsets,
               torch::Tensor sf_offsets, int64_t variant, int64_t sm_limit) {
  using namespace bmoe;
  switch (variant) {
    case 10: run_mm_ga<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 0, 0, 1>>(BMOE_GARGS); break;
    case 11: run_mm_ga<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 0, 1, 1>>(BMOE_GARGS); break;
    case 12: run_mm_ga<Cfg<128, 128, 128, 5, 4, 2, 0, 1, 2, 1, 1, 1>>(BMOE_GARGS); break;
    default: TORCH_CHECK(false, "blink_moegemm: no gather-A build of variant ", variant);
  }
}
#define BMOE_GFARGS q_out, q_sf, a_tok, a_tok_sf, a_map, b, b_sf, alphas, problem_sizes, expert_offsets, sf_offsets, q_gscale, (int)sm_limit
void w13_fused_ga(torch::Tensor q_out, torch::Tensor q_sf, torch::Tensor a_tok, torch::Tensor a_tok_sf,
                  torch::Tensor a_map, torch::Tensor b, torch::Tensor b_sf, torch::Tensor alphas,
                  torch::Tensor problem_sizes, torch::Tensor expert_offsets, torch::Tensor sf_offsets,
                  torch::Tensor q_gscale, int64_t cfg, int64_t sm_limit) {
  using namespace bmoe;
  switch (cfg) {
    case 1: run_w13_fused_ga<Cfg<128, 128, 128, 5, 4, 2, 1, 1, 2, 0, 0, 1>>(BMOE_GFARGS); break;
    case 3: run_w13_fused_ga<Cfg<128, 128, 128, 5, 4, 2, 1, 1, 2, 0, 1, 1>>(BMOE_GFARGS); break;
    default: TORCH_CHECK(false, "blink_moegemm: no gather-A build of fused cfg ", cfg);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_mm", &moe_mm, "NVFP4 grouped GEMM for MoE experts (SM120): CUTLASS variants 0-6, bmoe variants >= 10");
  m.def("w13_fused", &w13_fused, "bmoe: w13 NVFP4 grouped GEMM + gelu_tanh*up + NVFP4 quant in one kernel");
  m.def("moe_mm_ga", &moe_mm_ga, "bmoe GEMM, A gathered from the token rows through a_map");
  m.def("w13_fused_ga", &w13_fused_ga, "bmoe fused w13, A gathered from the token rows through a_map");
}
