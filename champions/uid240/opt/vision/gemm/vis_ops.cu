// SPDX-License-Identifier: Apache-2.0
// vis_ops: exact fused kernels for the Gemma4 VISION tower of the Blink vision engine (V3 levers E3 / E4).
// Every kernel reproduces, bit for bit, what the eager batch-invariant reference (vLLM 0.30.0 + HF transformers 5.17
// Gemma4 vision modules, VLLM_BATCH_INVARIANT=1) computes; only launches and memory round trips go away.
//
// E3 gelu_up: gate/up projections + GELU-tanh x up in ONE kernel. NVIDIA CUTLASS 2.x dual GEMM (examples/45_dual_gemm,
//    BSD-3-Clause, headers in vis_dual/) on SM80-class tensor ops: mma.sync m16n8k16 bf16 -> fp32, no split-k, no
//    sliced-k (warp K = threadblock K), so every gate and up element is the same increasing-k chain of 16-wide MMA
//    steps as our CUTLASS GEMM / vLLM's batch-invariant Triton matmul. Epilogue: gate and up are rounded to bf16 (the
//    tensors eager stores), then ATen's tanh-GELU on the bf16 gate in float opmath (aten/src/ATen/native/cuda/
//    ActivationGeluKernel.cu, GeluType::Tanh; same expression, same float constants), rounded to bf16 (the GELU output
//    tensor), times the bf16 up in float, rounded to bf16 (ATen's bf16 mul).
// E4 norm chains: HF Gemma4RMSNorm = x.float().pow(2).mean(-1) [vLLM's batch-invariant Triton mean_kernel] + eps,
//    pow(-0.5) [rsqrt], x * r, * w.float(), .type_as(x). The row sum replicates mean_kernel's compiled reduction tree
//    (decoded from its TTGIR/PTX, Triton 3.7.1, 4 warps, BLOCK 1024):
//    * N = 1152 (blocked layout sizePerThread 4): thread t of 128 holds, per 1024-wide chunk, elements 4t..4t+3 and
//      512+4t..512+4t+3 and sums them in that order; butterfly over the warp (xor 16, 8, 4, 2, 1; a = a + partner);
//      lane 0 of each warp -> smem; threads 0..3 butterfly (xor 2, 1): (w0 + w2) + (w1 + w3); chunks summed from 0.0.
//    * N = 72 (head norms; sizePerThread 1): thread t holds element t; per warp butterfly; (w0 + w2) + (w1 + w3).
//    Masked lanes load +0.0; every value summed is a square (>= +0), so a +0.0 term never changes a partial sum: the
//    kernels skip the all-zero partials and keep the association order of every nonzero one.
//    mean = div.full.f32(sum, N) (the Triton kernel's division), rsqrt.approx.f32 (ATen pow(x, -0.5) / tl.math.rsqrt),
//    then two round-to-nearest float multiplies and the bf16 rounding; residual adds are bf16 + bf16 in float.
//    Chains: (A) input norm; (B) post-attention norm + residual + pre-feed-forward norm; (C) post-feed-forward norm +
//    residual (+ the next layer's input norm). qkv_norm_rope: q/k/v head norms (q, k with weights; v without) + the
//    2-D axial RoPE of q and k (HF apply_multidimensional_rope: two 36-channel parts, rotate_half per part, bf16
//    rounding after each product and after the sum), reading q/k/v rows with any row stride (e.g. the fused QKV GEMM
//    output) and writing contiguous [rows, 16, 72] tensors (the stock layout attention expects).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/array.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include "vis_dual/device/dual_gemm.h"

using bf16 = __nv_bfloat16;

// ------------------------------------------------------------------------------------------------ ATen tanh-GELU
__device__ __forceinline__ float gelu_tanh_aten(float x) {
  // = GeluCUDAKernelImpl, GeluType::Tanh, opmath_t = float
  constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * float(0.5);
  constexpr float kKappa = 0.044715;
  auto x_cube = x * x * x;
  auto inner = kBeta * (x + kKappa * x_cube);
  return float(0.5) * x * (float(1) + tanhf(inner));
}

__device__ __forceinline__ float bf16_round(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// CUTLASS dual-GEMM output op 2: D2 = bf16( float(bf16(gelu(gate_bf16))) * float(up_bf16) )
template <int Count>
struct GeluTanhMul {
  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = cutlass::bfloat16_t;
  using ElementCompute = float;
  static int const kCount = Count;
  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  struct Params {};
  CUTLASS_HOST_DEVICE GeluTanhMul(Params const&) {}
  CUTLASS_HOST_DEVICE bool is_source_needed() const { return false; }
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}
  CUTLASS_DEVICE FragmentOutput operator()(FragmentAccumulator const& g, FragmentAccumulator const& u) const {
    FragmentOutput out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      float y = bf16_round(gelu_tanh_aten(float(g[i])));
      out[i] = cutlass::bfloat16_t::bitcast(__bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(y, float(u[i])))));
    }
    return out;
  }
};

template <int TBM, int TBN, int TBK, int WM, int WN, int STAGES>
struct DualCfg {
  using Op01 = cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, float, float,
                                                             cutlass::epilogue::thread::ScaleType::Nothing>;
  using Gemm = cutlass::gemm::device::DualGemm<
      cutlass::bfloat16_t, cutlass::layout::RowMajor,
      cutlass::bfloat16_t, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor,
      cutlass::bfloat16_t, cutlass::layout::RowMajor, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TBM, TBN, TBK>, cutlass::gemm::GemmShape<WM, WN, TBK>,
      cutlass::gemm::GemmShape<16, 8, 16>, Op01, Op01, GeluTanhMul<8>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, STAGES,
      /*StoreD0=*/false, /*StoreD1=*/false, /*SplitKSerial=*/false>;
};

template <typename G>
static void run_dual(const torch::Tensor& a, const torch::Tensor& wg, const torch::Tensor& wu, torch::Tensor& d) {
  int M = a.size(0), K = a.size(1), N = wg.size(0);
  using RefA = cutlass::TensorRef<cutlass::bfloat16_t const, cutlass::layout::RowMajor>;
  using RefB = cutlass::TensorRef<cutlass::bfloat16_t const, cutlass::layout::ColumnMajor>;
  using RefC = cutlass::TensorRef<cutlass::bfloat16_t const, cutlass::layout::RowMajor>;
  using RefD = cutlass::TensorRef<cutlass::bfloat16_t, cutlass::layout::RowMajor>;
  RefC c_null{};
  RefD d_null{};
  typename G::Arguments args(
      cutlass::gemm::DualGemmMode::kGemm, {M, N, K},
      RefA(reinterpret_cast<cutlass::bfloat16_t const*>(a.data_ptr()), a.stride(0)),
      RefB(reinterpret_cast<cutlass::bfloat16_t const*>(wg.data_ptr()), wg.stride(0)), c_null, d_null,
      RefB(reinterpret_cast<cutlass::bfloat16_t const*>(wu.data_ptr()), wu.stride(0)), c_null, d_null,
      RefD(reinterpret_cast<cutlass::bfloat16_t*>(d.data_ptr()), d.stride(0)));
  G gemm;
  auto st = gemm.can_implement(args);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "vis_ops.gelu_up: config cannot implement this problem");
  st = gemm.initialize(args, nullptr, c10::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(st == cutlass::Status::kSuccess, "vis_ops.gelu_up: initialize failed");
  st = gemm(c10::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(st == cutlass::Status::kSuccess, "vis_ops.gelu_up: launch failed");
}

// h = gelu_tanh(a @ wg.T) * (a @ wu.T), bf16 [M, N]; a [M, K] row-major, wg / wu [N, K] row-major (= nn.Linear weight)
torch::Tensor gelu_up(torch::Tensor a, torch::Tensor wg, torch::Tensor wu, int64_t cfg) {
  TORCH_CHECK(a.is_cuda() && wg.is_cuda() && wu.is_cuda());
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && wg.scalar_type() == at::kBFloat16 && wu.scalar_type() == at::kBFloat16);
  TORCH_CHECK(a.dim() == 2 && wg.dim() == 2 && wu.dim() == 2 && a.size(1) == wg.size(1) && wg.sizes() == wu.sizes());
  TORCH_CHECK(a.stride(1) == 1 && wg.stride(1) == 1 && wu.stride(1) == 1);
  auto d = torch::empty({a.size(0), wg.size(0)}, a.options());
  switch (cfg) {
    case 0: run_dual<DualCfg<128, 64, 32, 64, 32, 4>::Gemm>(a, wg, wu, d); break;
    case 1: run_dual<DualCfg<64, 128, 32, 32, 64, 4>::Gemm>(a, wg, wu, d); break;
    case 2: run_dual<DualCfg<128, 64, 64, 64, 32, 3>::Gemm>(a, wg, wu, d); break;
    case 3: run_dual<DualCfg<64, 64, 64, 32, 32, 4>::Gemm>(a, wg, wu, d); break;
    case 4: run_dual<DualCfg<128, 128, 32, 64, 32, 3>::Gemm>(a, wg, wu, d); break;
    case 5: run_dual<DualCfg<256, 64, 32, 64, 32, 3>::Gemm>(a, wg, wu, d); break;
    default: TORCH_CHECK(false, "vis_ops.gelu_up: unknown config");
  }
  return d;
}

// test entry: elementwise GeluTanhMul on bf16 tensors (gate, up) -> bf16; the exhaustive GELU check uses up = 1
__global__ void gelu_mul_kernel(const bf16* g, const bf16* u, bf16* o, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) {
    float y = bf16_round(gelu_tanh_aten(__bfloat162float(g[i])));
    o[i] = __float2bfloat16_rn(__fmul_rn(y, __bfloat162float(u[i])));
  }
}

torch::Tensor gelu_mul(torch::Tensor g, torch::Tensor u) {
  TORCH_CHECK(g.is_cuda() && g.scalar_type() == at::kBFloat16 && u.scalar_type() == at::kBFloat16);
  TORCH_CHECK(g.is_contiguous() && u.is_contiguous() && g.numel() == u.numel());
  auto o = torch::empty_like(g);
  int64_t n = g.numel();
  if (n) {
    gelu_mul_kernel<<<(unsigned)((n + 255) / 256), 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const bf16*>(g.data_ptr()), reinterpret_cast<const bf16*>(u.data_ptr()),
        reinterpret_cast<bf16*>(o.data_ptr()), n);
  }
  return o;
}

// ------------------------------------------------------------------------------------------------ E4 norm chains
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float rsqrt_approx(float a) {
  float r;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float bfly32(float a) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, o));
  return a;
}

struct V4 { float v[4]; };
__device__ __forceinline__ V4 ld4(const bf16* p) {     // 4 bf16 (8 bytes, aligned) -> float
  uint2 raw = *reinterpret_cast<const uint2*>(p);
  const bf16* h = reinterpret_cast<const bf16*>(&raw);
  V4 r;
#pragma unroll
  for (int i = 0; i < 4; ++i) r.v[i] = __bfloat162float(h[i]);
  return r;
}
__device__ __forceinline__ void st4(bf16* p, const float* v) {   // v already bf16-exact floats
  uint2 raw;
  bf16* h = reinterpret_cast<bf16*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __float2bfloat16_rn(v[i]);
  *reinterpret_cast<uint2*>(p) = raw;
}

// mean_kernel's row sum of squares for N = 1152 (see the header); q[0..7]: chunk 0 values of this thread in order,
// q[8..11]: chunk 1 values (threads 0..31 only). red: 5 floats of shared memory for this row.
__device__ __forceinline__ float rowsum1152(const float* q, float* red) {
  const int t = threadIdx.x & 127, lane = t & 31, warp = t >> 5;
  float p = q[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) p = __fadd_rn(p, q[i]);
  p = bfly32(p);
  if (lane == 0) red[warp] = p;
  // chunk 1: only warp 0 holds nonzero values; its butterfly is the chunk's sum ((W0 + 0) + (0 + 0) = W0)
  float c1 = 0.f;
  if (warp == 0) {
    float s = __fadd_rn(__fadd_rn(__fadd_rn(q[8], q[9]), q[10]), q[11]);
    c1 = bfly32(s);
  }
  __syncthreads();
  if (t < 4) {
    float w = red[t];
    w = __fadd_rn(w, __shfl_xor_sync(0x0000000fu, w, 2));
    w = __fadd_rn(w, __shfl_xor_sync(0x0000000fu, w, 1));
    if (t == 0) red[4] = __fadd_rn(w, c1);        // (0.0 + chunk0) + chunk1; 0.0 + x = x for x >= +0
  }
  __syncthreads();
  float total = red[4];
  __syncthreads();                                 // red is reused by a second norm of the same row
  return total;
}

// One 128-thread group per row of 1152. MODE 0: y = norm(x; w1). MODE 1: n1 = norm(x; w1), h = res + n1 (stored),
// y = norm(h; w2) (stored if w2). Rows of x / res / h / y have their own strides (elements, multiples of 4).
template <int MODE>
__global__ void __launch_bounds__(128) norm_chain_kernel(
    const bf16* __restrict__ x, int64_t sx, const bf16* __restrict__ res, int64_t sr, const bf16* __restrict__ w1,
    const bf16* __restrict__ w2, bf16* __restrict__ h, int64_t sh, bf16* __restrict__ y, int64_t sy, float eps1,
    float eps2, int rows) {
  __shared__ float red[5];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int t = threadIdx.x;
  const int c0 = 4 * t, c1 = 512 + 4 * t, c2 = 1024 + 4 * t;
  const bool tail = t < 32;
  float xv[12], sq[12];
  {
    V4 a = ld4(x + row * sx + c0), b = ld4(x + row * sx + c1);
#pragma unroll
    for (int i = 0; i < 4; ++i) { xv[i] = a.v[i]; xv[4 + i] = b.v[i]; }
    if (tail) {
      V4 c = ld4(x + row * sx + c2);
#pragma unroll
      for (int i = 0; i < 4; ++i) xv[8 + i] = c.v[i];
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) xv[8 + i] = 0.f;
    }
  }
#pragma unroll
  for (int i = 0; i < 12; ++i) sq[i] = __fmul_rn(xv[i], xv[i]);
  float r1 = rsqrt_approx(__fadd_rn(div_full(rowsum1152(sq, red), 1152.f), eps1));
  const int cols[3] = {c0, c1, c2};
  float nv[12];
#pragma unroll
  for (int g = 0; g < 3; ++g) {
    if (g == 2 && !tail) break;
    V4 w = ld4(w1 + cols[g]);
#pragma unroll
    for (int i = 0; i < 4; ++i) nv[4 * g + i] = bf16_round(__fmul_rn(__fmul_rn(xv[4 * g + i], r1), w.v[i]));
  }
  if (MODE == 0) {
#pragma unroll
    for (int g = 0; g < 3; ++g) {
      if (g == 2 && !tail) break;
      st4(y + row * sy + cols[g], nv + 4 * g);
    }
    return;
  }
  float hv[12];
#pragma unroll
  for (int g = 0; g < 3; ++g) {
    if (g == 2 && !tail) {
#pragma unroll
      for (int i = 0; i < 4; ++i) hv[8 + i] = 0.f;
      break;
    }
    V4 r = ld4(res + row * sr + cols[g]);
#pragma unroll
    for (int i = 0; i < 4; ++i) hv[4 * g + i] = bf16_round(__fadd_rn(r.v[i], nv[4 * g + i]));
    st4(h + row * sh + cols[g], hv + 4 * g);
  }
  if (w2 == nullptr) return;
#pragma unroll
  for (int i = 0; i < 12; ++i) sq[i] = __fmul_rn(hv[i], hv[i]);
  float r2 = rsqrt_approx(__fadd_rn(div_full(rowsum1152(sq, red), 1152.f), eps2));
#pragma unroll
  for (int g = 0; g < 3; ++g) {
    if (g == 2 && !tail) break;
    V4 w = ld4(w2 + cols[g]);
    float o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = __fmul_rn(__fmul_rn(hv[4 * g + i], r2), w.v[i]);
    st4(y + row * sy + cols[g], o);
  }
}

static void check_rows(const torch::Tensor& t, int64_t rows, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.dim() == 2 && t.size(0) == rows &&
                  t.size(1) == 1152 && t.stride(1) == 1 && t.stride(0) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(t.data_ptr()) % 8 == 0,
              "vis_ops: bad ", name);
}

// (A) y = rmsnorm(x; w), rows of 1152
torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, double eps) {
  int64_t rows = x.size(0);
  check_rows(x, rows, "x");
  TORCH_CHECK(w.is_contiguous() && w.numel() == 1152 && w.scalar_type() == at::kBFloat16);
  auto y = torch::empty({rows, 1152}, x.options());
  if (rows)
    norm_chain_kernel<0><<<(unsigned)rows, 128, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const bf16*>(x.data_ptr()), x.stride(0), nullptr, 0,
        reinterpret_cast<const bf16*>(w.data_ptr()), nullptr, nullptr, 0, reinterpret_cast<bf16*>(y.data_ptr()),
        1152, (float)eps, 0.f, (int)rows);
  return y;
}

// (B)/(C) h = res + rmsnorm(x; w1); y = rmsnorm(h; w2) (y undefined/empty when w2 is None)
std::vector<torch::Tensor> norm_res_norm(torch::Tensor x, torch::Tensor res, torch::Tensor w1,
                                         c10::optional<torch::Tensor> w2, double eps1, double eps2) {
  int64_t rows = x.size(0);
  check_rows(x, rows, "x");
  check_rows(res, rows, "res");
  TORCH_CHECK(w1.is_contiguous() && w1.numel() == 1152 && w1.scalar_type() == at::kBFloat16);
  bool has2 = w2.has_value() && w2->defined();
  if (has2) TORCH_CHECK(w2->is_contiguous() && w2->numel() == 1152 && w2->scalar_type() == at::kBFloat16);
  auto h = torch::empty({rows, 1152}, x.options());
  auto y = has2 ? torch::empty({rows, 1152}, x.options()) : torch::empty({0, 1152}, x.options());
  if (rows)
    norm_chain_kernel<1><<<(unsigned)rows, 128, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const bf16*>(x.data_ptr()), x.stride(0), reinterpret_cast<const bf16*>(res.data_ptr()),
        res.stride(0), reinterpret_cast<const bf16*>(w1.data_ptr()),
        has2 ? reinterpret_cast<const bf16*>(w2->data_ptr()) : nullptr, reinterpret_cast<bf16*>(h.data_ptr()), 1152,
        has2 ? reinterpret_cast<bf16*>(y.data_ptr()) : nullptr, 1152, (float)eps1, (float)eps2, (int)rows);
  return {h, y};
}

// ------------------------------------------------------------------------------------------------ q/k/v head norms + RoPE
// one warp per (row, head, which): lane L holds channels L, 32 + L and (L < 8) 64 + L of the 72-wide head.
__global__ void __launch_bounds__(256) qkv_norm_rope_kernel(
    const bf16* __restrict__ q, int64_t sq_, const bf16* __restrict__ k, int64_t sk_, const bf16* __restrict__ v,
    int64_t sv_, const bf16* __restrict__ wq, const bf16* __restrict__ wk, const bf16* __restrict__ cs,
    const bf16* __restrict__ sn, int64_t scs, bf16* __restrict__ qo, bf16* __restrict__ ko, bf16* __restrict__ vo,
    float eps, int rows) {
  __shared__ float ybuf[8][72];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int64_t item = (int64_t)blockIdx.x * 8 + warp;     // item = (row * 3 + which) * 16 + head
  if (item >= (int64_t)rows * 48) return;
  const int head = (int)(item % 16);
  const int which = (int)((item / 16) % 3);
  const int64_t row = item / 48;
  const bf16* src = which == 0 ? q + row * sq_ : which == 1 ? k + row * sk_ : v + row * sv_;
  src += head * 72;
  float x0 = __bfloat162float(src[lane]);
  float x1 = __bfloat162float(src[32 + lane]);
  float x2 = lane < 8 ? __bfloat162float(src[64 + lane]) : 0.f;
  // mean_kernel N = 72 tree: warp sums W0 (ch 0..31), W1 (32..63), W2 (64..71), W3 = 0; (W0 + W2) + (W1 + W3)
  float w0 = bfly32(__fmul_rn(x0, x0));
  float w1 = bfly32(__fmul_rn(x1, x1));
  float w2 = bfly32(__fmul_rn(x2, x2));
  float total = __fadd_rn(__fadd_rn(w0, w2), w1);
  float r = rsqrt_approx(__fadd_rn(div_full(total, 72.f), eps));
  float y0, y1, y2;
  if (which == 2) {                                         // v_norm: no scale
    y0 = __fmul_rn(x0, r);
    y1 = __fmul_rn(x1, r);
    y2 = __fmul_rn(x2, r);
    bf16* dst = vo + row * 1152 + head * 72;
    dst[lane] = __float2bfloat16_rn(y0);
    dst[32 + lane] = __float2bfloat16_rn(y1);
    if (lane < 8) dst[64 + lane] = __float2bfloat16_rn(y2);
    return;
  }
  const bf16* w = which == 0 ? wq : wk;
  y0 = bf16_round(__fmul_rn(__fmul_rn(x0, r), __bfloat162float(w[lane])));
  y1 = bf16_round(__fmul_rn(__fmul_rn(x1, r), __bfloat162float(w[32 + lane])));
  y2 = lane < 8 ? bf16_round(__fmul_rn(__fmul_rn(x2, r), __bfloat162float(w[64 + lane]))) : 0.f;
  float* yb = ybuf[warp];
  yb[lane] = y0;
  yb[32 + lane] = y1;
  if (lane < 8) yb[64 + lane] = y2;
  __syncwarp();
  const bf16* c = cs + row * scs;
  const bf16* s = sn + row * scs;
  bf16* dst = (which == 0 ? qo : ko) + row * 1152 + head * 72;
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    int ch = lane + 32 * j;
    if (ch >= 72) break;
    int jj = ch % 36;
    float yv = yb[ch];
    float xr = jj < 18 ? -yb[ch + 18] : yb[ch - 18];
    float a = bf16_round(__fmul_rn(yv, __bfloat162float(c[ch])));
    float b = bf16_round(__fmul_rn(xr, __bfloat162float(s[ch])));
    dst[ch] = __float2bfloat16_rn(__fadd_rn(a, b));
  }
}

// q, k, v: [rows, 1152] bf16 views (row strides free, unit column stride); wq, wk: [72]; cos, sin: [rows, 72] with the
// same row stride. Returns contiguous [rows, 1152] q, k, v (= [rows, 16, 72]).
std::vector<torch::Tensor> qkv_norm_rope(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor wq,
                                         torch::Tensor wk, torch::Tensor cos, torch::Tensor sin, double eps) {
  int64_t rows = q.size(0);
  for (auto* t : {&q, &k, &v})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16 && t->dim() == 2 && t->size(0) == rows &&
                    t->size(1) == 1152 && t->stride(1) == 1,
                "vis_ops.qkv_norm_rope: q/k/v must be [rows, 1152] bf16 with unit column stride");
  TORCH_CHECK(wq.is_contiguous() && wk.is_contiguous() && wq.numel() == 72 && wk.numel() == 72);
  TORCH_CHECK(cos.dim() == 2 && cos.size(0) == rows && cos.size(1) == 72 && cos.stride(1) == 1 &&
              sin.sizes() == cos.sizes() && sin.strides() == cos.strides() && cos.scalar_type() == at::kBFloat16 &&
              sin.scalar_type() == at::kBFloat16);
  auto qo = torch::empty({rows, 1152}, q.options());
  auto ko = torch::empty({rows, 1152}, q.options());
  auto vo = torch::empty({rows, 1152}, q.options());
  int64_t items = rows * 48;
  if (items)
    qkv_norm_rope_kernel<<<(unsigned)((items + 7) / 8), 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const bf16*>(q.data_ptr()), q.stride(0), reinterpret_cast<const bf16*>(k.data_ptr()),
        k.stride(0), reinterpret_cast<const bf16*>(v.data_ptr()), v.stride(0),
        reinterpret_cast<const bf16*>(wq.data_ptr()), reinterpret_cast<const bf16*>(wk.data_ptr()),
        reinterpret_cast<const bf16*>(cos.data_ptr()), reinterpret_cast<const bf16*>(sin.data_ptr()), cos.stride(0),
        reinterpret_cast<bf16*>(qo.data_ptr()), reinterpret_cast<bf16*>(ko.data_ptr()),
        reinterpret_cast<bf16*>(vo.data_ptr()), (float)eps, (int)rows);
  return {qo, ko, vo};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gelu_up", &gelu_up, "E3: gelu_tanh(a @ wg.T) * (a @ wu.T) (CUTLASS dual GEMM, exact epilogue)");
  m.def("gelu_mul", &gelu_mul, "E3 test: elementwise epilogue gelu_tanh(g) * u on bf16");
  m.def("rms_norm", &rms_norm, "E4 (A): rmsnorm rows of 1152");
  m.def("norm_res_norm", &norm_res_norm, "E4 (B)/(C): h = res + rmsnorm(x; w1), y = rmsnorm(h; w2)");
  m.def("qkv_norm_rope", &qkv_norm_rope, "E4: q/k/v head norms + 2-D RoPE of q, k");
}
