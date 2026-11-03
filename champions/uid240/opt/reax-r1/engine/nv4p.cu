// SPDX-License-Identifier: Apache-2.0
// nv4p.cu (kai lever N2, 10-28, KAI_PDL): programmatic dependent launch (PDL) twins of the NVFP4 CUTLASS GEMMs.
// nv4t.cu (kai lever B: rgh's nv4g.cu struct Nv4 verbatim, tiles 0-5 = nv4g variants 0-5, 6-10 more tiles) compiled with
// CUTLASS's own grid-dependency-control support (-DCUTLASS_ENABLE_GDC_FOR_SM100=1: for sm_120a CUTLASS's
// grid_dependency_control.h then emits griddepcontrol.wait before the kernel's first global load and
// griddepcontrol.launch_dependents after its mainloop) and run with GemmUniversalAdapter::run(stream, nullptr,
// launch_with_pdl = true), i.e. cudaLaunchKernelEx with cudaLaunchAttributeProgrammaticStreamSerialization.
// Review fix (10-28): CUTLASS's kernels wait only in their producer warps; the consumer warps build the epilogue, whose
// Sm90ScalarBroadcast reads *alpha_ptr, before any wait. alpha is written by the quant kernel right before the GEMM
// (nv4q._quant_dyn_kernel / actq._quant_deq_kernel), whose PDL twin releases its dependents at its top, so the GEMM could
// read a stale alpha. GemmP runs the same GemmUniversal behind WaitFirst: griddepcontrol.wait is the first statement of
// operator() in every thread (CUTLASS's own later waits are then already satisfied).
// EXACT: the same collective, tile, schedule and k order as nv4t / nv4g (no arithmetic changes; the GDC instructions only
// order the grid against its neighbours). nv4p_run_<t> is nv4t.cu's Nv4T::run with only the launch flag set (tests).
// C ABI (nv4p_dispatch.cu): nv4p_gemm(tile, raster, swizzle, ...) = nv4t_gemm's arguments and return codes.
#include "nv4t.cu"

template <class K>
struct WaitFirst : K {
  CUTLASS_DEVICE void operator()(typename K::Params const& params, char* smem_buf) {
    cutlass::arch::wait_on_dependent_grids();
    K::operator()(params, smem_buf);
  }
};
using GemmP = cutlass::gemm::device::GemmUniversalAdapter<WaitFirst<typename G::GemmKernel>>;

extern "C" int CAT(nv4p_run_, NV4T_TILE)(const void* A, const void* B, const void* SFA, const void* SFB,
                                         const float* alpha, void* D, int M, int N, int K, int ldd, void* wsp,
                                         size_t wsb, cudaStream_t st, int raster, int swz) {
  if (!G::sched_ok(raster, swz)) return 6;
  auto a = G::args(A, B, SFA, SFB, alpha, D, M, N, K, ldd, raster, swz);
  GemmP g;
  if (g.can_implement(a) != cutlass::Status::kSuccess) return 2;
  if (GemmP::get_workspace_size(a) > wsb) return 3;
  if (g.initialize(a, wsp, st) != cutlass::Status::kSuccess) return 4;
  if (g.run(st, nullptr, true) != cutlass::Status::kSuccess) return 5;
  return 0;
}
