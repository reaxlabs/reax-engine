# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 team ruj (Jeffan engine lever GQ, on pjs's E62 Jeffan engine)
"""Lever GQ: the MLP's GELU(tanh)-mul + per-token FP8 quantisation in ONE pass over the gate_up output.

The compiled graph of the J1 start state (vLLM 0.30.0 / torch 2.13 Inductor, custom_ops none) quantises the down
projection's input with the generated kernel
  triton_red_fused__to_copy_abs_clamp_cutlass_scaled_mm_div_gelu_max_mul_reciprocal_slice_unsqueeze_2
(source file cgpobg3yh3kxe3xpf6ph6naqwlf6ewwcojmg27pbmaikp3hifgka.py, pinned config XBLOCK 1 / R0_BLOCK 1024 /
8 warps): per token row of the [M, 43008] bf16 gate_up it computes act = gelu_tanh(gate) * up in fp32 (no bf16
rounding in between), the row's amax of |act| (loop 1), then again act, scale = max(amax * (1/448), 1/(448*512)),
q = fp8_e4m3(clamp(act * (1 / scale), -448, 448)) (loop 2). Loop 2 reads the row a second time.

This kernel computes the same values with the same Triton statements (copied from the generated kernel: the same
op order, constants, libdevice tanh, fp-fusion and ftz defaults, NaN-propagating max helpers of Inductor), but holds
the row's act values in registers, so the row is read once. Max is order-independent, so the amax (and with it the
scale and every q) is the same bits; checked on the GPU against the generated kernel with torch.equal on q and scale
(scripts/gq_test.py). Installed by jef_levers.py (JEF_GQ=1): CachingAutotuner.run of that one source file launches
this kernel instead (same arguments: in_ptr0, out_ptr2, out_ptr3, xnumel, r0_numel).
"""
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

N_HALF = 21504            # the intermediate size of Jeffan-31b (gate | up halves of gate_up)
GQ_FILE = "cgpobg3yh3kxe3xpf6ph6naqwlf6ewwcojmg27pbmaikp3hifgka.py"
GQ_NAME = "triton_red_fused__to_copy_abs_clamp_cutlass_scaled_mm_div_gelu_max_mul_reciprocal_slice_unsqueeze_2"


@triton.jit
def _act(tmp0, tmp16):
    # == the generated kernel's statements tmp1 .. tmp17 (loop 1) / tmp22 .. tmp38 (loop 2), verbatim
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1, 1], 0.5, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp1 * tmp1
    tmp5 = tmp4 * tmp1
    tmp6 = tl.full([1, 1], 0.044715, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 + tmp7
    tmp9 = tl.full([1, 1], 0.7978845608028654, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = tl.full([1, 1], 1.0, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp3 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 * tmp16
    return tmp17


@triton.jit
def _quant(tmp38, tmp20):
    # == the generated kernel's statements tmp39 .. tmp51 (loop 2), verbatim
    tmp33 = tl.full([1, 1], 1.0, tl.float32)
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp20.to(tl.float32)
    tmp41 = tl.full([1, 1], 0.002232142857142857, tl.float32)
    tmp42 = tmp40 * tmp41
    tmp43 = tl.full([1, 1], 4.359654017857143e-06, tl.float32)
    tmp44 = triton_helpers.maximum(tmp42, tmp43)
    tmp45 = (tmp33 / tmp44)
    tmp46 = tmp39 * tmp45
    tmp47 = tl.full([1, 1], -448.0, tl.float32)
    tmp48 = triton_helpers.maximum(tmp46, tmp47)
    tmp49 = tl.full([1, 1], 448.0, tl.float32)
    tmp50 = triton_helpers.minimum(tmp48, tmp49)
    tmp51 = tmp50.to(tl.float8e4nv)
    return tmp51


@triton.jit(do_not_specialize=["xnumel"])
def jef_gq_kernel(in_ptr0, out_ptr2, out_ptr3, xnumel):
    # one program per token row; the row's 21,504 act values = blocks of 16,384 + 4,096 + 1,024 (no masked lanes)
    x0 = tl.program_id(0).to(tl.int64)
    ra = tl.arange(0, 16384)[None, :]
    rb = 16384 + tl.arange(0, 4096)[None, :]
    rc = 20480 + tl.arange(0, 1024)[None, :]
    base = in_ptr0 + 43008 * x0
    ga = tl.load(base + ra).to(tl.float32)
    ua = tl.load(base + (21504 + ra)).to(tl.float32)
    gb = tl.load(base + rb).to(tl.float32)
    ub = tl.load(base + (21504 + rb)).to(tl.float32)
    gc = tl.load(base + rc).to(tl.float32)
    uc = tl.load(base + (21504 + rc)).to(tl.float32)
    acta = _act(ga, ua)
    actb = _act(gb, ub)
    actc = _act(gc, uc)
    # amax of |act| over the row (loop 1's tmp18 .. tmp20); max is order-independent, NaN-propagating like Inductor's
    ma = triton_helpers.max2(tl_math.abs(acta), 1)[:, None]
    mb = triton_helpers.max2(tl_math.abs(actb), 1)[:, None]
    mc = triton_helpers.max2(tl_math.abs(actc), 1)[:, None]
    tmp20 = triton_helpers.maximum(triton_helpers.maximum(ma, mb), mc)
    obase = out_ptr2 + 21504 * x0
    tl.store(obase + ra, _quant(acta, tmp20))
    tl.store(obase + rb, _quant(actb, tmp20))
    tl.store(obase + rc, _quant(actc, tmp20))
    # the scale (tmp52 .. tmp56)
    tmp52 = tmp20.to(tl.float32)
    tmp53 = tl.full([1, 1], 0.002232142857142857, tl.float32)
    tmp54 = tmp52 * tmp53
    tmp55 = tl.full([1, 1], 4.359654017857143e-06, tl.float32)
    tmp56 = triton_helpers.maximum(tmp54, tmp55)
    tl.store(out_ptr3 + x0 + tl.zeros([1, 1], tl.int64), tmp56)


NUM_WARPS = 32   # 1,024 threads per row (gq_test.py: 16 and 32 equal within noise, 32 a little faster at M = 1,024)


def gelu_quant(in_ptr0, out_ptr2, out_ptr3, xnumel, num_warps=None):
    """The replacement launch: same tensors / row count as the generated kernel's run()."""
    if xnumel > 0:
        jef_gq_kernel[(xnumel,)](in_ptr0, out_ptr2, out_ptr3, xnumel, num_warps=num_warps or NUM_WARPS,
                                 num_stages=1)
