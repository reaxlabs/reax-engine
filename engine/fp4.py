"""NVFP4 (e2m1 values, 16-element blocks, e4m3 block scales, fp32 per-tensor global scale) helpers for sm_120.

quantize_nvfp4(x) -> (packed uint8 viewed as float4_e2m1fn_x2 [M, K/2], swizzled e4m3 block scales, global scale)
mm_nvfp4(a, b) computes a @ b^T in bf16 via torch._scaled_mm (CUTLASS block-scaled FP4 MMA).
The quantizer is plain torch (torch.compile-able); values round-to-nearest-even onto the e2m1 grid.
"""
from __future__ import annotations

import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# midpoints between grid values; ties to even code index
MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def to_blocked(m: torch.Tensor) -> torch.Tensor:
    """Swizzle a [rows, cols] scale matrix into the 128x4 tiled layout cuBLAS/CUTLASS expect (SWIZZLE_32_4_4)."""
    rows, cols = m.shape
    rb, cb = -(-rows // 128), -(-cols // 4)
    pr, pc = rb * 128 - rows, cb * 4 - cols
    if pr or pc:
        m = torch.nn.functional.pad(m, (0, pc, 0, pr))
    blocks = m.view(rb, 128, cb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def f32_to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """x already scaled into [-6, 6]; returns uint8 codes (sign<<3 | idx)."""
    mid = MID.to(x.device)
    a = x.abs()
    idx = torch.bucketize(a, mid, right=False)  # a == mid goes left (to lower code)
    # ties to even: values exactly on a midpoint go to the even index
    on_mid = (idx < 7) & (a == mid[idx.clamp(max=6)])
    idx = torch.where(on_mid & (idx % 2 == 1), idx + 1, idx)
    sign = (x < 0).to(torch.uint8) << 3
    return sign | idx.to(torch.uint8)


def quantize_nvfp4(x: torch.Tensor, global_amax: torch.Tensor | None = None):
    M, K = x.shape
    assert K % 16 == 0
    xf = x.float()
    amax = xf.abs().amax() if global_amax is None else global_amax
    g = (amax / (448.0 * 6.0)).clamp(min=1e-12)  # global scale
    blk = xf.view(M, K // 16, 16)
    bamax = blk.abs().amax(-1)
    s = (bamax / (6.0 * g)).clamp(max=448.0).to(torch.float8_e4m3fn)  # block scale (e4m3)
    sf = s.float() * g
    q = blk / sf.clamp(min=1e-30)[..., None]
    codes = f32_to_e2m1_codes(q.clamp(-6, 6)).view(M, K)
    packed = (codes[:, 1::2] << 4) | codes[:, ::2]
    return packed.view(torch.float4_e2m1fn_x2), to_blocked(s), g


def mm_nvfp4(aq, a_s, a_g, bq, b_s, b_g, out_dtype=torch.bfloat16):
    """aq [M, K/2] fp4x2, bq [N, K/2] fp4x2 -> [M, N]."""
    y = torch._scaled_mm(aq, bq.t(), scale_a=a_s, scale_b=b_s, out_dtype=torch.float32 if out_dtype is None else out_dtype)
    return y * (a_g * b_g).to(y.dtype)


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    for M, N, K in [(8192, 34816, 5120), (8192, 5120, 17408)]:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K ** 0.5
        ref = torch.nn.functional.linear(x, w).float()
        wq, ws, wg = quantize_nvfp4(w)
        xq, xs, xg = quantize_nvfp4(x)
        try:
            y = mm_nvfp4(xq, xs, xg, wq, ws, wg).float()
            err = ((y - ref).norm() / ref.norm()).item()
            f = lambda: torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)  # noqa: E731
            f(); torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(20):
                f()
            torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 20
            qf = torch.compile(quantize_nvfp4, dynamic=True)
            qf(x); torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(20):
                qf(x)
            torch.cuda.synchronize(); dq = (time.perf_counter() - t) / 20
            print(f"M{M} N{N} K{K}: nvfp4 {2*M*N*K/dt/1e12:.0f} TFLOPS, relerr {err:.4f}, act-quant {dq*1e3:.2f} ms vs mm {dt*1e3:.2f} ms")
        except Exception as e:  # noqa: BLE001
            print("nvfp4 failed:", type(e).__name__, str(e)[:400])
