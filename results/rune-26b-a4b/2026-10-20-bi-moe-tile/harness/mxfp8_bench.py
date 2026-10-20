"""Does SM120 + this torch run block-scaled MXFP8 (1x32, e8m0) GEMMs, and how fast vs per-token/per-channel FP8 and bf16?"""
import torch, time, json
print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())
def t(f, n=50):
    for _ in range(5): f()
    torch.cuda.synchronize(); a = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - a) / n * 1e3
out = {}
for (M, K, N) in [(8192, 2560, 9728), (8192, 5120, 17408), (8192, 17408, 5120), (16384, 5120, 17408)]:
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    r = {"bf16": t(lambda: a @ b.t())}
    sa = a.abs().amax(1, keepdim=True).float() / 448; sb = b.abs().amax(1, keepdim=True).float() / 448
    a8 = (a / sa).to(torch.float8_e4m3fn); b8 = (b / sb).to(torch.float8_e4m3fn)
    r["fp8_rowwise"] = t(lambda: torch._scaled_mm(a8, b8.t(), scale_a=sa, scale_b=sb.view(1, -1), out_dtype=torch.bfloat16))
    try:
        from torch.nn.functional import scaled_mm, ScalingType, SwizzleType
        def to_blocked(m):
            r, c = m.shape; nr, nc = -(-r // 128), -(-c // 4)
            pad = torch.zeros(nr * 128, nc * 4, dtype=m.dtype, device=m.device); pad[:r, :c] = m
            b = pad.view(nr, 128, nc, 4).permute(0, 2, 1, 3)
            return b.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()
        def mx(x):
            xb = x.float().view(x.shape[0], -1, 32); am = xb.abs().amax(-1).clamp(min=1e-30)
            e = torch.ceil(torch.log2(am / 448)).clamp(-127, 127)
            q = (xb / torch.exp2(e)[..., None]).view(x.shape).to(torch.float8_e4m3fn)
            return q, (e + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
        qa, ea = mx(a); qb, eb = mx(b)
        ea_s, eb_s = to_blocked(ea), to_blocked(eb)
        f = lambda: scaled_mm(qa, qb.t(), ea_s, ScalingType.BlockWise1x32, eb_s, ScalingType.BlockWise1x32,
                              swizzle_a=SwizzleType.SWIZZLE_32_4_4, swizzle_b=SwizzleType.SWIZZLE_32_4_4, output_dtype=torch.bfloat16)
        y = f(); ref = (a.float() @ b.float().t())
        r["mxfp8"] = t(f); r["mxfp8_relerr"] = float((y.float() - ref).norm() / ref.norm())
        y2 = torch._scaled_mm(a8, b8.t(), scale_a=sa, scale_b=sb.view(1, -1), out_dtype=torch.bfloat16)
        r["fp8_rowwise_relerr"] = float((y2.float() - ref).norm() / ref.norm())
    except Exception as e:  # noqa
        r["mxfp8_error"] = repr(e)[:300]
    out[f"{M}x{K}x{N}"] = r; print(M, K, N, json.dumps(r), flush=True)
json.dump(out, open("mxfp8_bench.json", "w"), indent=1)
