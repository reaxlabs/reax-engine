# SPDX-License-Identifier: MIT
"""GPU check: fast_fp8_gemm returns the same bits as the finegrained-fp8 kernel. Run inside the runtime container."""
import torch
import fast_fp8_gemm as f
from autotune_guard import install
install()
from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel
def test_bits():
    mm=load_finegrained_fp8_kernel().matmul;torch.manual_seed(0)
    for N,K in [(21504,5376),(5376,21504),(8192,5376),(4096,5376),(5376,8192),(16384,5376),(2048,5376),(200,256)]:
        w=torch.randn(N,K,device='cuda')*.02;gn,gk=(N+127)//128,K//128
        s=torch.stack([torch.stack([w[i*128:(i+1)*128,j*128:(j+1)*128].abs().amax() for j in range(gk)]) for i in range(gn)])/448
        W=(w/s.repeat_interleave(128,0)[:N].repeat_interleave(128,1)).to(torch.float8_e4m3fn)
        for shape in [(1,1),(5,1),(1,17),(129,1),(1,1000),(4,300)]:
            A=(torch.randn(*shape,K,device='cuda')*torch.exp(torch.randn(*shape,K,device='cuda'))).to(torch.bfloat16)
            a=mm(A,W,s,[128,128],A.dtype);b=f.matmul(A,W,s,[128,128],A.dtype);torch.cuda.synchronize()
            assert torch.equal(a,b),(N,K,shape,int((a!=b).sum()))
    # zero rows, the 1e-12 floor
    A=torch.zeros(3,K,device='cuda',dtype=torch.bfloat16);assert torch.equal(mm(A,W,s,[128,128],A.dtype),f.matmul(A,W,s,[128,128],A.dtype))
if __name__=='__main__':
    test_bits();print('PASS')
