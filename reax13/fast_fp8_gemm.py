# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The HuggingFace Inc. team. All rights reserved (upstream finegrained-fp8, matmul.py).
# Modifications Copyright 2026 The REAX Engine authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 (full text: LICENSE and third_party/finegrained-fp8/LICENSE).
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# STATEMENT OF CHANGES (Apache-2.0 section 4(b)). This file is derived from w8a8_block_dynamic_fp8_matmul in
# third_party/finegrained-fp8/matmul.py (kernels-community/finegrained-fp8, pin in THIRD_PARTY_NOTICES.md). Changes: the activation
# quantization is a separate Triton kernel run once per call instead of inline for every N tile; the GEMM kernel is rewritten to read the
# quantized activations and their scales; tile sizes and launch parameters are fixed here; install() routes transformers' fp8_linear
# to this kernel only for the shapes and dtypes it supports and sends everything else to the original dispatcher with DeepGEMM disabled;
# the upstream K and Bs shape asserts are kept. Arithmetic is unchanged (test_fast_fp8_gemm.py compares bits).
"""Same block-scaled FP8 GEMM as finegrained-fp8 w8a8_block_dynamic_fp8_matmul (Apache-2.0, The HuggingFace Inc. team, third_party/finegrained-fp8/LICENSE),
with the activation quantization run once per call in its own kernel. Upstream repeats it inline for every N tile. Same fp32 scale and cast arithmetic,
same K tiles, same multiply order (dot*a_s)*b_s: results are bit-identical (test_fast_fp8_gemm.py)."""
import torch,triton,triton.language as tl
K_T=128
@triton.jit
def _quant(A,Q,S,M,K,sa_m,sa_k,sq_m,sq_k,ss_m,ss_k,BLOCK_M:tl.constexpr,BLOCK_K:tl.constexpr):
 pid_m=tl.program_id(0);pid_k=tl.program_id(1)
 offs_m=pid_m*BLOCK_M+tl.arange(0,BLOCK_M);offs_k=pid_k*BLOCK_K+tl.arange(0,BLOCK_K)
 mask_m=offs_m<M
 a_raw=tl.load(A+offs_m[:,None]*sa_m+offs_k[None,:]*sa_k,mask=mask_m[:,None]&(offs_k[None,:]<K),other=0.0).to(tl.float32)
 a_s=tl.max(tl.abs(a_raw),axis=1)/448.0
 q=(a_raw/tl.maximum(a_s[:,None],1e-12)).to(tl.float8e4nv)
 tl.store(Q+offs_m[:,None]*sq_m+offs_k[None,:]*sq_k,q,mask=mask_m[:,None]&(offs_k[None,:]<K))
 tl.store(S+offs_m*ss_m+pid_k*ss_k,a_s,mask=mask_m)
@triton.jit
def _gemm(Q,B,C,S,Bs,M,N,K,sq_m,sq_k,sb_k,sb_n,sc_m,sc_n,ss_m,ss_k,sbs_k,sbs_n,BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr,BLOCK_K:tl.constexpr,GROUP:tl.constexpr):
 pid_m=tl.program_id(0);pid_n=tl.program_id(1)
 pid_m,pid_n=tl.swizzle2d(pid_m,pid_n,tl.cdiv(M,BLOCK_M),tl.cdiv(N,BLOCK_N),GROUP)
 offs_m=(pid_m*BLOCK_M+tl.arange(0,BLOCK_M))%M;offs_n=(pid_n*BLOCK_N+tl.arange(0,BLOCK_N))%N;offs_k=tl.arange(0,BLOCK_K)
 a_ptrs=Q+offs_m[:,None]*sq_m+offs_k[None,:]*sq_k;b_ptrs=B+offs_k[:,None]*sb_k+offs_n[None,:]*sb_n
 s_ptrs=S+offs_m*ss_m;bs_ptr=Bs+pid_n*sbs_n
 acc=tl.zeros((BLOCK_M,BLOCK_N),dtype=tl.float32)
 for k in range(0,tl.cdiv(K,BLOCK_K)):
  a=tl.load(a_ptrs);b=tl.load(b_ptrs);a_s=tl.load(s_ptrs)
  acc+=tl.dot(a,b)*a_s[:,None]*tl.load(bs_ptr)
  a_ptrs+=BLOCK_K*sq_k;b_ptrs+=BLOCK_K*sb_k;s_ptrs+=ss_k;bs_ptr+=sbs_k
 c=acc.to(C.dtype.element_ty);om=pid_m*BLOCK_M+tl.arange(0,BLOCK_M);on=pid_n*BLOCK_N+tl.arange(0,BLOCK_N)
 tl.store(C+sc_m*om[:,None]+sc_n*on[None,:],c,mask=(om[:,None]<M)&(on[None,:]<N))
def _bm(M):return min(max(triton.next_power_of_2(M),16),128)
def matmul(A,B,Bs,block,out_dtype):
 assert len(block)==2,f"block_size must be [block_n, block_k], got {block}"
 bn,bk=block
 assert A.shape[-1]==B.shape[-1],f"K mismatch: A has K={A.shape[-1]}, B has K={B.shape[-1]}"
 assert B.ndim==2,f"B must be 2D (N, K), got ndim={B.ndim}"
 M=A.numel()//A.shape[-1];K=A.shape[-1];N=B.shape[0];A2=A.reshape(M,K)
 assert Bs.ndim==2 and tuple(Bs.shape)==(triton.cdiv(N,bn),triton.cdiv(K,bk)),f"Bs shape {tuple(Bs.shape)} != expected ({triton.cdiv(N,bn)}, {triton.cdiv(K,bk)})"
 Q=torch.empty(M,K,device=A.device,dtype=torch.float8_e4m3fn);S=torch.empty(M,K//K_T,device=A.device,dtype=torch.float32)
 _quant[(triton.cdiv(M,64),K//K_T)](A2,Q,S,M,K,A2.stride(0),A2.stride(1),Q.stride(0),Q.stride(1),S.stride(0),S.stride(1),BLOCK_M=64,BLOCK_K=K_T,num_warps=8,num_stages=3)
 BM=_bm(M);C=torch.empty(*A.shape[:-1],N,device=A.device,dtype=out_dtype)
 _gemm[(triton.cdiv(M,BM),triton.cdiv(N,bn))](Q,B,C,S,Bs,M,N,K,Q.stride(0),Q.stride(1),B.stride(1),B.stride(0),C.stride(-2),C.stride(-1),S.stride(0),S.stride(1),Bs.stride(1),Bs.stride(0),BLOCK_M=BM,BLOCK_N=bn,BLOCK_K=bk,GROUP=8,num_warps=16,num_stages=3)
 return C
def install():
 """Route FP8Linear through this kernel for the shape it supports; everything else keeps the original path."""
 import transformers.integrations.finegrained_fp8 as tf
 orig=tf.fp8_linear
 # Whole-weight blocks (N,K equal to the block) are tensor-wide upstream; not a shape this kernel was written for. DeepGEMM is never used: every call is this kernel or upstream Triton.
 def fp8_linear(input,weight,weight_scale_inv,block_size=None,bias=None,activation_scale=None,allow_deepgemm=True):
  if activation_scale is None and bias is None and block_size is not None and block_size[1]==K_T and input.ndim>=2 and input.numel()>0 and input.shape[-1]%K_T==0 and input.is_contiguous() and weight.dtype==torch.float8_e4m3fn and weight.is_contiguous() and weight.ndim==2 and not (weight.shape[0]==block_size[0] and weight.shape[1]==block_size[1]) and weight_scale_inv.dtype==torch.float32 and weight_scale_inv.ndim==2 and input.dtype==torch.bfloat16:
   return matmul(input,weight,weight_scale_inv,block_size,input.dtype)
  return orig(input,weight,weight_scale_inv,block_size,bias,activation_scale,False)
 tf.fp8_linear=fp8_linear
def precompile():
 """Compile every tile before serving: a first compile inside a request costs hundreds of ms."""
 W=torch.zeros(128,128,device='cuda',dtype=torch.float8_e4m3fn);S=torch.ones(1,1,device='cuda',dtype=torch.float32)
 for M in (16,32,64,128):matmul(torch.ones(M,128,device='cuda',dtype=torch.bfloat16),W,S,[128,128],torch.bfloat16)
 torch.cuda.synchronize()
