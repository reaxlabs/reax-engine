# SPDX-License-Identifier: MIT
"""Native K and U frontends in one isolated process/GPU, changed weight storage only."""
import os,sys,json,time,threading,traceback,contextlib
from pathlib import Path
assert Path('/.dockerenv').exists() and os.environ.get('DUO_ISOLATED')=='1'
import torch,numpy as np
from autotune_guard import install as install_autotune_guard,pin_answer_configs
install_autotune_guard()
import fast_fp8_gemm
fast_fp8_gemm.install()
import transformers
from concurrent.futures import ThreadPoolExecutor
K_STREAM=None
G_PATH=os.environ['REAX_PART_G']
Q_PATH=os.environ['REAX_PART_Q']
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM,AutoModelForImageTextToText,FineGrainedFP8Config,BitsAndBytesConfig
sys.path.insert(0,str(Path(__file__).resolve().parent/'third_party/knowline'))
import knowline_server as ks
from knowline_layout import Layout
mode,scheme="duo","fp8";torch.set_num_threads(4)
def quant():
 if scheme=='bf16':return None
 if scheme.startswith('fp8'):
  return FineGrainedFP8Config(weight_block_size=(128,128) if scheme=='fp8' else (64,64),modules_to_not_convert=['lm_head','visual','model.visual','vision_tower','model.vision_tower','multi_modal_projector','model.multi_modal_projector','audio_tower','model.audio_tower','embed_vision','model.embed_vision','embed_audio','model.embed_audio','in_proj_a','in_proj_b'])
 return BitsAndBytesConfig(load_in_8bit=True,llm_int8_threshold=6.0,llm_int8_skip_modules=['lm_head','visual','model.visual','vision_tower','model.vision_tower','multi_modal_projector','model.multi_modal_projector','audio_tower','model.audio_tower','embed_vision','model.embed_vision','embed_audio','model.embed_audio','in_proj_a','in_proj_b'])
def info(m):
 from collections import Counter
 return {'classes':dict(Counter(type(x).__name__ for x in m.modules())),'parameter_bytes':sum(p.numel()*p.element_size() for p in m.parameters()),'dtypes':dict(Counter(str(p.dtype) for p in m.parameters()))}
def kload():
 cfg=AutoConfig.from_pretrained(G_PATH);cls=getattr(transformers,'AutoModelForMultimodalLM',AutoModelForImageTextToText) if hasattr(cfg,'vision_config') or hasattr(cfg,'audio_config') else AutoModelForCausalLM
 kw={'dtype':torch.bfloat16,'device_map':{'':'cuda:0'},'attn_implementation':'flex_attention','local_files_only':True}
 if quant() is not None:kw['quantization_config']=quant()
 m=cls.from_pretrained(G_PATH,**kw).eval();tok=AutoTokenizer.from_pretrained(G_PATH)
 class Backend:
  def warm(self,*a):pass
  def score(self,text,media,ids):
   if media:raise ValueError('text only')
   with torch.inference_mode():
    inputs=torch.tensor([tok.encode(text,add_special_tokens=False)],device='cuda:0')
    logits=m(input_ids=inputs,logits_to_keep=1,use_cache=False,kernel_options={'BLOCK_M':32,'BLOCK_N':32}).logits[0,-1].float().log_softmax(-1)
   return [float(logits[i]) for i in ids],int(inputs.shape[1])
 temps=json.load(open(Path(G_PATH)/'temperatures.json' if (Path(G_PATH)/'temperatures.json').exists() else Path(__file__).resolve().parent/'third_party/knowline/temperatures.json'));engine=ks.KnowLine(tok,Backend(),1.0,1,temps,Layout(read_twice=True,twice_max_tokens=3000))
 return engine,m
try:
 k,km=kload();u=None
 tok=AutoTokenizer.from_pretrained(G_PATH);tiny=torch.tensor([tok.encode('Hello',add_special_tokens=False)],device='cuda:0')
 with torch.inference_mode():
  one=km(input_ids=tiny,logits_to_keep=1,use_cache=True,kernel_options={'BLOCK_M':32,'BLOCK_N':32}).logits.float()
  two=km(input_ids=tiny,logits_to_keep=1,use_cache=False,kernel_options={'BLOCK_M':32,'BLOCK_N':32}).logits.float()
  cache_parity=float((one-two).abs().max());assert torch.allclose(one,two,atol=.04,rtol=.04), 'cache-free prefill parity'
 del one,two,tiny

 if mode=='duo':
  sys.path.insert(0,Q_PATH)
  import d3_runtime
  original=d3_runtime.backbone_class
  def backbone(kind):
   cls=original(kind)
   class QuantizedLoader:
    @staticmethod
    def from_pretrained(*a,**kw):
     kw['quantization_config']=quant();kw['local_files_only']=True
     return cls.from_pretrained(*a,**kw)
   return QuantizedLoader
  d3_runtime.backbone_class=backbone
  # Same class used by AutoModel's native d3 entrypoint; wrapper delegates system_one directly.
  from modeling_d3 import D3Model
  u=D3Model.from_pretrained(Q_PATH,device='cuda:0',local_files_only=True,batch_size=1)
  original_run=u.runtime._run
  def bounded_run(prepared):
   wanted=u.runtime.batch_size
   longest=max([len(x) for x in prepared.sequences.values()]+[len(x) for x in prepared.reversed_sequences.values()]+[1])
   u.runtime.batch_size=min(wanted,max(1,4096//longest))
   try:return original_run(prepared)
   finally:u.runtime.batch_size=wanted
  u.runtime._run=bounded_run
  pin_answer_configs()
  ba_patches=[]
  state_once_backend=d3_runtime.StateOnce(u.backbone.language_model)
  opt_active='plain'
  from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_modeling
  from causal_conv1d import causal_conv1d_fn as official_conv
  def optimized_conv(x,weight,bias=None,activation=None,**kw):return official_conv(x,weight,bias,activation=activation)
  def reference_conv(x,weight,bias=None,activation=None,**kw):
   out=torch.nn.functional.conv1d(x.to(weight.dtype),weight[:,None,:],bias,padding=weight.shape[-1]-1,groups=weight.shape[0])[...,:x.shape[-1]]
   return (torch.nn.functional.silu(out) if activation in ['silu','swish'] else out).to(x.dtype)
  qwen_modeling.causal_conv1d_fn=reference_conv
  cx=torch.randn(1,512,1024,device='cuda',dtype=torch.bfloat16);cw=torch.randn(512,4,device='cuda',dtype=torch.bfloat16)
  ca=reference_conv(cx,cw,activation='silu');cb=optimized_conv(cx,cw,activation='silu')
  conv_probe={'max_abs':float((ca-cb).abs().max()),'mean_abs':float((ca-cb).abs().mean())}
  assert torch.allclose(ca,cb,atol=.08,rtol=.02), 'convolution parity prerequisite'
  del cx,cw,ca,cb
 if scheme!='bf16':
  for model in [km]+([u.backbone] if u else []):
   assert any('float8' in str(p.dtype) or p.dtype==torch.int8 for p in model.parameters()), 'quantization silently dequantized'
 torch.cuda.synchronize()
 fast_fp8_gemm.precompile()
 print(json.dumps({'ready':True,'autotuner_guard':True,'mode':mode,'scheme':scheme,'cache_parity_max_abs':cache_parity,'gpu':torch.cuda.get_device_name(0),'k':info(km),'u':info(u.backbone) if u else None,'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'peak_load_bytes':torch.cuda.max_memory_allocated()}),flush=True)
except BaseException:
 traceback.print_exc();raise

K_DONE=threading.Event()
def invoke_k(r):
 try:
  a,usage=k.run(r['state'],r['questions'])
  if opt_active in ['longcache','fastconv_longcache','fast_all_longcache']:torch.cuda.current_stream().synchronize()
  return {'answers':a,'usage':usage}
 finally:K_DONE.set()
def invoke_u(r):
 u.runtime.fast_route={'state_once':False,'oom_fallback':False,'serialized_for_memory':False}
 if opt_active in ['state_once','fastconv_state_once','fast_all','longcache','fastconv_longcache','fast_all_longcache'] and len(r['questions'])>1:
  prepared=u.runtime.prepare(r['state'],r['questions'])
  prefix=max(list(prepared.states.values())+[0]);long_mode=opt_active in ['longcache','fastconv_longcache','fast_all_longcache'];use=0<prefix<=(24576 if long_mode else 16384)
  u.runtime.fast_route.update(prefix_tokens=prefix,state_once=use)
  if use and long_mode and prefix>16384:
   # General length policy: avoid overlapping peak activations for large native prefix caches.
   if not K_DONE.wait(300):raise RuntimeError('K completion wait exceeded memory-scheduling budget')
   u.runtime.fast_route['serialized_for_memory']=True
  u.runtime.state_once=state_once_backend if use else None
  try:
   result=None
   if use:
    try:result=u.runtime.run(prepared)
    except torch.OutOfMemoryError:pass
   if result is None:
    if use:
     u.runtime.state_once=None;u.runtime.fast_route.update(state_once=False,oom_fallback=True);torch.cuda.empty_cache()
    result=u.runtime.run(prepared)
   probabilities,tokens=result
   return u.runtime.respond(prepared,probabilities,tokens)
  finally:u.runtime.state_once=None
 return u.system_one(state=r['state'],questions=r['questions'])

from probabilities import mix
norm_patches=[]
for mod_name,mod in km.named_modules():
 if type(mod).__name__.endswith('RMSNorm'):
  norm_patches.append((mod,mod.forward,None))
def apply_compile_norm(on):
 global norm_patches
 updated=[]
 for mod,plain,compiled in norm_patches:
  if on and compiled is None:compiled=torch.compile(plain,dynamic=True)
  mod.forward=compiled if on else plain
  updated.append((mod,plain,compiled))
 norm_patches=updated
stream_pool=ThreadPoolExecutor(2)
stream_k=torch.cuda.Stream();stream_u=torch.cuda.Stream()
stream_k.wait_stream(torch.cuda.current_stream());stream_u.wait_stream(torch.cuda.current_stream())
def probe_graph():
 apply_compile_norm(False)
 static=torch.tensor([AutoTokenizer.from_pretrained(G_PATH).encode('Hello, how are you?',add_special_tokens=False)],device='cuda:0')
 def forward():return km(input_ids=static,logits_to_keep=1,use_cache=False,kernel_options={'BLOCK_M':32,'BLOCK_N':32}).logits.float()
 with torch.inference_mode():
  for _ in range(3):ref=forward()
  torch.cuda.synchronize();ref=ref.clone()
  t=time.perf_counter()
  for _ in range(20):out=forward()
  torch.cuda.synchronize();eager=1000*(time.perf_counter()-t)/20
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):captured=forward()
  graph.replay();torch.cuda.synchronize();diff=float((ref-captured).abs().max())
  t=time.perf_counter()
  for _ in range(20):graph.replay()
  torch.cuda.synchronize();graphed=1000*(time.perf_counter()-t)/20
  return {'status':'ok','scope':'diagnostic repeated tiny synthetic K forward only, not representative latency','tokens':static.shape[1],'eager_ms':eager,'graph_ms':graphed,'max_abs_logit_diff':diff}
for line in sys.stdin:
 try:
  r=json.loads(line);kind=r.pop('_kind','concurrent')
  if kind=='graph_probe':
   try:probe=probe_graph()
   except BaseException as e:probe={'status':'error','scope':'K tiny diagnostic graph capture','error':repr(e)}
   print(json.dumps(probe),flush=True);continue
  ubatch=int(r.pop('_ubatch',8));opt_active=r.pop('_opt','fastconv_longcache');u.runtime.batch_size=ubatch
  for mod,plain,fast in ba_patches:mod.forward=fast if opt_active in ['miner_ba','fast_all','fast_all_longcache'] else plain
  apply_compile_norm(opt_active in ['compile_norm','fast_all','fast_all_longcache'])
  qwen_modeling.causal_conv1d_fn=optimized_conv if opt_active in ['fastconv','fastconv_state_once','fast_all','fastconv_longcache','fast_all_longcache'] else reference_conv
  K_DONE.clear()
  if kind=='single_u':K_DONE.set()
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  a=b=None
  if kind=='concurrent':
   def call_k():
    with torch.cuda.stream(stream_k):
     torch._dynamo.config.recompile_limit=128
     return invoke_k(r)
   def call_u():
    with torch.cuda.stream(stream_u):return invoke_u(r)
   fa=stream_pool.submit(call_k);fb=stream_pool.submit(call_u)
   # Drain both futures even on an error before accepting any next request.
   ae=be=None
   try:a=fa.result()
   except BaseException as e:ae=e
   try:b=fb.result()
   except BaseException as e:be=e
   torch.cuda.synchronize()
   if ae or be:raise (ae or be)
   mid=start
  elif kind=='single_k':
   a=invoke_k(r);torch.cuda.synchronize();mid=time.perf_counter()
  elif kind=='single_u':
   mid=start;b=invoke_u(r)
  else:
   a=invoke_k(r);torch.cuda.synchronize();mid=time.perf_counter();b=invoke_u(r)
  torch.cuda.synchronize();end=time.perf_counter()
  combined=mix(r,a,b) if a and b else None;duo_end=time.perf_counter()
  result={'K':a,'U':b,'Duo':combined,'timings_ms':{'K':1000*(mid-start),'U':1000*(end-mid),'Duo':1000*(duo_end-start)},'peak_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'optimization':{'name':opt_active,'miner_ba_patched_modules':len(ba_patches),'compile_norm_modules':len(norm_patches),'conv_probe':conv_probe,'u_route':getattr(u.runtime,'fast_route',None)}}
  print(json.dumps({'status':'ok',**result}),flush=True)
 except BaseException as e:
  traceback.print_exc();torch.cuda.synchronize();torch.cuda.empty_cache()
  print(json.dumps({'status':'error','error':str(e),'exception':type(e).__name__}),flush=True)
