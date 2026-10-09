"""Stop at layer0 K norm; locate first mismatch without further ability runs."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='2',ABOT_VRAM_RESERVE_GIB='18',HF_HUB_OFFLINE='1',
    TRANSFORMERS_OFFLINE='1',DIFFSYNTH_SKIP_DOWNLOAD='True',TOKENIZERS_PARALLELISM='false',
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
from run import BASE,RT,SOURCE,sha
import sys
sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'code/causal'),str(RT/'DiffSynth-Studio-h3-v2')]
import json,time
import torch
import infer as abot
from causal.h3_precision import configure_precision
from causal.local_topology import visible_inputs
from audit_core import prefill,forward,errors

class Captured(Exception): pass

torch.set_num_threads(4)
began=time.perf_counter()
pipe=abot.load_pipeline('cuda:0')
pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(RT/'checkpoints/H3-World/step-10000.safetensors'),hotload=True)
model=pipe.dit.eval().requires_grad_(False)
configure_precision(model,'h3_fp32',native_transformer_dir=RT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
pipe.load_models_to_device(['dit'])
data=torch.load(SOURCE/'source_coarse/inputs/parking_A.pt',map_location='cpu',weights_only=True)
h=torch.load(SOURCE/'source_coarse/coarse_A/window0_A.pt',map_location='cpu',weights_only=True).cuda()
z=torch.load(SOURCE/'C_A/sigma_noised_A_endpoint.pt',map_location='cpu',weights_only=True).cuda()
sigma=.939540;state=(1-sigma)*z+sigma*data['initial_noise'][:,:,12:17].cuda().float()
def move(x):
    if torch.is_tensor(x):return x.cuda()
    if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
    return x
packed=move(data['packed']);prompt=data['pairs'][0]['prompts']['A'].cuda()
def cond(stop):
    p,t=visible_inputs(packed,prompt,stop,390)
    return dict(packed=p,prompt=t,anchor=data['anchor'].cuda(),audio=data['audio_noise'].cuda())

capture={};ctx={};handles=[]
names=['video_patch_proj','time_embedder','blocks.0.norm1','blocks.0.attn.qkv_proj','blocks.0.attn.k_norm']
def hook(name,pre=False):
    def fn(module,args,output=None):
        x=args[0] if pre else output
        if name=='video_patch_proj':x=x[:390+12*390]
        elif name=='time_embedder':pass
        else:x=x[ctx['prefix']:ctx['prefix']+12*390]
        capture[name+('_input' if pre else '_output')]=x.detach().cpu().clone()
        if name.endswith('k_norm') and not pre:raise Captured()
    return fn
for name in names:
    module=model.get_submodule(name)
    handles.append(module.register_forward_pre_hook(hook(name,True)))
    handles.append(module.register_forward_hook(hook(name,False)))
def run(short):
    c=cond(12 if short else 17);ctx['prefix']=c['packed']['action_video_start'];capture.clear()
    try:
        with torch.no_grad():
            if short:prefill(model,h,sigma=sigma,**c)
            else:forward(model,state,history=h,sigma=sigma,kind='R2',**c)
    except Captured:pass
    return dict(capture)
result=dict(source_sha256=sha(__file__),initial_backend=dict(tf32=torch.backends.cuda.matmul.allow_tf32,
    bf16_reduced=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    float32_matmul_precision=torch.get_float32_matmul_precision()),cases=[])
for policy in ('initial','no_reduced_precision'):
    if policy=='no_reduced_precision':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    a=run(True);b=run(False)
    rows=[]
    for name in a:
        e=errors(a[name],b[name]);e.update(name=name,shape=list(a[name].shape),dtype=str(a[name].dtype),
            unequal=int((a[name]!=b[name]).sum()))
        rows.append(e);print(policy,name,e['relative_rms'],e['max_abs'],e['unequal'],flush=True)
    result['cases'].append(dict(policy=policy,stages=rows))
    (BASE/'numeric_debug.json').write_text(json.dumps(result,indent=2)+'\n')
result['wall_seconds']=time.perf_counter()-began
result['truncated_layer0_forwards']=4
(BASE/'numeric_debug.json').write_text(json.dumps(result,indent=2)+'\n')
