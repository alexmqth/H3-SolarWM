"""One real chunk2 history-gradient/VRAM probe, not a rollout-quality trial."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import gc
import json
import sys
import time

import torch

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
RUNTIME=OUT/'runtime'
sys.path[:0]=[str(RUNTIME/'code/abot'),str(RUNTIME/'code'),str(RUNTIME/'DiffSynth-Studio-h3-v2')]
import infer as abot
from causal.train_stage1_anyflow import prepare_real_cases, logical_batch
from causal.pretrained_lora import load_adapter, load_action_residual
from causal.h3_precision import configure_precision
from causal.anyflow import install_anyflow
from causal.stage1_lora import install_stage1_lora, save_stage1_lora


torch.set_num_threads(4)
torch.manual_seed(13)
torch.cuda.set_device('cuda:0')
started=time.perf_counter()
RESULT=dict(status='running',at=datetime.now().astimezone().isoformat(),
    scope='33B A/chunk2 matched detached/full-history gradients plus one update; no quality claim',
    gpu=0,init='fresh full-scope rank8 over original visual/action initialization',
    inference_changed=False)


def save():
    p=OUT/'gpu_probe/probe.tmp.json'
    p.write_text(json.dumps(RESULT,indent=2)+'\n');p.replace(OUT/'gpu_probe/probe.json')


def scalar_norm(values):
    return float(torch.sqrt(sum(x.double().square().sum() for x in values)))


save()
try:
    args=SimpleNamespace(actions=['A'],teacher_dir=[ROOT/'outputs/2026-10-02-03/action_A_teacher_39'],
        device='cuda:0',anchor_mode='rgb',chunk_frames=5,history_chunks=5,
        action_prefix_mode='causal',action_feedback=True,logical_batch=4,
        objective='anyflow',precision_profile='h3_fp32',flow_shift=2.22,
        finite_difference_epsilon=5.,smoke=False,history_gradient_mode='detached')
    pipe=abot.load_pipeline(args.device)
    pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(
        ROOT/'checkpoints/H3-World/step-10000.safetensors'),hotload=True)
    model=pipe.dit.requires_grad_(False).eval()
    pipe.load_models_to_device(['dit'])
    initial=ROOT/'outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2'
    loaded=load_adapter(model,initial/'causal_adapter.pt',args.device)
    action=load_action_residual(model,initial/'action_adapter.pt',args.device)['adapter']
    model.requires_grad_(False);action.requires_grad_(False)
    cases=prepare_real_cases(pipe,action,args)
    RESULT['precision']=configure_precision(model,'h3_fp32',native_transformer_dir=
        ROOT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
    for case in cases:case['clean']=case['clean'].float()
    time_module=install_anyflow(model,device=args.device).requires_grad_(False)
    bank=install_stage1_lora(model,rank=8,alpha=8.,device=args.device)
    parameters=bank.parameters()
    assert {id(p) for p in model.parameters() if p.requires_grad}=={id(p) for p in parameters}
    RESULT['bank']=bank.describe()
    visual=[p for i in loaded['block_indices'] for p in (
        model.blocks[i].attn.qkv_proj.base.lora_A,model.blocks[i].attn.qkv_proj.base.lora_B)]
    frozen=visual+list(time_module.parameters())+list(action.parameters())
    frozen_before=[p.detach().cpu().clone() for p in frozen]
    gradients={}
    for mode in ('detached','full'):
        args.history_gradient_mode=mode
        for p in parameters:p.grad=None
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick=time.perf_counter()
        record=logical_batch(model,cases[0],2,args,
            generator=torch.Generator().manual_seed(1001),backward=True)
        torch.cuda.synchronize()
        values=[torch.zeros_like(p,device='cpu') if p.grad is None else p.grad.detach().cpu().clone() for p in parameters]
        if not all(torch.isfinite(x).all() for x in values):raise FloatingPointError('Nonfinite adapter gradient')
        gradients[mode]=values
        RESULT[mode]=dict(metrics=record,seconds=time.perf_counter()-tick,
            grad_norm=scalar_norm(values),gpu_allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
            gpu_reserved_peak_MiB=torch.cuda.max_memory_reserved()/2**20)
        by_id=dict(zip(map(id,parameters),values))
        RESULT[mode]['gradient_group_norms']={group:scalar_norm([by_id[id(p)] for p in bank.parameters(group)])
            for group in ('qkv','out','ffn','refiner')}
        del by_id
        save()
    raw_a=[x['raw_loss'] for x in RESULT['detached']['metrics']['samples']]
    raw_b=[x['raw_loss'] for x in RESULT['full']['metrics']['samples']]
    errors=[abs(a-b) for a,b in zip(raw_a,raw_b)]
    if any(e>1e-4*max(1.,abs(a)) for e,a in zip(errors,raw_a)):
        raise RuntimeError('Full-history loss predictions differ from detached reference')
    difference=[a-b for a,b in zip(gradients['full'],gradients['detached'])]
    dot=sum((a.double()*b.double()).sum() for a,b in zip(gradients['full'],gradients['detached']))
    RESULT['comparison']=dict(raw_losses_identical=raw_a==raw_b,raw_loss_abs_errors=errors,
        gradient_difference_norm=scalar_norm(difference),
        gradient_cosine=float(dot/(RESULT['full']['grad_norm']*RESULT['detached']['grad_norm']+1e-30)),
        changed_gradient_tensors=sum(not torch.equal(a,b) for a,b in zip(gradients['full'],gradients['detached'])))
    if RESULT['comparison']['gradient_difference_norm']<=1e-8:raise RuntimeError('No real history gradient contribution detected')
    del difference,gradients,values
    before=[p.detach().cpu().clone() for p in parameters]
    optimizer=torch.optim.AdamW(parameters,lr=3e-5,betas=(.9,.95),weight_decay=.01)
    clipped=torch.nn.utils.clip_grad_norm_(parameters,1.)
    if not torch.isfinite(clipped):raise FloatingPointError('Nonfinite optimizer gradient')
    optimizer.step();torch.cuda.synchronize()
    RESULT['one_update']=dict(changed_tensors=sum(not torch.equal(p.detach().cpu(),b) for p,b in zip(parameters,before)),
        nonzero_B_projections=sum(bool(torch.count_nonzero(b).item()) for m in bank.modules for b in m.lora_B),
        frozen_visual_time_action_unchanged=all(torch.equal(p.detach().cpu(),b) for p,b in zip(frozen,frozen_before)),
        gpu_allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
        gpu_reserved_peak_MiB=torch.cuda.max_memory_reserved()/2**20)
    if not RESULT['one_update']['frozen_visual_time_action_unchanged']:raise RuntimeError('Frozen parameters changed')
    save_stage1_lora(OUT/'gpu_probe/stage1_lora_after_probe.pt',bank,
        {'scope':'single gradient/memory probe; not a resumable training checkpoint or accepted model'})
    RESULT.update(status='complete',wall_seconds=time.perf_counter()-started)
except BaseException as exc:
    RESULT.update(status='failed',error=repr(exc),wall_seconds=time.perf_counter()-started)
    raise
finally:
    save()
