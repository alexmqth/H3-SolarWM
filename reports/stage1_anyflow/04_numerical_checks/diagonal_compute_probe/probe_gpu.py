"""Same-state trained33B diagonal-shortcut compute probe; no optimizer update or video-quality claim."""
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
RUNTIME=ROOT/'outputs/2026-10-08-09/stage1_training_shift12/runtime'
sys.path[:0]=[str(RUNTIME/'code/abot'),str(RUNTIME/'code'),str(RUNTIME/'DiffSynth-Studio-h3-v2')]
import infer as abot
from causal.train_stage1_anyflow import prepare_real_cases, logical_batch
from causal.pretrained_lora import load_adapter, load_action_residual
from causal.h3_precision import configure_precision
from causal.anyflow import load_anyflow, anyflow_sample_loss
import causal.train_stage1_anyflow as trainer
from diagonal_candidate import sample_loss as shortcut_loss
from causal.stage1_lora import load_stage1_lora


torch.set_num_threads(4)
torch.manual_seed(13)
torch.cuda.set_device('cuda:0')
started=time.perf_counter()
RESULT=dict(status='running',at=datetime.now().astimezone().isoformat(),
    scope='same trained state A/chunk2: repeat reference then shortcut; no optimizer updates',
    gpu=4,init='shift12 full-history AnyFlow32 checkpoint',
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
        finite_difference_epsilon=5.,smoke=False,history_gradient_mode='full',
        training_timestep_shift=12.,validation_timestep_shift=2.22)
    pipe=abot.load_pipeline(args.device)
    pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(
        ROOT/'checkpoints/H3-World/step-10000.safetensors'),hotload=True)
    model=pipe.dit.requires_grad_(False).eval()
    pipe.load_models_to_device(['dit'])
    initial=ROOT/'outputs/2026-10-08-10/stage1_shift12_duration64/train_32/step_32'
    loaded=load_adapter(model,initial/'causal_adapter.pt',args.device)
    action=load_action_residual(model,initial/'action_adapter.pt',args.device)['adapter']
    model.requires_grad_(False);action.requires_grad_(False)
    cases=prepare_real_cases(pipe,action,args)
    RESULT['precision']=configure_precision(model,'h3_fp32',native_transformer_dir=
        ROOT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
    for case in cases:case['clean']=case['clean'].float()
    time_module,time_metadata=load_anyflow(model,initial/'anyflow_adapter.pt',args.device)
    time_module.requires_grad_(False)
    bank,bank_metadata=load_stage1_lora(model,initial/'stage1_lora.pt',device=args.device)
    assert bank_metadata==time_metadata
    assert bank_metadata['optimizer_step']==32
    assert bank_metadata['config']['training_timestep_shift']==12.
    assert bank_metadata['config']['history_gradient_mode']=='full'
    parameters=bank.parameters()
    assert {id(p) for p in model.parameters() if p.requires_grad}=={id(p) for p in parameters}
    RESULT['bank']=bank.describe()
    visual=[p for i in loaded['block_indices'] for p in (
        model.blocks[i].attn.qkv_proj.base.lora_A,model.blocks[i].attn.qkv_proj.base.lora_B)]
    frozen=visual+list(time_module.parameters())+list(action.parameters())
    frozen_before=[p.detach().cpu().clone() for p in frozen]
    RESULT['backend'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled())
    bank_before=[p.detach().cpu().clone() for p in parameters]
    gradients={}
    for label,function in [('reference_1',anyflow_sample_loss),('reference_2',anyflow_sample_loss),('shortcut',shortcut_loss)]:
        trainer.anyflow_sample_loss=function
        for p in parameters:p.grad=None
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
        record=logical_batch(model,cases[0],2,args,
            generator=torch.Generator().manual_seed(1001),backward=True)
        torch.cuda.synchronize()
        values=[torch.zeros_like(p,device='cpu') if p.grad is None else p.grad.detach().cpu().clone() for p in parameters]
        if not all(torch.isfinite(x).all() for x in values):raise FloatingPointError('Nonfinite gradient')
        gradients[label]=values
        RESULT[label]=dict(CPU_RNG_unchanged=torch.equal(cpu_rng,torch.get_rng_state()),
            CUDA_RNG_unchanged=torch.equal(cuda_rng,torch.cuda.get_rng_state()),metrics=record,seconds=time.perf_counter()-tick,grad_norm=scalar_norm(values),
            gpu_allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
            gpu_reserved_peak_MiB=torch.cuda.max_memory_reserved()/2**20)
        save()
    RESULT['comparison']={}
    for label in ['reference_2','shortcut']:
        a=gradients['reference_1'];b=gradients[label]
        diff=[x-y for x,y in zip(a,b)]
        dot=sum((x.double()*y.double()).sum() for x,y in zip(a,b))
        RESULT['comparison'][label]=dict(
            common_sample_values_identical=all(
                all(a[k]==b[k] for k in ('sigma','target_sigma','raw_loss','weight','sample_type','adaptive_scale','weighted_loss'))
                for a,b in zip(RESULT['reference_1']['metrics']['samples'],RESULT[label]['metrics']['samples'])),
            gradient_difference_norm=scalar_norm(diff),
            gradient_cosine=float(dot/(RESULT['reference_1']['grad_norm']*RESULT[label]['grad_norm']+1e-30)),
            changed_gradient_tensors=sum(not torch.equal(x,y) for x,y in zip(a,b)),
            max_gradient_abs_difference=max(float(x.abs().max()) for x in diff))
    RESULT['bank_parameters_unchanged']=all(torch.equal(p.detach().cpu(),b) for p,b in zip(parameters,bank_before))
    RESULT['frozen_visual_time_action_unchanged']=all(torch.equal(p.detach().cpu(),b) for p,b in zip(frozen,frozen_before))
    assert RESULT['bank_parameters_unchanged'] and RESULT['frozen_visual_time_action_unchanged']
    assert RESULT['shortcut']['metrics']['model_evaluations']==10
    assert RESULT['reference_1']['metrics']['model_evaluations']==16
    RESULT['interpretation']='Read-only compute/gradient probe; report GPU differences against repeated reference, not proof of bitwise CUDA equivalence or video quality.'
    RESULT.update(status='complete',wall_seconds=time.perf_counter()-started)
except BaseException as exc:
    RESULT.update(status='failed',error=repr(exc),wall_seconds=time.perf_counter()-started)
    raise
finally:
    save()
