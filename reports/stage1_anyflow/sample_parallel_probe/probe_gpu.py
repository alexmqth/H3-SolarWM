"""One read-only four-GPU trained-H3 batch, then same-state serial oracle.

No optimizer update, training continuation, or saved candidate model. Uses
the original four-forward loss, independently of the diagonal shortcut probe.
"""
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import gc
import hashlib
import json
import os
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUNTIME = ROOT / 'outputs/2026-10-08-09/stage1_training_shift12/runtime'
sys.path[:0] = [str(RUNTIME/'code/abot'), str(RUNTIME/'code'), str(RUNTIME/'DiffSynth-Studio-h3-v2')]
import torch
import torch.distributed as dist
import infer as abot
import causal.train_stage1_anyflow as trainer
from causal.pretrained_lora import load_adapter, load_action_residual
from causal.h3_precision import configure_precision
from causal.anyflow import load_anyflow
from causal.stage1_lora import load_stage1_lora
from parallel_batch import parallel_batch


def digest(values):
    h = hashlib.sha256()
    for p in values:
        h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def norm(values):
    return float(torch.sqrt(sum(x.double().square().sum() for x in values)))


local_rank = int(os.environ['LOCAL_RANK'])
device = f'cuda:{local_rank}'
torch.set_num_threads(4)
torch.manual_seed(13)
torch.cuda.set_device(device)
dist.init_process_group('nccl', timeout=timedelta(minutes=15), device_id=torch.device(device))
rank = dist.get_rank()
started = time.perf_counter()
result = dict(status='loading', rank=rank, local_rank=local_rank, pid=os.getpid(),
    at=datetime.now().astimezone().isoformat(), visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
    objective='same trained32 four-sample batch, parallel versus serial; zero optimizer updates')


def save():
    path = OUT / 'gpu_probe' / f'rank{rank}.tmp.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    path.replace(OUT / 'gpu_probe' / f'rank{rank}.json')


save()
try:
    args = SimpleNamespace(actions=['A'], teacher_dir=[ROOT/'outputs/2026-10-02-03/action_A_teacher_39'],
        device=device, anchor_mode='rgb', chunk_frames=5, history_chunks=5,
        action_prefix_mode='causal', action_feedback=True, logical_batch=4,
        objective='anyflow', precision_profile='h3_fp32', flow_shift=2.22,
        finite_difference_epsilon=5., smoke=False, history_gradient_mode='full',
        training_timestep_shift=12., validation_timestep_shift=2.22)
    pipe = abot.load_pipeline(device)
    pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(
        ROOT/'checkpoints/H3-World/step-10000.safetensors'), hotload=True)
    model = pipe.dit.requires_grad_(False).eval()
    pipe.load_models_to_device(['dit'])
    checkpoint = ROOT/'outputs/2026-10-08-10/stage1_shift12_duration64/train_32/step_32'
    loaded = load_adapter(model, checkpoint/'causal_adapter.pt', device)
    action = load_action_residual(model, checkpoint/'action_adapter.pt', device)['adapter']
    model.requires_grad_(False); action.requires_grad_(False)
    cases = trainer.prepare_real_cases(pipe, action, args)
    result['precision'] = configure_precision(model, 'h3_fp32', native_transformer_dir=
        ROOT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
    for case in cases:
        case['clean'] = case['clean'].float()
    time_module, time_metadata = load_anyflow(model, checkpoint/'anyflow_adapter.pt', device)
    time_module.requires_grad_(False)
    bank, bank_metadata = load_stage1_lora(model, checkpoint/'stage1_lora.pt', device=device)
    assert bank_metadata == time_metadata
    assert bank_metadata['optimizer_step']==32 and bank_metadata['config']['training_timestep_shift']==12.
    parameters = bank.parameters()
    assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in parameters}
    visual = [p for i in loaded['block_indices'] for p in (
        model.blocks[i].attn.qkv_proj.base.lora_A, model.blocks[i].attn.qkv_proj.base.lora_B)]
    frozen = visual+list(time_module.parameters())+list(action.parameters())
    frozen_before, bank_before = digest(frozen), digest(parameters)
    state_hashes = [None]*4
    dist.all_gather_object(state_hashes, (frozen_before, bank_before))
    assert len(set(state_hashes))==1
    result['bank'] = bank.describe()
    result['status'] = 'parallel_backward'
    save()
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    dist.barrier()
    tick = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    cpu_rng = torch.get_rng_state().clone(); cuda_rng = torch.cuda.get_rng_state(device).clone()
    generator = torch.Generator().manual_seed(1001)
    parallel = parallel_batch(model, cases[0], 2, args, generator=generator,
        backward=True, parameters=parameters, record_noise=True)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-tick
    gradients = [p.grad.detach().cpu().clone() for p in parameters]
    assert all(torch.isfinite(g).all() for g in gradients)
    result['parallel'] = dict(seconds=elapsed, metrics=parallel, gradient_norm=norm(gradients),
        gradient_sha256=digest(gradients),
        CPU_RNG_unchanged=torch.equal(cpu_rng,torch.get_rng_state()),
        CUDA_RNG_unchanged=torch.equal(cuda_rng,torch.cuda.get_rng_state(device)),
        logical_RNG_sha256=digest([generator.get_state()]),
        gpu_allocated_peak_MiB=torch.cuda.max_memory_allocated(device)/2**20,
        gpu_reserved_peak_MiB=torch.cuda.max_memory_reserved(device)/2**20)
    summaries=[None]*4
    dist.all_gather_object(summaries,{k:v for k,v in result['parallel'].items() if k!='metrics'})
    assert len({x['gradient_sha256'] for x in summaries})==1
    assert all(x['CPU_RNG_unchanged'] and x['CUDA_RNG_unchanged'] for x in summaries)
    result['all_rank_parallel']=summaries
    result['status']='serial_oracle_on_rank0'
    save()
    if rank==0:
        for p in parameters:
            p.grad=None
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device);tick=time.perf_counter()
        reference_noise=[]
        original_loss=trainer.anyflow_sample_loss
        def capture(velocity,clean,noise,*args,**kwargs):
            reference_noise.append(digest([noise]))
            return original_loss(velocity,clean,noise,*args,**kwargs)
        trainer.anyflow_sample_loss=capture
        serial_generator=torch.Generator().manual_seed(1001)
        serial=trainer.logical_batch(model,cases[0],2,args,generator=serial_generator,backward=True)
        trainer.anyflow_sample_loss=original_loss
        torch.cuda.synchronize(device)
        serial_gradients=[torch.zeros_like(p,device='cpu') if p.grad is None else p.grad.detach().cpu().clone() for p in parameters]
        diff=[p-s for p,s in zip(gradients,serial_gradients)]
        dot=sum((p.double()*s.double()).sum() for p,s in zip(gradients,serial_gradients))
        result['serial']=dict(seconds=time.perf_counter()-tick,metrics=serial,
            gradient_norm=norm(serial_gradients),
            gpu_allocated_peak_MiB=torch.cuda.max_memory_allocated(device)/2**20,
            gpu_reserved_peak_MiB=torch.cuda.max_memory_reserved(device)/2**20)
        result['comparison']=dict(
            raw_samples_identical=parallel['samples']==serial['samples'],
            actual_noise_identical=reference_noise==[r['actual_noise_sha256'] for r in parallel['ranks']],
            logical_RNG_identical=torch.equal(generator.get_state(),serial_generator.get_state()),
            CPU_RNG_unchanged=torch.equal(cpu_rng,torch.get_rng_state()),
            CUDA_RNG_unchanged=torch.equal(cuda_rng,torch.cuda.get_rng_state(device)),
            gradient_difference_norm=norm(diff),
            gradient_cosine=float(dot/(norm(gradients)*norm(serial_gradients)+1e-30)),
            max_gradient_abs_difference=max(float(x.abs().max()) for x in diff))
        for key in ('raw_samples_identical','actual_noise_identical','logical_RNG_identical','CPU_RNG_unchanged','CUDA_RNG_unchanged'):
            assert result['comparison'][key],key
        save()
    dist.barrier()
    result['frozen_visual_time_action_unchanged']=digest(frozen)==frozen_before
    result['bank_parameters_unchanged']=digest(parameters)==bank_before
    assert result['frozen_visual_time_action_unchanged'] and result['bank_parameters_unchanged']
    result.update(status='complete',wall_seconds=time.perf_counter()-started)
except BaseException as exc:
    result.update(status='failed',error=repr(exc),wall_seconds=time.perf_counter()-started)
    raise
finally:
    save()
    dist.destroy_process_group()
