"""Fresh full-history AnyFlow16: train shift12, common validation/inference2.22."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUNTIME = OUT / 'runtime'
SOURCE = ROOT / 'outputs/2026-10-08-08/stage1_anyflow39_full_history'
TEACHER = ROOT / 'outputs/2026-10-02-03'
sys.path.insert(0, str(RUNTIME / 'code/causal'))
from report_stage1_anyflow import audit_inputs
from evaluate_action_control import evaluate as evaluate_flow
from evaluate_videos import evaluate as evaluate_video
from summarize_action_experiment import rgb_boundary

ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='1', ABOT_VRAM_RESERVE_GIB='6',
    HF_HUB_OFFLINE='1', DIFFSYNTH_SKIP_DOWNLOAD='True', PYTHONUNBUFFERED='1',
    ABOT_DIFFSYNTH_ROOT=str(RUNTIME / 'DiffSynth-Studio-h3-v2'),
    DIFFSYNTH_ROOT=str(RUNTIME / 'DiffSynth-Studio-h3-v2'))
for key, name in [('HF_HOME','hf'),('TORCHINDUCTOR_CACHE_DIR','torchinductor'),
                  ('TRITON_CACHE_DIR','triton'),('XDG_CACHE_HOME','xdg')]:
    ENV[key] = str(ROOT / '.cache' / name)
STATE = dict(status='prepared_training_shift12', pid=os.getpid(), gpu=1,
    started_at=datetime.now().astimezone().isoformat(), active=None, precision_profile='h3_fp32', adapter_scope='all_qkvo_ffn',
    vram_reserve_gib=6, history_gradient_mode='full', training_timestep_shift=12.,
    validation_timestep_shift=2.22, inference_flow_shift=2.22,
    training_runs=[], evaluations=[], overall_gate='NOT_ACCEPTED')

def save():
    p=OUT/'run.tmp.json';p.write_text(json.dumps(STATE,indent=2)+'\n');p.replace(OUT/'run.json')

def live(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:
        return False

def verify():
    manifest=json.loads((OUT/'manifest.json').read_text())
    for path,sha in manifest['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:
            raise RuntimeError(f'Frozen source/input changed: {path}')

def child(label,command,directory):
    verify();directory.mkdir(parents=True,exist_ok=False)
    print('START',label,flush=True)
    with (directory/'run.log').open('w') as log:
        p=subprocess.Popen(command,cwd=RUNTIME,env=ENV,stdout=log,stderr=subprocess.STDOUT)
        STATE['active']=dict(label=label,pid=p.pid,command=command,directory=str(directory));save()
        code=p.wait()
    if code:
        raise RuntimeError(f'{label} exited {code}')
    STATE['active']=None;save()

def train(total,previous=None):
    initial=json.loads((OUT/'reference_initial.json').read_text())
    config=dict(initial['config'])
    directory=OUT/f'train_{total:02d}'
    start=0 if previous is None else 1
    config.update(out_dir=str(directory),steps=total,precision_profile='h3_fp32',objective='anyflow',history_gradient_mode='full',
        resume_from=None if previous is None else str(previous/'step_01'),
        checkpoint_every=1 if total==1 else 4, adapter_scope='all_qkvo_ffn', bank_rank=8, bank_alpha=8., training_timestep_shift=12., validation_timestep_shift=2.22)
    command=[sys.executable,'-u',str(RUNTIME/'code/causal/train_stage1_anyflow.py')]
    for key,value in config.items():
        if value is None:continue
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:command.append(flag)
            elif key in ('action_feedback','train_target_time'):command.append('--no-'+key.replace('_','-'))
        elif isinstance(value,list):command += [flag,*map(str,value)]
        else:command += [flag,str(value)]
    child(f'shift12_full_history_fp32_anyflow_train_{total:02d}',command,directory)
    report=json.loads((directory/'training.json').read_text())
    if (report['status']!='complete' or len(report['updates'])!=total
            or report['resumed_optimizer_step']!=start
            or report['target_time_parameters_changed'] or not report['qkv_parameters_changed']
            or report['target_time_trainable_parameters']!=0
            or report['precision']['profile']!='h3_fp32'
            or not report['precision']['native_fp32_weights_restored']
            or len(report['precision']['native_tensor_sha256'])!=12):
        raise RuntimeError('Native FP32 training contract failed')
    if (report['stage1_lora']['trainable_parameters']!=43237376
            or not report['frozen_visual_unchanged'] or not all(report['adapter_groups_changed'].values())
            or report['time_sampling'] != dict(training_shift=12.,validation_shift=2.22,inference_flow_shift=2.22)):
        raise RuntimeError('Full-scope training contract failed')
    # Initial weights/RNG and sigma sequence must match the AnyFlow candidate.
    import torch
    torch.set_num_threads(4)
    def equal(a,b):
        if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a,b)
        if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
        if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    checks={}
    for name in ('causal_adapter.pt','action_adapter.pt','stage1_lora.pt','anyflow_adapter.pt'):
        a=torch.load(directory/'step_00'/name,map_location='cpu',weights_only=True)
        b=torch.load(SOURCE/'train_16/step_00'/name,map_location='cpu',weights_only=True)
        a.pop('metadata');b.pop('metadata')
        checks[name]=equal(a,b)
        del a,b
    initial=torch.load(directory/'step_00/trainer_state.pt',map_location='cpu',weights_only=True)
    reference=torch.load(SOURCE/'train_16/step_00/trainer_state.pt',map_location='cpu',weights_only=True)
    for key in ('logical_rng_state','cpu_rng_state','cuda_rng_state'):
        checks[key]=equal(initial[key],reference[key])
    trained=json.loads((OUT/'expected_schedule.json').read_text())
    checks['differentiable_history_count']=all(x['differentiable_history_forwards']==x['chunk']*config['logical_batch'] for x in report['updates'])
    reference=json.loads((OUT/'reference_initial.json').read_text())
    checks['same_initial_validation_samples']=[x['samples'] for x in report['validation_before']]==[x['samples'] for x in reference['validation_before']]
    checks['matching_expected_shift12_training_sigmas']=all(
        x['action']==y['action'] and x['chunk']==y['chunk'] and
        [(a['sigma'],a['target_sigma']) for a in x['samples']]==[(a['sigma'],a['target_sigma']) for a in y['samples']]
        for x,y in zip(report['updates'],trained['updates']))
    (OUT/'gpu_matched_initialization.json').write_text(json.dumps(checks,indent=2)+'\n')
    if not all(checks.values()): raise RuntimeError('Training-shift matched initialization audit failed')
    STATE['training_runs'].append(dict(directory=str(directory),total_updates=total,
        wall_seconds=report['wall_seconds'],gpu_allocated_peak_MiB=report['gpu_allocated_peak_MiB']))
    save();return directory

def evaluate(checkpoint,step,action,nfe):
    directory=OUT/f'eval/anyflow/step_{step:02d}/generated_{nfe}step_native/{action}'
    command=[sys.executable,'-u',str(RUNTIME/'code/causal/benchmark.py'),
        '--modes','cached','--num-frames','39','--steps',str(nfe),'--seed','13',
        '--flow-shift','2.22','--anyflow-sigma-grid','native','--precision-profile','h3_fp32',
        '--chunk-frames','5','--history-chunks','5','--cache-device','cpu',
        '--action-preset',action,'--action-prefix-mode','causal','--action-feedback',
        '--anchor-mode','dynamic_last_frame_rgb_dual',
        '--causal-adapter',str(checkpoint/'causal_adapter.pt'),
        '--causal-action-adapter',str(checkpoint/'action_adapter.pt'),
        '--stage1-lora',str(checkpoint/'stage1_lora.pt'),
        '--anyflow-adapter',str(checkpoint/'anyflow_adapter.pt'),
        '--out-dir',str(directory),'--save-latents']
    child(f'shift12_full_history_fp32_step{step:02d}_{action}_{nfe}step',command,directory)
    runtime=json.loads((directory/'cached.json').read_text())
    setup=json.loads((directory/'setup.json').read_text())
    if (runtime['status']!='complete' or runtime['denoiser_forwards']!=nfe*3
            or runtime['commit_forwards']!=3 or runtime['video_sigma_grid']!='native'
            or setup['precision']['profile']!='h3_fp32'
            or not setup['precision']['native_fp32_weights_restored']
            or setup['stage1_lora']['trainable_parameters']!=43237376
            or setup['anyflow']['metadata']['config']['history_gradient_mode']!='full'
            or setup['anyflow']['metadata']['config']['training_timestep_shift']!=12.
            or setup['anyflow']['metadata']['config']['validation_timestep_shift']!=2.22
            or runtime['sampler']!='anyflow_finite_map'):
        raise RuntimeError('Native FP32 inference contract failed')
    video=directory/'cached.mp4';stats=evaluate_video(video);flow=evaluate_flow(video)
    if (stats['frames'],stats['fps'],stats['width'],stats['height'])!=(39,24,832,480):
        raise RuntimeError('Unexpected video format/count')
    fairness=audit_inputs(directory,TEACHER/f'action_{action}_teacher_39')
    if not fairness['exactly_equal']:
        raise RuntimeError('Saved inference conditioning differs from original H3')
    row=dict(objective='anyflow',history_gradient_mode='full',training_timestep_shift=12.,validation_timestep_shift=2.22,step=step,steps_per_chunk=nfe,action=action,history='generated',
        sigma_grid='native',precision_profile='h3_fp32',adapter_scope='all_qkvo_ffn',video=str(video),video_stats=stats,
        flow=flow,boundary=rgb_boundary(video,[17,34]),runtime=runtime,input_fairness=fairness)
    (directory/'evaluation.json').write_text(json.dumps(row,indent=2)+'\n')
    STATE['evaluations'].append(row);save()

if (OUT/'run.json').exists():
    raise SystemExit('Refusing duplicate pilot')
save()
try:
    verify()
    STATE['status']='running';save()
    trained=train(16)
    for nfe in (4,8):
        for action in 'AD':evaluate(trained/'step_16',16,action,nfe)
    STATE.update(status='complete',completed_at=datetime.now().astimezone().isoformat())
except BaseException as exc:
    STATE.update(status='failed',error=repr(exc));raise
finally:
    save()
