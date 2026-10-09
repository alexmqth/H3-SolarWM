"""Finite eight-update objective fork, then unchanged 8/4-step video evaluation."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
RUNTIME=OUT/'runtime'
SOURCE=ROOT/'outputs/2026-10-08-13/stage1_parallel_resume68_to128/train_128'
REFERENCE=OUT.parent/'stage1_finite_interval_refinement128'
VARIANT=sys.argv[1]
GPU=int(sys.argv[2])
if (VARIANT,GPU) not in [('control',6),('auxiliary',6)]:
    raise ValueError('Use the two reviewed objective forks')
WEIGHT=0. if VARIANT=='control' else .25
sys.path[:0]=[str(RUNTIME/'code/causal'),str(RUNTIME/'code')]
from evaluate_action_control import evaluate as evaluate_flow
from evaluate_videos import evaluate as evaluate_video
from summarize_action_experiment import rgb_boundary
from report_stage1_anyflow import audit_inputs

ENV=dict(os.environ,CUDA_VISIBLE_DEVICES=str(GPU),ABOT_VRAM_RESERVE_GIB='6',
    HF_HUB_OFFLINE='1',DIFFSYNTH_SKIP_DOWNLOAD='True',PYTHONUNBUFFERED='1',
    DIFFSYNTH_ROOT=str(RUNTIME/'DiffSynth-Studio-h3-v2'),
    ABOT_DIFFSYNTH_ROOT=str(RUNTIME/'DiffSynth-Studio-h3-v2'))
for key,suffix in [('HF_HOME','hf'),('TORCHINDUCTOR_CACHE_DIR','torchinductor'),
                   ('TRITON_CACHE_DIR','triton'),('XDG_CACHE_HOME','xdg')]:
    ENV[key]=str(ROOT/'.cache'/suffix)
STATE=dict(status='preflight',pid=os.getpid(),variant=VARIANT,gpu=GPU,
    started_at=datetime.now().astimezone().isoformat(),active=None,evaluations=[],
    maximum_new_updates=8,interval_consistency_weight=WEIGHT,
    overall_gate='NOT_REVIEWED',stage2=False)
PATH=OUT/f'run_{VARIANT}.json'


def save():
    temporary=PATH.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(STATE,indent=2)+'\n');temporary.replace(PATH)


def verify():
    for name,expected in json.loads((OUT/'auxiliary_manifest.json').read_text())['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Frozen input changed: {name}')
    for a in ('A','D'):
        r=json.loads((REFERENCE/f'probe_{a}.json').read_text())
        if r['status']!='complete' or len(r['records'])!=3 or not r['parameter_versions_unchanged']:
            raise RuntimeError('Completed reference required')


def child(label,command,directory):
    verify()
    used=int(subprocess.check_output(['nvidia-smi','-i',str(GPU),
        '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if used>=1024:
        raise RuntimeError(f'GPU{GPU} is busy ({used} MiB); no process interrupted')
    directory.mkdir(parents=True,exist_ok=False)
    with (directory/'run.log').open('w') as log:
        p=subprocess.Popen(command,cwd=RUNTIME,env=ENV,stdout=log,stderr=subprocess.STDOUT)
        STATE['active']=dict(label=label,pid=p.pid,command=command,directory=str(directory))
        save();code=p.wait()
    if code:
        raise RuntimeError(f'{label} exited {code}')
    STATE['active']=None;save()


def train():
    original=json.loads((SOURCE/'training.json').read_text())
    if original['status']!='complete' or len(original['updates'])!=128:
        raise RuntimeError('Expected completed128 source')
    config=dict(original['config'])
    directory=OUT/VARIANT/'train_136'
    config.update(out_dir=str(directory),resume_from=str(SOURCE/'step_128'),
        steps=136,checkpoint_every=4,device='cuda:0',
        interval_consistency_weight=WEIGHT,interval_reference_dir=str(REFERENCE))
    command=[sys.executable,'-u',str(RUNTIME/'code/causal/train_stage1_anyflow.py')]
    for key,value in config.items():
        if value is None:continue
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:command.append(flag)
            elif key in ('action_feedback','train_target_time'):
                command.append('--no-'+key.replace('_','-'))
        elif isinstance(value,list):command.extend([flag,*map(str,value)])
        else:command.extend([flag,str(value)])
    child('train_128_to136',command,directory)
    r=json.loads((directory/'training.json').read_text())
    if (r['status']!='complete' or len(r['updates'])!=136
            or r['updates'][:128]!=original['updates']
            or not r['frozen_visual_unchanged'] or r['target_time_parameters_changed']
            or not all(r['adapter_groups_changed'].values())):
        raise RuntimeError('Training protocol violated')
    STATE['training']=dict(directory=str(directory),new_updates=8,
        seconds=r['wall_seconds'],gpu_peak_MiB=r['gpu_allocated_peak_MiB'])
    save();return directory/'step_136'


def evaluate(checkpoint,action,nfe):
    directory=OUT/VARIANT/f'eval/{nfe}step/{action}'
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
    child(f'eval_{action}_{nfe}',command,directory)
    runtime=json.loads((directory/'cached.json').read_text())
    setup=json.loads((directory/'setup.json').read_text())
    if (runtime['status']!='complete' or runtime['sampler']!='anyflow_finite_map'
            or runtime['history_source']!='generated'
            or runtime['denoiser_forwards']!=nfe*3 or runtime['commit_forwards']!=3
            or setup['anyflow']['metadata']['config']['interval_consistency_weight']!=WEIGHT):
        raise RuntimeError('Evaluation sampler/protocol changed')
    video=directory/'cached.mp4'
    stats,flow=evaluate_video(video),evaluate_flow(video)
    if (stats['frames'],stats['fps'],stats['width'],stats['height'])!=(39,24,832,480):
        raise RuntimeError('Incomplete video')
    fairness=audit_inputs(directory,ROOT/f'outputs/2026-10-02-03/action_{action}_teacher_39')
    if not fairness['exactly_equal']:raise RuntimeError('Conditioning changed')
    row=dict(variant=VARIANT,objective='anyflow',objective_extension_weight=WEIGHT,
        step=136,steps_per_chunk=nfe,action=action,history='generated',video=str(video),
        video_stats=stats,flow=flow,boundary=rgb_boundary(video,[17,34]),runtime=runtime,
        input_fairness=fairness)
    (directory/'evaluation.json').write_text(json.dumps(row,indent=2)+'\n')
    STATE['evaluations'].append(row);save()


if PATH.exists():raise SystemExit('Refusing to duplicate a variant queue')
save()
try:
    verify();STATE['status']='running';save()
    checkpoint=train()
    for nfe in (8,4):
        for action in ('A','D'):evaluate(checkpoint,action,nfe)
    STATE.update(status='complete',completed_at=datetime.now().astimezone().isoformat())
except BaseException as exc:
    STATE.update(status='failed',error=repr(exc));raise
finally:
    save()
