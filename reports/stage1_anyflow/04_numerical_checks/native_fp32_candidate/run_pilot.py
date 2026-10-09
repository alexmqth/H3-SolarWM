"""Finite native-FP32 Stage1 pilot, serial after the current GPU0 work.

Fresh same visual/action initialization; precision is an explicit protocol
change, never an exact resume from a legacy checkpoint. One update checks
GPU execution, then exact-resume its Adam/RNG to16, evaluate4/8 A/D.
"""
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
PROBE = ROOT / 'outputs/2026-10-08-05/stage1_anyflow39_fp32_inputs'
DURATION = ROOT / 'outputs/2026-10-08-05/stage1_anyflow39_gpu0_extend64_fullmem'
SOURCE = ROOT / 'outputs/2026-10-08-05/stage1_anyflow39_gpu0'
TEACHER = ROOT / 'outputs/2026-10-02-03'
sys.path.insert(0, str(RUNTIME / 'code/causal'))
from report_stage1_anyflow import audit_inputs
from evaluate_action_control import evaluate as evaluate_flow
from evaluate_videos import evaluate as evaluate_video
from summarize_action_experiment import rgb_boundary

ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='0', ABOT_VRAM_RESERVE_GIB='6',
    HF_HUB_OFFLINE='1', DIFFSYNTH_SKIP_DOWNLOAD='True', PYTHONUNBUFFERED='1',
    ABOT_DIFFSYNTH_ROOT=str(RUNTIME / 'DiffSynth-Studio-h3-v2'),
    DIFFSYNTH_ROOT=str(RUNTIME / 'DiffSynth-Studio-h3-v2'))
for key, name in [('HF_HOME','hf'),('TORCHINDUCTOR_CACHE_DIR','torchinductor'),
                  ('TRITON_CACHE_DIR','triton'),('XDG_CACHE_HOME','xdg')]:
    ENV[key] = str(ROOT / '.cache' / name)
STATE = dict(status='waiting_for_input_precision_probe', pid=os.getpid(), gpu=0,
    started_at=datetime.now().astimezone().isoformat(), active=None, precision_profile='h3_fp32',
    vram_reserve_gib=6, training_runs=[], evaluations=[], overall_gate='NOT_ACCEPTED')

def save():
    p=OUT/'run.tmp.json';p.write_text(json.dumps(STATE,indent=2)+'\n');p.replace(OUT/'run.json')

def live(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:
        return False

def wait():
    while True:
        s=json.loads((PROBE/'run.json').read_text());alive=live(s['pid'])
        if s['status']=='complete' and not alive:
            if len(s['records'])!=24 or len(s['restoration_checks'])!=8:
                raise RuntimeError('Incomplete FP32 input diagnostic')
            return
        if s['status'] not in ('waiting_for_duration_control','running','complete') or not alive:
            raise RuntimeError('Preceding input diagnostic did not finish cleanly')
        time.sleep(20)

def existing_numeric_pass():
    s=json.loads((DURATION/'run.json').read_text())
    if live(s['pid']) or s['status'] not in ('complete','stopped_at32_pending_visual_review','skipped_pending_visual_review','completed32_deferred64'):
        raise RuntimeError('Duration control has not exited cleanly')
    groups={}
    for row in s['evaluations']:
        key=row['step'],row['steps_per_chunk'],row['sigma_grid']
        groups.setdefault(key,{})[row['action']]=row['flow']['horizontal_flow_px']['mean']
    return any(set(g)=={'A','D'} and g['A']>0 and g['D']<0 and g['A']-g['D']>1 for g in groups.values())

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
    initial=json.loads((SOURCE/'train_anyflow/training.json').read_text())
    config=dict(initial['config'])
    directory=OUT/f'train_{total:02d}'
    start=0 if previous is None else 1
    config.update(out_dir=str(directory),steps=total,precision_profile='h3_fp32',
        resume_from=None if previous is None else str(previous/'step_01'),
        checkpoint_every=1 if total==1 else 4)
    command=[sys.executable,'-u',str(RUNTIME/'code/causal/train_stage1_anyflow.py')]
    for key,value in config.items():
        if value is None:continue
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:command.append(flag)
            elif key in ('action_feedback','train_target_time'):command.append('--no-'+key.replace('_','-'))
        elif isinstance(value,list):command += [flag,*map(str,value)]
        else:command += [flag,str(value)]
    child(f'native_fp32_train_{total:02d}',command,directory)
    report=json.loads((directory/'training.json').read_text())
    if (report['status']!='complete' or len(report['updates'])!=total
            or report['resumed_optimizer_step']!=start
            or report['target_time_parameters_changed'] or not report['qkv_parameters_changed']
            or report['target_time_trainable_parameters']!=0
            or report['precision']['profile']!='h3_fp32'
            or not report['precision']['native_fp32_weights_restored']
            or len(report['precision']['native_tensor_sha256'])!=12):
        raise RuntimeError('Native FP32 training contract failed')
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
        '--anyflow-adapter',str(checkpoint/'anyflow_adapter.pt'),
        '--out-dir',str(directory),'--save-latents']
    child(f'native_fp32_step{step:02d}_{action}_{nfe}step',command,directory)
    runtime=json.loads((directory/'cached.json').read_text())
    setup=json.loads((directory/'setup.json').read_text())
    if (runtime['status']!='complete' or runtime['denoiser_forwards']!=nfe*3
            or runtime['commit_forwards']!=3 or runtime['video_sigma_grid']!='native'
            or setup['precision']['profile']!='h3_fp32'
            or not setup['precision']['native_fp32_weights_restored']):
        raise RuntimeError('Native FP32 inference contract failed')
    video=directory/'cached.mp4';stats=evaluate_video(video);flow=evaluate_flow(video)
    if (stats['frames'],stats['fps'],stats['width'],stats['height'])!=(39,24,832,480):
        raise RuntimeError('Unexpected video format/count')
    fairness=audit_inputs(directory,TEACHER/f'action_{action}_teacher_39')
    if not fairness['exactly_equal']:
        raise RuntimeError('Saved inference conditioning differs from original H3')
    row=dict(objective='anyflow',step=step,steps_per_chunk=nfe,action=action,history='generated',
        sigma_grid='native',precision_profile='h3_fp32',video=str(video),video_stats=stats,
        flow=flow,boundary=rgb_boundary(video,[17,34]),runtime=runtime,input_fairness=fairness)
    (directory/'evaluation.json').write_text(json.dumps(row,indent=2)+'\n')
    STATE['evaluations'].append(row);save()

if (OUT/'run.json').exists():
    raise SystemExit('Refusing duplicate pilot')
save()
try:
    wait()
    if existing_numeric_pass():
        STATE.update(status='skipped_pending_visual_review',reason='Existing duration control passed a numeric gate; review its videos first')
    else:
        STATE['status']='running';save()
        smoke=train(1)
        for action in 'AD':evaluate(smoke/'step_00',0,action,4)
        trained=train(16,smoke)
        for nfe in (4,8):
            for action in 'AD':evaluate(trained/'step_16',16,action,nfe)
        STATE.update(status='complete',completed_at=datetime.now().astimezone().isoformat())
except BaseException as exc:
    STATE.update(status='failed',error=repr(exc));raise
finally:
    save()
