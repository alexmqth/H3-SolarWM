"""Evaluate the saved full-history step04 on free GPU1 while GPU0 trains."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PILOT = ROOT / 'outputs/2026-10-08-08/stage1_anyflow39_full_history'
PROBE = ROOT / 'outputs/2026-10-08-09/stage1_gradient_repeatability'
RUNTIME = PILOT / 'runtime'
CKPT = PILOT / 'train_16/step_04'
TEACHER = ROOT / 'outputs/2026-10-02-03'
sys.path.insert(0, str(RUNTIME / 'code/causal'))
from report_stage1_anyflow import audit_inputs
from evaluate_action_control import evaluate as evaluate_flow
from evaluate_videos import evaluate as evaluate_video
from summarize_action_experiment import rgb_boundary

STATE = dict(status='waiting_for_gpu1_probe', pid=os.getpid(), gpu=1,
             started_at=datetime.now().astimezone().isoformat(), active=None,
             optimizer_step=4, evaluations=[], overall_gate='NOT_ACCEPTED')
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='1', ABOT_VRAM_RESERVE_GIB='6',
           HF_HUB_OFFLINE='1', DIFFSYNTH_SKIP_DOWNLOAD='True', PYTHONUNBUFFERED='1',
           ABOT_DIFFSYNTH_ROOT=str(RUNTIME/'DiffSynth-Studio-h3-v2'),
           DIFFSYNTH_ROOT=str(RUNTIME/'DiffSynth-Studio-h3-v2'))
for key, name in [('HF_HOME','hf'),('TORCHINDUCTOR_CACHE_DIR','torchinductor'),
                  ('TRITON_CACHE_DIR','triton'),('XDG_CACHE_HOME','xdg')]:
    ENV[key] = str(ROOT/'.cache'/name)

def save():
    p=OUT/'run.tmp.json';p.write_text(json.dumps(STATE,indent=2)+'\n');p.replace(OUT/'run.json')

def live(pid):
    try: return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError: return False

def verify():
    for p, sha in json.loads((OUT/'manifest.json').read_text())['files'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha:
            raise RuntimeError(f'Frozen input changed: {p}')

if (OUT/'run.json').exists(): raise SystemExit('Refusing duplicate controller')
save()
try:
    pid=json.loads((PROBE/'launch.json').read_text())['pid']
    while live(pid): time.sleep(20)
    receipt=json.loads((PROBE/'gpu_probe/probe.json').read_text())
    if receipt['status']!='complete': raise RuntimeError('GPU1 probe did not complete')
    STATE['status']='running';save()
    for action in 'AD':
        verify()
        directory=PILOT/f'eval/anyflow/step_04/generated_8step_native/{action}'
        directory.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,'-u',str(RUNTIME/'code/causal/benchmark.py'),
            '--modes','cached','--num-frames','39','--steps','8','--seed','13',
            '--flow-shift','2.22','--anyflow-sigma-grid','native','--precision-profile','h3_fp32',
            '--chunk-frames','5','--history-chunks','5','--cache-device','cpu',
            '--action-preset',action,'--action-prefix-mode','causal','--action-feedback',
            '--anchor-mode','dynamic_last_frame_rgb_dual',
            '--causal-adapter',str(CKPT/'causal_adapter.pt'),
            '--causal-action-adapter',str(CKPT/'action_adapter.pt'),
            '--stage1-lora',str(CKPT/'stage1_lora.pt'),
            '--anyflow-adapter',str(CKPT/'anyflow_adapter.pt'),
            '--out-dir',str(directory),'--save-latents']
        with (directory/'run.log').open('w') as log:
            p=subprocess.Popen(command,cwd=RUNTIME,env=ENV,stdout=log,stderr=subprocess.STDOUT)
            STATE['active']=dict(action=action,pid=p.pid,directory=str(directory),command=command);save()
            code=p.wait()
        if code: raise RuntimeError(f'{action} benchmark exited {code}')
        runtime=json.loads((directory/'cached.json').read_text())
        setup=json.loads((directory/'setup.json').read_text())
        if (runtime['status']!='complete' or runtime['denoiser_forwards']!=24
                or runtime['commit_forwards']!=3 or runtime['video_sigma_grid']!='native'
                or setup['precision']['profile']!='h3_fp32'
                or setup['stage1_lora']['trainable_parameters']!=43237376
                or setup['anyflow']['metadata']['optimizer_step']!=4
                or setup['anyflow']['metadata']['config']['history_gradient_mode']!='full'
                or runtime['sampler']!='anyflow_finite_map'):
            raise RuntimeError('Checkpoint or runtime protocol mismatch')
        video=directory/'cached.mp4';stats=evaluate_video(video);flow=evaluate_flow(video)
        if (stats['frames'],stats['fps'],stats['width'],stats['height'])!=(39,24,832,480):
            raise RuntimeError('Incomplete/unexpected video')
        fairness=audit_inputs(directory,TEACHER/f'action_{action}_teacher_39')
        if not fairness['exactly_equal']: raise RuntimeError('Conditioning mismatch')
        row=dict(objective='anyflow',history_gradient_mode='full',step=4,steps_per_chunk=8,
                 action=action,history='generated',sigma_grid='native',precision_profile='h3_fp32',
                 adapter_scope='all_qkvo_ffn',video=str(video),video_stats=stats,flow=flow,
                 boundary=rgb_boundary(video,[17,34]),runtime=runtime,input_fairness=fairness,
                 timing_caveat='GPU1 evaluation concurrent with GPU0 training; no speedup claim')
        (directory/'evaluation.json').write_text(json.dumps(row,indent=2)+'\n')
        STATE['evaluations'].append(row);STATE['active']=None;save()
    STATE.update(status='complete',completed_at=datetime.now().astimezone().isoformat())
except BaseException as exc:
    STATE.update(status='failed',error=repr(exc));raise
finally: save()
