"""Read-only generated-history 4-step evaluation of shift12 step32; matched FM32 control."""
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
SOURCE = ROOT / 'outputs/2026-10-08-10/stage1_shift12_duration64'
PREVIOUS = ROOT / 'outputs/2026-10-08-09/stage1_training_shift12'
RUNTIME = PREVIOUS / 'runtime'
TEACHER = ROOT / 'outputs/2026-10-02-03'
sys.path.insert(0, str(RUNTIME / 'code/causal'))
from report_stage1_anyflow import audit_inputs
from evaluate_action_control import evaluate as evaluate_flow
from evaluate_videos import evaluate as evaluate_video
from summarize_action_experiment import rgb_boundary

ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='4', ABOT_VRAM_RESERVE_GIB='6',
           HF_HUB_OFFLINE='1', DIFFSYNTH_SKIP_DOWNLOAD='True', PYTHONUNBUFFERED='1',
           ABOT_DIFFSYNTH_ROOT=str(RUNTIME/'DiffSynth-Studio-h3-v2'),
           DIFFSYNTH_ROOT=str(RUNTIME/'DiffSynth-Studio-h3-v2'))
for key, suffix in [('HF_HOME','hf'), ('TORCHINDUCTOR_CACHE_DIR','torchinductor'),
                    ('TRITON_CACHE_DIR','triton'), ('XDG_CACHE_HOME','xdg')]:
    ENV[key] = str(ROOT/'.cache'/suffix)
STATE = dict(status='waiting_for_shift12_training', pid=os.getpid(), gpu=4,
    started_at=datetime.now().astimezone().isoformat(), active=None,
    history_source='generated', training_timestep_shift=12., inference_flow_shift=2.22,
    overall_gate='NOT_ACCEPTED_PENDING_VIDEO_REVIEW', evaluations=[])


def save():
    p = OUT/'run.tmp.json'
    p.write_text(json.dumps(STATE, indent=2)+'\n')
    p.replace(OUT/'run.json')


def live(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def verify():
    for path, expected in json.loads((OUT/'manifest.json').read_text())['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen source/input changed: {path}')


def wait_for_training():
    while True:
        run = json.loads((SOURCE/'run.json').read_text())
        training = json.loads((SOURCE/'train_32/training.json').read_text())
        audit = PREVIOUS/'gpu_matched_initialization.json'
        if training['status'] == 'complete' and audit.exists():
            if not all(json.loads(audit.read_text()).values()):
                raise RuntimeError('Shift12 initialization/schedule audit failed')
            checkpoint = SOURCE/'train_32/step_32'
            hashes = {name: hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()
                      for name in ('causal_adapter.pt','action_adapter.pt',
                                   'stage1_lora.pt','anyflow_adapter.pt')}
            STATE['checkpoint_receipt'] = dict(path=str(checkpoint), hashes=hashes,
                optimizer_steps=len(training['updates']), audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest())
            if len(training['updates']) != 32:
                raise RuntimeError('Unexpected update count')
            cfg = training['config']
            if (cfg['objective'] != 'anyflow' or cfg['history_gradient_mode'] != 'full'
                    or cfg['training_timestep_shift'] != 12.0 or cfg['validation_timestep_shift'] != 2.22
                    or training['target_time_trainable'] or training['stage1_lora']['trainable_parameters'] != 43237376):
                raise RuntimeError('Unexpected source training protocol')
            STATE['training_sha256'] = hashlib.sha256((SOURCE/'train_32/training.json').read_bytes()).hexdigest()
            save()
            return checkpoint
        if training['status'] == 'failed' or run['status'] == 'failed' or not live(run['pid']):
            raise RuntimeError('Shift12 training did not finish successfully')
        STATE['dependency_wait'] = dict(checked_at=datetime.now().astimezone().isoformat(), pid=run['pid'], training_status=training['status'], observed_updates=len(training['updates']))
        save()
        time.sleep(20)


def evaluate(checkpoint, action):
    verify()
    for name, expected in STATE['checkpoint_receipt']['hashes'].items():
        if hashlib.sha256((checkpoint/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Checkpoint changed after training')
    used = int(subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=memory.used',
                                        '--format=csv,noheader,nounits'], text=True).strip())
    if used >= 1024:
        raise RuntimeError(f'GPU4 no longer idle ({used} MiB); no process interrupted')
    directory = OUT/f'eval/anyflow/step_32/generated_4step_native/{action}'
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', str(RUNTIME/'code/causal/benchmark.py'),
        '--modes','cached','--num-frames','39','--steps','4','--seed','13',
        '--flow-shift','2.22','--anyflow-sigma-grid','native','--precision-profile','h3_fp32',
        '--chunk-frames','5','--history-chunks','5','--cache-device','cpu',
        '--action-preset',action,'--action-prefix-mode','causal','--action-feedback',
        '--anchor-mode','dynamic_last_frame_rgb_dual','--history-source','generated',
        '--causal-adapter',str(checkpoint/'causal_adapter.pt'),
        '--causal-action-adapter',str(checkpoint/'action_adapter.pt'),
        '--stage1-lora',str(checkpoint/'stage1_lora.pt'),
        '--anyflow-adapter',str(checkpoint/'anyflow_adapter.pt'),
        '--out-dir',str(directory),'--save-latents']
    with (directory/'run.log').open('w') as log:
        p = subprocess.Popen(command, cwd=RUNTIME, env=ENV, stdout=log, stderr=subprocess.STDOUT)
        STATE['active'] = dict(label=f'shift12_step32_{action}_generated4', pid=p.pid,
                              command=command, directory=str(directory))
        save()
        code = p.wait()
    if code:
        raise RuntimeError(f'{action} generated-history evaluation exited {code}')
    STATE['active'] = None
    runtime = json.loads((directory/'cached.json').read_text())
    setup = json.loads((directory/'setup.json').read_text())
    if (runtime['status'] != 'complete' or runtime['history_source'] != 'generated'
            or runtime['denoiser_forwards'] != 12 or runtime['commit_forwards'] != 3
            or runtime['video_sigma_grid'] != 'native' or runtime['sampler'] != 'anyflow_finite_map'
            or setup['precision']['profile'] != 'h3_fp32'
            or setup['anyflow']['metadata']['config']['training_timestep_shift'] != 12.
            or setup['stage1_lora']['trainable_parameters'] != 43237376):
        raise RuntimeError('Inference contract failed')
    video = directory/'cached.mp4'
    stats, flow = evaluate_video(video), evaluate_flow(video)
    if (stats['frames'], stats['fps'], stats['width'], stats['height']) != (39,24,832,480):
        raise RuntimeError('Incomplete or different video format')
    fairness = audit_inputs(directory, TEACHER/f'action_{action}_teacher_39')
    if not fairness['exactly_equal']:
        raise RuntimeError('Initial conditioning differs from teacher')
    row = dict(objective='anyflow', step=32, steps_per_chunk=4, action=action,
        history='generated', history_gradient_mode='full', sigma_grid='native',
        precision_profile='h3_fp32', adapter_scope='all_qkvo_ffn', training_timestep_shift=12.,
        video=str(video), video_stats=stats, flow=flow,
        boundary=rgb_boundary(video,[17,34]), runtime=runtime, input_fairness=fairness)
    (directory/'evaluation.json').write_text(json.dumps(row,indent=2)+'\n')
    STATE['evaluations'].append(row)
    save()


def main():
    if (OUT/'run.json').exists():
        raise SystemExit('Refusing duplicate evaluation queue')
    save()
    try:
        verify()
        checkpoint = wait_for_training()
        STATE['status'] = 'running'
        save()
        for action in 'AD':
            evaluate(checkpoint, action)
        STATE.update(status='complete', completed_at=datetime.now().astimezone().isoformat())
    except BaseException as exc:
        STATE.update(status='failed',error=repr(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
