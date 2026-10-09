"""GPU0 exact continuation of frozen-time AnyFlow after a saved-checkpoint migration."""
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
SOURCE = ROOT / 'outputs/2026-10-08-04/stage1_anyflow39_frozen_time'
RUNTIME = SOURCE / 'runtime'
PILOT = ROOT / 'outputs/2026-10-08-02/stage1_anyflow39_pilot'
DEPENDENCY = ROOT / 'outputs/2026-10-08-03/stage1_anyflow39_uniform'
INIT = ROOT / 'outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2'
TEACHER = ROOT / 'outputs/2026-10-02-03'
sys.path.insert(0, str(RUNTIME / 'code/causal'))
from evaluate_action_control import evaluate as evaluate_flow
from evaluate_videos import evaluate as evaluate_video
from summarize_action_experiment import rgb_boundary
from report_stage1_anyflow import audit_inputs
from training_state import load_training_state

ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='0', ABOT_VRAM_RESERVE_GIB='20',
           DIFFSYNTH_SKIP_DOWNLOAD='True', HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1')
for key, suffix in [('HF_HOME', 'hf'), ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor'),
                    ('TRITON_CACHE_DIR', 'triton'), ('XDG_CACHE_HOME', 'xdg')]:
    ENV[key] = str(ROOT / '.cache' / suffix)
ENV['DIFFSYNTH_ROOT'] = ENV['ABOT_DIFFSYNTH_ROOT'] = str(ROOT / 'DiffSynth-Studio-h3-v2')
STATE = dict(status='validating_gpu0_migration', pid=os.getpid(), gpu=0,
             started_at=datetime.now().astimezone().isoformat(), active=None,
             evaluations=[], overall_gate='NOT_ACCEPTED')


def save():
    temporary = OUT / 'run.tmp.json'
    temporary.write_text(json.dumps(STATE, indent=2) + '\n')
    temporary.replace(OUT / 'run.json')


def live_process(pid):
    file = Path(f'/proc/{pid}/stat')
    try:
        return file.read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def wait_for_pilot():
    receipt = json.loads((OUT / 'migration_receipt.json').read_text())
    for pid in receipt['stopped_pids']:
        if live_process(pid):
            raise RuntimeError(f'Old GPU2 process {pid} is still alive')
    if receipt['gpu_after'] != 0:
        raise RuntimeError('Incorrect migration device')
    STATE['migration'] = receipt
    save()


def verify_sources():
    manifest = json.loads((OUT / 'runtime_manifest.json').read_text())
    for rel, expected in manifest['hashes'].items():
        if hashlib.sha256((RUNTIME / rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Isolated runtime changed: {rel}')
    for path, expected in manifest['shared_files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Shared dependency changed: {path}')


def evaluate(step, action, nfe, grid):
    verify_sources()
    checkpoint = OUT / f'train_anyflow/step_{step:02d}'
    directory = OUT / f'eval/anyflow/step_{step:02d}/generated_{nfe}step_{grid}/{action}'
    directory.mkdir(parents=True, exist_ok=False)
    args = [sys.executable, '-u', str(RUNTIME / 'code/causal/benchmark.py'),
        '--modes', 'cached', '--num-frames', '39', '--steps', str(nfe), '--seed', '13',
        '--flow-shift', '2.22', '--anyflow-sigma-grid', grid,
        '--chunk-frames', '5', '--history-chunks', '5', '--cache-device', 'cpu',
        '--action-preset', action, '--action-prefix-mode', 'causal', '--action-feedback',
        '--anchor-mode', 'dynamic_last_frame_rgb_dual',
        '--causal-adapter', str(checkpoint / 'causal_adapter.pt'),
        '--causal-action-adapter', str(checkpoint / 'action_adapter.pt'),
        '--anyflow-adapter', str(checkpoint / 'anyflow_adapter.pt'),
        '--out-dir', str(directory), '--save-latents']
    label = f'frozen_time_step{step:02d}_{action}_{nfe}step_{grid}'
    print('START', label, flush=True)
    with (directory / 'run.log').open('w') as log:
        child = subprocess.Popen(args, cwd=RUNTIME, env=ENV, stdout=log, stderr=subprocess.STDOUT)
        STATE['active'] = dict(label=label, pid=child.pid, command=args, directory=str(directory))
        save()
        code = child.wait()
    if code:
        raise RuntimeError(f'{label} exited {code}')
    runtime = json.loads((directory / 'cached.json').read_text())
    if runtime['status'] != 'complete' or runtime['video_sigma_grid'] != grid:
        raise RuntimeError('Incomplete output or incorrect finite-map grid')
    if runtime['denoiser_forwards'] != nfe * 3 or runtime['commit_forwards'] != 3:
        raise RuntimeError('Incorrect noisy/clean forward count')
    video = directory / 'cached.mp4'
    stats, flow = evaluate_video(video), evaluate_flow(video)
    if (stats['frames'], stats['fps'], stats['width'], stats['height']) != (39, 24, 832, 480):
        raise RuntimeError('Video protocol mismatch')
    fairness = audit_inputs(directory, TEACHER / f'action_{action}_teacher_39')
    if not fairness['exactly_equal']:
        raise RuntimeError('Conditioning changed in the frozen-time ablation')
    row = dict(objective='anyflow', step=step, steps_per_chunk=nfe, action=action,
        history='generated', sigma_grid=grid, video=str(video), video_stats=stats,
        flow=flow, boundary=rgb_boundary(video, [17, 34]), runtime=runtime, input_fairness=fairness)
    (directory / 'evaluation.json').write_text(json.dumps(row, indent=2) + '\n')
    STATE['evaluations'].append(row)
    STATE['active'] = None
    save()


def train():
    verify_sources()
    receipt = json.loads((OUT / 'migration_receipt.json').read_text())
    source = json.loads((SOURCE / 'train_anyflow/training.json').read_text())
    directory = OUT / 'train_anyflow'
    config = dict(source['config'])
    config.update(out_dir=str(directory), resume_from=receipt['checkpoint_path'])
    saved = load_training_state(Path(config['resume_from']), config)
    step = saved['optimizer_step']
    if step != receipt['checkpoint_step'] or saved['updates'] != source['updates'][:step]:
        raise RuntimeError('Migration checkpoint does not match recorded training history')
    args = [sys.executable, '-u', str(RUNTIME / 'code/causal/train_stage1_anyflow.py')]
    for key, value in config.items():
        if value is None:
            continue
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                args.append(flag)
            elif key in ('action_feedback', 'train_target_time'):
                args.append('--no-' + key.replace('_', '-'))
        elif isinstance(value, list):
            args += [flag, *map(str, value)]
        else:
            args += [flag, str(value)]
    directory.mkdir(parents=True, exist_ok=False)
    print(f'START GPU0 exact resume from{step} to16', flush=True)
    with (directory / 'run.log').open('w') as log:
        child = subprocess.Popen(args, cwd=RUNTIME, env=ENV, stdout=log, stderr=subprocess.STDOUT)
        STATE['active'] = dict(label='gpu0_frozen_time_resume_to16', pid=child.pid,
                              command=args, directory=str(directory))
        save()
        code = child.wait()
    if code:
        raise RuntimeError(f'GPU0 resumed training exited {code}')
    metrics = json.loads((directory / 'training.json').read_text())
    if (metrics['status'] != 'complete' or not metrics['qkv_parameters_changed']
            or metrics['target_time_parameters_changed'] or metrics['target_time_trainable_parameters'] != 0
            or metrics['resumed_optimizer_step'] != step or len(metrics['updates']) != 16
            or metrics['updates'][:step] != saved['updates'] or metrics['updates_this_run'] != 16-step):
        raise RuntimeError('GPU0 continuation violated the resume/frozen-time contract')
    STATE['training'] = {k: metrics[k] for k in ('status', 'trainable_parameters',
        'target_time_parameters_changed', 'qkv_parameters_changed', 'wall_seconds',
        'gpu_allocated_peak_MiB', 'resumed_optimizer_step', 'updates_this_run')}
    STATE['active'] = None
    save()


if (OUT / 'run.json').exists():
    raise SystemExit('Refusing duplicate queue')
save()
try:
    wait_for_pilot()
    STATE['status'] = 'running'
    save()
    train()
    for nfe, grid in ((4, 'native'), (8, 'native'), (4, 'uniform')):
        for action in 'AD':
            evaluate(16, action, nfe, grid)
    STATE.update(status='complete', completed_at=datetime.now().astimezone().isoformat())
except BaseException as exc:
    STATE.update(status='failed', error=repr(exc))
    raise
finally:
    save()
