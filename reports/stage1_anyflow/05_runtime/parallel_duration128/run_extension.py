"""Four-GPU same-batch duration candidate; source64 must finish first.

Wait for its complete 64-update evaluation. If any A/D configuration passes
the numeric short gate, defer to visual review rather than train needlessly.
Otherwise validate real parallel training to68, then96/eval8, at most128/eval4+8. This is a finite
duration-only experiment, not automatic visual acceptance or Stage2 work.
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
DEPENDENCY = ROOT / 'outputs/2026-10-08-10/stage1_shift12_duration64'
RUNTIME = OUT / 'training_runtime'
TEACHER = ROOT / 'outputs/2026-10-02-03'
sys.path.insert(0, str(RUNTIME / 'code/causal'))
from evaluate_action_control import evaluate as evaluate_flow
from evaluate_videos import evaluate as evaluate_video
from summarize_action_experiment import rgb_boundary
from report_stage1_anyflow import audit_inputs
from training_state import load_training_state

ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='3,4,5,6', ABOT_VRAM_RESERVE_GIB='6',
           DIFFSYNTH_SKIP_DOWNLOAD='True', HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1')
for key, suffix in [('HF_HOME', 'hf'), ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor'),
                    ('TRITON_CACHE_DIR', 'triton'), ('XDG_CACHE_HOME', 'xdg')]:
    ENV[key] = str(ROOT / '.cache' / suffix)
ENV['DIFFSYNTH_ROOT'] = ENV['ABOT_DIFFSYNTH_ROOT'] = str(RUNTIME / 'DiffSynth-Studio-h3-v2')
STATE = dict(status='waiting_for_shift12_64_evaluation', pid=os.getpid(), gpus=[3,4,5,6],
    started_at=datetime.now().astimezone().isoformat(), active=None, vram_reserve_gib=6,
    training_runs=[], evaluations=[], overall_gate='NOT_ACCEPTED',
    history_gradient_mode='full', precision_profile='h3_fp32', adapter_scope='all_qkvo_ffn',
    training_timestep_shift=12.0, validation_timestep_shift=2.22, inference_flow_shift=2.22)


def save():
    temporary = OUT / 'run.tmp.json'
    temporary.write_text(json.dumps(STATE, indent=2) + '\n')
    temporary.replace(OUT / 'run.json')


def live_process(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def wait_for_dependency():
    while True:
        previous = json.loads((DEPENDENCY / 'run.json').read_text())
        live = live_process(previous['pid'])
        if previous['status'] == 'complete' and not live:
            return previous
        if previous['status'] not in ('running', 'complete') or not live:
            raise RuntimeError('Frozen64 did not finish cleanly; review before continuation')
        STATE['dependency_wait'] = dict(checked_at=datetime.now().astimezone().isoformat(),
            pid=previous['pid'], status=previous['status'], active=(previous.get('active') or {}).get('label'))
        save()
        time.sleep(20)


def verify_sources():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen dependency changed: {name}')


def any_numeric_pass(evaluations):
    pairs = {}
    for r in evaluations:
        key = r['steps_per_chunk'], r['sigma_grid']
        pairs.setdefault(key, {})[r['action']] = r['flow']['horizontal_flow_px']['mean']
    expected = {(4, 'native'), (8, 'native')}
    if set(pairs) != expected or any(set(v) != {'A', 'D'} for v in pairs.values()):
        raise RuntimeError('Frozen64 evaluation is incomplete; refusing to infer gate failure')
    return any(p['A'] > 0 and p['D'] < 0 and p['A'] - p['D'] > 1. for p in pairs.values())


def train_config(source_directory, total):
    source = json.loads((source_directory / 'training.json').read_text())
    config = dict(source['config'])
    config.update(out_dir=str(OUT / f'train_{total:02d}'),
                  resume_from=str(source_directory / f"step_{len(source['updates']):02d}"),
                  steps=total, checkpoint_every=4 if total == 68 else 16)
    return source, config


def training_command(config):
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4', str(OUT / 'run_trainer.py')]
    for key, value in config.items():
        if value is None:
            continue
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(flag)
            elif key in ('action_feedback', 'train_target_time'):
                command.append('--no-' + key.replace('_', '-'))
        elif isinstance(value, list):
            command += [flag, *map(str, value)]
        else:
            command += [flag, str(value)]
    return command


def run_child(label, command, directory):
    verify_sources()
    parallel = 'torch.distributed.run' in command
    selected = [3,4,5,6] if parallel else [3]
    used = {gpu:int(subprocess.check_output(['nvidia-smi','-i',str(gpu),
        '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()) for gpu in selected}
    if any(value >= 1024 for value in used.values()):
        raise RuntimeError(f'Required GPUs no longer idle: {used}; no process interrupted')
    child_env = ENV if parallel else dict(ENV,CUDA_VISIBLE_DEVICES='3')
    directory.mkdir(parents=True, exist_ok=False)
    print('START', label, flush=True)
    with (directory / 'run.log').open('w') as log:
        child = subprocess.Popen(command, cwd=RUNTIME, env=child_env, stdout=log, stderr=subprocess.STDOUT)
        STATE['active'] = dict(label=label, pid=child.pid, command=command, directory=str(directory))
        save()
        code = child.wait()
    if code:
        raise RuntimeError(f'{label} exited {code}')
    STATE['active'] = None
    save()


def train(source_directory, total):
    source, config = train_config(source_directory, total)
    if source['status'] != 'complete' or source['target_time_trainable']:
        raise RuntimeError('Continuation requires completed frozen-time AnyFlow training')
    saved = load_training_state(Path(config['resume_from']), config)
    if saved['updates'] != source['updates']:
        raise RuntimeError('Optimizer checkpoint and source training history disagree')
    directory = Path(config['out_dir'])
    run_child(f'shift12_full_history_resume_to{total:02d}', training_command(config), directory)
    metrics = json.loads((directory / 'training.json').read_text())
    start = len(source['updates'])
    if (metrics['status'] != 'complete' or len(metrics['updates']) != total
            or metrics['resumed_optimizer_step'] != start
            or metrics['updates_this_run'] != total - start
            or metrics['updates'][:start] != source['updates']
            or not metrics['qkv_parameters_changed'] or metrics['target_time_parameters_changed']
            or metrics['target_time_trainable_parameters'] != 0
            or not metrics['frozen_visual_unchanged']
            or not all(metrics['adapter_groups_changed'].values())
            or metrics['stage1_lora']['trainable_parameters'] != 43237376
            or metrics['config']['history_gradient_mode'] != 'full'
            or metrics['time_sampling'] != dict(training_shift=12.,validation_shift=2.22,inference_flow_shift=2.22)
            or any(x['differentiable_history_forwards'] != x['chunk'] * 4 for x in metrics['updates'])):
        raise RuntimeError('Continuation violated its optimizer-step or trainability contract')
    from audit_resume import audit
    audit(Path(config['resume_from']), directory / f'step_{start:02d}', directory / 'resume_audit.json')
    if not all(metrics['replica_audit']['equality'].values()):
        raise RuntimeError('Replicas differ in model/Adam/RNG')
    STATE['training_runs'].append(dict(replica_audit=metrics['replica_audit'],directory=str(directory), resumed_step=start, total_updates=total,
        wall_seconds=metrics['wall_seconds'], allocated_peak_MiB=metrics['gpu_allocated_peak_MiB']))
    save()
    return directory


def evaluate(checkpoint, step, action, nfe):
    directory = OUT / f'eval/anyflow/step_{step:02d}/generated_{nfe}step_native/{action}'
    command = [sys.executable, '-u', str(RUNTIME / 'code/causal/benchmark.py'),
        '--modes', 'cached', '--num-frames', '39', '--steps', str(nfe), '--seed', '13',
        '--flow-shift', '2.22', '--anyflow-sigma-grid', 'native',
        '--precision-profile', 'h3_fp32',
        '--chunk-frames', '5', '--history-chunks', '5', '--cache-device', 'cpu',
        '--action-preset', action, '--action-prefix-mode', 'causal', '--action-feedback',
        '--anchor-mode', 'dynamic_last_frame_rgb_dual',
        '--causal-adapter', str(checkpoint / 'causal_adapter.pt'),
        '--causal-action-adapter', str(checkpoint / 'action_adapter.pt'),
        '--stage1-lora', str(checkpoint / 'stage1_lora.pt'),
        '--anyflow-adapter', str(checkpoint / 'anyflow_adapter.pt'),
        '--out-dir', str(directory), '--save-latents']
    run_child(f'shift12_full_history_step{step:02d}_{action}_{nfe}step_native', command, directory)
    runtime = json.loads((directory / 'cached.json').read_text())
    if (runtime['status'] != 'complete' or runtime['video_sigma_grid'] != 'native'
            or runtime['denoiser_forwards'] != nfe * 3 or runtime['commit_forwards'] != 3):
        raise RuntimeError('Incorrect output status/grid/forward counts')
    setup = json.loads((directory / 'setup.json').read_text())
    if (setup['precision']['profile'] != 'h3_fp32'
            or not setup['precision']['native_fp32_weights_restored']
            or setup['stage1_lora']['trainable_parameters'] != 43237376
            or setup['anyflow']['metadata']['config']['history_gradient_mode'] != 'full'
            or setup['anyflow']['metadata']['config']['training_timestep_shift'] != 12.
            or setup['anyflow']['metadata']['config']['validation_timestep_shift'] != 2.22
            or runtime['sampler'] != 'anyflow_finite_map'):
        raise RuntimeError('Full-history inference protocol mismatch')
    video = directory / 'cached.mp4'
    stats, flow = evaluate_video(video), evaluate_flow(video)
    if (stats['frames'], stats['fps'], stats['width'], stats['height']) != (39, 24, 832, 480):
        raise RuntimeError('Video protocol mismatch')
    fairness = audit_inputs(directory, TEACHER / f'action_{action}_teacher_39')
    if not fairness['exactly_equal']:
        raise RuntimeError('Conditioning changed during duration-only continuation')
    row = dict(objective='anyflow', step=step, steps_per_chunk=nfe, action=action,
        history='generated', sigma_grid='native', history_gradient_mode='full',
        precision_profile='h3_fp32', adapter_scope='all_qkvo_ffn', training_timestep_shift=12., video=str(video), video_stats=stats,
        flow=flow, boundary=rgb_boundary(video, [17, 34]), runtime=runtime, input_fairness=fairness)
    (directory / 'evaluation.json').write_text(json.dumps(row, indent=2) + '\n')
    STATE['evaluations'].append(row)
    save()


def main():
    if (OUT / 'run.json').exists():
        raise SystemExit('Refusing duplicate continuation queue')
    save()
    try:
        dependency = wait_for_dependency()
        if any_numeric_pass([e for e in dependency['evaluations'] if e['step']==64]):
            STATE.update(status='skipped_pending_visual_review',
                reason='Frozen64 passed a short numeric gate; review its actual videos before further training')
            return
        verify_sources()
        STATE['status'] = 'running'
        save()
        # Four real optimizer updates establish memory, all-replica Adam/RNG,
        # and exact source checkpoint restoration before a longer allocation.
        train68 = train(DEPENDENCY / 'train_64', 68)
        train96 = train(train68, 96)
        for action in 'AD':
            evaluate(train96 / 'step_96', 96, action, 8)
        pair = {r['action']:r['flow']['horizontal_flow_px']['mean'] for r in STATE['evaluations'] if r['step']==96}
        if pair['A'] > 0 and pair['D'] < 0 and pair['A']-pair['D'] > 1.:
            for action in 'AD':
                evaluate(train96 / 'step_96', 96, action, 4)
            STATE.update(status='stopped_at96_pending_visual_review',
                reason='Numeric8 gate passed; full visual, matched FM, independent seed and temporal tests still required')
            return
        train128 = train(train96, 128)
        for nfe in (4,8):
            for action in 'AD':
                evaluate(train128 / 'step_128', 128, action, nfe)
        STATE.update(status='complete', completed_at=datetime.now().astimezone().isoformat())
    except BaseException as exc:
        STATE.update(status='failed', error=repr(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
