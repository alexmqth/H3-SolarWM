"""Read-only action switch at chunk1 with the same teacher prefix and checkpoint128."""
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
SOURCE = ROOT / 'outputs/2026-10-08-13/stage1_parallel_resume68_to128'
ACTION = sys.argv[1]
GPU = int(sys.argv[2])
if (ACTION, GPU) not in [('A_to_D',5),('D_to_A',6)]:
    raise ValueError('Only the two frozen counterfactual jobs are allowed')
HISTORY_ACTION, CURRENT_ACTION = ACTION.split('_to_')
RUNTIME = SOURCE / 'training_runtime'
TEACHER = ROOT / 'outputs/2026-10-02-03'
sys.path.insert(0, str(RUNTIME / 'code/causal'))
from report_stage1_anyflow import audit_inputs
from evaluate_action_control import evaluate as evaluate_flow
from evaluate_videos import evaluate as evaluate_video
from summarize_action_experiment import rgb_boundary

ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU), ABOT_VRAM_RESERVE_GIB='6',
           HF_HUB_OFFLINE='1', DIFFSYNTH_SKIP_DOWNLOAD='True', PYTHONUNBUFFERED='1',
           ABOT_DIFFSYNTH_ROOT=str(RUNTIME/'DiffSynth-Studio-h3-v2'),
           DIFFSYNTH_ROOT=str(RUNTIME/'DiffSynth-Studio-h3-v2'))
for key, suffix in [('HF_HOME','hf'), ('TORCHINDUCTOR_CACHE_DIR','torchinductor'),
                    ('TRITON_CACHE_DIR','triton'), ('XDG_CACHE_HOME','xdg')]:
    ENV[key] = str(ROOT/'.cache'/suffix)
STATE = dict(status='waiting_for_shift12_training', pid=os.getpid(), gpu=GPU, action=ACTION,
    started_at=datetime.now().astimezone().isoformat(), active=None,
    history_source='clean', training_timestep_shift=12., inference_flow_shift=2.22,
    overall_gate='COUNTERFACTUAL_CHUNK1_DIAGNOSTIC_ONLY', evaluations=[])


def save():
    p = OUT/f'run_{ACTION}.tmp.json'
    p.write_text(json.dumps(STATE, indent=2)+'\n')
    p.replace(OUT/f'run_{ACTION}.json')


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
    training = json.loads((SOURCE/'train_128/training.json').read_text())
    audit = SOURCE/'train_128/resume_audit.json'
    if (training['status'] != 'complete' or len(training['updates']) != 128
            or not all(training['replica_audit']['equality'].values())
            or not all(json.loads(audit.read_text())['checks'].values())):
        raise RuntimeError('Completed and audited128 training is required')
    checkpoint = SOURCE/'train_128/step_128'
    hashes = {name: hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()
              for name in ('causal_adapter.pt','action_adapter.pt','stage1_lora.pt','anyflow_adapter.pt')}
    STATE['checkpoint_receipt'] = dict(path=str(checkpoint),hashes=hashes,
        optimizer_steps=128,audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest())
    save()
    return checkpoint


def evaluate(checkpoint, action):
    verify()
    for name, expected in STATE['checkpoint_receipt']['hashes'].items():
        if hashlib.sha256((checkpoint/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Checkpoint changed after training')
    used = int(subprocess.check_output(['nvidia-smi','-i',str(GPU),'--query-gpu=memory.used',
                                        '--format=csv,noheader,nounits'], text=True).strip())
    if used >= 1024:
        raise RuntimeError(f'GPU{GPU} no longer idle ({used} MiB); no process interrupted')
    directory = OUT/f'eval/anyflow/step_128/clean_8step_native/{action}'
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', str(RUNTIME/'code/causal/benchmark.py'),
        '--modes','cached','--num-frames','39','--steps','8','--seed','13',
        '--flow-shift','2.22','--anyflow-sigma-grid','native','--precision-profile','h3_fp32',
        '--chunk-frames','5','--history-chunks','5','--cache-device','cpu',
        '--action-preset',HISTORY_ACTION,'--action-schedule',f'{HISTORY_ACTION}:1,{CURRENT_ACTION}:2','--action-prefix-mode','causal','--action-feedback',
        '--anchor-mode','dynamic_last_frame_rgb_dual','--history-source','clean',
        '--teacher-latents',str(TEACHER/f'action_{HISTORY_ACTION}_teacher_39/baseline_latents.pt'),
        '--causal-adapter',str(checkpoint/'causal_adapter.pt'),
        '--causal-action-adapter',str(checkpoint/'action_adapter.pt'),
        '--stage1-lora',str(checkpoint/'stage1_lora.pt'),
        '--anyflow-adapter',str(checkpoint/'anyflow_adapter.pt'),
        '--out-dir',str(directory),'--save-latents']
    with (directory/'run.log').open('w') as log:
        p = subprocess.Popen(command, cwd=RUNTIME, env=ENV, stdout=log, stderr=subprocess.STDOUT)
        STATE['active'] = dict(label=f'shift12_step128_{action}_clean8', pid=p.pid,
                              command=command, directory=str(directory))
        save()
        code = p.wait()
    if code:
        raise RuntimeError(f'{action} clean-history evaluation exited {code}')
    STATE['active'] = None
    runtime = json.loads((directory/'cached.json').read_text())
    setup = json.loads((directory/'setup.json').read_text())
    if (runtime['status'] != 'complete' or runtime['history_source'] != 'clean'
            or runtime['denoiser_forwards'] != 24 or runtime['commit_forwards'] != 3
            or runtime['video_sigma_grid'] != 'native' or runtime['sampler'] != 'anyflow_finite_map'
            or setup['precision']['profile'] != 'h3_fp32'
            or setup['anyflow']['metadata']['config']['training_timestep_shift'] != 12.
            or setup['stage1_lora']['trainable_parameters'] != 43237376):
        raise RuntimeError('Inference contract failed')
    video = directory/'cached.mp4'
    stats, flow = evaluate_video(video), evaluate_flow(video)
    if (stats['frames'], stats['fps'], stats['width'], stats['height']) != (39,24,832,480):
        raise RuntimeError('Incomplete or different video format')
    import torch
    torch.set_num_threads(4)
    reference = ROOT / f'outputs/2026-10-08-14/stage1_shift12_history128/eval/anyflow/step_128/clean_8step_native/{HISTORY_ACTION}'
    current = torch.load(directory/'conditioning.pt',map_location='cpu',weights_only=True)
    original = torch.load(reference/'conditioning.pt',map_location='cpu',weights_only=True)
    checks = {key:torch.equal(current[key],original[key]) for key in ['initial_noise','audio_noise','anchor']}
    cut = int(original['packed']['action_text_rows'][4,1])
    checks['scene_prompt_and_first_five_action_embeddings_equal'] = torch.equal(current['prompt_embeds'][:cut],original['prompt_embeds'][:cut])
    checks['later_action_embeddings_changed'] = not torch.equal(current['prompt_embeds'][cut:],original['prompt_embeds'][cut:])
    checks.update({'packed:'+key:torch.equal(value,original['packed'][key]) for key,value in current['packed'].items() if isinstance(value,torch.Tensor)})
    a = torch.load(directory/'cached_latents.pt',map_location='cpu',weights_only=True)
    b = torch.load(reference/'cached_latents.pt',map_location='cpu',weights_only=True)
    checks['first_five_latents_equal'] = torch.equal(a[:,:,:5],b[:,:,:5])
    if not all(checks.values()):
        raise RuntimeError(f'Counterfactual shared-prefix contract failed: {checks}')
    fairness = dict(reference=str(reference),history_action=HISTORY_ACTION,current_action=CURRENT_ACTION,
        checks=checks,shared_prefix_contract_passed=True,
        first_chunk_max_abs=float((a[:,:,:5]-b[:,:,:5]).abs().max()),
        chunk1_output_max_abs=float((a[:,:,5:10]-b[:,:,5:10]).abs().max()),
        scope='Only chunk1 has identical committed teacher history and historical actions. Chunk2 cache may differ after the changed-action commit. Full-clip flow is descriptive only.')
    row = dict(objective='anyflow', step=128, steps_per_chunk=8, action=action,
        history='clean',history_action=HISTORY_ACTION,current_action=CURRENT_ACTION,attribution_chunk=1, history_gradient_mode='full', sigma_grid='native',
        precision_profile='h3_fp32', adapter_scope='all_qkvo_ffn', training_timestep_shift=12.,
        video=str(video), video_stats=stats, flow=flow,
        boundary=rgb_boundary(video,[17,34]), runtime=runtime, input_fairness=fairness)
    (directory/'evaluation.json').write_text(json.dumps(row,indent=2)+'\n')
    STATE['evaluations'].append(row)
    save()


def main():
    if (OUT/f'run_{ACTION}.json').exists():
        raise SystemExit('Refusing duplicate evaluation queue')
    save()
    try:
        verify()
        checkpoint = wait_for_training()
        STATE['status'] = 'running'
        save()
        for action in [ACTION]:
            evaluate(checkpoint, action)
        STATE.update(status='complete', completed_at=datetime.now().astimezone().isoformat())
    except BaseException as exc:
        STATE.update(status='failed',error=repr(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
