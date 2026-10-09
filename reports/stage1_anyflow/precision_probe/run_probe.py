"""Same-state 33B numerical diagnostic after frozen16 evaluation, on GPU0.

Uses fixed clean legacy KV, identical noisy latents and a shared legacy FM
direction for the finite difference. Only compute precision varies. No
optimizer update and no claim of a video-quality fix.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DEPENDENCY = ROOT / 'outputs/2026-10-08-05/stage1_anyflow39_gpu0'
RUNTIME = ROOT / 'outputs/2026-10-08-04/stage1_anyflow39_frozen_time/runtime'
sys.path[:0] = [str(RUNTIME / 'code'), str(RUNTIME / 'code/abot'),
               str(RUNTIME / 'code/causal'), str(ROOT / 'DiffSynth-Studio-h3-v2')]
os.environ.update(CUDA_VISIBLE_DEVICES='0', ABOT_VRAM_RESERVE_GIB='6',
    DIFFSYNTH_SKIP_DOWNLOAD='True', HF_HUB_OFFLINE='1',
    ABOT_DIFFSYNTH_ROOT=str(ROOT / 'DiffSynth-Studio-h3-v2'))
for key, suffix in [('HF_HOME', 'hf'), ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor'),
                    ('TRITON_CACHE_DIR', 'triton'), ('XDG_CACHE_HOME', 'xdg')]:
    os.environ[key] = str(ROOT / '.cache' / suffix)

STATE = dict(status='waiting_for_frozen16_evaluation', pid=os.getpid(), gpu=0,
    vram_reserve_gib=6, started_at=datetime.now().astimezone().isoformat(), records=[],
    scope='Same-state compute-precision diagnostic; no training or video acceptance')

def save():
    tmp = OUT / 'run.tmp.json'
    tmp.write_text(json.dumps(STATE, indent=2) + '\n')
    tmp.replace(OUT / 'run.json')

def live(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False

def wait():
    while True:
        previous = json.loads((DEPENDENCY / 'run.json').read_text())
        alive = live(previous['pid'])
        if previous['status'] == 'complete' and not alive:
            return
        if previous['status'] not in ('running', 'complete') or not alive:
            raise RuntimeError('Preceding experiment failed; refusing to overlap GPU jobs')
        time.sleep(20)

def verify_sources():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for path, expected in manifest['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Probe source or input changed: {path}')
    manifest = json.loads((DEPENDENCY / 'runtime_manifest.json').read_text())
    for rel, expected in manifest['hashes'].items():
        if hashlib.sha256((RUNTIME / rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Runtime changed: {rel}')
    for path, expected in manifest['shared_files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Shared dependency changed: {path}')

def main():
    import torch
    import torch.nn.functional as F
    import infer as abot
    from causal.train_stage1_anyflow import prepare_real_cases, condition, clean_cache
    from causal.pretrained_lora import load_adapter, load_action_residual
    from causal.anyflow import load_anyflow, logical_time_pairs
    from causal.anyflow_reference import bounded_difference_timesteps, central_difference_derivative
    from causal.h3_cached import chunk_forward
    from precision_profile import precision_profile
    verify_sources()
    torch.cuda.set_device(0)
    torch.manual_seed(13)
    tick = time.perf_counter()
    pipe = abot.load_pipeline('cuda:0')
    pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(
        ROOT / 'checkpoints/H3-World/step-10000.safetensors'), hotload=True)
    model = pipe.dit.requires_grad_(False).eval()
    checkpoint = DEPENDENCY / 'train_anyflow/step_16'
    for name in ('causal_adapter.pt', 'action_adapter.pt', 'anyflow_adapter.pt'):
        STATE.setdefault('checkpoint_sha256', {})[name] = hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
    load_adapter(model, checkpoint / 'causal_adapter.pt', 'cuda:0')
    action_adapter = load_action_residual(model, checkpoint / 'action_adapter.pt', 'cuda:0')['adapter']
    args = SimpleNamespace(actions=['A', 'D'], teacher_dir=[
        ROOT / f'outputs/2026-10-02-03/action_{a}_teacher_39' for a in 'AD'],
        device='cuda:0', chunk_frames=5, history_chunks=5, anchor_mode='rgb',
        action_prefix_mode='causal', action_feedback=True)
    cases = prepare_real_cases(pipe, action_adapter, args)
    load_anyflow(model, checkpoint / 'anyflow_adapter.pt', 'cuda:0')
    model.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    STATE.update(status='running', preparation_seconds=time.perf_counter()-tick,
        shared_history='legacy clean CPU KV, read-only across precision profiles',
        shared_fd_direction='legacy diagonal FM direction, fixed across profiles')
    save()
    with torch.no_grad():
        for case in cases:
            for chunk in (0, 2):
                clean = case['clean'][:, :, chunk*5:chunk*5+5]
                cache = clean_cache(model, case, chunk, args)
                common = condition(case, chunk, args)
                generator = torch.Generator().manual_seed(1001)
                pairs = logical_time_pairs(generator)
                def velocity(x, sigma, target):
                    return chunk_forward(model, x, sigma=sigma, target_sigma=target,
                        index=chunk, cache=cache, **common)
                try:
                    for i in range(4):
                        noise = torch.randn(clean.shape, generator=generator).to(clean)
                        if i < 2:
                            continue
                        t, r = float(pairs.t[i]), float(pairs.r[i])
                        noisy = (1-t)*clean.float() + t*noise.float()
                        diagonal = velocity(noisy.to(clean), t, t).float()
                        raw_t, raw_r = torch.tensor([t*1000], device=clean.device), torch.tensor([r*1000], device=clean.device)
                        plus_t, minus_t = bounded_difference_timesteps(raw_t, raw_r, epsilon=5.)
                        plus_sigma, minus_sigma = min(1., float(plus_t[0])/1000), max(r, float(minus_t[0])/1000)
                        plus = (noisy + diagonal*(plus_sigma-t)).to(clean)
                        minus = (noisy - diagonal*(t-minus_sigma)).to(clean)
                        baseline = None
                        for profile in ('legacy', 'time_fp32', 'boundary_fp32'):
                            start = time.perf_counter()
                            commits = cache.commits
                            with precision_profile(model, profile) as metadata:
                                pv = velocity(plus, plus_sigma, r).float()
                                mv = velocity(minus, minus_sigma, r).float()
                                v = velocity(noisy.to(clean), t, r).float()
                                derivative = central_difference_derivative(pv, mv, plus_t, minus_t)
                                residual = v + (raw_t-raw_r).view(*([1]*v.ndim))*derivative - (noise.float()-clean.float())
                            if cache.commits != commits or not torch.isfinite(residual).all():
                                raise RuntimeError('Probe mutated KV or produced nonfinite output')
                            if baseline is None:
                                baseline = (v.clone(), derivative.clone())
                            row = dict(metadata, action=case['label'], chunk=chunk,
                                sample_type='endpoint' if i == 2 else 'flow_map', sigma=t, target_sigma=r,
                                raw_loss=float(residual.square().mean()), fd_norm=float(derivative.norm()),
                                velocity_relative_l2=float((v-baseline[0]).norm()/baseline[0].norm().clamp_min(1e-12)),
                                derivative_relative_l2=float((derivative-baseline[1]).norm()/baseline[1].norm().clamp_min(1e-12)),
                                derivative_cosine=float(F.cosine_similarity(derivative.flatten(), baseline[1].flatten(), dim=0)),
                                cpu_kv_MiB=cache.peak_bytes/2**20,
                                gpu_allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
                                seconds=time.perf_counter()-start)
                            STATE['records'].append(row)
                            print(json.dumps(row), flush=True)
                            save()
                        # Removing every hook/policy must return the original output exactly.
                        restored = velocity(noisy.to(clean), t, r).float()
                        if not torch.equal(restored, baseline[0]):
                            raise RuntimeError('Legacy output did not restore bitwise')
                        STATE.setdefault('restoration_checks', []).append(dict(action=case['label'], chunk=chunk, sample=i, bitwise=True))
                        save()
                finally:
                    cache.clear()
    STATE.update(status='complete', completed_at=datetime.now().astimezone().isoformat(),
        wall_seconds=time.perf_counter()-tick,
        gpu_allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
        gpu_reserved_peak_MiB=torch.cuda.max_memory_reserved()/2**20,
        interpretation='Numerical sensitivity only; derivative truth and video quality are not established')

if (OUT / 'run.json').exists():
    raise SystemExit('Refusing duplicate probe')
save()
try:
    wait()
    main()
except BaseException as exc:
    STATE.update(status='failed', error=repr(exc))
    raise
finally:
    save()
