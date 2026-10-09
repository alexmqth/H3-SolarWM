"""Read-only finite-map consistency probe; diagonal integration is NOT ground truth.

One action/checkpoint per process; noise-stratified self-consistency AND pseudo-target distance.
No optimizer, rollout acceptance, architecture change, or generated-video claim.
"""
from datetime import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / 'outputs/2026-10-08-13/stage1_parallel_resume68_to128'
RUNTIME = SOURCE / 'training_runtime'
CHECKPOINT = SOURCE / 'train_128/step_128'
VARIANTS = {'initial128': CHECKPOINT, **{v+'136': ROOT / 'outputs/2026-10-08-15/stage1_interval_consistency_candidate' / v / 'train_136/step_136' for v in ('control','auxiliary')}}
SIGMAS = (1., .9395404663085938, .8694517211914062, .7872340698242187,
          .6894410400390625, .5711835327148438, .4252873229980469,
          .24078089904785155, 0.)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rmse(a, b):
    return float((a.double() - b.double()).square().mean().sqrt())


def interval(velocity, initial, sigma, target, subdivisions=(8, 16)):
    """All paths start at the same state; r=t is used only by the references."""
    from causal.anyflow import finite_map_step
    with torch.no_grad():
        original = initial.clone()
        predicted = velocity(initial.clone(), sigma, target).float()
        direct = finite_map_step(initial, predicted, sigma, target)
        endpoints = {}
        for count in subdivisions:
            state = initial.clone()
            for i in range(count):
                t = sigma + (target - sigma) * i / count
                r = sigma + (target - sigma) * (i + 1) / count
                state = finite_map_step(state, velocity(state, t, t), t, r)
            endpoints[count] = state
        if not torch.equal(initial, original):
            raise RuntimeError('The reference mutated the shared initial state')
        if not all(torch.isfinite(x).all() for x in [predicted, direct, *endpoints.values()]):
            raise FloatingPointError('Nonfinite finite-map or reference')
        fine = endpoints[subdivisions[-1]]
        average = (fine - initial) / (target - sigma)
        global INTERVAL_TENSORS
        INTERVAL_TENSORS = dict(initial=initial.detach().cpu(), finite_velocity=predicted.detach().cpu(),
            reference_velocity=average.detach().cpu(),
            reference_endpoints={n:x.detach().cpu() for n,x in endpoints.items()})
        a, b = predicted.double().flatten(), average.double().flatten()
        fine_rms = float(b.square().mean().sqrt())
        error = rmse(direct, fine)
        refinement = rmse(endpoints[subdivisions[-2]], fine)
        return dict(sigma=sigma, target_sigma=target,
            initial_sha256=tensor_sha(initial), initial_state_unchanged=True,
            finite_velocity_rms=float(a.square().mean().sqrt()),
            diagonal_average_velocity_rms=fine_rms,
            velocity_rmse=rmse(predicted, average),
            velocity_relative_rmse=rmse(predicted, average) / max(fine_rms, 1e-12),
            velocity_cosine=float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30)),
            finite_vs_diagonal_endpoint_rmse={str(n): rmse(direct, x) for n, x in endpoints.items()},
            diagonal_refinement_endpoint_rmse=refinement,
            endpoint_error_over_refinement=error / max(refinement, 1e-12),
            direct_endpoint_sha256=tensor_sha(direct),
            reference_endpoint_sha256={str(n): tensor_sha(x) for n, x in endpoints.items()},
            subdivisions=list(subdivisions), model_forwards=1 + sum(subdivisions),
            caveat='8/16 refinement difference is a numerical diagnostic, not a rigorous error bound; neither reference is ground truth')


def self_test():
    """Analytic vector fields check direction, map conditioning, and convergence."""
    x = torch.tensor([[[[[.3, -.8]]]]], dtype=torch.float32)
    constant = interval(lambda z, t, r: torch.full_like(z, .25), x, 1., 0.)
    assert constant['velocity_rmse'] < 1e-6
    # dz/dsigma = 2*sigma has exact interval-average velocity sigma+r.
    affine = interval(lambda z, t, r: torch.full_like(z, t + r), x, .8, .2)
    assert affine['finite_vs_diagonal_endpoint_rmse']['16'] < affine['finite_vs_diagonal_endpoint_rmse']['8']
    assert abs(affine['finite_vs_diagonal_endpoint_rmse']['8'] / affine['finite_vs_diagonal_endpoint_rmse']['16'] - 2.) < 1e-4
    # Deliberate finite-map bias must be visible; diagonal references stay fixed.
    biased = interval(lambda z, t, r: torch.full_like(z, .25 + (.5 if t != r else 0.)), x, 1., 0.)
    assert abs(biased['velocity_rmse'] - .5) < 1e-6
    assert biased['reference_endpoint_sha256'] == constant['reference_endpoint_sha256']
    return dict(status='passed', scope='CPU analytic numerical probe, not real H3 validation',
                constant=constant, affine=affine, deliberate_bias=biased)


def cache_digest(cache):
    digest = hashlib.sha256()
    for layer, entries in sorted(cache.layers.items()):
        for entry in entries:
            digest.update(str((layer, entry.index)).encode())
            for tensor in (entry.key, entry.value, entry.rope):
                digest.update(tensor_sha(tensor).encode())
    return digest.hexdigest()


def main(args):
    global CHECKPOINT
    CHECKPOINT = VARIANTS[args.variant]
    expected_step = 128 if args.variant == 'initial128' else 136
    destination = OUT / args.variant
    destination.mkdir(exist_ok=True)
    path = destination / f'probe_{args.action}.json'
    if path.exists():
        raise FileExistsError('Choose a fresh experiment; do not overwrite a probe receipt')
    result = dict(status='preflight', pid=os.getpid(), action=args.action, gpu=args.gpu,
        started_at=datetime.now().astimezone().isoformat(), variant=args.variant, records=[],
        scope='clean-history teacher interpolants; no optimizer or video generation',
        limitations='Same-model diagonal references measure self-consistency, not correctness or Stage2 efficacy')
    started = time.perf_counter()
    def save():
        temporary = path.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(path)
    save()
    try:
        for filename, expected in json.loads((OUT/f'manifest_{args.variant}.json').read_text())['files'].items():
            if sha(filename) != expected:
                raise RuntimeError(f'Frozen input changed: {filename}')
        result['manifest_sha256'] = sha(OUT/f'manifest_{args.variant}.json')
        training = json.loads((CHECKPOINT.parent/'training.json').read_text())
        if training['status'] != 'complete' or len(training['updates']) != expected_step:
            raise RuntimeError('Completed training at the expected checkpoint is required')
        usage = int(subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if usage < 42000:
            result['status'] = 'not_started_gpu_busy'
            result['gpu_free_MiB'] = usage
            return
        result['status'] = 'running'; save()
        torch.set_num_threads(4); torch.manual_seed(13); torch.cuda.set_device('cuda:0')
        import infer as abot
        from causal.train_stage1_anyflow import prepare_real_cases, clean_cache, condition
        from causal.pretrained_lora import load_adapter, load_action_residual
        from causal.h3_precision import configure_precision, validate_precision_checkpoint
        from causal.anyflow import load_anyflow
        from causal.stage1_lora import load_stage1_lora
        from causal.h3_cached import chunk_forward
        config = SimpleNamespace(actions=[args.action], teacher_dir=[ROOT /
            f'outputs/2026-10-02-03/action_{args.action}_teacher_39'], device='cuda:0',
            anchor_mode='rgb', chunk_frames=5, history_chunks=5,
            action_prefix_mode='causal', action_feedback=True)
        pipe = abot.load_pipeline(config.device)
        pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(
            ROOT/'checkpoints/H3-World/step-10000.safetensors'), hotload=True)
        model = pipe.dit.requires_grad_(False).eval()
        pipe.load_models_to_device(['dit'])
        loaded = load_adapter(model, CHECKPOINT/'causal_adapter.pt', config.device)
        action = load_action_residual(model, CHECKPOINT/'action_adapter.pt', config.device)['adapter']
        action.requires_grad_(False)
        case = prepare_real_cases(pipe, action, config)[0]
        result['precision'] = configure_precision(model, 'h3_fp32', native_transformer_dir=
            ROOT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        validate_precision_checkpoint(loaded['metadata'], result['precision'])
        time_module, time_meta = load_anyflow(model, CHECKPOINT/'anyflow_adapter.pt', config.device)
        bank, bank_meta = load_stage1_lora(model, CHECKPOINT/'stage1_lora.pt', device=config.device)
        validate_precision_checkpoint(bank_meta, result['precision'])
        for metadata in (loaded['metadata'], time_meta, bank_meta):
            assert metadata['optimizer_step'] == expected_step
            assert metadata['config'] == loaded['metadata']['config']
        model.requires_grad_(False); time_module.requires_grad_(False)
        case['clean'] = case['clean'].float()
        saved = torch.load(config.teacher_dir[0]/'conditioning.pt', map_location='cpu', weights_only=True)
        noise = saved['initial_noise'].to(device=config.device, dtype=torch.float32)
        result.update(checkpoint=str(CHECKPOINT), bank=bank.describe(),
            clean_sha256=tensor_sha(case['clean']), noise_sha256=tensor_sha(noise),
            action_sha256=tensor_sha(case['actions']),
            anchor_sha256=[tensor_sha(x) for x in case['anchors']],
            prompt_sha256=tensor_sha(case['prompt']), audio_sha256=tensor_sha(case['audio']))
        versions = [(name, p, p._version) for name, p in model.named_parameters()]
        save()
        with torch.no_grad():
            for chunk in range(3):
                cache = clean_cache(model, case, chunk, config)
                before = cache_digest(cache); commits = cache.commits
                common = condition(case, chunk, config)
                clean = case['clean'][:, :, chunk*5:(chunk+1)*5]
                eps = noise[:, :, chunk*5:(chunk+1)*5]
                for index in (0, 4, 7):
                    t, r = SIGMAS[index:index+2]
                    z = (1-t)*clean + t*eps
                    def velocity(state, sigma, target):
                        return chunk_forward(model, state, sigma=sigma, target_sigma=target,
                            index=chunk, cache=cache, **common)
                    tick = time.perf_counter()
                    record = interval(velocity, z, t, r, subdivisions=(8,16) if index == 7 else (4,8))
                    tensors = INTERVAL_TENSORS
                    direct = tensors['initial'] + (r-t)*tensors['finite_velocity']
                    target_at_r = ((1-r)*clean + r*eps).cpu()
                    record['pseudo_target_at_r_sha256'] = tensor_sha(target_at_r)
                    record['endpoint_rmse_to_teacher_interpolant_at_r'] = {
                        'finite': rmse(direct, target_at_r),
                        **{f'diagonal{n}': rmse(x, target_at_r) for n,x in tensors['reference_endpoints'].items()}}
                    if r == 0.:
                        clean_endpoint = direct
                    else:
                        from causal.anyflow import finite_map_step
                        clean_endpoint = finite_map_step(z, velocity(z,t,0.),t,0.).cpu()
                        record['model_forwards'] += 1
                    record['finite_t_to_zero_rmse_to_teacher_clean'] = rmse(clean_endpoint,clean.cpu())
                    record['finite_t_to_zero_sha256'] = tensor_sha(clean_endpoint)
                    record['noise_region'] = {0:'high',4:'middle',7:'low'}[index]
                    record['pseudo_target_note'] = 'Original H3 pseudo-GT, not real GT; at r>0 use (1-r)*clean+r*same_noise, not clean. Separate t->0 is a diagnostic map, not the native inference step.'
                    # Auxiliary fitted a frozen128 target; keep that distinct from current diagonal self-consistency.
                    if index == 7:
                        frozen_dir = ROOT/'outputs/2026-10-08-15/stage1_finite_interval_refinement128'
                        frozen_report = json.loads((frozen_dir/f'probe_{args.action}.json').read_text())
                        old = next(x for x in frozen_report['records'] if x['chunk']==chunk)
                        assert old['initial_sha256'] == record['initial_sha256']
                        assert sha(old['target_file']) == old['target_sha256']
                        frozen = torch.load(old['target_file'],map_location='cpu',weights_only=True)['tensors']
                        record['finite_velocity_rmse_to_frozen128_training_target'] = rmse(tensors['finite_velocity'], frozen['reference_velocity'])
                        record['finite_endpoint_rmse_to_frozen128_training_target'] = rmse(direct, frozen['reference_endpoints'][16])
                    record.update(chunk=chunk, native_step=index,
                        seconds=time.perf_counter()-tick, cpu_kv_MiB=cache.nbytes/2**20)
                    directory = destination / args.action / 'targets'
                    directory.mkdir(parents=True, exist_ok=True)
                    target_path = directory / f'chunk_{chunk}_step_{index}.pt'
                    torch.save(dict(tensors={**tensors, 'finite_clean_endpoint': clean_endpoint, 'pseudo_target_at_r': target_at_r}, record=record, action=args.action,
                        checkpoint=str(CHECKPOINT), manifest_sha256=result['manifest_sha256']), target_path)
                    record['target_file'] = str(target_path)
                    record['target_sha256'] = sha(target_path)
                    result['records'].append(record); save()
                    print(json.dumps(record), flush=True)
                if cache_digest(cache) != before or cache.commits != commits:
                    raise RuntimeError('Read-only interval probe changed clean KV history')
                result.setdefault('cache_checks', []).append(dict(chunk=chunk,
                    hash_before_and_after=before, commits=commits, unchanged=True))
                cache.clear(); save()
        if any(p._version != version for _, p, version in versions):
            raise RuntimeError('A model parameter changed during the read-only probe')
        result.update(status='complete', parameter_versions_unchanged=True,
            model_forwards=sum(r['model_forwards'] for r in result['records']),
            clean_commit_forwards=3,
            gpu_allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as exc:
        result.update(status='failed', error=repr(exc)); raise
    finally:
        result['wall_seconds'] = time.perf_counter()-started
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['A', 'D'])
    parser.add_argument('--variant', choices=tuple(VARIANTS), default='initial128')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if not args.self_test and (args.action is None or args.gpu is None):
        parser.error('--action and --gpu are required for real inference')
    os.environ.update(CUDA_VISIBLE_DEVICES='' if args.self_test else str(args.gpu),
        ABOT_VRAM_RESERVE_GIB='6', HF_HUB_OFFLINE='1', DIFFSYNTH_SKIP_DOWNLOAD='True',
        ABOT_DIFFSYNTH_ROOT=str(RUNTIME/'DiffSynth-Studio-h3-v2'),
        DIFFSYNTH_ROOT=str(RUNTIME/'DiffSynth-Studio-h3-v2'))
    for key, suffix in [('HF_HOME','hf'), ('TORCHINDUCTOR_CACHE_DIR','torchinductor'),
                        ('TRITON_CACHE_DIR','triton'), ('XDG_CACHE_HOME','xdg')]:
        os.environ[key] = str(ROOT/'.cache'/suffix)
    sys.path[:0] = [str(RUNTIME/'code'), str(RUNTIME/'code/abot'),
                   str(RUNTIME/'DiffSynth-Studio-h3-v2')]
    import torch
    if args.self_test:
        result = self_test()
        (OUT/'cpu_self_test.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
    else:
        main(args)
