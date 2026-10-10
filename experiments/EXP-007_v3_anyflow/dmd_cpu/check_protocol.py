"""Preparation only: tiny actual H3, distinct roles, full 8-map DMD gradient.

No pretrained model, GPU calls, video capability claim, or DMD GPU permission.
Uses the already checked V3 interval entry with unchanged prefix/time routing.
"""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import sys
import time

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[2]
FROZEN = ROOT / 'H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime'
SW = EXP.parent / 'EXP-005_v3_sliding_window'
PREFIX = ROOT / 'submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate'
sys.path[:0] = [str(FROZEN / 'code'), str(FROZEN / 'DiffSynth-Studio-h3-v2'),
                str(SW), str(PREFIX), str(EXP)]

import torch
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.h3_cached import H3ChunkCache
from causal.local_topology import visible_inputs
from causal.anyflow import install_anyflow, finite_map_step
from chunk_plan import cache_identity
from current_prefix import current_prefix_feedback
from interval_student import interval_student

DMD_PATH = ROOT / 'H3-World/code/causal/dmd.py'
spec = importlib.util.spec_from_file_location('v3_dmd_math', DMD_PATH)
dmd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dmd)


def snapshot(cache):
    return [(e.key.clone(), e.value.clone()) for es in cache.layers.values() for e in es]


def main():
    tick = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(170008)
    teacher = make_small_h3(rank=2).eval()
    fake = copy.deepcopy(teacher)
    student = copy.deepcopy(teacher)
    teacher.requires_grad_(False)
    target = install_anyflow(student, device='cpu', gate=.25)
    trainables = {name: [p for p in model.parameters() if p.requires_grad]
                  for name, model in [('fake', fake), ('student', student)]}
    assert not hasattr(fake, 'anyflow_conditioner')
    assert not ({id(p) for p in teacher.parameters()} & {id(p) for p in fake.parameters()})
    assert not ({id(p) for p in fake.parameters()} & {id(p) for p in student.parameters()})
    base_versions = [(p, p._version) for model in (teacher, fake, student)
                     for p in model.parameters() if not p.requires_grad]
    batch = synthetic_h3_batch(frames=17, seed=7008)
    clean, noise = batch['clean_video'], batch['noise']
    layouts = {i: visible_inputs(batch['packed'], batch['prompt_embeds'], n, 4)
               for i, n in [(0, 12), (1, 17)]}
    checks = {'independent_roles_and_fake_has_no_target_module': True}
    calls = []

    def call(role, model, cache, index, x, sigma, r=None, commit=False):
        packed, prompt = layouts[index]
        calls.append(dict(role=role, index=index, sigma=float(sigma),
                          r=None if r is None else float(r), commit=commit,
                          grad=torch.is_grad_enabled()))
        return interval_student(model, x, start=0 if index == 0 else 12,
            index=index, cache=cache, full_packed=packed, prompt=prompt,
            anchor=batch['anchor_rows'], audio=batch['audio_latents'], sigma=sigma,
            target_sigma=r, commit=commit, expected_layers=2,
            use_gradient_checkpointing=torch.is_grad_enabled())

    def prefill(role, model):
        cache = H3ChunkCache(5, 'cpu')
        with torch.no_grad():
            call(role, model, cache, 0, clean[:, :, :12], 0,
                 0 if role == 'student' else None, commit=True)
        return cache

    with current_prefix_feedback():
        caches = {name: prefill(name, model) for name, model in
                  [('teacher', teacher), ('fake', fake), ('student', student)]}
        first_keys = [cache.layers[0][0].key for cache in caches.values()]
        assert len({x.data_ptr() for x in first_keys}) == 3
        assert all(not e.key.requires_grad and not e.value.requires_grad
                   for cache in caches.values() for es in cache.layers.values() for e in es)
        checks['each_role_owns_detached_history_KV'] = True
        gen_signature = cache_identity(caches['student'])
        gen_snapshot = snapshot(caches['student'])

        # An actual eight-call differentiable finite-map chain, no detached replay.
        sigmas = [float(2.22 * t / (1 + 1.22 * t)) for t in torch.linspace(1, 0, 9)]
        x = noise[:, :, 12:17].clone()
        states, velocities = [], []
        for sigma, r in zip(sigmas[:-1], sigmas[1:]):
            v = call('student', student, caches['student'], 1, x, sigma, r)
            v.retain_grad()
            x = finite_map_step(x, v, sigma, r)
            x.retain_grad()
            states.append(x)
            velocities.append(v)
        endpoint = x
        assert endpoint.requires_grad

        # Fake score update uses a detached on-policy endpoint and ordinary FM.
        generator = torch.Generator().manual_seed(170108)
        score_noise = torch.randn(endpoint.shape, generator=generator)
        sigma = .6
        z = dmd.score_sample(endpoint, score_noise, sigma)
        assert not z.requires_grad
        fake_opt = torch.optim.AdamW(trainables['fake'], lr=1e-3)
        fake_signature = cache_identity(caches['fake'])
        pred = call('fake', fake, caches['fake'], 1, z, sigma)
        fake_loss = dmd.critic_flow_loss(pred, endpoint, score_noise)
        fake_loss.backward()
        assert all(p.grad is None for p in trainables['student'])
        assert all(p.grad is None for p in teacher.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainables['fake'])
        assert cache_identity(caches['fake']) == fake_signature
        fake_opt.step()
        fake_opt.zero_grad(set_to_none=True)
        fresh_fake = prefill('fake', fake)
        checks['fake_update_changes_own_rebuilt_KV'] = any(
            not torch.equal(a.key, b.key)
            for layer in fresh_fake.layers
            for a, b in zip(caches['fake'].layers[layer], fresh_fake.layers[layer]))
        assert checks['fake_update_changes_own_rebuilt_KV']
        caches['fake'].clear()
        caches['fake'] = fresh_fake
        checks['fake_FM_backward_isolated_from_generator_and_teacher'] = True

        # Both scoring roles evaluate the exact same detached noisy endpoint.
        score_z = z.clone()
        with torch.no_grad():
            real = call('teacher', teacher, caches['teacher'], 1, z, sigma)
            fake_v = call('fake', fake, caches['fake'], 1, z, sigma)
        assert torch.equal(z, score_z)
        loss, direction, stats = dmd.distribution_matching_loss(endpoint, z, sigma,
            real_velocity=real, fake_velocity=fake_v, normalize=True)
        raw_direction = sigma * (real - fake_v)
        normalizer = (endpoint.detach() - (z - sigma * real)).abs().mean()
        torch.testing.assert_close(direction, raw_direction / normalizer.clamp_min(1e-8),
                                   rtol=1e-4, atol=1e-6)
        assert direction.norm() > 0
        assert not direction.requires_grad
        checks['matched_scores_and_noise_minus_clean_direction'] = True
        loss.backward()
        torch.testing.assert_close(endpoint.grad, direction / direction.numel(), rtol=1e-5, atol=1e-8)
        assert all(v.grad is not None and v.grad.norm() > 0 for v in velocities)
        assert all(x.grad is not None and x.grad.norm() > 0 for x in states)
        checks['all_eight_actual_maps_receive_generator_gradient'] = True
        checks['map_velocity_grad_norms'] = [float(v.grad.norm()) for v in velocities]
        assert all(p.grad is None for p in teacher.parameters())
        assert all(p.grad is None for p in trainables['fake'])
        for name, ps in [('target', list(target.parameters())),
                         ('student_qkv', [p for n, p in student.named_parameters() if 'lora_' in n])]:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps)
            norm = sum(float(p.grad.square().sum()) for p in ps) ** .5
            assert norm > 0
            checks[name + '_grad_norm'] = norm
        checks['generator_backward_isolated_from_scores'] = True
        assert cache_identity(caches['student']) == gen_signature
        for old, new in zip(gen_snapshot, snapshot(caches['student'])):
            assert all(torch.equal(a, b) for a, b in zip(old, new))
        assert all(p._version == version for p, version in base_versions)
        checks['generator_history_immutable_and_all_base_weights_frozen'] = True
        try:
            dmd.distribution_matching_loss(endpoint.detach(), z, sigma,
                real_velocity=real, fake_velocity=fake_v)
        except ValueError:
            checks['disconnected_endpoint_rejected'] = True
        else:
            raise AssertionError('Detached endpoint accepted as generator graph')

    result = dict(status='CPU_PASS', scope='random tiny actual H3; CPU DMD preparation only',
        gpu_forwards=0, seconds=time.perf_counter()-tick, torch=torch.__version__,
        fake_loss=float(fake_loss.detach()), generator_loss=float(loss.detach()),
        direction_stats=stats, checks=checks, calls=calls,
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [Path(__file__), DMD_PATH, EXP/'interval_student.py',
                           PREFIX/'current_prefix.py', FROZEN/'code/causal/anyflow.py']})
    (HERE/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
