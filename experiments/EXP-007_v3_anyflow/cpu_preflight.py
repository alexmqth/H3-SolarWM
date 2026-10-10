"""AF0: real tiny H3 forwards/backward, no pretrained weights or GPU."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = ROOT / 'H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime'
EXP005 = HERE.parent / 'EXP-005_v3_sliding_window'
ROUTER = ROOT / 'submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate'
sys.path[:0] = [str(FROZEN / 'code'), str(FROZEN / 'DiffSynth-Studio-h3-v2'), str(EXP005), str(ROUTER), str(HERE)]

import torch
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.h3_cached import H3ChunkCache
from causal.local_topology import visible_inputs
from causal.anyflow import install_anyflow, anyflow_sample_loss, finite_map_step
from current_prefix import current_prefix_feedback
from chunk_plan import cache_identity
from interval_sw import interval_sw
from interval_student import interval_student


def main():
    tick = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(170007)
    model = make_small_h3(rank=2).eval()
    batch = synthetic_h3_batch(frames=22, seed=7007)
    full, prompt = batch['packed'], batch['prompt_embeds']
    clean, noise = batch['clean_video'], batch['noise']
    common = dict(anchor=batch['anchor_rows'], audio=batch['audio_latents'], expected_layers=2)
    checks = {}

    def call(fn, cache, index, current, sigma, **kwargs):
        start = 0 if index == 0 else 12 + 5 * (index - 1)
        packed, text = visible_inputs(full, prompt, start + current.shape[2], 4)
        return fn(model, current, start=start, index=index, cache=cache,
                  full_packed=packed, prompt=text, sigma=sigma, **common, **kwargs)

    with current_prefix_feedback():
        old_cache = H3ChunkCache(5, 'cpu')
        with torch.no_grad():
            # Exact ordinary-FM regression before attaching new modules.
            for index, sl in [(0, slice(0, 12)), (1, slice(12, 17))]:
                x = torch.lerp(clean[:, :, sl], noise[:, :, sl], .6)
                a = call(interval_sw, old_cache, index, x, .6)
                b = call(interval_student, old_cache, index, x, .6)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                checks[f'ordinary_entry_exact_C{index+1}'] = True
                if index == 0:
                    call(interval_sw, old_cache, 0, clean[:, :, :12], 0, commit=True)
            base = b.clone()
        rng = torch.random.get_rng_state().clone()
        module = install_anyflow(model, device='cpu', gate=.25)
        assert torch.equal(rng, torch.random.get_rng_state())
        checks['conditioner_preserves_rng'] = True
        student_cache = H3ChunkCache(5, 'cpu')
        with torch.no_grad():
            call(interval_student, student_cache, 0, clean[:, :, :12], 0, commit=True)
            diag = call(interval_student, student_cache, 1, x, .6, target_sigma=.6)
            torch.testing.assert_close(diag, base, rtol=1e-5, atol=1e-5)
        checks['initialized_diagonal_max_abs'] = float((diag-base).abs().max())
        def time_rows_hook(_module, args, kwargs):
            current_times = kwargs['unique_timesteps'][kwargs['inverse_indices']]
            target_times = kwargs['unique_target_timesteps'][kwargs['inverse_indices']]
            prefix = kwargs['action_video_start']
            torch.testing.assert_close(current_times[prefix:], torch.full_like(current_times[prefix:], .4))
            torch.testing.assert_close(target_times[prefix:], torch.full_like(target_times[prefix:], .8))
            assert torch.equal(current_times[:prefix], target_times[:prefix])
            text_rows = kwargs['text_pos_info']['position_ids']
            torch.testing.assert_close(current_times[text_rows], torch.full_like(current_times[text_rows], .4))
            audio_rows = kwargs['audio_pos_info']['position_ids']
            assert torch.count_nonzero(current_times[audio_rows]) == 0
            assert len(kwargs['action_text_rows']) == 17
            checks['actual_native_time_and_target_prefix_rows'] = True
        hook = model.register_forward_pre_hook(time_rows_hook, with_kwargs=True)
        with torch.no_grad():
            call(interval_student, student_cache, 1, x, .6, target_sigma=.2)
        hook.remove()
        signature = cache_identity(student_cache)
        entries = [(e.key.clone(), e.value.clone()) for es in student_cache.layers.values() for e in es]
        calls = []

        def velocity(sample, sigma, target):
            calls.append((torch.is_grad_enabled(), sigma, target))
            return call(interval_student, student_cache, 1, sample, sigma,
                        target_sigma=target, use_gradient_checkpointing=torch.is_grad_enabled())

        loss, weight, stats = anyflow_sample_loss(velocity, clean[:, :, 12:17],
            noise[:, :, 12:17], .6, .2, preserve_fp32_inputs=True)
        (loss * weight).backward()
        assert [c[0] for c in calls] == [False, False, False, True]
        checks['finite_map_loss'] = float(loss.detach())
        checks['three_detached_one_grad_prediction'] = True
        grads = {'target_time': [p.grad for p in module.parameters()],
                 'qkv': [p.grad for n,p in model.named_parameters() if 'lora_' in n]}
        for name, values in grads.items():
            assert all(v is not None and torch.isfinite(v).all() for v in values)
            norm = sum(float(v.float().square().sum()) for v in values)**.5
            assert norm > 0, name
            checks[name + '_grad_norm'] = norm
        assert cache_identity(student_cache) == signature
        now = [(e.key, e.value) for es in student_cache.layers.values() for e in es]
        for before, after in zip(entries, now):
            for a,b in zip(before,after):
                assert torch.equal(a,b) and not b.requires_grad
        checks['checkpoint_backward_cache_immutable_and_detached'] = True
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        optimizer.step()
        fresh = H3ChunkCache(5, 'cpu')
        with torch.no_grad():
            call(interval_student, fresh, 0, clean[:, :, :12], 0, commit=True)
        changed = any(not torch.equal(a.key,b.key) for layer in fresh.layers
                      for a,b in zip(fresh.layers[layer],student_cache.layers[layer]))
        assert changed
        checks['student_update_changes_rebuilt_history_KV'] = True

    # Finite-map sign and raw-time derivative scale, on an analytic constant field.
    target = clean[:, :, 12:17]
    eps = noise[:, :, 12:17]
    current = .4 * target + .6 * eps
    torch.testing.assert_close(finite_map_step(current, eps-target, .6, .2), .8*target+.2*eps)
    constant_loss, _, _ = anyflow_sample_loss(lambda x,t,r: eps-target, target, eps, .6,.2)
    assert float(constant_loss) == 0
    checks['analytic_map_sign_and_constant_field_residual'] = True
    linear_loss, _, _ = anyflow_sample_loss(lambda x,t,r: torch.ones_like(x)*t,
                                          target, eps, .6, .2)
    expected = (torch.ones_like(target)*(2*.6-.2)-(eps-target)).square().mean()
    torch.testing.assert_close(linear_loss, expected, rtol=1e-5, atol=1e-5)
    checks['normalized_derivative_raw1000_unit_cancellation'] = True
    result = dict(task='EXP-007/v1', stage='AF0', status='CPU_PASS', gpu_forwards=0,
        model='random tiny actual H3, 2 layers; not pretrained 33B capability',
        seconds=time.perf_counter()-tick, torch=torch.__version__, checks=checks,
        interval_sha256=hashlib.sha256((HERE/'interval_student.py').read_bytes()).hexdigest())
    (HERE/'cpu_preflight.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
