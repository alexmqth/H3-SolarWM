"""Real tiny-H3 integration; no pretrained weights, data or GPU required."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'DiffSynth-Studio-h3-v2')]
from causal.anyflow import finite_map_step, install_anyflow
from causal.fmbs import shortcut_intervals, simulate_chunk, simulate_h3_chunk
from causal.h3_cached import H3ChunkCache, chunk_forward
from causal.h3_precision import configure_precision
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.stage1_lora import install_stage1_lora


def setup(dtype=torch.float32):
    torch.set_num_threads(2); torch.manual_seed(151)
    model = make_small_h3().eval().to(dtype).requires_grad_(False)
    configure_precision(model, 'h3_fp32')
    install_anyflow(model).requires_grad_(False)
    bank = install_stage1_lora(model, rank=2, alpha=2)
    with torch.no_grad():
        for module in bank.modules:
            for weight in module.lora_B: weight.normal_(std=.03)
    batch = synthetic_h3_batch(frames=12, seed=71)
    common = dict(full_packed=batch['packed'], prompt=batch['prompt_embeds'].to(dtype),
        anchor=batch['anchor_rows'].to(dtype), audio=batch['audio_latents'].to(dtype), chunk_frames=5,
        action_prefix_mode='causal', action_feedback=True)
    # A shifted 8-step grid: use actual adjacent endpoints, not t-1/N.
    grid = (1., *(2.22 * t / (1 + 1.22 * t) for t in [.875, .75, .625, .5, .375, .25, .125]), 0.)
    cache = H3ChunkCache(2, 'cpu')
    # Two actual student-generated history chunks, detached at clean commit.
    # History uses full N-step inference, not GT or the current FMBS shortcut.
    with torch.no_grad():
        for index in range(2):
            current = batch['noise'][:, :, index*5:index*5+5].clone()
            for t, r in zip(grid, grid[1:]):
                v = chunk_forward(model, current, sigma=t, target_sigma=r,
                                  index=index, cache=cache, **common)
                current = finite_map_step(current, v, t, r)
            chunk_forward(model, current, sigma=0, index=index, cache=cache,
                          commit=True, **common)
    return model, bank, batch, common, cache, grid


def gradient(model, bank, batch, common, cache, grid, *, checkpoint, detach_first=False):
    params = bank.parameters()
    for p in params: p.grad = None
    noise = batch['noise'][:, :, 10:].clone().requires_grad_()
    if detach_first:
        current = noise
        for i, (t, r) in enumerate(shortcut_intervals(grid, 3)):
            v = chunk_forward(model, current, sigma=t, target_sigma=r, index=2,
                cache=cache, allow_grad_read=True, **common)
            current = finite_map_step(current, v, t, r)
            if i == 0: current = current.detach()
        endpoint = current
    else:
        endpoint = simulate_h3_chunk(model, noise, sigmas=grid, interval_index=3,
            chunk_index=2, cache=cache, conditions=common, checkpoint=checkpoint, offload=checkpoint)
    loss = endpoint.square().mean()
    loss.backward()
    grad = [torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in params]
    return endpoint.detach(), grad, noise.grad


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_checkpoint_offload_preserves_h3_fmbs_state_and_full_gradient(dtype):
    model, bank, batch, common, cache, grid = setup(dtype)
    snapshots = [(x, x.clone(), x._version) for entries in cache.layers.values()
                 for e in entries for x in (e.key, e.value, e.rope)]
    commits = cache.commits
    plain, grad_plain, noise_grad = gradient(model, bank, batch, common, cache, grid, checkpoint=False)
    checked, grad_checked, checked_noise_grad = gradient(model, bank, batch, common, cache, grid, checkpoint=True)
    assert plain.dtype == checked.dtype == torch.float32
    torch.testing.assert_close(plain, checked, atol=0, rtol=0)
    for a, b in zip(grad_plain, grad_checked): torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(noise_grad, checked_noise_grad, atol=1e-7, rtol=1e-5)
    assert noise_grad.norm() > 0 and cache.commits == commits
    assert all(torch.equal(x, old) and x._version == version for x, old, version in snapshots)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    cut, grad_cut, _ = gradient(model, bank, batch, common, cache, grid, checkpoint=False, detach_first=True)
    torch.testing.assert_close(plain, cut, atol=0, rtol=0)
    delta = torch.sqrt(sum((a - b).square().sum() for a, b in zip(grad_plain, grad_cut)))
    norm = torch.sqrt(sum(a.square().sum() for a in grad_plain))
    assert delta > norm * .01
    print(f'full_parameter_gradient_norm={norm.item():.8f} detach_first_gradient_gap={delta.item():.8f}')


def test_fmbs_parameter_jacobian_matches_finite_difference_with_fixed_history():
    model, bank, batch, common, cache, grid = setup()
    _, grads, _ = gradient(model, bank, batch, common, cache, grid, checkpoint=False)
    norm = torch.sqrt(sum(x.square().sum() for x in grads))
    directions = [x / norm for x in grads]; params = bank.parameters()
    originals = [p.detach().clone() for p in params]
    values = []; eps = 2e-3
    with torch.no_grad():
        for sign in (1, -1):
            for p, initial, direction in zip(params, originals, directions):
                p.copy_(initial + sign * eps * direction)
            # Preserve the observed self-generated history while differentiating
            # the current chunk. Rebuilding history would test a different J.
            value = simulate_h3_chunk(model, batch['noise'][:, :, 10:], sigmas=grid,
                interval_index=3, chunk_index=2, cache=cache, conditions=common,
                checkpoint=False, offload=False).square().mean()
            values.append(value)
        for p, initial in zip(params, originals): p.copy_(initial)
    finite_difference = (values[0] - values[1]) / (2 * eps)
    torch.testing.assert_close(finite_difference, norm, atol=2e-4, rtol=.02)
    print(f'autograd_directional={norm.item():.8f} finite_difference={finite_difference.item():.8f}')


def test_checkpoint_history_snapshot_survives_later_commit_and_eviction():
    model, bank, batch, common, cache, grid = setup()
    _, expected, _ = gradient(model, bank, batch, common, cache, grid, checkpoint=True)
    params = bank.parameters()
    for p in params: p.grad = None
    endpoint = simulate_h3_chunk(model, batch['noise'][:, :, 10:], sigmas=grid,
        interval_index=3, chunk_index=2, cache=cache, conditions=common)
    with torch.no_grad():
        chunk_forward(model, endpoint.detach(), sigma=0, index=2, cache=cache,
                      commit=True, **common)
    assert all([e.index for e in entries] == [1, 2] for entries in cache.layers.values())
    # Backward still needs chunks0/1, not the caller's now-advanced chunks1/2.
    endpoint.square().mean().backward()
    for p, grad in zip(params, expected):
        actual = torch.zeros_like(p) if p.grad is None else p.grad
        torch.testing.assert_close(actual, grad, atol=1e-7, rtol=1e-5)


def test_h3_integration_rejects_nonanyflow_mutating_args_and_live_history_graph():
    model, _, batch, common, cache, grid = setup()
    args = dict(sigmas=grid, interval_index=3, chunk_index=2, cache=cache, conditions=common)
    noise = batch['noise'][:, :, 10:]
    with pytest.raises(ValueError, match='owns time'):
        simulate_h3_chunk(model, noise, **dict(args, conditions=dict(common, commit=True)))
    with pytest.raises(ValueError, match='detached'):
        cache.layers[0][0].key.requires_grad_(True)
        simulate_h3_chunk(model, noise, **args)
    cache.layers[0][0].key.requires_grad_(False)
    with pytest.raises(ValueError, match='strictly decreasing'):
        simulate_h3_chunk(model, noise, **dict(args, sigmas=(1, .5, .5, 0)))
    del model.anyflow_conditioner
    with pytest.raises(ValueError, match='AnyFlow conditioner'):
        simulate_h3_chunk(model, noise, **args)


def test_shortcut_skips_zero_intervals_and_uses_actual_shifted_endpoints():
    grid = (1, .9, .65, .3, 0)
    assert shortcut_intervals(grid, 0) == ((1, .9), (.9, 0))
    assert shortcut_intervals(grid, 2) == ((1, .65), (.65, .3), (.3, 0))
    assert shortcut_intervals(grid, 3) == ((1, .3), (.3, 0))
    assert shortcut_intervals((1, 0), 0) == ((1, 0),)
    noise = torch.ones(1, requires_grad=True)
    p = torch.tensor(.3, requires_grad=True)
    calls = []
    def velocity(x, t, r):
        calls.append((t, r)); return x * p
    endpoint = simulate_chunk(noise, velocity, grid, 2)
    endpoint.sum().backward()
    assert calls == list(shortcut_intervals(grid, 2)) and noise.grad is not None and p.grad is not None
