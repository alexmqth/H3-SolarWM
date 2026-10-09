from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'DiffSynth-Studio-h3-v2'), str(ROOT / 'code')]
from causal.anyflow import (install_anyflow, save_anyflow, load_anyflow,
    logical_time_pairs, anyflow_sample_loss, adaptive_scale, finite_map_step,
    validate_checkpoint_protocol)
from causal.anyflow_reference import adaptive_rescale_non_diffusion_losses
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.h3_cached import H3ChunkCache, chunk_forward


def test_logical_batch_covers_all_types_and_rescale_matches_reference():
    pairs = logical_time_pairs(torch.Generator().manual_seed(4))
    assert pairs.sample_type.tolist() == [0, 0, 1, 2]
    assert torch.equal(pairs.r[:2], pairs.t[:2])
    assert pairs.r[2] == 0 and 0 < pairs.r[3] < pairs.t[3] < 1
    raw = torch.tensor([2., 4., 10., 8.], requires_grad=True)
    expected = adaptive_rescale_non_diffusion_losses(raw, pairs.is_diffusion)
    actual = torch.stack([raw[i] * adaptive_scale(raw[i], is_diffusion=i < 2,
                            diffusion_losses=[2., 4.]) for i in range(4)])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(ValueError):
        logical_time_pairs(torch.Generator(), batch_size=1)


def test_inference_rejects_checkpoint_and_anchor_protocol_mismatch():
    config = dict(anchor_mode='rgb', chunk_frames=5, history_chunks=5,
        action_prefix_mode='causal', action_feedback=True, flow_shift=2.22,
        action_adapter='initial_action.pt')
    meta = dict(objective='TF-AnyFlow v1.5', config=config, optimizer_step=4)
    infer = dict(config, anchor_mode='dynamic_last_frame_rgb_dual',
                 causal_action_adapter='saved_action.pt')
    validate_checkpoint_protocol(meta, meta, infer)
    with pytest.raises(ValueError, match='steps differ'):
        validate_checkpoint_protocol(meta, dict(meta, optimizer_step=8), infer)
    with pytest.raises(ValueError, match='anchor_mode mismatch'):
        validate_checkpoint_protocol(meta, meta, dict(infer, anchor_mode='dynamic_last_frame_dual'))
    with pytest.raises(ValueError, match='flow_shift mismatch'):
        validate_checkpoint_protocol(meta, meta, dict(infer, flow_shift=12.))


@pytest.mark.parametrize('t,r', [(.7, .7), (.7, 0.), (.7, .23), (1., .99), (.002, 0.),
                               (.5731241703033447, .5731241703033447)])
def test_loss_and_gradient_match_solarwm_native_sign_conversion(t, r):
    # Numerical oracle is the official implementation, not duplicated expected
    # values from this port. SolarWM itself is only needed for this oracle test.
    sys.path.insert(0, str(ROOT.parent / 'SolarWM/src'))
    from solarwm.backends.minimax_h3.anyflow_loss import h3_anyflow_v15_loss
    torch.manual_seed(10)
    clean, noise = torch.randn(1, 3, 4), torch.randn(1, 3, 4)
    p = torch.nn.Parameter(torch.tensor(.37))
    calls = []
    def velocity(x, s, target):
        assert float(target) <= float(s)
        calls.append(torch.is_grad_enabled())
        return torch.tanh(p * x + torch.as_tensor(s) * .11 + torch.as_tensor(target) * .19)
    raw, weight, _ = anyflow_sample_loss(velocity, clean, noise, t, r)
    ours = raw * weight
    ours_grad, = torch.autograd.grad(ours, p)
    assert calls == [False, False, False, True]
    def native(x, data_t, data_r):
        return -velocity(x, 1 - data_t, 1 - data_r)
    reference = h3_anyflow_v15_loss(native, clean, noise, torch.tensor([t]),
        torch.tensor([r]), torch.tensor([t == r]), epsilon=5., shift=2.22)
    reference_grad, = torch.autograd.grad(reference, p)
    torch.testing.assert_close(ours, reference, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(ours_grad, reference_grad, atol=2e-6, rtol=2e-5)


def test_diagonal_is_fm_and_finite_map_sign_hits_clean_endpoint():
    clean, noise = torch.randn(1, 2, 3), torch.randn(1, 2, 3)
    p = torch.nn.Parameter(torch.tensor(.2))
    fn = lambda x, t, r: p * x
    loss, _, _ = anyflow_sample_loss(fn, clean, noise, .6, .6)
    expected = ((p * (.4 * clean + .6 * noise)) - (noise - clean)).square().mean()
    torch.testing.assert_close(loss, expected)
    current = noise.clone()
    for t, r in [(1., .75), (.75, .5), (.5, .25), (.25, 0.)]:
        current = finite_map_step(current, noise - clean, t, r)
    torch.testing.assert_close(current, clean)
    tiny = finite_map_step(torch.ones(1, dtype=torch.bfloat16),
                           torch.ones(1, dtype=torch.bfloat16), .501, .5)
    assert tiny.dtype == torch.float32
    torch.testing.assert_close(tiny, torch.tensor([.999]))


def test_fp32_trajectory_with_bf16_h3_prediction_and_clean_commit():
    torch.set_num_threads(2)
    torch.manual_seed(22)
    model = make_small_h3().eval().to(torch.bfloat16)
    install_anyflow(model)
    batch = synthetic_h3_batch(frames=7)
    common = dict(full_packed=batch['packed'],
        prompt=batch['prompt_embeds'].to(torch.bfloat16),
        anchor=batch['anchor_rows'].to(torch.bfloat16),
        audio=batch['audio_latents'].to(torch.bfloat16), chunk_frames=5)
    cache = H3ChunkCache(2, 'cpu')
    current = batch['noise'][:, :, :5].clone()
    with torch.no_grad():
        for t, r in [(1., .5), (.5, 0.)]:
            before = current.clone()
            velocity = chunk_forward(model, current, sigma=t, target_sigma=r,
                index=0, cache=cache, **common)
            assert current.dtype == torch.float32
            torch.testing.assert_close(current, before, atol=0, rtol=0)
            current = finite_map_step(current, velocity, t, r)
        chunk_forward(model, current, sigma=0., index=0, cache=cache,
                      commit=True, **common)
        velocity = chunk_forward(model, batch['noise'][:, :, 5:], sigma=.5,
            target_sigma=0., index=1, cache=cache, **common)
    assert torch.isfinite(velocity).all()
    assert cache.commits == len(model.blocks)


def test_rgb_anchor_honors_requested_dtype_with_fp32_history():
    from causal.h3_cached import last_frame_image_anchor
    class VAE:
        def decode_video(self, history, **kwargs):
            return torch.zeros(1, 3, 5, 4, 4, dtype=kwargs['dtype'])
        def encode_video(self, rgb, **kwargs):
            return torch.zeros(1, 24, 1, 4, 4, dtype=kwargs['dtype'])
    anchor = last_frame_image_anchor(VAE(), torch.zeros(1, 24, 5, 4, 4),
                                     dtype=torch.bfloat16)
    assert anchor.dtype == torch.bfloat16


def test_actual_h3_pair_times_prefix_invariance_gradients_and_reload(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(11)
    model = make_small_h3().eval()
    batch = synthetic_h3_batch(frames=7)
    common = dict(full_packed=batch['packed'], prompt=batch['prompt_embeds'],
        anchor=batch['anchor_rows'], audio=batch['audio_latents'], chunk_frames=5)
    current = batch['noise'][:, :, :5]
    with torch.no_grad():
        original = chunk_forward(model, current, sigma=1., index=0,
            cache=H3ChunkCache(2, 'cpu'), **common)
    rng = torch.get_rng_state().clone()
    module = install_anyflow(model)
    assert torch.equal(rng, torch.get_rng_state())
    cache = H3ChunkCache(2, 'cpu')
    captured = []
    def hook(_, args, kw):
        captured.append(kw)
    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    with torch.no_grad():
        diagonal = chunk_forward(model, current, sigma=1., target_sigma=1.,
            index=0, cache=cache, **common)
        # Random diagonal times must stay equal after conversion to H3's
        # float32 0..1000 time units (float/tensor mixing used to reject r=t).
        chunk_forward(model, current, sigma=.5731241703033447,
            target_sigma=.5731241703033447, index=0, cache=cache, **common)
        shifted = chunk_forward(model, current, sigma=1., target_sigma=.4,
            index=0, cache=cache, **common)
    handle.remove()
    torch.testing.assert_close(diagonal, original, atol=2e-6, rtol=2e-5)
    assert float((diagonal - shifted).abs().max()) > 1e-5
    row_times = captured[-1]['unique_timesteps'][captured[-1]['inverse_indices']]
    targets = captured[-1]['unique_target_timesteps'][captured[-1]['inverse_indices']]
    video_start = int(captured[-1]['action_video_start'])
    torch.testing.assert_close(targets[:video_start], row_times[:video_start], atol=0, rtol=0)
    torch.testing.assert_close(targets[video_start:], torch.full_like(targets[video_start:], .6))
    # Audio and video both have native t=0 at sigma=1 but need distinct r.
    assert len(captured[-1]['unique_timesteps']) > len(torch.unique(row_times))
    with torch.no_grad():
        chunk_forward(model, batch['clean_video'][:, :, :5], sigma=0., index=0,
            cache=cache, commit=True, **common)
    before = cache.commits
    def velocity(x, t, r):
        return chunk_forward(model, x, sigma=t, target_sigma=r, index=1,
            cache=cache, allow_grad_read=True, use_gradient_checkpointing=True, **common)
    loss, weight, _ = anyflow_sample_loss(velocity, batch['clean_video'][:, :, 5:],
        batch['noise'][:, :, 5:], .7, .2)
    (loss * weight).backward()
    assert module.delta.proj_out.weight.grad.abs().sum() > 0
    assert model.blocks[-1].attn.qkv_proj.lora_B.grad.abs().sum() > 0
    assert cache.commits == before
    assert all(not e.key.requires_grad and not e.value.requires_grad
               for entries in cache.layers.values() for e in entries)
    path = tmp_path / 'anyflow.pt'
    save_anyflow(path, module, {'test': True})
    with torch.no_grad():
        expected = velocity(batch['noise'][:, :, 5:], .7, .2)
    del model.anyflow_conditioner
    loaded, meta = load_anyflow(model, path, 'cpu')
    assert meta['test']
    with torch.no_grad():
        actual = velocity(batch['noise'][:, :, 5:], .7, .2)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match='target_sigma'):
        chunk_forward(model, current, sigma=.5, index=0,
            cache=H3ChunkCache(2, 'cpu'), **common)
