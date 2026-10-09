"""TF-AnyFlow v1.5 for H3-World's *noise-ward* scheduler velocity.

The finite-difference loss follows SolarWM's H3 implementation, after its
explicit native-H3 sign/time conversion. Here chunk_forward already returns
noise-clean, so applying that sign conversion a second time would be wrong.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from .anyflow_reference import (
    apply_timestep_shift, sample_anyflow_time_pairs, bounded_difference_timesteps,
    central_difference_derivative, gaussian_timestep_weights,
)


class H3AnyFlowConditioner(nn.Module):
    """Independent FP32 target-time MLP, cloned from the loaded H3 time MLP.

    Matches SolarWM's (1-gate)*time_embedding + gate*target_embedding.
    The fixed gate is not a trainable action/anchor gain.
    """

    def __init__(self, base, gate=0.25, device='cpu'):
        super().__init__()
        from diffsynth.models.minimax_h3_dit import MiniMaxH3TimeEmbedder
        if not math.isfinite(gate) or not 0 < gate <= 1:
            raise ValueError('AnyFlow gate must lie in (0, 1]')
        self.gate = float(gate)
        # Do not deepcopy AutoTorchModule: optimizer parameters must not be
        # offloaded/replaced behind Adam's back during VAE/model switching.
        self.delta = MiniMaxH3TimeEmbedder(
            base.frequency_embedding_size, base.proj_in.out_features,
            base.proj_out.out_features).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            for name in ('proj_in', 'proj_out'):
                source, dest = getattr(base, name), getattr(self.delta, name)
                if getattr(source, 'disk_offload', False):
                    source.preparing()
                if source.weight.is_meta:
                    raise ValueError('Time embedder must be materialized before AnyFlow cloning')
                dest.weight.copy_(source.weight.float().to(device))
                dest.bias.copy_(source.bias.float().to(device))
                # Preserve a loaded hot-LoRA if a future H3 checkpoint has it.
                for a, b in zip(getattr(source, 'lora_A_weights', ()),
                                getattr(source, 'lora_B_weights', ())):
                    dest.weight.add_(b.float().to(device) @ a.float().to(device))

    def forward(self, current_embedding, native_target_time, *, dtype):
        if native_target_time.ndim != 1 or current_embedding.shape[0] != native_target_time.numel():
            raise ValueError('Target times must match the current time embedding rows')
        with torch.autocast(device_type=current_embedding.device.type, enabled=False):
            target = self.delta(native_target_time.float(), dtype=torch.float32)
            return ((1 - self.gate) * current_embedding.float() + self.gate * target).to(dtype)


def install_anyflow(dit, *, device='cpu', gate=0.25):
    if hasattr(dit, 'anyflow_conditioner'):
        raise ValueError('AnyFlow is already installed; refusing to reset learned target times')
    # Cloning a module must not shift the subsequent noise RNG sequence.
    devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        module = H3AnyFlowConditioner(dit.time_embedder, gate, device)
    dit.anyflow_conditioner = module
    return module


def save_anyflow(path, module, metadata):
    path = Path(path)
    state = dict(format='h3world_anyflow_time_v1', gate=module.gate,
                 velocity_convention='noise-clean', native_time='1-sigma',
                 weights={k: v.detach().cpu() for k, v in module.state_dict().items()},
                 metadata=metadata)
    tmp = path.with_suffix('.tmp.pt')
    torch.save(state, tmp)
    tmp.replace(path)


def load_anyflow(dit, path, device):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state.get('format') != 'h3world_anyflow_time_v1'
            or state.get('velocity_convention') != 'noise-clean'):
        raise ValueError('Unsupported AnyFlow checkpoint/sign convention')
    module = install_anyflow(dit, device=device, gate=state['gate'])
    module.load_state_dict(state['weights'], strict=True)
    return module, state.get('metadata', {})


def logical_time_pairs(generator, *, shift=2.22, batch_size=4):
    """Logical batch of 2 FM + 1 endpoint + 1 finite-map samples per four.

    Materialize these sequentially on one GPU with gradient accumulation.
    Calling the official type allocator with physical batch=1, DP=1 instead
    would round both reference classes to zero and train only general maps.
    """
    if batch_size < 4 or batch_size % 4:
        raise ValueError('Logical AnyFlow batch must be a positive multiple of four')
    pairs = sample_anyflow_time_pairs(batch_size, generator=generator, device='cpu')
    return pairs._replace(t=apply_timestep_shift(pairs.t, shift),
                          r=apply_timestep_shift(pairs.r, shift))


def anyflow_sample_loss(velocity, clean, noise, sigma, target_sigma, *,
                        shift=2.22, epsilon=5.0, num_train_timesteps=1000, preserve_fp32_inputs=False):
    """Return UNRESCALED loss and weight for one logical-batch member.

    velocity(sample, sigma, target_sigma) returns noise-ward velocity. All four
    calls must use identical clean history, anchor, prompt, audio and action.
    Three detached target evaluations precede the single gradient prediction.
    Adaptive rescaling is done by the trainer across the logical batch, before
    applying this sample's Gaussian weight (exactly the official order).
    """
    sigma, target_sigma = float(sigma), float(target_sigma)
    if clean.shape != noise.shape or clean.shape[0] != 1:
        raise ValueError('Expected matching physical batch-one clean/noise tensors')
    if not 0 <= target_sigma <= sigma <= 1 or sigma == 0:
        raise ValueError('Training requires 0 <= target_sigma <= sigma <= 1 and sigma > 0')
    raw_t = torch.tensor([sigma * num_train_timesteps], device=clean.device)
    raw_r = torch.tensor([target_sigma * num_train_timesteps], device=clean.device)
    noisy = (1 - sigma) * clean.float() + sigma * noise.float()
    input_dtype = torch.float32 if preserve_fp32_inputs else clean.dtype
    with torch.no_grad():
        diagonal = velocity(noisy.to(input_dtype), sigma, sigma).float()
        plus_t, minus_t = bounded_difference_timesteps(
            raw_t, raw_r, epsilon=epsilon, num_train_timesteps=num_train_timesteps)
        plus_sigma = min(1.0, float(plus_t[0]) / num_train_timesteps)
        # Returning a bounded float32 raw time to Python can round it just
        # below r, particularly on the r=t diagonal. Keep the legal interval.
        minus_sigma = max(target_sigma, float(minus_t[0]) / num_train_timesteps)
        plus = noisy + diagonal * (plus_sigma - sigma)
        minus = noisy - diagonal * (sigma - minus_sigma)
        plus_v = velocity(plus.to(input_dtype), plus_sigma, target_sigma).float()
        minus_v = velocity(minus.to(input_dtype), minus_sigma, target_sigma).float()
        derivative = central_difference_derivative(plus_v, minus_v, plus_t, minus_t)
    del diagonal, plus, minus, plus_v, minus_v
    prediction = velocity(noisy.to(input_dtype), sigma, target_sigma).float()
    residual = prediction + (raw_t - raw_r).view(*([1] * prediction.ndim)) * derivative - (noise.float() - clean.float())
    loss = residual.square().mean()
    weight = gaussian_timestep_weights(raw_t, shift=shift, num_train_timesteps=num_train_timesteps)[0]
    stats = dict(sigma=sigma, target_sigma=target_sigma,
                 raw_loss=float(loss.detach()), weight=float(weight),
                 finite_difference_norm=float(derivative.norm()), model_evaluations=4)
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite AnyFlow residual')
    return loss, weight, stats


def adaptive_scale(raw_loss, *, is_diffusion, diffusion_losses):
    if is_diffusion:
        return raw_loss.new_tensor(1.0)
    if not diffusion_losses:
        raise ValueError('Logical batch must evaluate diffusion references first')
    reference = raw_loss.new_tensor(diffusion_losses).mean()
    return (reference / (raw_loss.detach() + 1e-5)).detach()


def finite_map_step(current, velocity, sigma, target_sigma):
    if not 0 <= float(target_sigma) <= float(sigma) <= 1:
        raise ValueError('Invalid finite-map interval')
    # Match the official finite-map sampler's FP32 trajectory state; the
    # backbone may still evaluate in BF16. Avoid rounding every map update.
    return current.float() + (float(target_sigma) - float(sigma)) * velocity.float()


def validate_checkpoint_protocol(time_metadata, qkv_metadata, inference):
    """Reject silent mixing of checkpoint steps or anchor/routing protocols."""
    if time_metadata.get('objective') != 'TF-AnyFlow v1.5':
        raise ValueError('Target-time checkpoint is not a TF-AnyFlow training result')
    if qkv_metadata.get('objective') != time_metadata['objective']:
        raise ValueError('AnyFlow requires its matching trained QKV checkpoint')
    if time_metadata.get('optimizer_step') != qkv_metadata.get('optimizer_step'):
        raise ValueError('AnyFlow time/QKV checkpoint steps differ')
    config = time_metadata['config']
    if config != qkv_metadata.get('config'):
        raise ValueError('AnyFlow time/QKV training protocols differ')
    anchor_modes = {'fixed': 'fixed', 'latent': 'dynamic_last_frame_dual',
                    'rgb': 'dynamic_last_frame_rgb_dual'}
    expected = dict(anchor_mode=anchor_modes[config['anchor_mode']],
        chunk_frames=config['chunk_frames'], history_chunks=config['history_chunks'],
        action_prefix_mode=config['action_prefix_mode'],
        action_feedback=config['action_feedback'], flow_shift=config['flow_shift'])
    for key, value in expected.items():
        if inference[key] != value:
            raise ValueError(f'AnyFlow {key} mismatch: trained={value}, inference={inference[key]}')
    if config.get('action_adapter') and not inference.get('causal_action_adapter'):
        raise ValueError('This AnyFlow checkpoint requires the frozen action adapter used in training')
