"""Isolated r=t compute shortcut for deterministic, read-only H3 velocity calls.

No main trainer or running experiment is changed. Off-diagonal samples use
the existing AnyFlow implementation. The skipped derivative is unmeasured,
not zero; its coefficient in the diagonal residual is exactly zero.
"""
import math
import torch
from causal.anyflow import anyflow_sample_loss as reference_loss
from causal.anyflow_reference import gaussian_timestep_weights


def sample_loss(velocity, clean, noise, sigma, target_sigma, *, shift=2.22,
                epsilon=5.0, num_train_timesteps=1000, preserve_fp32_inputs=False):
    sigma, target_sigma = float(sigma), float(target_sigma)
    if sigma != target_sigma:
        return reference_loss(velocity, clean, noise, sigma, target_sigma,
            shift=shift, epsilon=epsilon, num_train_timesteps=num_train_timesteps,
            preserve_fp32_inputs=preserve_fp32_inputs)
    if clean.shape != noise.shape or clean.shape[0] != 1:
        raise ValueError('Expected matching physical batch-one clean/noise tensors')
    if not 0 < sigma <= 1:
        raise ValueError('Diagonal training requires 0 < sigma <= 1')
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be positive and finite')
    if not isinstance(num_train_timesteps, int) or num_train_timesteps <= 0:
        raise ValueError('num_train_timesteps must be a positive integer')
    noisy = (1 - sigma) * clean.float() + sigma * noise.float()
    dtype = torch.float32 if preserve_fp32_inputs else clean.dtype
    prediction = velocity(noisy.to(dtype), sigma, target_sigma).float()
    loss = (prediction - (noise.float() - clean.float())).square().mean()
    raw_t = torch.tensor([sigma * num_train_timesteps], device=clean.device)
    weight = gaussian_timestep_weights(raw_t, shift=shift,
        num_train_timesteps=num_train_timesteps)[0]
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite diagonal AnyFlow residual')
    return loss, weight, dict(sigma=sigma, target_sigma=target_sigma,
        raw_loss=float(loss.detach()), weight=float(weight),
        finite_difference_norm=None, finite_difference_skipped=True,
        model_evaluations=1)
