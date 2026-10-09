"""Inference sigma grids, separate from AnyFlow's training time distribution."""
from __future__ import annotations

import math

import torch


def configure_video_schedule(scheduler, *, steps, grid='native', flow_shift=2.22):
    """Configure native H3 or the uniform finite-map grid used by SolarWM Stage1.

    Training's ``flow_shift`` still defines its sampled time-pair distribution
    and loss weights. It need not equal an inference grid's effective shift.
    This function never changes the audio scheduler, model, or random state.
    The returned list includes the final zero endpoint used by finite maps.
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError('steps must be a positive integer')
    if grid not in ('native', 'uniform'):
        raise ValueError('grid must be native or uniform')
    if not math.isfinite(flow_shift) or flow_shift <= 0:
        raise ValueError('flow_shift must be positive and finite')
    if scheduler.num_train_timesteps != 1000:
        raise ValueError('H3 benchmark uses timesteps scaled by 1000')
    if grid == 'native':
        # Preserve the exact legacy path, including its FP32 rounding.
        scheduler.set_timesteps(steps, shift=flow_shift)
    else:
        # Official H3 Stage1: torch.linspace(1, 0, 5) for four maps.
        scheduler.sigmas = torch.linspace(1., 0., steps + 1, dtype=torch.float32)[:-1]
        scheduler.timesteps = scheduler.sigmas * scheduler.num_train_timesteps
        scheduler.training = False
    return [float(t) / 1000 for t in scheduler.timesteps] + [0.]
