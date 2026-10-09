"""Additional isolated ablation: preserve FP32 perturbed video/audio inputs."""
from contextlib import contextmanager
import torch
from boundary_profile import precision_profile as boundary_profile

@contextmanager
def precision_profile(model, profile):
    if profile != 'boundary_fp32_inputs':
        with boundary_profile(model, profile) as metadata:
            yield metadata
        return
    if hasattr(model, '_diagnostic_input_dtype'):
        raise ValueError('Input precision profile is already active')
    with boundary_profile(model, 'boundary_fp32') as metadata:
        model._diagnostic_input_dtype = torch.float32
        try:
            yield dict(metadata, profile=profile, input_latent_rounding='FP32 through input projection',
                       noise_amplitudes='unchanged from legacy BF16 noise; only interpolation/perturbation retained in FP32')
        finally:
            del model._diagnostic_input_dtype
