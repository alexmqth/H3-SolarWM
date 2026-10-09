"""Explicit legacy or native FP32 H3 boundary policy, with CPU offload.

The main transformer stack remains in its loaded dtype. Native H3 stores
the six input/output/time projections in FP32; preserve these values and
keep interpolated/noisy inputs and time SiLU in FP32 until their intended
projection/packing boundary. No optimizer parameter is changed here.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

import torch
from torch import nn

OWNERS = ('video_patch_proj', 'audio_patch_proj', 'time_embedder.proj_in',
          'time_embedder.proj_out', 'final_layer.video_out', 'final_layer.audio_out')


def _cast_input(module, args):
    return (args[0].to(getattr(module, 'computation_dtype', module.weight.dtype)), *args[1:])


def configure_precision(model, profile, *, native_transformer_dir=None):
    if profile not in ('legacy', 'h3_fp32'):
        raise ValueError(f'Unknown H3 precision profile: {profile}')
    if hasattr(model, '_h3_precision_report'):
        raise ValueError('H3 precision policy is already installed')
    if profile == 'legacy':
        report = dict(profile='legacy', native_fp32_weights_restored=False)
        model._h3_precision_report = report
        return report
    linears = {name: model.get_submodule(name) for name in OWNERS}
    weights = {}
    if native_transformer_dir is not None:
        from safetensors import safe_open
        required = {f'{name}.{slot}' for name in OWNERS for slot in ('weight', 'bias')}
        for path in sorted(Path(native_transformer_dir).glob('*.safetensors')):
            with safe_open(path, framework='pt', device='cpu') as file:
                for name in required.intersection(file.keys()):
                    if name in weights:
                        raise ValueError(f'Duplicate native tensor: {name}')
                    if file.get_slice(name).get_dtype() != 'F32':
                        raise ValueError(f'Native boundary tensor is not FP32: {name}')
                    weights[name] = file.get_tensor(name)
        if set(weights) != required:
            raise ValueError(f'Missing native FP32 boundary tensors: {sorted(required-set(weights))}')
    # Validate the entire restore before any mutation, including offload mode.
    for name, module in linears.items():
        if module.weight.is_meta or getattr(module, 'disk_offload', False):
            raise ValueError('FP32 policy requires materialized CPU/GPU boundary weights')
        for slot in ('weight', 'bias'):
            if weights and weights[f'{name}.{slot}'].shape != getattr(module, slot).shape:
                raise ValueError(f'Native tensor shape mismatch: {name}.{slot}')
    handles = []
    with torch.no_grad():
        for name, module in linears.items():
            for field in ('offload_dtype', 'onload_dtype', 'preparing_dtype', 'computation_dtype'):
                if hasattr(module, field):
                    setattr(module, field, torch.float32)
            module.to(dtype=torch.float32)
            for slot in ('weight', 'bias'):
                if weights:
                    getattr(module, slot).copy_(weights[f'{name}.{slot}'].to(getattr(module, slot).device))
            handles.append(module.register_forward_pre_hook(_cast_input))
    for projection in [b.adaln_proj for b in model.blocks] + [model.final_layer.adaln_proj]:
        # Activation happens before this hook, at FP32 time-embedding dtype.
        handles.append(projection.linear.register_forward_pre_hook(_cast_input))
    model._h3_input_dtype = torch.float32
    model._h3_time_dtype = torch.float32
    model._h3_precision_hooks = handles
    report = dict(profile='h3_fp32', native_fp32_weights_restored=bool(weights),
        boundary_owners=list(OWNERS), interpolated_inputs='float32', time_condition='float32',
        transformer_stack='unchanged loaded dtype',
        native_tensor_sha256={name: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
                              for name, value in weights.items()})
    model._h3_precision_report = report
    return report


def validate_precision_checkpoint(metadata, installed_report):
    expected = metadata.get('config', {}).get('precision_profile', 'legacy')
    if expected != installed_report['profile']:
        raise ValueError(f'Checkpoint precision mismatch: trained={expected}, installed={installed_report["profile"]}')
    if expected == 'h3_fp32':
        saved = metadata.get('precision')
        if saved is None or saved['native_tensor_sha256'] != installed_report['native_tensor_sha256']:
            raise ValueError('Native FP32 weight provenance differs from training')
