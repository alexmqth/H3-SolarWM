"""Process-local precision ablation; no edits to the frozen training runtime.

Keep identical (already BF16-rounded) parameter VALUES. Change only compute
precision at the time MLP / mixture / activation and optional input/output
heads. This is not native-FP32 weight restoration or a full SolarWM port.
"""
from contextlib import contextmanager

import torch


@contextmanager
def precision_profile(model, profile):
    if profile not in ('legacy', 'time_fp32', 'boundary_fp32'):
        raise ValueError(profile)
    if profile == 'legacy':
        yield {'profile': profile, 'native_fp32_weights_restored': False}
        return
    handles, saved = [], []
    dtype_fields = ('offload_dtype', 'onload_dtype', 'preparing_dtype', 'computation_dtype')

    def cast_input(module, args):
        dtype = getattr(module, 'computation_dtype', module.weight.dtype)
        return (args[0].to(dtype), *args[1:])

    def time_input(module, args, kwargs):
        return args, dict(kwargs, dtype=torch.float32)

    def preserve_mix(module, args, kwargs):
        return args, dict(kwargs, dtype=torch.float32)

    linears = [model.time_embedder.proj_in, model.time_embedder.proj_out]
    if profile == 'boundary_fp32':
        linears += [model.video_patch_proj, model.audio_patch_proj,
                    model.final_layer.video_out, model.final_layer.audio_out]
    try:
        for module in linears:
            if getattr(module, 'disk_offload', False) or module.weight.is_meta:
                raise ValueError('Probe requires materialized CPU/GPU weights')
            saved.append((module, module.weight.dtype,
                          {name: getattr(module, name) for name in dtype_fields if hasattr(module, name)}))
            for name in saved[-1][2]:
                setattr(module, name, torch.float32)
            module.to(dtype=torch.float32)
            handles.append(module.register_forward_pre_hook(cast_input))
        handles.append(model.time_embedder.register_forward_pre_hook(time_input, with_kwargs=True))
        conditioner = getattr(model, 'anyflow_conditioner', None)
        if conditioner is not None:
            handles.append(conditioner.register_forward_pre_hook(preserve_mix, with_kwargs=True))
        # SiLU still runs in FP32; only its output is rounded for BF16 AdaLN
        # matrix multiplication, as in official H3's modulation classes.
        for proj in [b.adaln_proj for b in model.blocks] + [model.final_layer.adaln_proj]:
            handles.append(proj.linear.register_forward_pre_hook(cast_input))
        yield {'profile': profile, 'fp32_linears': len(linears),
               'native_fp32_weights_restored': False,
               'packed_hidden_dtype': 'unchanged', 'input_latent_rounding': 'unchanged'}
    finally:
        for handle in handles:
            handle.remove()
        for module, original, policies in saved:
            for name, value in policies.items():
                setattr(module, name, value)
            module.to(dtype=original)
