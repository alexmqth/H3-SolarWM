"""Original directed visible-window inference with explicit global interval.

Isolated evaluation, no persistent hidden KV or training. All known history
is recomputed; original I0 coordinates and native text times are retained.
"""
import torch
from causal.h3_cached import slice_packed
from causal.local_topology import OriginalWindowAttention


def interval_forward(dit, current, *, start, history, history_noise, mode,
                     full_packed, prompt, anchor, audio, sigma):
    from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3
    if torch.is_grad_enabled() or hasattr(dit, 'anyflow_conditioner'):
        raise RuntimeError('Ordinary FM, no-grad inference only')
    if not isinstance(start, int) or start < 0 or history.shape[2] != start:
        raise ValueError('History must be exactly global latent [0:start]')
    if history.shape != history_noise.shape or history.dtype != history_noise.dtype:
        raise ValueError('History noise shape/dtype mismatch')
    if mode not in ('clean', 'sigma_noised') or not 0 <= sigma <= 1:
        raise ValueError('Invalid history mode or sigma')
    stop = start + current.shape[2]
    rows = (current.shape[-2]//2)*(current.shape[-1]//2)
    if anchor.shape[0] != rows or len(full_packed['action_text_rows']) != stop:
        raise ValueError('Require single original I0 and already visibility-trimmed layout')
    dtype = getattr(dit, '_h3_input_dtype', prompt.dtype)
    temporary = history if mode == 'clean' else (1-sigma)*history + sigma*history_noise
    whole = torch.cat([temporary, current], dim=2).to(dtype)
    packed = slice_packed(full_packed, 0, stop, rows)
    update = torch.ones_like(whole)
    update[:, :, :start] = 0
    control = OriginalWindowAttention(packed, stop, rows, 0)
    out = model_fn_minimax_h3(dit, whole, audio.to(dtype), packed, prompt,
        timestep_video=torch.tensor(sigma*1000, device=current.device),
        timestep_audio=torch.tensor(1000., device=current.device),
        keyframe_cond_anchor=anchor.to(dtype),
        input_latents_video=(whole if mode == 'clean' else None),
        denoise_mask_video=update, fixed_prefix_timesteps=False,
        causal_control=control)[0]
    return out[:, :, start:]
