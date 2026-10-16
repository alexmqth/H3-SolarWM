"""Differentiable Original-visible factual C2 FM probe with clean GT C1.

The visible 17 latents are jointly recomputed under Original Single-Egress
bidirectional attention. C1 is detached and marked fixed so its native video
timestep is sigma=0; only C2 velocity enters the factual loss. This path has
no KV cache, and keeps the ordinary sigma-dependent text/image prefix time.
"""
from __future__ import annotations

import torch
from causal.h3_cached import slice_packed
from causal.local_transition import DifferentiableWindowAttention


def original_clean_forward(dit, current, *, history, full_packed, prompt,
                           anchor, audio, sigma: float, checkpoint: bool = False,
                           offload: bool = False):
    from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3
    if not 0 < float(sigma) < 1 or history.shape[2] != 12 or current.shape[2] != 5:
        raise ValueError("Original-visible probe requires GT C1=12 and factual C2=5")
    if history.shape[0:2] != current.shape[0:2] or history.shape[-2:] != current.shape[-2:]:
        raise ValueError("history/current latent shapes differ")
    if len(full_packed["action_text_rows"]) != 17:
        raise ValueError("future action spans must be cropped to 17")
    rows = (current.shape[-2]//2) * (current.shape[-1]//2)
    if int(full_packed["seq_len"]) != int(full_packed["action_video_start"]) + 17*rows:
        raise ValueError("future video rows or padding remain visible")
    dtype = getattr(dit, "_h3_input_dtype", prompt.dtype)
    whole = torch.cat((history.detach(), current), dim=2).to(dtype)
    fixed = torch.ones_like(whole)
    fixed[:, :, :12] = 0
    packed = slice_packed(full_packed, 0, 17, rows)
    control = DifferentiableWindowAttention(packed, 17, rows, 0)
    output = model_fn_minimax_h3(dit, whole, audio.to(dtype), packed, prompt,
        timestep_video=torch.tensor(float(sigma)*1000, device=current.device),
        timestep_audio=torch.tensor(1000., device=current.device),
        keyframe_cond_anchor=anchor.to(dtype),
        input_latents_video=whole, denoise_mask_video=fixed,
        fixed_prefix_timesteps=False, causal_control=control,
        use_gradient_checkpointing=checkpoint,
        use_gradient_checkpointing_offload=offload)[0]
    return output[:, :, 12:]
