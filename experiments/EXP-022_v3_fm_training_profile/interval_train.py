"""Differentiable current interval under the frozen V3 attention protocol.

This is intentionally separate from EXP-002's inference-only interval_cached.
The caller must physically trim the layout with visible_inputs and install
current_prefix_feedback. Clean commits must run under torch.no_grad().
"""
from __future__ import annotations

import torch
from causal.h3_cached import ChunkAttention, H3ChunkCache, slice_packed


def interval_train(dit, current, *, start: int, index: int, cache: H3ChunkCache,
                   full_packed: dict, prompt: torch.Tensor, anchor: torch.Tensor,
                   audio: torch.Tensor, sigma: float, commit: bool = False,
                   checkpoint: bool = False, offload: bool = False):
    from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3

    attend = ChunkAttention.attend
    if attend.__module__ != "current_prefix" or attend.__qualname__ != "current_prefix_feedback.<locals>.checked":
        raise RuntimeError("frozen current_prefix_feedback context required")
    if hasattr(dit, "anyflow_conditioner"):
        raise RuntimeError("ordinary FM only; no AnyFlow conditioner")
    if not isinstance(current, torch.Tensor) or current.ndim != 5 or current.shape[0] != 1:
        raise ValueError("current must be [1,C,T,H,W]")
    if (start, index, current.shape[2]) not in ((0, 0, 12), (12, 1, 5)):
        raise ValueError("only native V3 [12,5] intervals permitted")
    if not 0 <= float(sigma) <= 1 or (commit and float(sigma) != 0):
        raise ValueError("clean commit requires sigma=0")
    if commit and torch.is_grad_enabled():
        raise RuntimeError("clean history commit must be no_grad")
    if commit and (checkpoint or offload):
        raise ValueError("checkpoint/offload not used for no-grad commit")
    if offload and not checkpoint:
        raise ValueError("offload requires checkpoint")
    stop = start + current.shape[2]
    rows = (current.shape[-2] // 2) * (current.shape[-1] // 2)
    if anchor.shape[0] != rows or len(full_packed["action_text_rows"]) != stop:
        raise ValueError("Single I0 and physically trimmed visible action layout required")
    if len(prompt) != len(full_packed["text_pos"]):
        raise ValueError("visible prompt/layout mismatch")
    prefix = int(full_packed["action_video_start"])
    if int(full_packed["seq_len"]) != prefix + stop * rows:
        raise ValueError("future video rows or padding remain visible")
    if any([e.index for e in cache.history(layer, index)] != list(range(index)) for layer in cache.layers):
        raise RuntimeError("incomplete past cache")
    packed = slice_packed(full_packed, start, stop, rows)
    # Exact global coordinates are copied, not rebased to current chunk.
    assert torch.equal(packed["img_position_ids"][:, :prefix], full_packed["img_position_ids"][:, :prefix])
    assert torch.equal(packed["img_position_ids"][:, prefix:],
                       full_packed["img_position_ids"][:, prefix + start * rows:prefix + stop * rows])
    dtype = getattr(dit, "_h3_input_dtype", prompt.dtype)
    control = ChunkAttention(cache, index, prefix, rows, start,
                             packed["action_text_rows"], "own", True,
                             commit=commit, allow_grad_read=not commit)
    return model_fn_minimax_h3(
        dit, current.to(dtype), audio.to(dtype), packed, prompt,
        timestep_video=torch.tensor(float(sigma) * 1000, device=current.device),
        timestep_audio=torch.tensor(1000., device=current.device),
        keyframe_cond_anchor=anchor.to(dtype), fixed_prefix_timesteps=False,
        use_gradient_checkpointing=checkpoint,
        use_gradient_checkpointing_offload=offload,
        causal_control=control)[0]
