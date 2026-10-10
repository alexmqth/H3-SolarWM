"""EXP-005 CPU-audited explicit interval entry for V3-SW-G and V3-SW-L.

No model is loaded by importing this file. A future GPU runner needs a
separately approved >37-latent fixture before index 6 can be used.
"""
from __future__ import annotations

import torch

from chunk_plan import PLAN, ChunkPlan, cache_identity, validate_cache
from position_sw import (assert_frozen_past_preserved, assert_native_video_grid,
                         localize_window, video_origin)


def interval_sw(dit, current, *, start: int, index: int, cache,
                full_packed: dict, prompt: torch.Tensor, anchor: torch.Tensor,
                audio: torch.Tensor, sigma: float, commit: bool = False,
                mode: str = "global", plan: ChunkPlan = PLAN,
                expected_layers: int = 50,
                frozen_reference: dict | None = None,
                frozen_reference_prompt: torch.Tensor | None = None,
                structural_probe_only: bool = False,
                return_audit: bool = False):
    from causal.h3_cached import ChunkAttention, slice_packed
    from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3

    # The accepted V3 path installs this attention edge through an external
    # context manager. Refuse a forward if the caller forgot that context.
    attend = ChunkAttention.attend
    if (attend.__module__ != "current_prefix" or
            attend.__qualname__ != "current_prefix_feedback.<locals>.checked"):
        raise RuntimeError("current_prefix_feedback() context is required")
    if torch.is_grad_enabled() or hasattr(dit, "anyflow_conditioner"):
        raise RuntimeError("EXP-005 supports no-grad ordinary FM inference only")
    if mode not in {"global", "local"}:
        raise ValueError("mode must be global or local")
    if not isinstance(current, torch.Tensor) or current.ndim != 5 or current.shape[0] != 1:
        raise ValueError("current must be [1,C,T,H,W]")
    if current.shape[-2] % 2 or current.shape[-1] % 2:
        raise ValueError("spatial latents must patchify by 2×2")
    expected_start, stop = plan.span(index)
    if start != expected_start or current.shape[2] != stop - start:
        raise ValueError("current latent interval differs from the explicit ChunkPlan")
    if not 0 <= float(sigma) <= 1 or (commit and float(sigma) != 0):
        raise ValueError("clean commit requires sigma=0")
    rows = (current.shape[-2] // 2) * (current.shape[-1] // 2)
    prefix = int(full_packed["action_video_start"])
    if anchor.shape[0] != rows or len(full_packed["action_text_rows"]) != stop:
        raise ValueError("require one I0 and physically visibility-trimmed actions")
    if len(prompt) != len(full_packed["text_pos"]):
        raise ValueError("visible prompt and text layout disagree")
    if int(full_packed["seq_len"]) != prefix + stop * rows:
        raise ValueError("require physically trimmed video; no future/padding rows")
    if stop > 37:
        if not structural_probe_only or current.device.type != "cpu":
            raise RuntimeError("post-37 GPU inference needs separately approved native fixture")
        if frozen_reference is None or frozen_reference_prompt is None:
            raise RuntimeError("post-37 inference requires a certified long native fixture")
        assert_frozen_past_preserved(full_packed, frozen_reference, frame_rows=rows)
        old_text_len = len(frozen_reference["text_pos"])
        if len(frozen_reference_prompt) != old_text_len or not torch.equal(
                prompt[:old_text_len], frozen_reference_prompt):
            raise RuntimeError("old prompt embeddings changed in long extension")
    origin = video_origin(frozen_reference or full_packed, rows)
    assert_native_video_grid(full_packed, frame_rows=rows, stop=stop, origin=origin)
    cache_audit = validate_cache(cache, plan=plan, index=index,
                                 frame_rows=rows, expected_layers=expected_layers)
    before = cache_identity(cache)
    packed = slice_packed(full_packed, start, stop, rows)
    assert torch.equal(packed["img_position_ids"][:, :prefix],
                       full_packed["img_position_ids"][:, :prefix])
    assert torch.equal(packed["img_position_ids"][:, prefix:],
                       full_packed["img_position_ids"][:, prefix + start*rows:prefix + stop*rows])
    position_audit = dict(index=index, remapped=False, ancestors=list(plan.ancestors(index)))
    attention_cache = cache
    if mode == "local":
        if not hasattr(dit, "rope"):
            raise TypeError("local mode needs actual H3 MM-RoPE callable")
        packed, attention_cache, position_audit = localize_window(
            canonical_cache=cache, packed=packed, visible_layout=full_packed,
            plan=plan, index=index, frame_rows=rows, rope_fn=dit.rope, origin=origin)
    dtype = getattr(dit, "_h3_input_dtype", prompt.dtype)
    control = ChunkAttention(attention_cache, index, prefix, rows, start,
                             packed["action_text_rows"], "own", True, commit=commit)
    output = model_fn_minimax_h3(
        dit, current.to(dtype), audio.to(dtype), packed, prompt,
        timestep_video=torch.tensor(float(sigma) * 1000, device=current.device),
        timestep_audio=torch.tensor(1000., device=current.device),
        keyframe_cond_anchor=anchor.to(dtype), fixed_prefix_timesteps=False,
        causal_control=control)[0]
    if commit:
        after_audit = validate_cache(cache, plan=plan, index=index + 1,
            frame_rows=rows, expected_layers=expected_layers)
    else:
        if cache_identity(cache) != before:
            raise RuntimeError("sampling mutated canonical history cache")
        after_audit = cache_audit
    audit = dict(mode=mode, span=[start, stop], cache_before=cache_audit,
                 cache_after=after_audit, position=position_audit,
                 prefix_rows=prefix, video_sigma=float(sigma), audio_timestep=1000.,
                 fixed_prefix_timesteps=False, action_prefix_mode="own",
                 action_feedback=True, anchor_rows=int(anchor.shape[0]))
    return (output, audit) if return_audit else output
