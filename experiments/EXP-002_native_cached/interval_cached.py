"""Native-time, explicit-global-interval H3 inference with persistent raw video KV.

This is deliberately separate from the fixed-size/fixed-prefix-time historical
``chunk_forward`` entry. The current-prefix attention implementation and cache
data structure are reused unchanged. Future action/video rows must already be
removed with ``visible_inputs`` before calling this function.
"""
import torch

from causal.h3_cached import ChunkAttention, H3ChunkCache, slice_packed


def interval_cached(dit, current, *, start, index, cache: H3ChunkCache,
                    full_packed, prompt, anchor, audio, sigma, commit=False):
    from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3

    if torch.is_grad_enabled() or hasattr(dit, "anyflow_conditioner"):
        raise RuntimeError("EXP-002 supports no-grad ordinary FM inference only")
    if not isinstance(start, int) or start < 0 or index not in (0, 1, 2):
        raise ValueError("invalid global start or cache chunk index")
    if not 0 <= float(sigma) <= 1 or (commit and float(sigma) != 0):
        raise ValueError("clean cache commit requires sigma=0")
    stop = start + current.shape[2]
    if (start, stop, index) not in ((0, 12, 0), (12, 17, 1), (17, 22, 2)):
        raise ValueError("outside registered 12→5→5 partition")
    rows = (current.shape[-2] // 2) * (current.shape[-1] // 2)
    if anchor.shape[0] != rows or len(full_packed["action_text_rows"]) != stop:
        raise ValueError("require Single I0 and a physically visibility-trimmed layout")
    if not all(len(cache.history(layer, index)) == index for layer in cache.layers):
        raise RuntimeError("incomplete cache history")
    packed = slice_packed(full_packed, start, stop, rows)
    # Global RoPE is copied by slice_packed, never rebased to the local chunk.
    assert torch.equal(packed["img_position_ids"][:, -current.shape[2]*rows:],
                       full_packed["img_position_ids"][:, -current.shape[2]*rows:])
    dtype = getattr(dit, "_h3_input_dtype", prompt.dtype)
    control = ChunkAttention(cache, index, int(packed["action_video_start"]), rows,
                             start, packed["action_text_rows"], "own", True,
                             commit=commit)
    return model_fn_minimax_h3(
        dit, current.to(dtype), audio.to(dtype), packed, prompt,
        timestep_video=torch.tensor(float(sigma)*1000, device=current.device),
        timestep_audio=torch.tensor(1000., device=current.device),
        keyframe_cond_anchor=anchor.to(dtype), fixed_prefix_timesteps=False,
        causal_control=control)[0]
