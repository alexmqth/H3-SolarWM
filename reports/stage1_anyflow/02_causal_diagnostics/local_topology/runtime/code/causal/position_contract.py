"""Keep H3's media/action RoPE origin independent of future clause lengths.

This utility is deliberately opt-in. A reference text length must be chosen
before receiving future actions and retained for the whole rollout. It does
not change token visibility or the within-video/action temporal coordinates.
Existing benchmark/trainer paths do not enable it automatically.
"""
from __future__ import annotations

import torch


def initial_action_text_reference(packed: dict) -> int:
    """Calibrate from the static head, known horizon and FIRST action only.

    Constant-action clips keep their native coordinates exactly; changing
    later clauses cannot change this reference. The frame horizon is assumed
    known before rollout, as in the current 39/124-frame benchmark.
    """
    spans = packed.get('action_text_rows')
    if spans is None or len(spans) == 0:
        raise ValueError('Expected the H3 per-latent action text layout')
    spans = torch.as_tensor(spans)
    first_start, first_end = map(int, spans[0])
    if first_start < 1 or first_end <= first_start:
        raise ValueError('Invalid first action span')
    return first_start + len(spans) * (first_end - first_start)


def fixed_action_position_origin(packed: dict, *, reference_text_length: int) -> dict:
    """Reposition a FULL packed layout using a fixed calibrated text origin.

    Native H3 puts media at ``text_len`` and action k at
    ``text_len - video_span - 1 + offset[k]``. Thus future action text lengths
    can alter past query/head-key relative positions even with causal masks.
    Shift valid action/media rows together while retaining head positions.

    For a layout whose text length equals the reference, this is exactly a
    no-op numerically. The caller must record the reference in the inference
    and training protocol; selecting it from each new full action sequence
    would reintroduce the same future dependency.
    """
    if not isinstance(reference_text_length, int) or reference_text_length < 1:
        raise ValueError('reference_text_length must be a positive integer')
    if 'action_position_reference' in packed:
        if packed['action_position_reference'] != reference_text_length:
            raise ValueError('Cannot change an established action position reference')
        return dict(packed)
    spans = packed.get('action_text_rows')
    if spans is None or len(spans) == 0:
        raise ValueError('Expected the H3 per-latent action text layout')
    spans = torch.as_tensor(spans)
    head_rows = int(spans[0, 0])
    text_rows = int(packed['text_pos'].numel())
    if head_rows < 1 or reference_text_length < head_rows:
        raise ValueError('Reference must leave room for the static head')
    positions = packed['img_position_ids']
    first_video = int(packed['action_video_start'])
    if float(positions[0, first_video, 0]) != text_rows:
        raise ValueError('Apply the position contract to the full native layout before slicing')
    delta = reference_text_length - text_rows
    if float(positions[0, head_rows, 0]) + delta < head_rows:
        raise ValueError('Reference would put action positions inside the static head')
    result = dict(packed)
    result['img_position_ids'] = positions.clone()
    indices = torch.arange(positions.shape[1], device=positions.device)
    move = (indices >= head_rows) & (packed['token_tags'].to(positions.device) >= 0)
    result['img_position_ids'][0, move, 0] += delta
    result['action_position_reference'] = reference_text_length
    result['action_position_native_text_length'] = text_rows
    return result
