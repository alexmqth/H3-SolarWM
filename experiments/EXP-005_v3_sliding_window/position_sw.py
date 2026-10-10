"""V3-SW-L research-only read-time MM-RoPE remapping.

Raw K/V and their canonical *global* RoPE frequencies stay in H3ChunkCache.
At read time only retained video K and current video Q/K use a local native
temporal grid. Static text/action/I0/audio prefix positions are untouched.
This is a candidate geometry, not an equivalence claim to full recomputation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from chunk_plan import ChunkPlan


def native_times(count: int, origin: float, *, device=None) -> torch.Tensor:
    """Use the frozen H3 builder's nonuniform (1,4,4,4,4)×5/3 grid."""
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder

    if not isinstance(count, int) or count < 1:
        raise ValueError("native grid count must be positive")
    builder = MiniMaxH3Unit_PackedSequenceBuilder()
    return builder._video_t_grid(count, float(origin)).to(device=device)


def video_origin(frozen, frame_rows: int) -> float:
    p = int(frozen["action_video_start"])
    if frame_rows < 1 or frozen["img_position_ids"].shape[1] < p + frame_rows:
        raise ValueError("missing first complete video latent frame")
    origin = float(frozen["img_position_ids"][0, p, 0])
    return origin


def assert_native_video_grid(layout, *, frame_rows: int, stop: int, origin: float) -> None:
    p = int(layout["action_video_start"])
    positions = layout["img_position_ids"]
    if positions.shape[1] != p + stop * frame_rows:
        raise ValueError("layout must be visibility-trimmed with complete video frames")
    times = native_times(stop, origin, device=positions.device)
    actual = positions[0, p:, 0].reshape(stop, frame_rows)
    torch.testing.assert_close(actual, times[:, None].expand_as(actual), rtol=0, atol=1e-9)
    spatial = positions[0, p:, 1:].reshape(stop, frame_rows, 2)
    torch.testing.assert_close(spatial, spatial[0:1].expand_as(spatial), rtol=0, atol=0)


def assert_frozen_past_preserved(candidate, frozen, *, frame_rows: int,
                                 frozen_latents: int = 37) -> None:
    """Fail if a proposed >37 layout changes old text, I0, audio or video coords.

    The physical prefix can grow with new action tokens, so compare semantic
    row selectors rather than requiring equal packed offsets or equal length.
    This does not certify new action embeddings or a usable long fixture.
    """
    if len(candidate["action_text_rows"]) <= frozen_latents:
        raise ValueError("candidate must contain new action spans")
    if not torch.equal(candidate["action_text_rows"][:frozen_latents],
                       frozen["action_text_rows"][:frozen_latents]):
        raise RuntimeError("existing action row spans changed")
    selectors = {
        "old_text": (frozen["text_pos"], candidate["text_pos"][:len(frozen["text_pos"])]),
        "I0": (frozen["img_pos"][:frame_rows], candidate["img_pos"][:frame_rows]),
        "audio": (frozen["audio_pos"], candidate["audio_pos"][:len(frozen["audio_pos"])]),
        "first37_video": (
            frozen["img_pos"][frame_rows:frame_rows * (frozen_latents + 1)],
            candidate["img_pos"][frame_rows:frame_rows * (frozen_latents + 1)]),
    }
    for name, (old_rows, new_rows) in selectors.items():
        if len(old_rows) != len(new_rows):
            raise RuntimeError(f"{name} row count changed")
        old_pos = frozen["img_position_ids"][0, old_rows]
        new_pos = candidate["img_position_ids"][0, new_rows]
        if not torch.equal(old_pos, new_pos):
            raise RuntimeError(f"{name} RoPE coordinates changed")
        if not torch.equal(frozen["token_tags"][old_rows], candidate["token_tags"][new_rows]):
            raise RuntimeError(f"{name} token tags changed")


def _local_video_positions(layout, *, start: int, stop: int, oldest_start: int,
                           frame_rows: int, origin: float) -> torch.Tensor:
    p = int(layout["action_video_start"])
    positions = layout["img_position_ids"][:, p + start * frame_rows:p + stop * frame_rows].clone()
    if positions.shape[1] != (stop - start) * frame_rows or start < oldest_start:
        raise ValueError("local video interval outside retained native window")
    times = native_times(stop - oldest_start, origin, device=positions.device)
    positions[0, :, 0] = times[start-oldest_start:stop-oldest_start].repeat_interleave(frame_rows)
    return positions


@dataclass
class PositionedCacheView:
    """Zero-copy raw K/V view; only per-entry RoPE buffers are remapped."""
    canonical: object
    local_ropes: dict[int, torch.Tensor]
    canonical_ropes: dict[int, torch.Tensor]
    canonical_current_rope: torch.Tensor

    def history(self, layer, index):
        from causal.h3_cached import RawEntry
        result = []
        for e in self.canonical.history(layer, index):
            if (e.rope.device != self.canonical_ropes[e.index].device or
                    e.rope.dtype != self.canonical_ropes[e.index].dtype):
                raise RuntimeError(f"layer {layer} chunk {e.index} has incompatible RoPE metadata")
            if not torch.equal(e.rope, self.canonical_ropes[e.index]):
                raise RuntimeError(f"layer {layer} chunk {e.index} lost canonical RoPE metadata")
            result.append(RawEntry(e.index, e.key, e.value, self.local_ropes[e.index]))
        return result

    def commit(self, layer, index, key, value, _local_rope):
        if _local_rope.shape != self.canonical_current_rope.shape:
            raise ValueError("local and canonical current RoPE shapes differ")
        # Store canonical metadata. Raw K/V are already pre-RoPE in H3.
        self.canonical.commit(layer, index, key, value, self.canonical_current_rope)


def localize_window(*, canonical_cache, packed, visible_layout, plan: ChunkPlan,
                    index: int, frame_rows: int, rope_fn, origin: float):
    """Return (current packed, cache view, geometry audit) for SW-L.

    Before first eviction (index<=5), return originals by identity to make
    SW-L exactly the frozen SW-G positional protocol. After eviction, b is
    the first retained ancestor's latent start, not a constant time offset.
    """
    ancestors = plan.ancestors(index)
    if index <= plan.history_chunks:
        return packed, canonical_cache, dict(index=index, oldest_latent=0,
                                              remapped=False, ancestors=list(ancestors))
    if not ancestors:
        raise RuntimeError("post-eviction local window needs history")
    oldest_start = plan.span(ancestors[0])[0]
    start, stop = plan.span(index)
    assert_native_video_grid(visible_layout, frame_rows=frame_rows, stop=stop, origin=origin)
    local_ropes = {}
    canonical_ropes = {}
    for ancestor in ancestors:
        lo, hi = plan.span(ancestor)
        coords = _local_video_positions(visible_layout, start=lo, stop=hi,
            oldest_start=oldest_start, frame_rows=frame_rows, origin=origin)
        local_ropes[ancestor] = rope_fn(coords)
        p = int(visible_layout["action_video_start"])
        canonical_ropes[ancestor] = rope_fn(visible_layout["img_position_ids"][:,
            p + lo * frame_rows:p + hi * frame_rows]).to(
                device=canonical_cache.storage_device or visible_layout["img_position_ids"].device)
    current = _local_video_positions(visible_layout, start=start, stop=stop,
        oldest_start=oldest_start, frame_rows=frame_rows, origin=origin)
    out = dict(packed)
    out["img_position_ids"] = packed["img_position_ids"].clone()
    prefix = int(packed["action_video_start"])
    out["img_position_ids"][:, prefix:] = current
    # Current raw K/V must be committed with canonical metadata even though
    # this forward's attention applied local video rotations.
    p = int(visible_layout["action_video_start"])
    canonical_current = rope_fn(visible_layout["img_position_ids"][:,
        p + start * frame_rows:p + stop * frame_rows])
    view = PositionedCacheView(canonical_cache, local_ropes, canonical_ropes,
                               canonical_current)
    assert torch.equal(out["img_position_ids"][:, :prefix], packed["img_position_ids"][:, :prefix])
    return out, view, dict(index=index, oldest_latent=oldest_start,
                           remapped=True, ancestors=list(ancestors),
                           canonical_current_shape=list(canonical_current.shape))
