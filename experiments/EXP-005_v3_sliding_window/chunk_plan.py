"""Explicit native H3 12→5 chunk geometry and strict five-ancestor cache checks.

This module is CPU-safe. It never constructs a new H3 conditioning fixture.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkPlan:
    first_latents: int = 12
    later_latents: int = 5
    history_chunks: int = 5
    first_rgb: int = 39
    later_rgb: int = 17

    def __post_init__(self):
        if (self.first_latents, self.later_latents, self.history_chunks,
                self.first_rgb, self.later_rgb) != (12, 5, 5, 39, 17):
            raise ValueError("EXP-005 is frozen to H3 native 12→5 / W5 geometry")

    def span(self, index: int) -> tuple[int, int]:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("chunk index must be a nonnegative integer")
        if index == 0:
            return 0, self.first_latents
        start = self.first_latents + (index - 1) * self.later_latents
        return start, start + self.later_latents

    def ancestors(self, index: int) -> tuple[int, ...]:
        self.span(index)
        return tuple(range(max(0, index - self.history_chunks), index))

    def history_start(self, index: int) -> int:
        ancestors = self.ancestors(index)
        return self.span(ancestors[0])[0] if ancestors else 0

    def rgb_stop(self, index: int) -> int:
        self.span(index)
        return self.first_rgb + self.later_rgb * index


PLAN = ChunkPlan()


def cache_identity(cache) -> tuple:
    """Stable in-process identity of raw ancestral entries without cloning K/V."""
    return (cache.commits, tuple((layer, tuple(
        (e.index, id(e), tuple((id(t), t.data_ptr(), t._version, tuple(t.shape))
                                  for t in (e.key, e.value, e.rope))) for e in entries))
        for layer, entries in sorted(cache.layers.items())))


def validate_cache(cache, *, plan: ChunkPlan, index: int, frame_rows: int,
                   expected_layers: int = 50) -> dict:
    """Reject vacuous/missing layers and verify the exact retained ancestry.

    H3ChunkCache.history itself checks individual layers, but iterating over an
    empty dict silently passes. The model has 50 layers in the frozen V3 path.
    Tests may use a small explicit expected_layers, never an inferred count.
    """
    from causal.h3_cached import H3ChunkCache

    ancestors = plan.ancestors(index)
    if not isinstance(cache, H3ChunkCache) or cache.max_history != plan.history_chunks:
        raise TypeError("require frozen H3ChunkCache(max_history=5)")
    if not isinstance(frame_rows, int) or frame_rows < 1:
        raise ValueError("frame_rows must be positive")
    if not isinstance(expected_layers, int) or expected_layers < 1:
        raise ValueError("expected_layers must be explicit and positive")
    required = set(range(expected_layers)) if index else set()
    if set(cache.layers) != required:
        raise RuntimeError(f"cache layers {sorted(cache.layers)} != expected {sorted(required)}")
    if cache.commits != index * expected_layers:
        raise RuntimeError(f"cache commits={cache.commits}, expected={index * expected_layers}")
    expected_rows = tuple((plan.span(i)[1] - plan.span(i)[0]) * frame_rows for i in ancestors)
    for layer in range(expected_layers):
        entries = cache.history(layer, index)
        if tuple(e.index for e in entries) != ancestors:
            raise RuntimeError(f"layer {layer} has wrong ancestor indices")
        if tuple(e.key.shape[0] for e in entries) != expected_rows:
            raise RuntimeError(f"layer {layer} has wrong ancestor row counts")
        for entry in entries:
            if entry.key.shape != entry.value.shape or entry.rope.shape[0] != entry.key.shape[0]:
                raise RuntimeError(f"layer {layer} has malformed raw K/V/RoPE")
            if any(t.requires_grad for t in (entry.key, entry.value, entry.rope)):
                raise RuntimeError("history entries must be detached")
    return dict(index=index, ancestors=list(ancestors),
                latent_spans=[list(plan.span(i)) for i in ancestors],
                rows_per_layer=sum(expected_rows), layer_count=expected_layers,
                nbytes=cache.nbytes, commits=cache.commits)
