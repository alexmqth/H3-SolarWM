"""AnyFlow Flow Map Backward Simulation with H3's noise-clean velocity.

The shortcut 1 -> t -> r -> 0 retains the entire current-chunk gradient chain.
It approximates an N-step flow map; it is not an exact N-step sampler. This
module supplies the generator path only, not a teacher/critic or DMD trainer.
Method: AnyFlow, arXiv:2605.13724, section 4.2.2 / Algorithm 2.
"""
from __future__ import annotations

import math


def shortcut_intervals(sigmas, index):
    times = tuple(float(x) for x in sigmas)
    if (len(times) < 2 or times[0] != 1 or times[-1] != 0
            or any(not math.isfinite(x) for x in times)
            or any(not 0 <= b < a <= 1 for a, b in zip(times, times[1:]))):
        raise ValueError('Require a strictly decreasing 1-to-0 sigma grid')
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(times) - 1:
        raise ValueError('Gradient interval index lies outside the grid')
    candidates = ((times[0], times[index]), (times[index], times[index + 1]),
                  (times[index + 1], times[-1]))
    return tuple((t, r) for t, r in candidates if t != r)


def simulate_chunk(noise, velocity, sigmas, index):
    """Apply up to three nonzero maps, without detaching any segment."""
    current = noise.float()
    for t, r in shortcut_intervals(sigmas, index):
        prediction = velocity(current, t, r)
        if prediction.shape != current.shape:
            raise ValueError('Finite-map velocity must match current latent shape')
        current = current + (r - t) * prediction.float()
    return current


def simulate_h3_chunk(model, noise, *, sigmas, interval_index, chunk_index,
                      cache, conditions, checkpoint=True, offload=True):
    """Differentiable H3 FMBS with a fixed, detached history-cache snapshot.

    ``conditions`` supplies the same packed layout, prompt, anchor, audio,
    chunk size and action controls to every map. The caller constructs these
    conditions using the inference protocol. This helper neither commits the
    endpoint nor computes new RGB anchors.

    Copying entry lists allows the caller's cache to commit/evict *new* entries
    before backward without changing checkpoint recomputation. Existing entry
    tensors and condition tensors must remain immutable through backward.
    Student parameters and adapter-enable flags must likewise be restored to
    their forward role before backward if a shared teacher/critic was used.
    No teacher/critic role switching is implemented by this helper.
    """
    from .h3_cached import H3ChunkCache, chunk_forward

    sigmas = tuple(float(x) for x in sigmas)
    shortcut_intervals(sigmas, interval_index)  # Validate before any model call.
    if not hasattr(model, 'anyflow_conditioner'):
        raise ValueError('H3 FMBS requires an installed AnyFlow conditioner')
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError('chunk_index must be a nonnegative integer')
    reserved = {'sigma', 'target_sigma', 'index', 'cache', 'commit', 'allow_grad_read',
                'use_gradient_checkpointing', 'use_gradient_checkpointing_offload',
                'clean_graph_entries'}
    if reserved.intersection(conditions):
        raise ValueError('FMBS owns time, cache, commit and gradient arguments')
    if offload and not checkpoint:
        raise ValueError('Activation offload requires checkpoint=True')
    for layer in range(len(model.blocks)):
        for entry in cache.history(layer, chunk_index):
            if any(x.requires_grad or x.grad_fn is not None for x in
                   (entry.key, entry.value, entry.rope)):
                raise ValueError('FMBS historical KV must be detached')
    # The snapshot owns its lists. H3 clean commit allocates fresh tensors;
    # there is no extra copy of large, immutable raw KV tensors here.
    fixed_cache = H3ChunkCache(cache.max_history, cache.storage_device,
        {layer: list(entries) for layer, entries in cache.layers.items()},
        cache.peak_bytes, cache.commits)
    fixed_conditions = dict(conditions)

    def velocity(current, t, r):
        return chunk_forward(model, current, index=chunk_index, cache=fixed_cache,
            sigma=t, target_sigma=r, commit=False, allow_grad_read=True,
            use_gradient_checkpointing=checkpoint,
            use_gradient_checkpointing_offload=offload, **fixed_conditions)

    return simulate_chunk(noise, velocity, sigmas, interval_index)
