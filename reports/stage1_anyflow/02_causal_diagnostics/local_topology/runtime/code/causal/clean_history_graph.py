"""Differentiable clean-history prefill with immutable per-chunk cache reads.

This preserves the existing chunk/anchor/action forward semantics. It restores
gradients through clean history; it is not SolarWM's fused two-stream operator.
Inference cache commits retain their original strict no_grad contract.
"""
import torch

from .h3_cached import H3ChunkCache, CleanGraphCapture, chunk_forward


def build_clean_history_graph(model, clean, target_chunk, condition, *,
                             chunk_frames=5, history_chunks=5,
                             storage_device='cpu', checkpoint=True, offload=False):
    if not torch.is_grad_enabled():
        raise RuntimeError('Differentiable history requires gradients enabled')
    if target_chunk < 0 or target_chunk * chunk_frames >= clean.shape[2]:
        raise ValueError('Target chunk outside the supplied teacher trajectory')
    cache = H3ChunkCache(history_chunks, storage_device)
    for index in range(target_chunk):
        start = index * chunk_frames
        capture = CleanGraphCapture()
        chunk_forward(model, clean[:, :, start:start+chunk_frames],
            index=index, cache=cache, sigma=0.,
            target_sigma=0. if hasattr(model, 'anyflow_conditioner') else None,
            allow_grad_read=True, use_gradient_checkpointing=checkpoint,
            use_gradient_checkpointing_offload=offload,
            clean_graph_entries=capture, **condition(index))
        captured = capture.entries
        if set(captured) != set(range(len(model.blocks))):
            raise RuntimeError('Incomplete per-layer clean-history capture')
        # Do not change the cache seen by this forward's checkpoint closures.
        following = H3ChunkCache(history_chunks, storage_device)
        following.layers = {layer: [*cache.history(layer, index), entry][-history_chunks:]
                            for layer, entry in captured.items()}
        following.commits = cache.commits + len(captured)
        following.peak_bytes = max(cache.peak_bytes, following.nbytes)
        capture.sealed = True
        capture.entries.clear()
        cache = following
    return cache
