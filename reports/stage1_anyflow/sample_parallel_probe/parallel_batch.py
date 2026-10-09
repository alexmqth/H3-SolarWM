"""Isolated four-rank evaluation of the SAME four-sample logical batch.

Every rank owns one sample and a replicated frozen H3. No sampler, data,
architecture, optimizer, or global batch change. This is not sequence parallel.
The original four-forward AnyFlow loss is retained, including on the diagonal.
"""
from contextlib import nullcontext
import hashlib

import torch
import torch.distributed as dist

from causal.train_stage1_anyflow import clean_cache, condition, timestep_shift
from causal.anyflow import logical_time_pairs, anyflow_sample_loss, adaptive_scale
from causal.anyflow_reference import gaussian_timestep_weights
from causal.h3_cached import chunk_forward


def parallel_batch(model, case, chunk, args, *, generator, backward,
                   parameters, record_noise=False):
    if not dist.is_initialized() or dist.get_world_size() != 4 or args.logical_batch != 4:
        raise ValueError('This isolated probe requires four ranks and global logical batch4')
    rank = dist.get_rank()
    cache = clean_cache(model, case, chunk, args)
    start = chunk * args.chunk_frames
    clean = case['clean'][:, :, start:start + args.chunk_frames]
    common = condition(case, chunk, args)
    shift = timestep_shift(args, training=backward)
    pairs = logical_time_pairs(generator, shift=shift, batch_size=4)
    # Consume the same CPU RNG sequence on every rank as the serial trainer.
    # Only this rank's one noise tensor is transferred to its accelerator.
    noise = None
    for i in range(4):
        sample = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
        if i == rank:
            noise = sample.to(clean)
    del sample
    t, r = float(pairs.t[rank]), float(pairs.r[rank])
    initial_commits = cache.commits
    graph_forwards, combined_cpu_kv_peak = 0, cache.peak_bytes

    def velocity(sample, sigma, target_sigma):
        nonlocal graph_forwards, combined_cpu_kv_peak
        grad = torch.is_grad_enabled()
        read_cache = cache
        if grad and args.history_gradient_mode == 'full' and chunk > 0:
            from causal.clean_history_graph import build_clean_history_graph
            read_cache = build_clean_history_graph(model, case['clean'], chunk,
                lambda i: condition(case, i, args), chunk_frames=args.chunk_frames,
                history_chunks=args.history_chunks, checkpoint=True,
                offload=sample.device.type == 'cuda')
            graph_forwards += chunk
            combined_cpu_kv_peak = max(combined_cpu_kv_peak, cache.nbytes + read_cache.peak_bytes)
        return chunk_forward(model, sample, sigma=sigma,
            target_sigma=target_sigma if args.objective == 'anyflow' else None,
            index=chunk, cache=read_cache, allow_grad_read=grad,
            use_gradient_checkpointing=grad,
            use_gradient_checkpointing_offload=grad and sample.device.type == 'cuda',
            **common)

    try:
        with nullcontext() if backward else torch.no_grad():
            if args.objective == 'anyflow':
                raw, weight, item = anyflow_sample_loss(velocity, clean, noise, t, r,
                    shift=shift, epsilon=args.finite_difference_epsilon,
                    preserve_fp32_inputs=args.precision_profile == 'h3_fp32')
                is_diffusion = bool(pairs.is_diffusion[rank])
            else:
                noisy = ((1-t)*clean.float() + t*noise.float()).to(clean)
                prediction = velocity(noisy, t, t).float()
                raw = (prediction - (noise.float()-clean.float())).square().mean()
                weight = gaussian_timestep_weights(torch.tensor([t*1000], device=clean.device), shift=shift)[0]
                item = dict(sigma=t, target_sigma=t, raw_loss=float(raw.detach()),
                    weight=float(weight), adaptive_scale=1., sample_type='diffusion', model_evaluations=1)
                is_diffusion = True
            # Gather RAW FM losses BEFORE weighting, exactly as in serial
            # logical_batch; never use each rank's own reference or total loss.
            gathered = [torch.empty_like(raw.detach().reshape(1)) for _ in range(4)]
            dist.all_gather(gathered, raw.detach().reshape(1))
            if args.objective == 'anyflow':
                reference = [float(gathered[i][0]) for i in range(4) if bool(pairs.is_diffusion[i])]
                scale = adaptive_scale(raw, is_diffusion=is_diffusion, diffusion_losses=reference)
                item['sample_type'] = ('diffusion', 'endpoint', 'flow_map')[int(pairs.sample_type[rank])]
                item['adaptive_scale'] = float(scale)
            else:
                scale = raw.new_tensor(1.)
            value = raw * scale * weight / 4
            if not torch.isfinite(value):
                raise FloatingPointError('Non-finite sample-parallel loss')
            item['weighted_loss'] = float(value.detach()) * 4
            if backward:
                value.backward()
        if cache.commits != initial_commits:
            raise RuntimeError('Read-only loss changed committed history')
        if backward:
            # Loss was already divided by GLOBAL batch4. SUM, do not average
            # a second time. The caller can then clip and apply one Adam step.
            for p in parameters:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        local = dict(rank=rank, sample=item, differentiable_history_forwards=graph_forwards,
            clean_commit_forwards=chunk, cpu_kv_peak_MiB=combined_cpu_kv_peak/2**20)
        if record_noise:
            local['actual_noise_sha256'] = hashlib.sha256(noise.detach().float().cpu().numpy().tobytes()).hexdigest()
        all_rows = [None]*4
        dist.all_gather_object(all_rows, local)
        metrics = [row['sample'] for row in all_rows]
        return dict(action=case['label'], chunk=chunk, samples=metrics,
            loss=sum(x['weighted_loss'] for x in metrics)/4,
            model_evaluations=sum(x['model_evaluations'] for x in metrics),
            differentiable_history_forwards=sum(x['differentiable_history_forwards'] for x in all_rows),
            clean_commit_forwards_physical=sum(x['clean_commit_forwards'] for x in all_rows),
            cpu_kv_peak_MiB_per_rank=[x['cpu_kv_peak_MiB'] for x in all_rows], ranks=all_rows)
    finally:
        cache.clear()
