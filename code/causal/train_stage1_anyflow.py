#!/usr/bin/env python3
"""Stage1 TF-AnyFlow: clean teacher history, target times, finite-map loss.

Uses one physical sample and a logical batch of four (2 FM / 1 endpoint /
1 general map), with exact within-batch adaptive scaling. All four velocity
evaluations use the same clean-history values. The optional full gradient
mode rebuilds a differentiable history for each gradient prediction. Rebuild
history after every optimizer update. No self-rollout, critic, or DMD here.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/abot'), str(ROOT / 'code'),
               str(ROOT / 'DiffSynth-Studio-h3-v2')]
import torch
from causal.training_state import serialize_config, load_training_state, restore_random_state
from causal.anyflow import (install_anyflow, load_anyflow, save_anyflow, logical_time_pairs,
    anyflow_sample_loss, adaptive_scale)
from causal.h3_cached import (H3ChunkCache, chunk_forward, expand_packed_two_anchors,
    last_frame_anchor, last_frame_image_anchor)
from causal.pretrained_lora import (install_adapters, load_adapter, save_adapters,
    load_action_residual, save_action_residual)


def move_tree(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_tree(v, device) for k, v in value.items()}
    return value


def timestep_shift(args, *, training):
    """Separate training density from common validation and inference grids."""
    key = 'training_timestep_shift' if training else 'validation_timestep_shift'
    shift = getattr(args, key, None)
    shift = float(args.flow_shift if shift is None else shift)
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError(f'{key} must be positive and finite')
    return shift


def timestep_weight_shift(args, *, training):
    """Choose the Gaussian weight grid independently of sample density.

    Absent overrides preserve the historical coupled policy exactly. A fixed
    weight grid means the same function of sigma, not identical weights for
    samples drawn at different sigmas or an importance-corrected objective.
    """
    key = 'training_weight_shift' if training else 'validation_weight_shift'
    shift = getattr(args, key, None)
    shift = timestep_shift(args, training=training) if shift is None else float(shift)
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError(f'{key} must be positive and finite')
    return shift


def condition(case, chunk, args):
    start = chunk * args.chunk_frames
    count = case['clean'].shape[2]
    return dict(full_packed=case['packed'], prompt=case['prompt'],
        audio=case['audio'], anchor=case['anchors'][chunk],
        anchor_frame_index=start - 1 if chunk and args.anchor_mode != 'fixed' else None,
        anchor_slot=1 if args.anchor_mode != 'fixed' else 0,
        chunk_frames=args.chunk_frames, action_prefix_mode=args.action_prefix_mode,
        action_feedback=args.action_feedback,
        action_cond=(case['actions'][start:min(start + args.chunk_frames, count)]
                     if case['actions'] is not None else None),
        action_adapter=case['action_adapter'])


def clean_cache(model, case, target_chunk, args):
    """No future/target latents committed; each past chunk uses its own anchor."""
    cache = H3ChunkCache(args.history_chunks, 'cpu')
    with torch.no_grad():
        for chunk in range(target_chunk):
            start = chunk * args.chunk_frames
            chunk_forward(model, case['clean'][:, :, start:start + args.chunk_frames],
                index=chunk, cache=cache, sigma=0., commit=True,
                **condition(case, chunk, args))
    return cache


def logical_batch(model, case, chunk, args, *, generator, backward):
    cache = clean_cache(model, case, chunk, args)
    start = chunk * args.chunk_frames
    clean = case['clean'][:, :, start:start + args.chunk_frames]
    common = condition(case, chunk, args)
    sample_shift = timestep_shift(args, training=backward)
    weight_shift = timestep_weight_shift(args, training=backward)
    pairs = logical_time_pairs(generator, shift=sample_shift,
                               batch_size=args.logical_batch)
    diffusion_losses, metrics = [], []
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
        for i in range(args.logical_batch):
            noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(clean)
            t, r = float(pairs.t[i]), float(pairs.r[i])
            context = nullcontext() if backward else torch.no_grad()
            with context:
                if args.objective == 'anyflow':
                    raw, weight, item = anyflow_sample_loss(
                        velocity, clean, noise, t, r, shift=weight_shift,
                        epsilon=args.finite_difference_epsilon,
                        preserve_fp32_inputs=args.precision_profile == "h3_fp32")
                    is_diffusion = bool(pairs.is_diffusion[i])
                    scale = adaptive_scale(raw, is_diffusion=is_diffusion,
                                           diffusion_losses=diffusion_losses)
                    if is_diffusion:
                        diffusion_losses.append(float(raw.detach()))
                    value = raw * scale * weight / args.logical_batch
                    item['sample_type'] = ('diffusion', 'endpoint', 'flow_map')[int(pairs.sample_type[i])]
                    item['adaptive_scale'] = float(scale)
                else:
                    # Same data/noise/time grid and QKV initialization, regular
                    # FM control. No target-time MLP and no finite-map claim.
                    from causal.anyflow_reference import gaussian_timestep_weights
                    noisy = ((1 - t) * clean.float() + t * noise.float()).to(clean)
                    prediction = velocity(noisy, t, t).float()
                    raw = (prediction - (noise.float() - clean.float())).square().mean()
                    weight = gaussian_timestep_weights(
                        torch.tensor([t * 1000], device=clean.device), shift=weight_shift)[0]
                    value = raw * weight / args.logical_batch
                    item = dict(sigma=t, target_sigma=t, raw_loss=float(raw.detach()),
                        weight=float(weight), adaptive_scale=1., sample_type='diffusion',
                        model_evaluations=1)
                if not torch.isfinite(value):
                    raise FloatingPointError('Non-finite logical AnyFlow loss')
                item['weighted_loss'] = float(value.detach()) * args.logical_batch
                if backward:
                    value.backward()
            metrics.append(item)
            if not args.smoke:
                print(f"[TF-{args.objective} sample] phase={'train' if backward else 'validation'} "
                      f"action={case['label']} chunk={chunk} sample={i+1}/{args.logical_batch} "
                      f"type={item['sample_type']} raw={item['raw_loss']:.6f} "
                      f"weighted={item['weighted_loss']:.6f}", flush=True)
            del value, raw, noise
        if cache.commits != initial_commits:
            raise RuntimeError('Loss evaluations unexpectedly mutated the clean-history cache')
        return dict(action=case['label'], chunk=chunk, samples=metrics,
            loss=sum(x['weighted_loss'] for x in metrics) / len(metrics),
            model_evaluations=sum(x['model_evaluations'] for x in metrics),
            clean_commit_forwards=chunk, differentiable_history_forwards=graph_forwards,
            cpu_kv_peak_MiB=combined_cpu_kv_peak / 2**20)
    finally:
        cache.clear()


def prepare_real_cases(pipe, action_adapter, args):
    import infer as abot
    from causal.train_online_selfrollout import action_condition
    cases = []
    for label, teacher_dir in zip(args.actions, args.teacher_dir):
        teacher_dir = Path(teacher_dir)
        setup = json.loads((teacher_dir / 'baseline.json').read_text())
        expected = {'A': 'strafe-left', 'D': 'strafe-right', 'W': 'forward', 'S': 'back'}[label]
        if setup.get('action_preset_resolved', setup.get('action_preset')) not in (label, expected):
            raise ValueError(f'{teacher_dir}: teacher action label does not match {label}')
        if setup.get('status') != 'complete' or setup.get('steps') != 30:
            raise ValueError('Stage1 targets must be completed original H3 30-step references')
        cond = move_tree(torch.load(teacher_dir / 'conditioning.pt', map_location='cpu', weights_only=True), args.device)
        clean = torch.load(teacher_dir / 'baseline_latents.pt', map_location=args.device, weights_only=True)
        if clean.shape != cond['initial_noise'].shape or clean.shape[0] != 1 or not torch.isfinite(clean).all():
            raise ValueError('Teacher latent/noise shape or finiteness mismatch')
        count = (clean.shape[2] + args.chunk_frames - 1) // args.chunk_frames
        rows = (clean.shape[-2] // 2) * (clean.shape[-1] // 2)
        original = cond['anchor']
        anchors = [original]
        packed = cond['packed']
        if args.anchor_mode != 'fixed':
            packed = expand_packed_two_anchors(packed, frame_rows=rows)
            anchors = [torch.cat((original, original.clone()))]
        for chunk in range(1, count):
            history = clean[:, :, :chunk * args.chunk_frames]
            with torch.no_grad():
                if args.anchor_mode == 'rgb':
                    pipe.load_models_to_device(['video_vae'])
                    tail = last_frame_image_anchor(pipe.video_vae, history, dtype=clean.dtype)
                elif args.anchor_mode == 'latent':
                    tail = last_frame_anchor(history[:, :, -1:])
                else:
                    tail = None
                anchors.append(original if tail is None else torch.cat((original, tail)))
        cases.append(dict(label=label, clean=clean, packed=packed, prompt=cond['prompt_embeds'],
            audio=cond['audio_noise'], anchors=anchors, chunks=count, action_adapter=action_adapter,
            actions=action_condition(label, clean.shape[2], args.device, clean.dtype),
            teacher_dir=str(teacher_dir.resolve()),
            teacher_sha256=hashlib.sha256((teacher_dir / 'baseline_latents.pt').read_bytes()).hexdigest()))
    pipe.load_models_to_device(['dit'])
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--resume-from', type=Path, help='Checkpoint directory with adapters, Adam and RNG; --steps is total updates')
    ap.add_argument('--teacher-dir', type=Path, nargs='+')
    ap.add_argument('--real-data-manifest', type=Path, help='Verified real ABot cache; disjoint train/validation episodes')
    ap.add_argument('--actions', nargs='+', default=['A', 'D'])
    ap.add_argument('--causal-adapter', type=Path)
    ap.add_argument('--action-adapter', type=Path)
    ap.add_argument('--objective', choices=['anyflow', 'fm'], default='anyflow')
    ap.add_argument('--precision-profile', choices=['legacy', 'h3_fp32'], default='legacy')
    ap.add_argument('--adapter-scope', choices=['tail_qkv','all_qkvo_ffn'], default='tail_qkv')
    ap.add_argument('--bank-rank', type=int, default=8)
    ap.add_argument('--bank-alpha', type=float, default=8.)
    ap.add_argument('--history-gradient-mode', choices=['detached','full'], default='detached',
                    help='full rebuilds a differentiable clean history for each gradient prediction; inference unchanged')
    ap.add_argument('--train-target-time', action=argparse.BooleanOptionalAction, default=False,
                    help='Experimental variant; official H3 Stage1 freezes the cloned time MLP')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--smoke', action='store_true', help='CPU random SMALL H3; not a pretrained/video-quality result')
    ap.add_argument('--steps', type=int, default=16)
    ap.add_argument('--checkpoint-every', type=int, default=4)
    ap.add_argument('--logical-batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--tail-blocks', type=int, default=16)
    ap.add_argument('--chunk-frames', type=int, default=5)
    ap.add_argument('--history-chunks', type=int, default=5)
    ap.add_argument('--anchor-mode', choices=['fixed', 'latent', 'rgb'], default='rgb')
    ap.add_argument('--action-prefix-mode', choices=['own', 'causal'], default='causal')
    ap.add_argument('--action-feedback', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--flow-shift', type=float, default=2.22)
    ap.add_argument('--training-timestep-shift', type=float,
                    help='Training time-pair density; defaults to --flow-shift')
    ap.add_argument('--validation-timestep-shift', type=float,
                    help='Validation time density; defaults to --flow-shift independently of training')
    ap.add_argument('--training-weight-shift', type=float,
                    help='Gaussian training weight grid; defaults to the training time density shift')
    ap.add_argument('--validation-weight-shift', type=float,
                    help='Gaussian validation weight grid; defaults to the validation time density shift')
    ap.add_argument('--finite-difference-epsilon', type=float, default=5.)
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--validation-seed', type=int, default=1001)
    ap.add_argument('--validation-cases', type=int, default=2)
    args = ap.parse_args()
    try:
        timestep_shift(args, training=True)
        timestep_shift(args, training=False)
        timestep_weight_shift(args, training=True)
        timestep_weight_shift(args, training=False)
    except ValueError as exc:
        ap.error(str(exc))
    if args.logical_batch < 4 or args.logical_batch % 4 or min(args.steps, args.checkpoint_every, args.chunk_frames, args.history_chunks, args.validation_cases) < 1:
        ap.error('Invalid steps/chunks/checkpoint interval or logical batch (must be 4k)')
    if min(args.lr, args.flow_shift, args.finite_difference_epsilon, args.rank, args.tail_blocks, args.bank_rank, args.bank_alpha) <= 0:
        ap.error('Learning rate, shift, finite-difference epsilon, rank and tail blocks must be positive')
    if args.real_data_manifest and (args.teacher_dir or args.smoke):
        ap.error('Real ABot and teacher/smoke inputs are mutually exclusive')
    if args.real_data_manifest and (args.anchor_mode != 'rgb' or args.chunk_frames != 5 or args.history_chunks != 5):
        ap.error('Real ABot cache requires RGB dual anchors and 5-frame/5-history protocol')
    if not args.smoke and not args.real_data_manifest and (not args.teacher_dir or len(args.teacher_dir) != len(args.actions)):
        ap.error('Provide one original H3 teacher directory per action')
    if any(a not in ('W', 'S', 'A', 'D') for a in args.actions):
        ap.error('Actions must be W/S/A/D')
    if args.smoke:
        args.device, args.anchor_mode = 'cpu', 'fixed'
    if (args.out_dir / 'training.json').exists():
        ap.error('Output already contains a run; choose a new experiment directory')
    args.real_manifest_sha256 = (hashlib.sha256(args.real_data_manifest.read_bytes()).hexdigest() if args.real_data_manifest else None)
    config = serialize_config(args)
    resume = load_training_state(args.resume_from, config) if args.resume_from else None
    start_step = resume['optimizer_step'] if resume else 0
    qkv_path = args.resume_from / 'causal_adapter.pt' if resume else args.causal_adapter
    action_path = args.resume_from / 'action_adapter.pt' if resume and args.action_adapter else args.action_adapter
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    result = dict(status='running', started_at=datetime.now().astimezone().isoformat(),
        objective='TF-AnyFlow v1.5' if args.objective == 'anyflow' else 'TF-FM control',
        scope='random-small-H3 CPU integration' if args.smoke else 'pretrained-H3 clean teacher history',
        config=config, resumed_from=str(args.resume_from) if resume else None,
        time_sampling=dict(training_shift=timestep_shift(args, training=True),
                           validation_shift=timestep_shift(args, training=False),
                           training_weight_shift=timestep_weight_shift(args, training=True),
                           validation_weight_shift=timestep_weight_shift(args, training=False),
                           inference_flow_shift=args.flow_shift),
        resumed_optimizer_step=start_step,
        history_protocol=('clean pseudo-GT from original H3; differentiable CPU history rebuilt per gradient prediction; detached no-grad targets'
            if args.history_gradient_mode == 'full' else
            'clean pseudo-GT from original H3; detached CPU cache rebuilt per optimizer step'),
        velocity_convention='noise-clean; native H3 time=1-sigma',
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        vram_reserve_gib=float(os.environ.get('ABOT_VRAM_RESERVE_GIB', '5')),
        updates=list(resume['updates']) if resume else [])
    def save_result():
        tmp = args.out_dir / 'training.tmp.json'
        tmp.write_text(json.dumps(result, indent=2) + '\n')
        tmp.replace(args.out_dir / 'training.json')
    if args.real_data_manifest:
        result.update(scope='pretrained H3 on REAL ABot video latents',
            history_protocol=('REAL ABot clean GT history; CPU KV committed clean, differentiable history rebuilt per gradient prediction'
                if args.history_gradient_mode == 'full' else 'REAL ABot clean GT history; detached CPU KV rebuilt per update'),
            data_source_kind='real_ABot_episode', validation_protocol='episode-disjoint validation cache',
            curriculum='case=step%N; chunk=(epoch+case_index)%3; 3N updates cover every case/chunk')
    save_result()
    try:
        action_adapter = None
        if args.smoke:
            from causal.h3_training import make_small_h3, synthetic_h3_batch
            model = make_small_h3().eval()
            for block in model.blocks:
                block.attn.qkv_proj = block.attn.qkv_proj.base
            indices = list(range(len(model.blocks)))
            if resume:
                loaded = load_adapter(model, qkv_path, args.device)
                indices = loaded['block_indices']
                adapters = [model.blocks[i].attn.qkv_proj for i in indices]
            else:
                adapters, indices = install_adapters(model, rank=4, block_indices=indices)
            b = synthetic_h3_batch(frames=12, seed=31)
            cases = [dict(label='synthetic', clean=b['clean_video'], packed=b['packed'],
                prompt=b['prompt_embeds'], audio=b['audio_latents'],
                anchors=[b['anchor_rows']] * 3, chunks=3, actions=None, action_adapter=None)]
        else:
            import infer as abot
            torch.cuda.set_device(args.device)
            pipe = abot.load_pipeline(args.device)
            pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(
                ROOT / 'checkpoints/H3-World/step-10000.safetensors'), hotload=True)
            model = pipe.dit.requires_grad_(False).eval()
            pipe.load_models_to_device(['dit'])
            if qkv_path:
                loaded = load_adapter(model, qkv_path, args.device)
                trained_anchor = loaded.get('metadata', {}).get('anchor_mode')
                aliases = {'dynamic_last_frame_dual': 'latent',
                           'dynamic_last_frame_rgb_dual': 'rgb'}
                if trained_anchor and aliases.get(trained_anchor, trained_anchor) != args.anchor_mode:
                    raise ValueError(f'Adapter anchor {trained_anchor} differs from training anchor {args.anchor_mode}')
                indices = loaded['block_indices']
                adapters = [model.blocks[i].attn.qkv_proj for i in indices]
            else:
                if args.tail_blocks > len(model.blocks):
                    raise ValueError('Requested more QKV blocks than the H3 model contains')
                indices = list(range(len(model.blocks) - args.tail_blocks, len(model.blocks)))
                adapters, indices = install_adapters(model, args.rank, indices, args.device)
            if action_path:
                action_adapter = load_action_residual(model, action_path, args.device)['adapter']
                action_adapter.requires_grad_(False)
            if args.real_data_manifest:
                from causal.real_video_data import load_cases
                cases, real_validation_pool = load_cases(args.real_data_manifest, args.device, action_adapter)
                pipe.load_models_to_device(['dit'])
            else:
                cases = prepare_real_cases(pipe, action_adapter, args)
        validation_pool = real_validation_pool if args.real_data_manifest else cases
        all_data_cases = cases + validation_pool if args.real_data_manifest else cases
        from causal.h3_precision import configure_precision
        precision = configure_precision(model, args.precision_profile,
            native_transformer_dir=(None if args.smoke else
                ROOT / 'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer'))
        result['precision'] = precision
        if resume:
            from causal.h3_precision import validate_precision_checkpoint
            validate_precision_checkpoint(torch.load(qkv_path, map_location='cpu', weights_only=True)['metadata'], precision)
        if args.precision_profile == 'h3_fp32':
            # VAE anchor preparation above follows its existing BF16 policy.
            # AnyFlow clean/noise/interpolation now use the official FP32 path.
            for case in all_data_cases:
                case['clean'] = case['clean'].float()
        time_module = None
        if args.objective == 'anyflow':
            time_module = (load_anyflow(model, args.resume_from / 'anyflow_adapter.pt', args.device)[0]
                           if resume else install_anyflow(model, device=args.device))
        if time_module is not None:
            time_module.requires_grad_(args.train_target_time)
        bank = None
        if args.adapter_scope == 'all_qkvo_ffn':
            from causal.stage1_lora import install_stage1_lora, load_stage1_lora, save_stage1_lora
            for adapter in adapters:
                adapter.requires_grad_(False)
            if resume:
                bank, bank_metadata = load_stage1_lora(model, args.resume_from/'stage1_lora.pt', device=args.device)
                validate_precision_checkpoint(bank_metadata, precision)
            else:
                bank = install_stage1_lora(model, rank=args.bank_rank, alpha=args.bank_alpha, device=args.device)
            params = bank.parameters()
            qkv_params = bank.parameters('qkv')
            result['stage1_lora'] = bank.describe()
        else:
            params = []
            for adapter in adapters:
                for p in (adapter.lora_A, adapter.lora_B):
                    p.requires_grad_(True); params.append(p)
            qkv_params = list(params)
        time_params = list(time_module.parameters()) if time_module is not None and args.train_target_time else []
        params.extend(time_params)
        if {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in params}:
            raise RuntimeError('Unexpected trainable parameters outside the selected Stage1 adapter scope')
        optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(.9, .95), weight_decay=.01)
        result.update(trainable_parameters=sum(p.numel() for p in params),
            qkv_block_indices=indices, target_time_parameters=sum(p.numel() for p in time_module.parameters()) if time_module else 0,
            teacher_artifacts=[{k: c[k] for k in ('label', 'teacher_dir', 'teacher_sha256', 'source_kind', 'source_file', 'source_sha256', 'sample_id', 'split') if k in c} for c in all_data_cases])
        if args.real_data_manifest:
            result.update(data_artifacts=result['teacher_artifacts'], train_cases=len(cases),
                heldout_cases=len(validation_pool), optimizer_targets=('real noisy-minus-clean latent; no teacher generated target'
                    if args.objective == 'fm' else 'TF-AnyFlow finite-map objective over real video latents; no teacher generated dataset'))
        if resume:
            if result['teacher_artifacts'] != resume['teacher_artifacts']:
                raise ValueError('Teacher artifacts changed since the saved optimizer state')
            optimizer.load_state_dict(resume['optimizer'])
        result['target_time_trainable'] = bool(time_module and args.train_target_time)
        result['target_time_trainable_parameters'] = (sum(p.numel() for p in time_module.parameters()) if time_module and args.train_target_time else 0)
        save_result()
        generator = torch.Generator().manual_seed(args.seed + 10000)
        def checkpoint(directory):
            directory.mkdir(parents=True, exist_ok=True)
            metadata = {k: result[k] for k in ('objective', 'scope', 'config', 'history_protocol', 'velocity_convention', 'precision')}
            metadata['optimizer_step'] = len(result['updates'])
            save_adapters(directory / 'causal_adapter.pt', adapters, indices, metadata)
            if bank is not None:
                save_stage1_lora(directory / 'stage1_lora.pt', bank, metadata)
            if time_module is not None:
                save_anyflow(directory / 'anyflow_adapter.pt', time_module, metadata)
            if action_adapter is not None:
                save_action_residual(directory / 'action_adapter.pt', action_adapter, metadata)
            # Save optimizer/RNG along with adapters so a later extension need
            # not silently reset Adam moments or the logical noise sequence.
            files = ['causal_adapter.pt']
            if bank is not None:
                files.append('stage1_lora.pt')
            if time_module is not None:
                files.append('anyflow_adapter.pt')
            if action_adapter is not None:
                files.append('action_adapter.pt')
            training_state = dict(format='h3world_stage1_training_state_v1',
                optimizer=move_tree(optimizer.state_dict(), 'cpu'),
                optimizer_step=len(result['updates']), updates=result['updates'],
                config=result['config'], teacher_artifacts=result['teacher_artifacts'],
                logical_rng_state=generator.get_state(), cpu_rng_state=torch.get_rng_state(),
                cuda_rng_state=None if args.smoke else torch.cuda.get_rng_state(args.device),
                weight_sha256={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                               for name in files})
            temporary = directory / 'trainer_state.tmp.pt'
            torch.save(training_state, temporary)
            temporary.replace(directory / 'trainer_state.pt')
        def validate():
            output = []
            for c in validation_pool[:args.validation_cases]:
                output.append(logical_batch(model, c, c['chunks'] - 1, args,
                    generator=torch.Generator().manual_seed(args.validation_seed), backward=False))
            return output
        if resume:
            restore_random_state(resume, generator, device=args.device)
        checkpoint(args.out_dir / f'step_{start_step:02d}')
        result['validation_before'] = validate()
        save_result()
        if not args.smoke:
            torch.cuda.reset_peak_memory_stats(args.device)
        initial = [p.detach().cpu().clone() for p in params]
        initial_by_id = dict(zip(map(id, params), initial))
        frozen_visual = ([p.detach().cpu().clone() for a in adapters for p in (a.lora_A,a.lora_B)]
                         if bank is not None else None)
        initial_time = [p.detach().cpu().clone() for p in time_module.parameters()] if time_module else []
        if resume:
            # Validation owns a separate noise generator. Re-establish global
            # state too, so diagnostics cannot perturb continuation RNG.
            restore_random_state(resume, generator, device=args.device)
        for step in range(start_step, args.steps):
            tick = time.perf_counter()
            case = cases[step % len(cases)]
            chunk = ((step // len(cases) + step % len(cases)) if args.real_data_manifest else (step // len(cases))) % case['chunks']
            optimizer.zero_grad(set_to_none=True)
            metrics = logical_batch(model, case, chunk, args, generator=generator, backward=True)
            def group_grad_norm(group):
                norms = [p.grad.detach().float().norm() for p in group if p.grad is not None]
                return float(torch.stack(norms).norm()) if norms else 0.0
            metrics['qkv_grad_norm'] = group_grad_norm(qkv_params)
            metrics['target_time_grad_norm'] = group_grad_norm(time_params) if time_module else None
            if bank is not None:
                metrics['adapter_group_grad_norms'] = {g:group_grad_norm(bank.parameters(g)) for g in ('qkv','out','ffn','refiner')}
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Non-finite AnyFlow gradient norm')
            optimizer.step()
            if not args.smoke:
                torch.cuda.synchronize(args.device)
            metrics.update(step=step + 1, grad_norm=float(norm), seconds=time.perf_counter() - tick)
            result['updates'].append(metrics)
            print(f"[TF-{args.objective}] {step+1}/{args.steps} {case['label']} chunk={chunk} loss={metrics['loss']:.6f} grad={float(norm):.4f}", flush=True)
            save_result()
            # Preserve early real-data checkpoints after the first current-only
            # update and after all three history lengths have run. This changes
            # checkpoint I/O only, not the optimizer or sample budget.
            if (step + 1) % args.checkpoint_every == 0 or (args.real_data_manifest and step + 1 in (1, 3)):
                checkpoint(args.out_dir / f'step_{step+1:02d}')
        result['validation_after'] = validate()
        result['parameters_changed'] = any(not torch.equal(p.detach().cpu(), old) for p, old in zip(params, initial))
        result['qkv_parameters_changed'] = any(not torch.equal(p.detach().cpu(), initial_by_id[id(p)]) for p in qkv_params)
        if bank is not None:
            result['adapter_groups_changed'] = {g:any(not torch.equal(p.detach().cpu(), initial_by_id[id(p)])
                for p in bank.parameters(g)) for g in ('qkv','out','ffn','refiner')}
            result['frozen_visual_unchanged'] = all(torch.equal(p.detach().cpu(), old)
                for p,old in zip((p for a in adapters for p in (a.lora_A,a.lora_B)), frozen_visual))
            if not result['frozen_visual_unchanged'] or not all(result['adapter_groups_changed'].values()):
                raise RuntimeError('Full-scope update violated frozen initialization or failed to update a target group')
        result['target_time_parameters_changed'] = (any(not torch.equal(p.detach().cpu(), old)
            for p, old in zip(time_module.parameters(), initial_time)) if time_module else None)
        if not result['qkv_parameters_changed']:
            raise RuntimeError('Optimizer did not update QKV parameters')
        if time_module and result['target_time_parameters_changed'] != args.train_target_time:
            raise RuntimeError('Target-time parameter changes violate the configured frozen/trainable policy')
        checkpoint(args.out_dir)
        result.update(status='complete', updates_this_run=len(result['updates']) - start_step,
            wall_seconds=time.perf_counter() - started,
            gpu_allocated_peak_MiB=torch.cuda.max_memory_allocated(args.device) / 2**20 if not args.smoke else 0,
            quality_gate='NOT_EVALUATED: training completion is not Stage1 quality acceptance')
        save_result()
    except Exception as exc:
        result.update(status='failed', error=repr(exc), wall_seconds=time.perf_counter() - started)
        save_result()
        raise


if __name__ == '__main__':
    main()
