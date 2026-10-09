#!/usr/bin/env python3
"""H3 baseline and incremental raw-KV rollout with explicit measurements.

Generated video is silent for both modes. Cached mode fixes audio NOISE and
prefix times; it is an untrained causal ablation, NOT a Stage2 checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/abot'))
sys.path.insert(0, str(ROOT/'code'))
# infer sets offline model resolution and patched DiffSynth path.
import infer as abot
import torch
import numpy as np
import av
from causal.h3_cached import (H3ChunkCache, chunk_forward, expand_packed_two_anchors,
                               last_frame_anchor, last_frame_image_anchor,
                               recompute_forward)


def write_video(frames, path):
    temporary = path.with_suffix('.tmp.mp4')
    with av.open(str(temporary), 'w', options={'movflags': '+faststart'}) as container:
        stream = container.add_stream('libx264', rate=24)
        stream.width, stream.height = frames[0].size
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'medium'}
        for im in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(im), format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    os.replace(temporary, path)


def save_json(path, data):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temporary, path)


def parse_action_schedule(spec, chunk_count):
    """Expand ``W:3,A:2,D:3`` into one held action per latent chunk."""
    if not spec:
        return None
    aliases = {
        'W': 'forward', 'S': 'back', 'A': 'strafe-left', 'D': 'strafe-right',
    }
    expanded = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f'action schedule item must be ACTION:COUNT, got {item!r}')
        action, count_text = (part.strip() for part in item.split(':', 1))
        action = aliases.get(action, action)
        if action not in abot.ACTION_PRESETS:
            raise ValueError(f'unknown action {action!r}; use W/S/A/D or a named H3 preset')
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(f'invalid action schedule count {count_text!r}') from exc
        if count < 1:
            raise ValueError('action schedule counts must be positive')
        expanded.extend([action] * count)
    if len(expanded) != chunk_count:
        raise ValueError(
            f'action schedule expands to {len(expanded)} chunks, expected {chunk_count}; '
            'the 124-frame default has 8 chunks')
    return expanded


@torch.no_grad()
def main():
    from causal.h3_precision import configure_precision, validate_precision_checkpoint
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--modes', nargs='+', choices=['baseline', 'cached', 'recompute'], default=['baseline', 'cached'])
    ap.add_argument('--steps', type=int, default=4, help='per chunk steps in cached/recompute mode, no distillation')
    ap.add_argument('--flow-shift', type=float, default=12., help='Euler noise schedule shift; native H3 default 12')
    ap.add_argument('--anyflow-sigma-grid', choices=['native', 'uniform'], default='native',
                    help='AnyFlow inference grid; uniform is independent of the checkpoint training shift')
    ap.add_argument('--baseline-steps', type=int, default=30)
    ap.add_argument('--num-frames', type=int, default=124)
    ap.add_argument('--chunk-frames', type=int, default=5)
    ap.add_argument('--history-chunks', type=int, default=5)
    ap.add_argument('--action-preset', default='W',
                    choices=['W', 'S', 'A', 'D'] + sorted(abot.ACTION_PRESETS),
                    help=('held action for the whole clip. W/S/A/D are convenient '
                          'aliases for forward/back/strafe-left/strafe-right; '
                          'the named H3 presets are also accepted'))
    ap.add_argument('--action-schedule', default=None,
                    help=('optional chunk-local schedule such as W:3,A:2,D:3; it must '
                          'expand to one action per latent chunk and overrides --action-preset'))
    ap.add_argument('--cache-device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--anchor-mode', default='fixed',
                    choices=['fixed', 'dynamic_last_frame', 'dynamic_last_frame_dual',
                             'dynamic_last_frame_rgb_dual'],
                    help=('fixed H3 image anchor; dynamic_last_frame replaces it with the '
                          'previous chunk tail; dynamic_last_frame_dual keeps the original '
                          'anchor and adds the previous chunk tail as a second slot; '
                          'dynamic_last_frame_rgb_dual performs the second slot via '
                          'H3 RGB decode/re-encode image conditioning'))
    ap.add_argument('--boundary-blend', type=float, default=0.0,
                    help='blend each new chunk first latent frame with previous last frame (0..1)')
    ap.add_argument('--boundary-overlap-frames', type=int, default=1,
                    help='number of leading latent frames receiving a decaying soft boundary blend')
    ap.add_argument('--boundary-clamp', action='store_true',
                    help=('fix each new chunk first latent frame to the previous chunk last '
                          'latent during every denoising step and clean KV commit'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=2)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--save-latents', action='store_true')
    ap.add_argument('--causal-adapter', type=Path, help='additional trained causal tail adapter (cached/recompute only)')
    ap.add_argument('--anyflow-adapter', type=Path,
                    help='trained target-time MLP; enables explicit finite-map sampling (cached only)')
    ap.add_argument('--causal-adapter-scope', choices=['all', 'commit', 'last_step_commit'],
                    default='all',
                    help=('when to enable the trained causal visual tail adapter: all applies '
                          'it to every denoising and clean-commit forward; commit keeps it only '
                          'for clean KV commits; last_step_commit applies it on the final '
                          'denoising step and the clean commit'))
    ap.add_argument('--causal-action-adapter', type=Path,
                    help=('zero-initialized action residual trained for the causal cache path; '
                          'adds action-dependent Q/K/V residuals to the current chunk'))
    ap.add_argument('--causal-action-prefix-adapter', type=Path,
                    help=('action-token prefix residual trained for causal action rows; '
                          'adds a per-latent hidden residual to current action text rows'))
    ap.add_argument('--h3-lora-adapter', type=Path,
                    help=('online update to selected released H3 LoRA matrices; '
                          'causal modes only'))
    ap.add_argument('--history-source', choices=['generated', 'clean'], default='generated',
                    help=('causal history used after each chunk. generated is the normal free '
                          'rollout; clean commits detached teacher latents for diagnosis.'))
    ap.add_argument('--action-prefix-mode', choices=['own', 'causal', 'all'], default='own',
                    help=('which action annotation rows a video query can read: own keeps the '
                          'original prototype, causal exposes past/current controls, all is a '
                          'diagnostic upper bound'))
    ap.add_argument('--action-feedback', action='store_true',
                    help=('restore the directed H3 edge from each action row to its bound '
                          'current video latent inside causal attention; future/history video '
                          'rows remain invisible'))
    ap.add_argument('--teacher-latents', type=Path, default=None,
                    help='teacher latent tensor [1,24,T,H,W] for --history-source clean')
    ap.add_argument('--precision-profile', choices=['legacy', 'h3_fp32'], default='legacy')
    ap.add_argument('--stage1-lora', type=Path, default=None)
    args = ap.parse_args()
    if args.precision_profile != 'legacy' and args.modes != ['cached']:
        ap.error('FP32 prototype currently requires --modes cached')
    if args.stage1_lora and (not args.causal_adapter or args.modes != ['cached']
                            or args.causal_adapter_scope != 'all'):
        ap.error('Full-scope Stage1 bank requires its frozen visual checkpoint, cached mode and all scope')
    if args.anyflow_adapter and (args.modes != ['cached'] or args.causal_adapter_scope != 'all'):
        ap.error('AnyFlow requires --modes cached and --causal-adapter-scope all')
    if args.anyflow_sigma_grid == 'uniform' and not args.anyflow_adapter:
        ap.error('Uniform AnyFlow grid requires an AnyFlow checkpoint')
    if args.anyflow_adapter and not args.causal_adapter:
        ap.error('AnyFlow requires the matching --causal-adapter checkpoint')
    action_aliases = {
        'W': 'forward', 'S': 'back', 'A': 'strafe-left', 'D': 'strafe-right',
    }
    action_preset = action_aliases.get(args.action_preset, args.action_preset)
    chunk_count = (abot.A.latent_t_for(args.num_frames) + args.chunk_frames - 1) // args.chunk_frames
    try:
        action_schedule = parse_action_schedule(args.action_schedule, chunk_count)
    except ValueError as exc:
        ap.error(str(exc))
    if action_schedule is None:
        action_schedule = [action_preset] * chunk_count
    if args.causal_adapter and 'baseline' in args.modes:
        ap.error('causal adapter cannot be applied to the original baseline')
    if args.causal_action_adapter and 'baseline' in args.modes:
        ap.error('causal action adapter cannot be applied to the original baseline')
    if args.h3_lora_adapter and 'baseline' in args.modes:
        ap.error('H3 online LoRA adapter cannot be applied to the original baseline')
    if args.history_source == 'clean' and ('baseline' in args.modes or args.teacher_latents is None):
        ap.error('--history-source clean requires causal mode and --teacher-latents')
    if args.history_source == 'generated' and args.teacher_latents is not None:
        ap.error('--teacher-latents is only valid with --history-source clean')
    if args.anchor_mode in ('dynamic_last_frame_dual', 'dynamic_last_frame_rgb_dual') and 'baseline' in args.modes:
        ap.error('dual-anchor mode is a causal prototype and cannot be used with baseline')
    if not 0.0 <= args.boundary_blend <= 1.0:
        ap.error('--boundary-blend must be between 0 and 1')
    if args.boundary_clamp and args.boundary_blend > 0:
        ap.error('--boundary-clamp and --boundary-blend are mutually exclusive')
    if args.boundary_overlap_frames < 1 or args.boundary_overlap_frames > args.chunk_frames:
        ap.error('--boundary-overlap-frames must be between 1 and chunk_frames')
    if args.num_frames < 5 or (args.num_frames-5)%17 or min(args.steps,args.baseline_steps,args.chunk_frames,args.history_chunks)<1:
        ap.error('invalid steps, chunks or frame count (must be 17k+5)')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    def sync():
        torch.cuda.synchronize(args.device)
    def memory():
        return dict(allocated_peak_MiB=torch.cuda.max_memory_allocated(args.device)/2**20,
                    reserved_peak_MiB=torch.cuda.max_memory_reserved(args.device)/2**20)
    @contextmanager
    def phase(results, name):
        sync()
        start = time.perf_counter()
        yield
        sync()
        results[name] = time.perf_counter()-start
        print(f'[timing] {name}: {results[name]:.3f}s', flush=True)

    setup={}
    with phase(setup, 'load_models_and_lora_seconds'):
        pipe=abot.load_pipeline(args.device)
        lora=abot.load_checkpoint_lora(ROOT/'checkpoints/H3-World/step-10000.safetensors')
        pipe.load_lora(pipe.dit, state_dict=lora, hotload=True)
        setup['precision'] = configure_precision(pipe.dit, args.precision_profile,
            native_transformer_dir=ROOT / 'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        if args.anyflow_adapter:
            from causal.anyflow import load_anyflow
            _, anyflow_metadata = load_anyflow(pipe.dit, args.anyflow_adapter, args.device)
            validate_precision_checkpoint(anyflow_metadata, setup['precision'])
            setup['anyflow'] = dict(path=str(args.anyflow_adapter), metadata=anyflow_metadata,
                                   objective='anyflow_forward_map_v1_5')
        if args.h3_lora_adapter:
            from causal.pretrained_lora import load_h3_lora_adapter
            loaded_h3_lora = load_h3_lora_adapter(
                pipe.dit, args.h3_lora_adapter, args.device)
            setup['h3_lora_adapter'] = {
                'path': str(args.h3_lora_adapter),
                'module_names': loaded_h3_lora['module_names'],
                'metadata': loaded_h3_lora['metadata'],
            }
        causal_adapter_modules = []
        if args.causal_adapter:
            from causal.pretrained_lora import load_adapter
            loaded_causal = load_adapter(pipe.dit, args.causal_adapter, args.device)
            validate_precision_checkpoint(loaded_causal.get('metadata', {}), setup['precision'])
            setup['causal_adapter'] = loaded_causal
            if args.anyflow_adapter:
                from causal.anyflow import validate_checkpoint_protocol
                validate_checkpoint_protocol(anyflow_metadata, loaded_causal['metadata'], vars(args))
            for block_index in loaded_causal['block_indices']:
                module = pipe.dit.blocks[int(block_index)].attn.qkv_proj
                if not hasattr(module, 'enabled'):
                    raise RuntimeError(
                        f'loaded causal adapter block {block_index} has no runtime enabled flag')
                causal_adapter_modules.append(module)

        def set_causal_adapter_enabled(enabled):
            for module in causal_adapter_modules:
                module.enabled = bool(enabled)

        # Keep visual-history stabilization independently gateable from the
        # action residual.  The commit-only ablation lets the trained visual
        # adapter write clean historical K/V without changing the noisy action
        # score field used to generate the current chunk.
        set_causal_adapter_enabled(args.causal_adapter_scope == 'all')
        action_adapter = None
        bank_metadata = None
        if args.causal_adapter:
            full_scope = loaded_causal.get('metadata', {}).get('config', {}).get('adapter_scope') == 'all_qkvo_ffn'
            if full_scope != bool(args.stage1_lora):
                raise ValueError('Checkpoint requires its matching full-scope Stage1 adapter bank')
        if args.stage1_lora:
            from causal.stage1_lora import load_stage1_lora
            bank, bank_metadata = load_stage1_lora(pipe.dit,args.stage1_lora,device=args.device)
            validate_precision_checkpoint(bank_metadata, setup['precision'])
            for parameter in bank.parameters():
                parameter.requires_grad_(False)
            setup['stage1_lora'] = dict(path=str(args.stage1_lora), **bank.describe())
        if args.causal_action_adapter:
            from causal.pretrained_lora import load_action_residual
            loaded_action = load_action_residual(pipe.dit, args.causal_action_adapter, args.device)
            action_adapter = loaded_action['adapter']
            setup['causal_action_adapter'] = {
                'path': str(args.causal_action_adapter),
                'block_indices': loaded_action['block_indices'],
                'metadata': loaded_action['metadata'],
            }
        if args.causal_adapter:
            from causal.stage1_protocol import validate_stage1_protocol
            validate_stage1_protocol(loaded_causal.get('metadata', {}), bank_metadata,
                loaded_action['metadata'] if args.causal_action_adapter else None,
                anyflow_metadata if args.anyflow_adapter else None, vars(args))
        action_prefix_adapter = None
        if args.causal_action_prefix_adapter:
            from causal.pretrained_lora import load_action_prefix_residual
            loaded_prefix = load_action_prefix_residual(
                pipe.dit, args.causal_action_prefix_adapter, args.device)
            action_prefix_adapter = loaded_prefix['adapter']
            setup['causal_action_prefix_adapter'] = {
                'path': str(args.causal_action_prefix_adapter),
                'block_indices': loaded_prefix['block_indices'],
                'metadata': loaded_prefix['metadata'],
            }
    with phase(setup, 'conditioning_seconds'):
        latent_t=abot.A.latent_t_for(args.num_frames)
        keys=np.zeros((latent_t,len(abot.S.KEYS9)),dtype=np.int64)
        for chunk, scheduled_action in enumerate(action_schedule):
            start = chunk * args.chunk_frames
            stop = min(start + args.chunk_frames, latent_t)
            for key in abot.ACTION_PRESETS[scheduled_action]:
                keys[start:stop, abot.S.KEYS9.index(key)] = 1
        shared=dict(cfg_scale=1., height=abot.HEIGHT, width=abot.WIDTH,
                    num_frames=args.num_frames, seed=args.seed, rand_device='cpu',
                    keyframes=[abot.load_first_frame(ROOT/'examples/first_frame.png')],
                    keyframe_indices=[0], imgvid_cond_noise_aug=.999, audio_cond_noise_aug=1.)
        pos=dict(prompt='A man in a yellow floral shirt stands in a dim, multi-level concrete parking garage.',
                 action_script=abot.S.annotate_from_keys9(keys))
        neg={}
        for unit in pipe.units:
            shared,pos,neg=pipe.unit_runner(unit,pipe,shared,pos,neg)
    setup['memory']=memory()
    setup['completed_at']=datetime.now().astimezone().isoformat()
    setup['config']={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    setup['gpu']=torch.cuda.get_device_name(args.device)
    setup['visible_devices']=os.environ.get('CUDA_VISIBLE_DEVICES')
    setup['torch_version']=torch.__version__
    schedule_label = ','.join(action_schedule)
    setup['conditioning']=(f'same prompt, image, chunk action schedule [{schedule_label}], '
                           'initial video/audio noise reused across modes')
    setup['action_preset']=args.action_preset
    setup['action_preset_resolved']=action_preset
    setup['action_schedule']=action_schedule
    initial=shared['video_latents'].clone()
    audio_noise=shared['audio_latents'].clone()
    # Byte-identical inputs can be audited across separately launched runs.
    # Hashing is read-only and does not consume RNG state or change sampling.
    setup['input_fingerprints'] = {
        'initial_image_sha256': hashlib.sha256(
            (ROOT/'examples/first_frame.png').read_bytes()).hexdigest(),
        'prompt': pos['prompt'],
        'video_noise_sha256': hashlib.sha256(
            initial.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        'audio_noise_sha256': hashlib.sha256(
            audio_noise.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        'video_noise_shape': list(initial.shape),
        'audio_noise_shape': list(audio_noise.shape),
    }
    setup['vram_reserve_gib'] = float(os.environ.get('ABOT_VRAM_RESERVE_GIB', '5'))
    setup['compile_action_block_mask'] = os.environ.get('H3_COMPILE_BLOCK_MASK') == '1'
    action_cond_full = torch.from_numpy(keys.astype(np.float32)).to(
        device=initial.device, dtype=initial.dtype)
    teacher_history = None
    if args.history_source == 'clean':
        teacher_history = torch.load(args.teacher_latents, map_location=args.device,
                                     weights_only=True).to(device=args.device, dtype=initial.dtype)
        if (teacher_history.ndim != 5 or teacher_history.shape[0] != initial.shape[0]
                or teacher_history.shape[1] != initial.shape[1]
                or teacher_history.shape[2] < latent_t
                or teacher_history.shape[-2:] != initial.shape[-2:]
                or not torch.isfinite(teacher_history[:, :, :latent_t]).all()):
            raise ValueError('--teacher-latents must be finite [1,24,T,H,W] and cover this run')
        teacher_history = teacher_history[:, :, :latent_t].contiguous()
        setup['history_source'] = 'clean_teacher_latents'
        setup['teacher_latents'] = str(args.teacher_latents)
    rows_per_frame = (initial.shape[-2] // 2) * (initial.shape[-1] // 2)
    dual_packed = None
    dual_anchor = None
    if args.anchor_mode in ('dynamic_last_frame_dual', 'dynamic_last_frame_rgb_dual'):
        # The production H3 packed builder normally has one image condition
        # frame.  Extend that immutable layout once, then use a second anchor
        # slot for the previous chunk tail during causal rollout.  Chunk 0
        # uses a duplicate first-frame anchor until a generated tail exists.
        dual_packed = expand_packed_two_anchors(pos['packed'], frame_rows=rows_per_frame)
        dual_anchor = torch.cat((shared['keyframe_cond_anchor'],
                                 shared['keyframe_cond_anchor'].clone()), dim=0)
        setup['dual_anchor_protocol'] = {
            'slots': ['original_h3_first_frame', 'previous_chunk_last_latent_frame'],
            'chunk0_second_slot': 'duplicate_original_anchor',
            'frame_rows': rows_per_frame,
            'packed_seq_len': int(dual_packed['seq_len']),
            'slot1_encoding': ('rgb_decode_reencode_image'
                               if args.anchor_mode == 'dynamic_last_frame_rgb_dual'
                               else 'normalized_temporal_latent_patchify'),
        }
    save_json(args.out_dir/'setup.json',setup)
    # A corrupted text-encoder/offload execution can otherwise produce a tiny
    # but apparently successful MP4 full of NaNs.  Refuse to save teacher
    # latents in that case so later causal training cannot silently consume an
    # invalid target clip.
    finite_checks = {
        'prompt_embeds': pos['prompt_embeds'],
        'video_latents': initial,
        'audio_latents': audio_noise,
        'keyframe_cond_anchor': shared['keyframe_cond_anchor'],
    }
    bad = [name for name, value in finite_checks.items()
           if not torch.isfinite(value).all()]
    if bad:
        raise FloatingPointError(
            'conditioning produced non-finite tensors: ' + ', '.join(bad))
    if args.save_latents:
        torch.save(dict(prompt_embeds=pos['prompt_embeds'].cpu(),
                        packed={k:v.cpu() if torch.is_tensor(v) else v for k,v in pos['packed'].items()},
                        anchor=shared['keyframe_cond_anchor'].cpu(), initial_noise=initial.cpu(),
                        audio_noise=audio_noise.cpu()), args.out_dir/'conditioning.pt')

    for mode in args.modes:
        result=dict(mode=mode, status='running', started_at=datetime.now().astimezone().isoformat(),
                    steps=args.baseline_steps if mode=='baseline' else args.steps,
                    num_frames=args.num_frames, seed=args.seed, gpu=setup['gpu'],
                    visible_devices=setup['visible_devices'], cache_device=args.cache_device,
                    chunk_frames=args.chunk_frames, history_chunks=args.history_chunks,
                    trained_for_causal=bool(args.causal_adapter),
                    causal_adapter_scope=args.causal_adapter_scope,
                    history_source=args.history_source,
                    anchor_mode=args.anchor_mode,
                    action_prefix_mode=args.action_prefix_mode,
                    boundary_blend=args.boundary_blend,
                    boundary_overlap_frames=args.boundary_overlap_frames,
                    boundary_clamp=args.boundary_clamp,
                    action_preset=args.action_preset,
                    action_preset_resolved=action_preset,
                    action_schedule=action_schedule,
                    action_feedback=args.action_feedback,
                    flow_shift=args.flow_shift,
                    anyflow_adapter=str(args.anyflow_adapter) if args.anyflow_adapter else None,
                    sampler='anyflow_finite_map' if args.anyflow_adapter else 'flow_matching_euler',
                    anchor_protocol=(
                        'global_retimed_rgb_prefix_last_image_dual_v2'
                        if args.anchor_mode == 'dynamic_last_frame_rgb_dual'
                        else ('global_retimed_latent_dual_v1'
                        if args.anchor_mode == 'dynamic_last_frame_dual'
                        else ('global_retimed_latent_v1' if args.anchor_mode == 'dynamic_last_frame'
                              else 'fixed_image_v1'))),
                    causal_adapter=str(args.causal_adapter) if args.causal_adapter else None,
                    causal_action_adapter=str(args.causal_action_adapter)
                    if args.causal_action_adapter else None,
                    cfg_scale=1., audio_output='silent',
                    attention_backend=(
                        'PyTorch SDPA'
                        if mode == 'cached' or os.environ.get('H3_CAUSAL_EAGER_SDPA') == '1'
                        else 'FlexAttention'))
        result['prefix_contract']=('original joint audio/video denoising' if mode=='baseline' else
                                  'text t=1, anchor t=.999; fixed audio noise at native t=0; global H3 RoPE')
        metrics_path=args.out_dir/f'{mode}.json'
        save_json(metrics_path,result)
        torch.cuda.reset_peak_memory_stats(args.device)
        run_start=time.perf_counter()
        cache=H3ChunkCache(args.history_chunks, 'cpu' if args.cache_device=='cpu' else args.device)
        try:
            with phase(result, 'dit_prepare_seconds'):
                pipe.load_models_to_device(['dit'])
            steps=result['steps']
            from causal.anyflow_sampling import configure_video_schedule
            result['video_sigmas'] = configure_video_schedule(pipe.scheduler,
                steps=steps, grid=args.anyflow_sigma_grid, flow_shift=args.flow_shift)
            result['video_sigma_grid'] = args.anyflow_sigma_grid
            result['effective_video_sampling_shift'] = (1.0 if args.anyflow_sigma_grid == 'uniform' else args.flow_shift)
            if args.anyflow_adapter:
                training_shift = anyflow_metadata['config'].get('training_timestep_shift')
                result['anyflow_training_shift'] = (
                    anyflow_metadata['config']['flow_shift']
                    if training_shift is None else training_shift)
            save_json(metrics_path, result)
            pipe.scheduler_audio.set_timesteps(steps,shift=3.)
            video=initial.clone()
            audio=audio_noise.clone()
            result['denoiser_forwards']=0
            result['commit_forwards']=0
            with phase(result, 'sampling_seconds'):
                if mode=='baseline':
                    for j,t in enumerate(pipe.scheduler.timesteps):
                        vp,apred=pipe.model_fn(dit=pipe.dit, **dict(shared,video_latents=video,audio_latents=audio),
                                              **pos,timestep_video=t.to(args.device),
                                              timestep_audio=pipe.scheduler_audio.timesteps[j].to(args.device))
                        video=pipe.scheduler.step(vp,t,video)
                        audio=pipe.scheduler_audio.step(apred,pipe.scheduler_audio.timesteps[j],audio)
                        result['denoiser_forwards']+=1
                        if j%5==0 or j==steps-1:
                            print(f'[baseline] step {j+1}/{steps}',flush=True)
                else:
                    done=[]
                    mode_packed = (dual_packed if args.anchor_mode in
                                   ('dynamic_last_frame_dual', 'dynamic_last_frame_rgb_dual')
                                   else pos['packed'])
                    mode_anchor = (dual_anchor if args.anchor_mode in
                                   ('dynamic_last_frame_dual', 'dynamic_last_frame_rgb_dual')
                                   else shared['keyframe_cond_anchor'])
                    common=dict(full_packed=mode_packed,prompt=pos['prompt_embeds'],
                                anchor=mode_anchor,audio=audio_noise,
                                chunk_frames=args.chunk_frames)
                    chunks=[]
                    for i,start in enumerate(range(0,latent_t,args.chunk_frames)):
                        sync(); chunk_start=time.perf_counter()
                        current=initial[:,:,start:start+args.chunk_frames].clone()
                        # The explicit last-frame anchor is a visual handoff
                        # between chunks.  It is only a diagnostic prefix
                        # replacement for now; cache rows still contain the
                        # causal history and the repeated frame is not
                        # committed as a new video chunk.
                        anchor_slot = 0
                        previous_history = (done[-1] if args.history_source == 'generated' and i > 0
                                            else (teacher_history[:, :, :start]
                                                  if args.history_source == 'clean'
                                                  else initial[:, :, :0]))
                        boundary_frame = (previous_history[:, :, -1:].clone()
                                          if args.boundary_clamp and i > 0 else None)
                        if args.anchor_mode == 'dynamic_last_frame' and i > 0:
                            chunk_anchor = last_frame_anchor(previous_history[:, :, -1:])
                            anchor_frame_index = start - 1
                        elif args.anchor_mode in ('dynamic_last_frame_dual',
                                                  'dynamic_last_frame_rgb_dual') and i > 0:
                            # Preserve the original scene anchor in slot 0;
                            # replace only slot 1 with the previous generated
                            # tail.  The RGB mode follows H3's actual image
                            # condition path instead of patchifying a temporal
                            # latent directly.
                            if args.anchor_mode == 'dynamic_last_frame_rgb_dual':
                                rgb_start = time.perf_counter()
                                pipe.load_models_to_device(['video_vae'])
                                tail_anchor = last_frame_image_anchor(
                                    pipe.video_vae, (torch.cat(done, dim=2)
                                                     if args.history_source == 'generated'
                                                     else previous_history),
                                    dtype=pipe.torch_dtype)
                                pipe.load_models_to_device(['dit'])
                                result.setdefault('rgb_anchor_seconds', []).append(
                                    time.perf_counter() - rgb_start)
                            else:
                                tail_anchor = last_frame_anchor(previous_history[:, :, -1:])
                            chunk_anchor = torch.cat((mode_anchor[:rows_per_frame], tail_anchor), dim=0)
                            anchor_frame_index = start - 1
                            anchor_slot = 1
                        else:
                            chunk_anchor = mode_anchor
                            anchor_frame_index = None
                        chunk_common = dict(common, anchor=chunk_anchor)
                        chunk_action_cond = action_cond_full[start:start + current.shape[2]]
                        for j,t in enumerate(pipe.scheduler.timesteps):
                            sigma=float(t)/1000
                            target_sigma = (float(pipe.scheduler.timesteps[j+1]) / 1000
                                            if j+1 < len(pipe.scheduler.timesteps) else 0.0)
                            if causal_adapter_modules:
                                set_causal_adapter_enabled(
                                    args.causal_adapter_scope == 'all'
                                    or (args.causal_adapter_scope == 'last_step_commit'
                                        and j == len(pipe.scheduler.timesteps) - 1))
                            if mode=='cached':
                                vp=chunk_forward(pipe.dit,current,index=i,cache=cache,sigma=sigma,
                                                  target_sigma=target_sigma if args.anyflow_adapter else None,
                                                  anchor_frame_index=anchor_frame_index,
                                                  anchor_slot=anchor_slot,
                                                  fixed_boundary=boundary_frame,
                                                  action_prefix_mode=args.action_prefix_mode,
                                                  action_feedback=args.action_feedback,
                                                  action_cond=chunk_action_cond,
                                                  action_adapter=action_adapter,
                                                  action_prefix_adapter=action_prefix_adapter,
                                                  **chunk_common)
                            else:
                                history=(torch.cat(done, dim=2)
                                         if args.history_source == 'generated'
                                         else teacher_history[:, :, :start])
                                vp=recompute_forward(pipe.dit,current,history=history,sigma=sigma,
                                                     window_chunks=args.history_chunks+1,
                                                     anchor_frame_index=anchor_frame_index,
                                                     anchor_slot=anchor_slot,
                                                     fixed_boundary=boundary_frame,
                                                     action_cond=chunk_action_cond,
                                                     action_adapter=action_adapter,
                                                     action_prefix_mode=args.action_prefix_mode,
                                                     **chunk_common)
                            # Advance the current chunk at every scheduler
                            # point.  Keeping this update inside the loop is
                            # essential: evaluating all velocity fields and
                            # applying only the final one degenerates an
                            # advertised N-step rollout into a one-step jump.
                            if args.anyflow_adapter:
                                from causal.anyflow import finite_map_step
                                current=finite_map_step(current, vp, sigma, target_sigma)
                            else:
                                current=pipe.scheduler.step(vp,t,current)
                            if boundary_frame is not None:
                                # The H3 model receives this row with
                                # denoise_mask=0. Re-apply it after the
                                # scheduler update so the fixed state cannot
                                # drift between denoiser evaluations.
                                current[:, :, :1] = boundary_frame
                            if not torch.isfinite(current).all():
                                raise FloatingPointError(f'chunk {i}, step {j}: non-finite latents')
                            result['denoiser_forwards']+=1
                            if j % 5 == 0 or j == steps - 1:
                                print(f'[{mode}] chunk {i+1} step {j+1}/{steps}', flush=True)
                        if (mode != 'baseline' and i > 0 and args.boundary_blend > 0):
                            # A one-frame overlap is a deterministic temporal
                            # continuity constraint. It is deliberately
                            # exposed as a separate ablation: it cannot replace
                            # Stage1/Stage2 training, but prevents a noisy
                            # chunk boundary from becoming an instantaneous
                            # spatial jump in the final decoded video.
                            previous_last = done[-1][:, :, -1:]
                            overlap = min(args.boundary_overlap_frames, current.shape[2])
                            decay = (1.0 - torch.arange(
                                overlap, device=current.device, dtype=current.dtype
                            ) / overlap).view(1, 1, overlap, 1, 1)
                            alpha = args.boundary_blend * decay
                            current[:, :, :overlap] = (
                                alpha * previous_last
                                + (1.0 - alpha) * current[:, :, :overlap])
                        done.append(current)
                        # Separate clean forward is essential: the final noisy
                        # step's K/V does not represent the generated clean chunk.
                        if mode=='cached':
                            set_causal_adapter_enabled(
                                args.causal_adapter_scope in ('all', 'commit', 'last_step_commit'))
                            commit_latents = (current if args.history_source == 'generated'
                                              else teacher_history[:, :, start:start + current.shape[2]])
                            chunk_forward(pipe.dit,commit_latents,index=i,cache=cache,sigma=0.,commit=True,
                                          anchor_frame_index=anchor_frame_index,
                                          anchor_slot=anchor_slot,
                                          action_prefix_mode=args.action_prefix_mode,
                                          action_feedback=args.action_feedback,
                                          action_cond=chunk_action_cond,
                                          action_adapter=action_adapter,
                                          action_prefix_adapter=action_prefix_adapter,
                                          fixed_boundary=boundary_frame, **chunk_common)
                            result['commit_forwards']+=1
                        sync()
                        chunks.append(time.perf_counter()-chunk_start)
                        print(f'[{mode}] chunk {i+1}/{(latent_t+args.chunk_frames-1)//args.chunk_frames} '
                              f'{chunks[-1]:.2f}s, cache {cache.nbytes/2**30:.2f} GiB',flush=True)
                        result['chunk_seconds']=chunks
                        save_json(metrics_path,result)
                    video=torch.cat(done,dim=2)
            result['sampling_memory']=memory()
            result['kv_cache_peak_MiB']=cache.peak_bytes/2**20
            result['per_layer_commits']=cache.commits
            cache.clear()
            if args.save_latents:
                torch.save(video.cpu(),args.out_dir/f'{mode}_latents.pt')
            with phase(result, 'vae_prepare_seconds'):
                pipe.load_models_to_device(['video_vae'])
            with phase(result, 'vae_decode_seconds'):
                decoded=pipe.video_vae.decode_video(video,dtype=pipe.torch_dtype,
                                                     tiled=True,tile_size=256,tile_overlap=64)
            with phase(result, 'frames_to_cpu_seconds'):
                frames=pipe.vae_output_to_video(decoded,min_value=0,max_value=1)
            del decoded
            with phase(result, 'encode_mp4_seconds'):
                write_video(frames,args.out_dir/f'{mode}.mp4')
            result['wall_after_conditioning_seconds']=time.perf_counter()-run_start
            result['wall_including_shared_setup_seconds']=result['wall_after_conditioning_seconds']+sum(
                setup[k] for k in ['load_models_and_lora_seconds','conditioning_seconds'])
            result['memory_after_setup']=memory()
            result['memory_entire_run_peak_MiB']=max(setup['memory']['allocated_peak_MiB'],
                                                    result['memory_after_setup']['allocated_peak_MiB'])
            result['status']='complete'
            result['completed_at']=datetime.now().astimezone().isoformat()
            save_json(metrics_path,result)
        except Exception as e:
            result['status']='failed'
            result['error']=repr(e)
            save_json(metrics_path,result)
            raise
        finally:
            cache.clear()


if __name__=='__main__':
    main()
