#!/usr/bin/env python3
"""Multi-action Stage1-style clean-history training on pretrained H3.

The script freezes the released action LoRA, then trains rank-r QKV LoRA on a
configurable tail of DiT blocks using both first and later chunks with clean
history.  It is deliberately a feasibility prototype:
there is no AnyFlow, student rollout, SGF, DMD, or held-out-scene claim.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/abot"))
sys.path.insert(0, str(ROOT / "code"))
import infer as abot
import torch
try:
    # Each target chunk has a different packed sequence length.  FlexAttention
    # specializes those shapes through torch._dynamo; the default limit of
    # eight recompilations aborts a long-clip teacher-forcing run before the
    # optimizer ever sees a batch.
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 128)
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 128)
except AttributeError:
    pass
from causal.h3_cached import (H3ChunkCache, chunk_forward, expand_packed_two_anchors,
                              last_frame_anchor, last_frame_image_anchor,
                              recompute_forward)
from causal.pretrained_lora import (install_action_residual, install_adapters,
                                    replay_tail, save_action_residual, save_adapters)
from diffsynth.models.minimax_h3_dit import patchify_video
from diffsynth.core.vram.layers import AutoTorchModule


def move_tree(value, device, clone=False):
    if torch.is_tensor(value):
        return value.detach().to(device=device, copy=clone)
    if isinstance(value, dict):
        return {k: move_tree(v, device, clone) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move_tree(v, device, clone) for v in value)
    return value


def _normalise_action_key(label):
    aliases = {'w': 'W', 's': 'S', 'a': 'A', 'd': 'D'}
    key = aliases.get(str(label).strip().lower(), str(label).strip().upper())
    if key not in abot.S.KEYS9:
        raise ValueError(f'cannot construct action condition for {label!r}')
    return key


def action_condition(spec, latent_count, device, dtype, chunk_frames=5):
    """One-hot H3 action buttons at the original latent-interval granularity.

    A plain ``W``/``A``/``S``/``D`` applies to every latent interval.  A
    schedule such as ``W:3,A:2,D:3`` applies the action per causal chunk while
    still producing one action row per latent interval, matching H3's native
    action binding.  The schedule is therefore training-time supervision, not
    a chunk-level replacement for the model's latent-level rows.
    """
    raw = str(spec).strip()
    if not raw:
        raise ValueError('empty action specification')
    if ':' not in raw:
        key = _normalise_action_key(raw)
        keys = [key] * int(latent_count)
    else:
        chunks = []
        for item in raw.split(','):
            item = item.strip()
            if not item or ':' not in item:
                raise ValueError(f'invalid action schedule item {item!r}')
            label, count_text = (part.strip() for part in item.split(':', 1))
            try:
                count = int(count_text)
            except ValueError as exc:
                raise ValueError(f'invalid action schedule count {count_text!r}') from exc
            if count < 1:
                raise ValueError('action schedule counts must be positive')
            chunks.extend([_normalise_action_key(label)] * count)
        expected_chunks = (int(latent_count) + int(chunk_frames) - 1) // int(chunk_frames)
        if len(chunks) != expected_chunks:
            raise ValueError(
                f'action schedule {raw!r} expands to {len(chunks)} chunks; '
                f'teacher has {latent_count} latent frames and needs {expected_chunks}')
        keys = []
        for chunk_key in chunks:
            keys.extend([chunk_key] * int(chunk_frames))
        keys = keys[:int(latent_count)]
    value = torch.zeros((int(latent_count), len(abot.S.KEYS9)), device=device, dtype=dtype)
    for row, key in enumerate(keys):
        value[row, abot.S.KEYS9.index(key)] = 1
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-dirs", type=Path, nargs="+", required=True,
                    help="one or more action-conditioned teacher directories; all share the adapter")
    ap.add_argument("--action-specs", nargs="+", default=None,
                    help=("one action spec per teacher dir: W/S/A/D for a held action or "
                          "a latent-preserving chunk schedule such as W:3,A:2,D:3"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--history-latents", type=Path, default=None,
                    help="detached generated-history latent tensor; keep teacher targets")
    ap.add_argument("--history-latent-dirs", type=Path, nargs="+", default=None,
                    help=('one generated-history directory per teacher dir; each must contain '
                          'cached_latents.pt (or generated_latents.pt)'))
    ap.add_argument("--history-mix", type=float, default=1.0,
                    help=("when --history-latents is supplied, mix it with clean teacher "
                          "history: 0=clean, 1=generated; this is a scheduled-history "
                          "diagnostic, not Stage2 distillation"))
    ap.add_argument("--history-mix-schedule", type=str, default=None,
                    help=("curriculum schedule for generated-history probability: 'START:END' "
                          "linearly interpolates the probability of sampling a generated-history "
                          "feature over training steps (e.g. '0.0:0.5' starts clean and reaches "
                          "50%% generated by the final step). Requires --history-latent-dirs. "
                          "Overrides --history-mix when set."))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target-chunks", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--train-noise-seeds", type=int, nargs="+", default=[101, 102],
                    help="CPU noise seeds used for teacher-forcing samples")
    ap.add_argument("--validation-noise-seeds", type=int, nargs="+", default=[1001, 1002],
                    help="CPU noise seeds reserved for validation samples")
    ap.add_argument("--history-noise-std", type=float, default=0.0,
                    help="std of perturbation added to clean history (scheduled-history proxy)")
    ap.add_argument("--tail-blocks", type=int, default=1,
                    help="number of final DiT blocks receiving causal QKV LoRA")
    ap.add_argument("--history-chunks", type=int, default=5,
                    help="clean-history chunks retained during teacher forcing")
    ap.add_argument("--anchor-mode",
                    choices=["fixed", "dynamic_last_frame", "dynamic_last_frame_dual",
                             "dynamic_last_frame_rgb_dual"],
                    default="fixed",
                    help=("fixed H3 image anchor; dynamic_last_frame replaces it with the "
                          "clean previous chunk tail; dynamic_last_frame_dual keeps the "
                          "original anchor and adds that tail as a second slot; "
                          "dynamic_last_frame_rgb_dual encodes the clean prefix tail through "
                          "H3's RGB image branch"))
    ap.add_argument("--scheduler-steps", type=int, default=8,
                    help="H3 scheduler grid used to choose teacher-forcing noise points")
    ap.add_argument("--scheduler-shift", type=float, default=12.0,
                    help="H3 flow shift, matched to the causal benchmark")
    ap.add_argument('--sigma-grid', type=float, nargs='+', help='explicit FM sigma coverage, in (0,1]')
    ap.add_argument('--history-protocol', choices=['recompute', 'cached'], default='recompute',
                    help='cached preserves each historical chunk original anchor and K/V; tail replay uses detached base history')
    ap.add_argument('--action-feedback', action='store_true',
                    help=('train with the causal attention path that restores each action row\'s '
                          'read edge to its bound current video latent'))
    ap.add_argument('--boundary-weight', type=float, default=1.0,
                    help='relative loss weight for the first latent frame of every chunk')
    ap.add_argument('--action-pair-weight', type=float, default=0.0,
                    help=('weight for paired W/S and A/D velocity-difference loss at matching '
                          'chunk, noise seed and sigma; zero keeps plain flow loss'))
    ap.add_argument('--action-base-pair-weight', type=float, default=0.0,
                    help=('weight for preserving the zero-adapter causal action difference. '
                          'This explicitly prevents the adapter from collapsing the pretrained '
                          'causal A-D/W-S response'))
    ap.add_argument('--base-output-reg-weight', type=float, default=0.0,
                    help=('weight for preserving the zero-adapter causal output. This keeps '
                          'the pretrained action geometry while the adapter learns the '
                          'teacher target; zero disables the regularizer'))
    ap.add_argument('--action-residual', action='store_true',
                    help=('train a zero-initialized current-chunk action Q/K/V residual in the '
                          'causal tail; save it separately as action_adapter.pt'))
    ap.add_argument('--action-residual-mode', choices=['qkv', 'hidden'], default='qkv',
                    help='where the causal action residual is injected')
    ap.add_argument('--no-qkv-adapter', action='store_true',
                    help=('freeze the causal QKV path and train only --action-residual; this '
                          'isolates action repair from generic denoising adaptation'))
    args = ap.parse_args()
    if args.no_qkv_adapter and not args.action_residual:
        ap.error('--no-qkv-adapter requires --action-residual')
    if args.action_specs is not None and len(args.action_specs) != len(args.teacher_dirs):
        ap.error('--action-specs must contain exactly one spec per --teacher-dirs entry')
    if args.history_mix_schedule is not None:
        if args.history_latent_dirs is None:
            ap.error('--history-mix-schedule requires --history-latent-dirs')
        try:
            start_str, end_str = args.history_mix_schedule.split(':')
            history_mix_start = float(start_str)
            history_mix_end = float(end_str)
            if not (0.0 <= history_mix_start <= 1.0 and 0.0 <= history_mix_end <= 1.0):
                ap.error('--history-mix-schedule values must be in [0.0, 1.0]')
        except (ValueError, AttributeError):
            ap.error('--history-mix-schedule must be START:END (e.g. "0.0:0.5")')
    else:
        history_mix_start = history_mix_end = args.history_mix
    if (args.steps < 1 or args.rank < 1 or args.tail_blocks < 1 or args.history_chunks < 1
            or args.tail_blocks > 50 or args.lr <= 0 or args.history_noise_std < 0
            or args.boundary_weight <= 0 or args.action_pair_weight < 0
            or args.action_base_pair_weight < 0
            or args.base_output_reg_weight < 0
            or not 0.0 <= args.history_mix <= 1.0
            or any(x < 0 for x in args.target_chunks)):
        ap.error("steps/rank must be positive, lr must be positive, chunks non-negative")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.cuda.set_device(args.device)
    started = time.perf_counter()
    result = dict(
        status="running", kind="pretrained multi-chunk QKV LoRA",
        objective="Stage1-style clean-history flow matching, no AnyFlow or DMD",
        data="multiple action-conditioned H3 teacher clips; clean-history teacher forcing, held-out noise",
        teacher_dirs=[str(p) for p in args.teacher_dirs], steps=args.steps, rank=args.rank, lr=args.lr,
        action_specs=list(args.action_specs) if args.action_specs is not None else None,
        history_latents=str(args.history_latents) if args.history_latents else None,
        history_latent_dirs=[str(p) for p in args.history_latent_dirs]
        if args.history_latent_dirs else None,
        history_mix_schedule=args.history_mix_schedule,
        history_mix_start=history_mix_start,
        history_mix_end=history_mix_end,
        boundary_weight=args.boundary_weight,
        action_pair_weight=args.action_pair_weight,
        action_base_pair_weight=args.action_base_pair_weight,
        base_output_reg_weight=args.base_output_reg_weight,
        action_residual=args.action_residual,
        action_residual_mode=args.action_residual_mode,
        qkv_adapter=not args.no_qkv_adapter,
        target_chunks=args.target_chunks, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        anchor_mode=args.anchor_mode, scheduler_steps=args.scheduler_steps,
        scheduler_shift=args.scheduler_shift,
        train_noise_seeds=args.train_noise_seeds,
        validation_noise_seeds=args.validation_noise_seeds,
        history_noise_std=args.history_noise_std,
        tail_blocks=args.tail_blocks,
        history_chunks=args.history_chunks,
        requested_history_protocol=args.history_protocol,
        action_feedback=args.action_feedback,
    )

    def save_result():
        p = args.out_dir / "training.json"
        tmp = p.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(p)

    save_result()
    try:
        cases = []
        rows_per_frame = None
        if args.history_latents is not None:
            raise ValueError('--history-latents is not supported by the multi-action trainer; use --history-latent-dirs')
        if args.history_latent_dirs is not None and len(args.history_latent_dirs) != len(args.teacher_dirs):
            raise ValueError('--history-latent-dirs must align one-to-one with --teacher-dirs')
        for case_index, teacher_dir in enumerate(args.teacher_dirs):
            cond = torch.load(teacher_dir / "conditioning.pt", map_location="cpu", weights_only=True)
            clean = torch.load(teacher_dir / "baseline_latents.pt", map_location="cpu", weights_only=True)
            if clean.ndim != 5 or clean.shape[0] != 1 or clean.shape[2] < 5:
                raise ValueError(f"{teacher_dir}: teacher must be [1,24,T,H,W] with at least five latent frames")
            for chunk in args.target_chunks:
                if chunk * 5 >= clean.shape[2]:
                    raise ValueError(f"{teacher_dir}: target chunk {chunk} is outside {clean.shape[2]} latent frames")
            clean = clean.to(args.device)
            # When --history-mix-schedule is active, we keep both clean and
            # generated history separately, then interpolate per training step.
            history_latents_clean = clean
            history_latents_generated = None
            if args.history_latent_dirs is not None:
                history_dir = args.history_latent_dirs[case_index]
                history_path = history_dir / 'cached_latents.pt'
                if not history_path.exists():
                    history_path = history_dir / 'generated_latents.pt'
                if not history_path.exists():
                    raise ValueError(f'{history_dir}: expected cached_latents.pt or generated_latents.pt')
                history_latents_generated = torch.load(history_path, map_location=args.device,
                                                weights_only=True).to(device=args.device, dtype=clean.dtype)
                if (history_latents_generated.shape != clean.shape
                        or not torch.isfinite(history_latents_generated).all()):
                    raise ValueError(f'{history_path}: generated history shape/finite check failed')
            rows_per_frame = int(clean.shape[-2] // 2) * int(clean.shape[-1] // 2)
            packed = move_tree(cond["packed"], args.device)
            original_anchor = cond["anchor"].to(args.device)
            if args.anchor_mode == "dynamic_last_frame_dual":
                packed = expand_packed_two_anchors(packed, frame_rows=rows_per_frame)
                anchor = torch.cat((original_anchor, original_anchor.clone()), dim=0)
            elif args.anchor_mode == "dynamic_last_frame_rgb_dual":
                packed = expand_packed_two_anchors(packed, frame_rows=rows_per_frame)
                anchor = torch.cat((original_anchor, original_anchor.clone()), dim=0)
            else:
                anchor = original_anchor
            common = dict(full_packed=packed,
                          prompt=cond["prompt_embeds"].to(args.device),
                          anchor=anchor,
                          audio=cond["audio_noise"].to(args.device),
                          chunk_frames=5, window_chunks=args.history_chunks+1)
            teacher_name = teacher_dir.name.lower()
            if args.action_specs is not None:
                action_spec = str(args.action_specs[case_index])
            else:
                action_spec = next((x.upper() for x in ('w', 's', 'a', 'd')
                                    if f'action_{x}_' in teacher_name), None)
                if action_spec is None:
                    raise ValueError(
                        f'{teacher_dir}: cannot infer action from directory name; '
                        'pass --action-specs explicitly')
            # Fixed actions retain the short pair-loss labels. Schedule cases
            # get a stable distinct label and therefore do not accidentally
            # enter the W/S/A/D pair lookup.
            action_label = (action_spec.strip().lower()
                            if ':' not in action_spec
                            else 'schedule_' + action_spec.strip().lower().replace(',', '_').replace(':', 'x'))
            cases.append(dict(teacher_dir=str(teacher_dir), action_label=action_label,
                              action_spec=action_spec, clean=clean,
                              history_latents_clean=history_latents_clean,
                              history_latents_generated=history_latents_generated,
                              packed=packed,
                              common=common))
        result["history_protocol_detail"] = (
            "scheduled_sampling_clean_to_generated_history"
            if args.history_mix_schedule is not None else
            "mixed_clean_and_generated_history_per_action_clip"
            if args.history_latent_dirs is not None else
            "clean_teacher_history_for_each_action_clip")
        result["history_mix"] = args.history_mix
        result["action_teacher_dirs"] = [case["teacher_dir"] for case in cases]
        result["action_labels"] = [case["action_label"] for case in cases]
        result["action_specs"] = [case["action_spec"] for case in cases]
        result["action_clip_count"] = len(cases)
        result["dual_anchor_protocol"] = {
            "slots": ["original_h3_first_frame", "previous_chunk_last_latent_frame"],
            "chunk0_second_slot": "duplicate_original_anchor",
            "frame_rows": rows_per_frame,
        }
        with torch.no_grad():
            pipe = abot.load_pipeline(args.device)
            pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(
                ROOT / "checkpoints/H3-World/step-10000.safetensors"), hotload=True)
            pipe.dit.requires_grad_(False).eval()
            pipe.load_models_to_device(["dit"])
            # Use the same shifted H3 flow schedule as the few-step rollout
            # instead of three hand-picked sigma values.  We still keep a
            # tiny held-out set because this script is a feasibility prototype.
            pipe.scheduler.set_timesteps(args.scheduler_steps, shift=args.scheduler_shift)
            schedule_sigmas = [float(x) for x in pipe.scheduler.sigmas]
        if len(schedule_sigmas) < 4:
            raise ValueError("scheduler grid must provide at least four sigma values")

        # RGB image anchors must be constructed from the same available clean
        # prefix used by the training sample.  Keep the VAE resident while
        # making one anchor per chunk, then restore the DiT before extracting
        # cached features. This makes RGB-prefix training and rollout use the
        # same temporal-VAE context rule.
        if args.anchor_mode == "dynamic_last_frame_rgb_dual":
            with torch.no_grad():
                pipe.load_models_to_device(["video_vae"])
                # Construct anchors independently for each action clip.  The
                # old loop reused the last case's history_latents, silently
                # giving every action the final clip's RGB anchor.
                for case_index, case in enumerate(cases):
                    anchors = {}
                    history_latents = case["history_latents"]
                    for chunk in range(1, max(args.target_chunks) + 1):
                        prefix = history_latents[:, :, :chunk * 5]
                        anchors[chunk] = last_frame_image_anchor(
                            pipe.video_vae, prefix, dtype=pipe.torch_dtype)
                        print(f"[anchor] case={case_index} RGB prefix chunk={chunk} "
                              f"latent_frames={prefix.shape[2]}", flush=True)
                    case["rgb_tail_anchors"] = anchors
                pipe.load_models_to_device(["dit"])

        tail_start = len(pipe.dit.blocks) - args.tail_blocks
        tail_blocks = list(pipe.dit.blocks[tail_start:])
        final = pipe.dit.final_layer
        action_adapter = (install_action_residual(
            pipe.dit, range(tail_start, len(pipe.dit.blocks)),
            num_buttons=len(abot.S.KEYS9), device=args.device,
            mode=args.action_residual_mode)
            if args.action_residual else None)
        if action_adapter is not None:
            result['action_residual_blocks'] = list(action_adapter.block_indices)
            result['action_residual_parameters'] = sum(
                p.numel() for p in action_adapter.parameters())
        capture = {}

        def block_hook(module, positional, keyword):
            capture["x"] = move_tree(positional[0], "cpu", clone=True)
            capture["block_kwargs"] = move_tree(keyword, "cpu", clone=True)
            control = keyword.get('causal_control')
            if control is not None:
                capture['block_kwargs']['causal_control'] = replace(
                    control, cache=control.cache.snapshot(range(tail_start, len(pipe.dit.blocks))),
                    commit=False, allow_grad_read=True, _masks={})

        def final_hook(module, positional, keyword):
            capture["final_kwargs"] = move_tree(keyword, "cpu", clone=True)

        hooks = [tail_blocks[0].register_forward_pre_hook(block_hook, with_kwargs=True),
                 final.register_forward_pre_hook(final_hook, with_kwargs=True)]
        features = []
        rows_per_frame = (clean.shape[-2] // 2) * (clean.shape[-1] // 2)
        # Cover the actual inference schedule, while reserving two distinct
        # noise seeds for validation.  The scheduler values are descending
        # sigmas in the same normalized flow-matching convention used by H3.
        train_sigmas = [schedule_sigmas[i] for i in
                        sorted(set([0, len(schedule_sigmas)//3,
                                    (2*len(schedule_sigmas))//3,
                                    len(schedule_sigmas)-1]))]
        if args.sigma_grid:
            if len(args.sigma_grid) < 4 or any(not 0 < s <= 1 for s in args.sigma_grid):
                raise ValueError('at least four sigmas in (0,1] are required')
            train_sigmas = args.sigma_grid
        result['train_sigmas'] = train_sigmas
        result['history_protocol'] = ('sequential_clean_commit_original_anchor_frozen_base_history'
                                      if args.history_protocol == 'cached' else
                                      'full_ancestors_with_windowed_attention')
        if args.history_protocol == 'cached' and args.history_noise_std:
            raise ValueError('cached history noise requires consistent per-sequence augmentation, not per-target noise')
        specs = [(seed, sigma, "train") for seed in args.train_noise_seeds for sigma in train_sigmas]
        specs += [(seed, train_sigmas[len(train_sigmas)//2 + i % 2], "validation")
                  for i, seed in enumerate(args.validation_noise_seeds)]
        with torch.no_grad():
            for case_index, case in enumerate(cases):
                clean = case["clean"]
                history_latents_clean = case["history_latents_clean"]
                history_latents_generated = case["history_latents_generated"]
                packed = case["packed"]
                common = case["common"]
                action_cond_full = action_condition(
                    case["action_spec"], clean.shape[2], args.device, clean.dtype)
                cache = H3ChunkCache(args.history_chunks, 'cpu')
                # For curriculum learning, extract features with multiple history mix ratios.
                # During training, we'll sample from the appropriate mix based on step progress.
                if args.history_mix_schedule is not None and history_latents_generated is not None:
                    # Extract at 3 mix points: start, middle, end of the schedule
                    history_mix_points = [history_mix_start,
                                          (history_mix_start + history_mix_end) / 2,
                                          history_mix_end]
                    history_latents_variants = [
                        ((1.0 - mix) * history_latents_clean + mix * history_latents_generated)
                        for mix in history_mix_points
                    ]
                elif history_latents_generated is not None:
                    # Fixed mix, single variant
                    history_latents_variants = [
                        (1.0 - args.history_mix) * history_latents_clean + args.history_mix * history_latents_generated
                    ]
                else:
                    # Clean only
                    history_latents_variants = [history_latents_clean]

                for variant_index, history_latents in enumerate(history_latents_variants):
                    cache.clear()
                    for chunk in (range(max(args.target_chunks)+1) if args.history_protocol == 'cached'
                              else args.target_chunks):
                        target_start = chunk * 5
                        target_stop = min(target_start + 5, clean.shape[2])
                        history_start = 0  # Historical hidden states retain their own causal ancestors.
                        local_target_start = target_start - history_start
                        output_start = int(packed["action_video_start"]) + local_target_start * rows_per_frame
                        if args.history_protocol == 'cached':
                            output_start = int(packed['action_video_start'])
                        output_stop = output_start + (target_stop - target_start) * rows_per_frame
                        # Even skipped loss chunks must be committed causally, with
                        # their own anchor. They cannot be rebuilt with a later image.
                        for seed, sigma, split in (specs if chunk in args.target_chunks else [(101, 1., 'skip')]):
                            g = torch.Generator(device="cpu").manual_seed(seed)
                            noise = torch.randn(clean.shape, generator=g, dtype=torch.float32).to(clean)
                            current = ((1 - sigma) * clean + sigma * noise)[:, :, target_start:target_stop]
                            history = history_latents[:, :, history_start:target_start]
                            if args.history_noise_std > 0 and target_start > 0:
                                hg = torch.Generator(device="cpu").manual_seed(seed + 1000003 * (chunk + 1))
                                history_noise = torch.randn(history.shape, generator=hg,
                                                             dtype=torch.float32).to(clean)
                                history = history + args.history_noise_std * history_noise
                            anchor_slot = 0
                            if args.anchor_mode == "dynamic_last_frame" and chunk > 0:
                                anchor = last_frame_anchor(history[:, :, -1:])
                                anchor_frame_index = target_start - 1
                            elif args.anchor_mode == "dynamic_last_frame_dual" and chunk > 0:
                                anchor = torch.cat((common["anchor"][:rows_per_frame],
                                                    last_frame_anchor(history[:, :, -1:])), dim=0)
                                anchor_frame_index = target_start - 1
                                anchor_slot = 1
                            elif args.anchor_mode == "dynamic_last_frame_rgb_dual" and chunk > 0:
                                anchor = torch.cat((common["anchor"][:rows_per_frame],
                                                    case["rgb_tail_anchors"][chunk]), dim=0)
                                anchor_frame_index = target_start - 1
                                anchor_slot = 1
                            else:
                                anchor = common["anchor"]
                                anchor_frame_index = None
                            condition = dict(anchor=anchor, anchor_frame_index=anchor_frame_index,
                                             anchor_slot=anchor_slot,
                                             **{k: v for k, v in common.items() if k not in ['anchor', 'window_chunks']})
                            current_action_cond = action_cond_full[target_start:target_stop]
                            if args.history_protocol == 'cached':
                                prediction = chunk_forward(pipe.dit, current, sigma=sigma, index=chunk,
                                                           cache=cache,
                                                           action_feedback=args.action_feedback,
                                                           action_cond=current_action_cond,
                                                           action_adapter=action_adapter,
                                                           **condition)
                            else:
                                prediction = recompute_forward(
                                    pipe.dit, current, history=history, sigma=sigma,
                                    history_start=history_start, window_chunks=common['window_chunks'],
                                    action_cond=current_action_cond,
                                    action_adapter=action_adapter, **condition)
                            sample = dict(capture)
                            target_rows = patchify_video(clean - noise)[target_start * rows_per_frame:target_stop * rows_per_frame]
                            sample.update(action_case=case["teacher_dir"], action_label=case["action_label"],
                                         case_index=case_index,
                                         chunk=chunk, seed=seed, sigma=sigma, split=split,
                                         output_start=output_start, output_stop=output_stop,
                                         target=target_rows.cpu(), reference=patchify_video(prediction).cpu(),
                                         action_cond=current_action_cond.detach().cpu(),
                                         boundary_rows=rows_per_frame,
                                         boundary_weight=args.boundary_weight,
                                         target_start=target_start, target_stop=target_stop,
                                         history_mix_variant=variant_index if len(history_latents_variants) > 1 else None)
                            if split != 'skip':
                                features.append(sample)
                            print(f"[extract] case={case_index} chunk={chunk} {split} noise={seed} sigma={sigma}", flush=True)
                        if args.history_protocol == 'cached':
                            # Commit the same detached generated-history chunk that
                            # will be visible to the next target. Teacher latents are
                            # still used only as the supervised current target.
                            chunk_forward(pipe.dit, history_latents[:, :, target_start:target_stop], sigma=0.,
                                          index=chunk, cache=cache, commit=True,
                                          action_feedback=args.action_feedback,
                                          action_cond=current_action_cond,
                                          action_adapter=action_adapter, **condition)
                    cache.clear()
        for hook in hooks:
            hook.remove()
        cache.clear()
        result["feature_extraction_seconds"] = time.perf_counter() - started
        pipe.load_models_to_device([])
        for tail in (*tail_blocks, final):
            for module in list(tail.modules()):
                if isinstance(module, AutoTorchModule):
                    module.preparing()
        block_indices = list(range(tail_start, len(pipe.dit.blocks)))
        if args.no_qkv_adapter:
            adapters = []
        else:
            adapters, block_indices = install_adapters(
                pipe.dit, args.rank, block_indices, args.device)
        params = [p for adapter in adapters for p in (adapter.lora_A, adapter.lora_B)]
        if action_adapter is not None:
            params += list(action_adapter.parameters())
        optimizer = torch.optim.AdamW(params, lr=args.lr)
        initial = [p.detach().clone() for p in params]

        def get_history_mix_for_step(step):
            """Curriculum learning: linearly interpolate history_mix over training."""
            if args.history_mix_schedule is None or args.history_latent_dirs is None:
                return None  # No dynamic sampling
            progress = step / max(args.steps - 1, 1)
            return history_mix_start + progress * (history_mix_end - history_mix_start)

        def predict(sample, history_mix_override=None):
            """Predict with optional dynamic history sampling.

            If history_mix_override is provided and the sample's case has generated
            history, we recompute the history prefix on-the-fly by interpolating
            between clean and generated history at the requested mix ratio.
            """
            hidden = sample["x"].to(args.device)
            block_kwargs = move_tree(sample["block_kwargs"], args.device)
            hidden = replay_tail(tail_blocks, block_indices, hidden, block_kwargs)
            video, _ = final(hidden, **move_tree(sample["final_kwargs"], args.device))
            control = block_kwargs.get('causal_control')
            if control is not None:
                # Graph keeps the necessary masks until backward completes.
                # Do not retain GPU masks for every frozen training sample.
                control._masks.clear()
            return video[sample["output_start"]:sample["output_stop"]]

        def loss(sample):
            prediction = predict(sample).float()
            target = sample["target"].to(args.device).float()
            weight = float(sample.get("boundary_weight", 1.0))
            if weight == 1.0:
                base = torch.nn.functional.mse_loss(prediction, target)
            else:
                rows = int(sample["boundary_rows"])
                weights = torch.ones(prediction.shape[0], device=prediction.device,
                                     dtype=prediction.dtype)
                weights[:rows] = weight
                error = (prediction - target).square().mean(dim=-1)
                base = (error * weights).sum() / weights.sum()
            total = base
            if args.base_output_reg_weight > 0:
                # The captured zero-adapter output is stored with the sign used
                # by H3's final flow head.  replay_exact above verifies this
                # relation before training begins.  Regularizing in output
                # space is preferable to an adapter-weight penalty: it keeps
                # the action-conditioned base geometry while still allowing a
                # small correction where the causal target requires one.
                base_prediction = -sample["reference"].to(args.device).float()
                total = total + args.base_output_reg_weight * torch.nn.functional.mse_loss(
                    prediction, base_prediction)
            if args.action_pair_weight <= 0:
                return total
            partner_label = {'w': 's', 's': 'w', 'a': 'd', 'd': 'a'}.get(sample['action_label'])
            partner = action_pair_lookup.get((partner_label, int(sample['chunk']),
                                              int(sample['seed']), float(sample['sigma'])))
            if partner is None:
                return total
            partner_prediction = predict(partner).float()
            partner_target = partner['target'].to(args.device).float()
            pair_loss = torch.nn.functional.mse_loss(
                prediction - partner_prediction, target - partner_target)
            total = total + args.action_pair_weight * pair_loss
            if args.action_base_pair_weight > 0:
                base_prediction = -sample['reference'].to(args.device).float()
                base_partner = -partner['reference'].to(args.device).float()
                base_pair_loss = torch.nn.functional.mse_loss(
                    (prediction - partner_prediction) - (base_prediction - base_partner),
                    torch.zeros_like(base_prediction))
                total = total + args.action_base_pair_weight * base_pair_loss
            return total

        train = [s for s in features if s["split"] == "train"]
        validation = [s for s in features if s["split"] == "validation"]
        action_pair_lookup = {
            (s['action_label'], int(s['chunk']), int(s['seed']), float(s['sigma'])): s
            for s in features
        }
        pair_labels = {'w': 's', 's': 'w', 'a': 'd', 'd': 'a'}
        result['action_pair_samples'] = sum(
            1 for s in features
            if (pair_labels.get(s['action_label']), int(s['chunk']),
                int(s['seed']), float(s['sigma'])) in action_pair_lookup)

        @torch.no_grad()
        def evaluate(samples):
            return sum(float(loss(s)) for s in samples) / len(samples)

        with torch.no_grad():
            replay_errors = []
            for sample in features:
                replay = predict(sample)
                reference = -sample["reference"].to(replay)
                replay_errors.append(float((replay.float() - reference.float()).abs().max()))
                result["replay_max_error"] = max(replay_errors)
                # Zero-initialized adapters must reproduce the captured full
                # forward exactly. A failure means the training path differs;
                # do not train or publish an adapter from that path.
                torch.testing.assert_close(replay, reference, atol=0, rtol=0)
        result["replay_exact"] = max(replay_errors, default=0.0) == 0.0
        result.update(block_indices=block_indices, block_index=block_indices[-1],
                      replay_max_error=max(replay_errors),
                      trainable_parameters=sum(p.numel() for p in params),
                      train_before=evaluate(train), validation_before=evaluate(validation),
                      history=[], train_samples=len(train), validation_samples=len(validation),
                      extracted_chunks=sorted(set(s["chunk"] for s in features)))
        save_result()
        torch.cuda.reset_peak_memory_stats(args.device)
        train_started = time.perf_counter()

        # Group training samples by (case, chunk, seed, sigma) for curriculum sampling
        if args.history_mix_schedule is not None:
            train_groups = {}
            for s in train:
                if s.get('history_mix_variant') is not None:
                    key = (s['case_index'], s['chunk'], s['seed'], float(s['sigma']))
                    if key not in train_groups:
                        train_groups[key] = []
                    train_groups[key].append(s)
            # Verify each group has 3 variants (start, middle, end)
            for key, group in train_groups.items():
                if len(group) != 3:
                    raise RuntimeError(f"Expected 3 history mix variants for {key}, got {len(group)}")

        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)

            # Select training sample with curriculum-based history mix
            if args.history_mix_schedule is not None and train_groups:
                # Pick a group, then select variant based on curriculum progress
                base_sample = train[step % len(train)]
                key = (base_sample['case_index'], base_sample['chunk'],
                       base_sample['seed'], float(base_sample['sigma']))
                if key in train_groups:
                    current_mix = get_history_mix_for_step(step)
                    # Find closest variant: 0=start, 1=middle, 2=end
                    if current_mix <= (history_mix_start + history_mix_end) / 2:
                        variant_idx = 0 if current_mix <= history_mix_start + (history_mix_end - history_mix_start) / 4 else 1
                    else:
                        variant_idx = 2
                    sample = train_groups[key][variant_idx]
                else:
                    sample = base_sample
            else:
                sample = train[step % len(train)]

            value = loss(sample)
            value.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(value) or not torch.isfinite(norm):
                raise FloatingPointError("non-finite flow loss or gradient")
            optimizer.step()
            result["history"].append(dict(step=step + 1, loss=float(value.detach()), grad_norm=float(norm)))
            if (step + 1) % 10 == 0:
                print(f"[train] {step + 1}/{args.steps} loss={float(value.detach()):.6f}", flush=True)
                save_result()
        torch.cuda.synchronize(args.device)
        result.update(train_seconds=time.perf_counter() - train_started,
                      train_after=evaluate(train), validation_after=evaluate(validation),
                      allocated_peak_MiB=torch.cuda.max_memory_allocated(args.device) / 2**20,
                      parameters_changed=any(not torch.equal(p.detach(), old)
                                             for p, old in zip(params, initial)))
        if not result["parameters_changed"]:
            raise RuntimeError("adapter parameters did not change")
        metadata = {k: v for k, v in result.items() if k not in ["history", "status"]}
        if adapters:
            save_adapters(args.out_dir / "adapter.pt", adapters, block_indices, metadata)
        if action_adapter is not None:
            save_action_residual(args.out_dir / "action_adapter.pt", action_adapter, metadata)
        result.update(status="complete", wall_seconds=time.perf_counter() - started)
        save_result()
        print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
        save_result()
        raise


if __name__ == "__main__":
    main()
