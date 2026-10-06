#!/usr/bin/env python3
"""Minimal multi-chunk Stage1-style clean-history training on pretrained H3.

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
from causal.pretrained_lora import install_adapters, replay_tail, save_adapters
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--history-latents", type=Path, default=None,
                    help="detached generated-history latent tensor; keep teacher targets")
    ap.add_argument("--history-mix", type=float, default=1.0,
                    help=("when --history-latents is supplied, mix it with clean teacher "
                          "history: 0=clean, 1=generated; this is a scheduled-history "
                          "diagnostic, not Stage2 distillation"))
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
    ap.add_argument('--boundary-weight', type=float, default=1.0,
                    help='relative loss weight for the first latent frame of every chunk')
    args = ap.parse_args()
    if (args.steps < 1 or args.rank < 1 or args.tail_blocks < 1 or args.history_chunks < 1
            or args.tail_blocks > 50 or args.lr <= 0 or args.history_noise_std < 0
            or args.boundary_weight <= 0 or not 0.0 <= args.history_mix <= 1.0
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
        data="generated H3 teacher clip; selected chunks, clean-history teacher forcing, held-out noise",
        teacher_dir=str(args.teacher_dir), steps=args.steps, rank=args.rank, lr=args.lr,
        history_latents=str(args.history_latents) if args.history_latents else None,
        boundary_weight=args.boundary_weight,
        target_chunks=args.target_chunks, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        anchor_mode=args.anchor_mode, scheduler_steps=args.scheduler_steps,
        scheduler_shift=args.scheduler_shift,
        train_noise_seeds=args.train_noise_seeds,
        validation_noise_seeds=args.validation_noise_seeds,
        history_noise_std=args.history_noise_std,
        tail_blocks=args.tail_blocks,
        history_chunks=args.history_chunks,
        requested_history_protocol=args.history_protocol,
    )

    def save_result():
        p = args.out_dir / "training.json"
        tmp = p.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(p)

    save_result()
    try:
        cond = torch.load(args.teacher_dir / "conditioning.pt", map_location="cpu", weights_only=True)
        clean = torch.load(args.teacher_dir / "baseline_latents.pt", map_location="cpu", weights_only=True)
        if clean.ndim != 5 or clean.shape[0] != 1 or clean.shape[2] < 5:
            raise ValueError("teacher must be [1,24,T,H,W] with at least five latent frames")
        for chunk in args.target_chunks:
            if chunk * 5 >= clean.shape[2]:
                raise ValueError(f"target chunk {chunk} is outside {clean.shape[2]} latent frames")
        clean = clean.to(args.device)
        history_latents = clean
        if args.history_latents is not None:
            generated_history = torch.load(
                args.history_latents, map_location=args.device, weights_only=True
            ).to(device=args.device, dtype=clean.dtype)
            if generated_history.shape != clean.shape or not torch.isfinite(generated_history).all():
                raise ValueError("--history-latents must be finite and match teacher latent shape")
            history_latents = ((1.0 - args.history_mix) * clean
                               + args.history_mix * generated_history)
            result["history_protocol_detail"] = (
                "mixed_clean_and_generated_history_with_teacher_current_target")
            result["history_mix"] = args.history_mix
        else:
            result["history_protocol_detail"] = "clean_teacher_history"
            if args.history_mix != 1.0:
                raise ValueError("--history-mix requires --history-latents")
        packed = move_tree(cond["packed"], args.device)
        original_anchor = cond["anchor"].to(args.device)
        rows_per_frame = (clean.shape[-2] // 2) * (clean.shape[-1] // 2)
        if args.anchor_mode == "dynamic_last_frame_dual":
            packed = expand_packed_two_anchors(packed, frame_rows=rows_per_frame)
            # Chunk 0 has no generated predecessor, so the second slot is a
            # duplicate of the H3 first-frame anchor.  Later chunks replace
            # only this slot with the clean teacher-history tail.
            anchor = torch.cat((original_anchor, original_anchor.clone()), dim=0)
            result["dual_anchor_protocol"] = {
                "slots": ["original_h3_first_frame", "previous_chunk_last_latent_frame"],
                "chunk0_second_slot": "duplicate_original_anchor",
                "frame_rows": rows_per_frame,
            }
        elif args.anchor_mode == "dynamic_last_frame_rgb_dual":
            packed = expand_packed_two_anchors(packed, frame_rows=rows_per_frame)
            anchor = torch.cat((original_anchor, original_anchor.clone()), dim=0)
            result["dual_anchor_protocol"] = {
                "slots": ["original_h3_first_frame", "previous_prefix_last_rgb_image"],
                "chunk0_second_slot": "duplicate_original_anchor",
                "frame_rows": rows_per_frame,
                "slot1_encoding": "prefix_decode_last_rgb_encode_process_image",
            }
        else:
            anchor = original_anchor
        common = dict(full_packed=packed,
                      prompt=cond["prompt_embeds"].to(args.device),
                      anchor=anchor,
                      audio=cond["audio_noise"].to(args.device),
                      chunk_frames=5, window_chunks=args.history_chunks+1)
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
        rgb_tail_anchors = {}
        if args.anchor_mode == "dynamic_last_frame_rgb_dual":
            with torch.no_grad():
                pipe.load_models_to_device(["video_vae"])
                for chunk in range(1, max(args.target_chunks) + 1):
                    prefix = history_latents[:, :, :chunk * 5]
                    rgb_tail_anchors[chunk] = last_frame_image_anchor(
                        pipe.video_vae, prefix, dtype=pipe.torch_dtype)
                    print(f"[anchor] RGB prefix chunk={chunk} latent_frames={prefix.shape[2]}", flush=True)
                pipe.load_models_to_device(["dit"])

        tail_start = len(pipe.dit.blocks) - args.tail_blocks
        tail_blocks = list(pipe.dit.blocks[tail_start:])
        final = pipe.dit.final_layer
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
        cache = H3ChunkCache(args.history_chunks, 'cpu')
        specs = [(seed, sigma, "train") for seed in args.train_noise_seeds for sigma in train_sigmas]
        specs += [(seed, train_sigmas[len(train_sigmas)//2 + i % 2], "validation")
                  for i, seed in enumerate(args.validation_noise_seeds)]
        with torch.no_grad():
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
                                            rgb_tail_anchors[chunk]), dim=0)
                        anchor_frame_index = target_start - 1
                        anchor_slot = 1
                    else:
                        anchor = common["anchor"]
                        anchor_frame_index = None
                    condition = dict(anchor=anchor, anchor_frame_index=anchor_frame_index,
                                     anchor_slot=anchor_slot,
                                     **{k: v for k, v in common.items() if k not in ['anchor', 'window_chunks']})
                    if args.history_protocol == 'cached':
                        prediction = chunk_forward(pipe.dit, current, sigma=sigma, index=chunk,
                                                   cache=cache, **condition)
                    else:
                        prediction = recompute_forward(
                            pipe.dit, current, history=history, sigma=sigma,
                            history_start=history_start, window_chunks=common['window_chunks'], **condition)
                    sample = dict(capture)
                    target_rows = patchify_video(clean - noise)[target_start * rows_per_frame:target_stop * rows_per_frame]
                    sample.update(chunk=chunk, seed=seed, sigma=sigma, split=split,
                                 output_start=output_start, output_stop=output_stop,
                                 target=target_rows.cpu(), reference=patchify_video(prediction).cpu(),
                                 boundary_rows=rows_per_frame,
                                 boundary_weight=args.boundary_weight)
                    if split != 'skip':
                        features.append(sample)
                    print(f"[extract] chunk={chunk} {split} noise={seed} sigma={sigma}", flush=True)
                if args.history_protocol == 'cached':
                    # Commit the same detached generated-history chunk that
                    # will be visible to the next target. Teacher latents are
                    # still used only as the supervised current target.
                    chunk_forward(pipe.dit, history_latents[:, :, target_start:target_stop], sigma=0.,
                                  index=chunk, cache=cache, commit=True, **condition)
        for hook in hooks:
            hook.remove()
        cache.clear()
        result["feature_extraction_seconds"] = time.perf_counter() - started
        pipe.load_models_to_device([])
        for tail in (*tail_blocks, final):
            for module in list(tail.modules()):
                if isinstance(module, AutoTorchModule):
                    module.preparing()
        adapters, block_indices = install_adapters(
            pipe.dit, args.rank, range(tail_start, len(pipe.dit.blocks)), args.device)
        params = [p for adapter in adapters for p in (adapter.lora_A, adapter.lora_B)]
        optimizer = torch.optim.AdamW(params, lr=args.lr)
        initial = [p.detach().clone() for p in params]

        def predict(sample):
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
                return torch.nn.functional.mse_loss(prediction, target)
            rows = int(sample["boundary_rows"])
            weights = torch.ones(prediction.shape[0], device=prediction.device,
                                 dtype=prediction.dtype)
            weights[:rows] = weight
            error = (prediction - target).square().mean(dim=-1)
            return (error * weights).sum() / weights.sum()

        train = [s for s in features if s["split"] == "train"]
        validation = [s for s in features if s["split"] == "validation"]

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
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            value = loss(train[step % len(train)])
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
        save_adapters(args.out_dir / "adapter.pt", adapters, block_indices, metadata)
        result.update(status="complete", wall_seconds=time.perf_counter() - started)
        save_result()
        print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
        save_result()
        raise


if __name__ == "__main__":
    main()
