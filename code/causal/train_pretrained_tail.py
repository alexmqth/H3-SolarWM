#!/usr/bin/env python3
"""Train last-block QKV LoRA on a generated H3 clip with clean history.

Frozen predecessor activations are exact for this restricted trainable subset.
This is a single-clip, Stage1-style flow-matching feasibility experiment, NOT
AnyFlow, Stage2, or a held-out scene quality evaluation.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/abot"))
sys.path.insert(0, str(ROOT / "code"))
import infer as abot
import torch
from causal.h3_cached import recompute_forward
from causal.pretrained_lora import install_adapter, save_adapter
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
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target-chunk", type=int, default=1,
                    help="latent chunk index to train (22-frame teacher has chunks 0 and 1)")
    args = ap.parse_args()
    if args.steps < 1 or args.rank < 1 or args.lr <= 0 or args.target_chunk < 0:
        ap.error("steps, rank and target-chunk must be non-negative/positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.cuda.set_device(args.device)
    started = time.perf_counter()
    result = dict(status="running", kind="pretrained last-block QKV LoRA",
                  objective="clean-history flow matching, no AnyFlow or DMD",
                  data="one generated 30-step H3 clip; noise held out within same clip",
                  teacher_dir=str(args.teacher_dir), steps=args.steps, rank=args.rank,
                  lr=args.lr, target_chunk=args.target_chunk,
                  visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))

    def save_result():
        p = args.out_dir / "training.json"
        tmp = p.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(p)

    save_result()
    try:
        cond = torch.load(args.teacher_dir / "conditioning.pt", map_location="cpu", weights_only=True)
        clean = torch.load(args.teacher_dir / "baseline_latents.pt", map_location="cpu", weights_only=True)
        if clean.shape[2] != 7:
            raise ValueError("this minimal experiment expects 22 decoded frames / 7 latent frames")
        target_start = args.target_chunk * 5
        target_stop = min(target_start + 5, clean.shape[2])
        if target_start >= clean.shape[2]:
            raise ValueError(f"target chunk {args.target_chunk} is outside {clean.shape[2]} latent frames")
        clean = clean.to(args.device)
        common = dict(full_packed=move_tree(cond["packed"], args.device),
                      prompt=cond["prompt_embeds"].to(args.device),
                      anchor=cond["anchor"].to(args.device),
                      audio=cond["audio_noise"].to(args.device), chunk_frames=5, window_chunks=6)
        with torch.no_grad():
            pipe = abot.load_pipeline(args.device)
            pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(
                ROOT / "checkpoints/H3-World/step-10000.safetensors"), hotload=True)
            pipe.dit.requires_grad_(False).eval()
            pipe.load_models_to_device(["dit"])

        # Capture frozen predecessor output once per noisy example. The last
        # block and final projection are replayed with gradients below.
        block, final = pipe.dit.blocks[-1], pipe.dit.final_layer
        capture = {}

        def block_hook(module, positional, keyword):
            capture["x"] = move_tree(positional[0], "cpu", clone=True)
            # BlockMask uses live CUDA buffers. All examples deliberately use
            # the same layout; retain this mask rather than serializing it.
            capture["block_kwargs"] = move_tree(keyword, "cpu", clone=True)

        def final_hook(module, positional, keyword):
            capture["final_kwargs"] = move_tree(keyword, "cpu", clone=True)

        hooks = [block.register_forward_pre_hook(block_hook, with_kwargs=True),
                 final.register_forward_pre_hook(final_hook, with_kwargs=True)]
        features = []
        rows_per_frame = (clean.shape[-2] // 2) * (clean.shape[-1] // 2)
        output_start = int(cond["packed"]["action_video_start"]) + target_start * rows_per_frame
        output_stop = output_start + (target_stop - target_start) * rows_per_frame
        specs = [(seed, sigma, "train") for seed in [101, 102] for sigma in [.3, .6, .9]]
        specs += [(1001, .6, "validation"), (1002, .45, "validation")]
        with torch.no_grad():
            for seed, sigma, split in specs:
                g = torch.Generator(device="cpu").manual_seed(seed)
                noise = torch.randn(clean.shape, generator=g, dtype=torch.float32).to(clean)
                current = ((1-sigma)*clean + sigma*noise)[:, :, target_start:target_stop]
                prediction = recompute_forward(pipe.dit, current, history=clean[:, :, :target_start],
                                               sigma=sigma, **common)
                sample = dict(capture)
                target_rows = patchify_video(clean-noise)[target_start*rows_per_frame:target_stop*rows_per_frame]
                sample.update(seed=seed, sigma=sigma, split=split,
                              target=target_rows.cpu(),
                              reference=patchify_video(prediction).cpu())
                features.append(sample)
                print(f"[extract] {split} noise={seed} sigma={sigma}", flush=True)
        for hook in hooks:
            hook.remove()
        result["feature_extraction_seconds"] = time.perf_counter() - started
        pipe.load_models_to_device([])
        # Keep only the tail resident for gradient computation. H3 action LoRA
        # in AutoWrappedLinear remains active in addition to the new adapter.
        for tail in (block, final):
            for module in list(tail.modules()):
                if isinstance(module, AutoTorchModule):
                    module.preparing()
        adapter, block_index = install_adapter(pipe.dit, args.rank, -1, args.device)
        params = [adapter.lora_A, adapter.lora_B]
        optimizer = torch.optim.AdamW(params, lr=args.lr)
        initial = [p.detach().clone() for p in params]

        def predict(sample):
            hidden = block(sample["x"].to(args.device),
                           **move_tree(sample["block_kwargs"], args.device))
            video, _ = final(hidden, **move_tree(sample["final_kwargs"], args.device))
            return video[output_start:output_stop]

        def loss(sample):
            return torch.nn.functional.mse_loss(predict(sample).float(), sample["target"].to(args.device).float())

        train = [s for s in features if s["split"] == "train"]
        validation = [s for s in features if s["split"] == "validation"]

        @torch.no_grad()
        def evaluate(samples):
            return sum(float(loss(s)) for s in samples) / len(samples)

        with torch.no_grad():
            replay = predict(features[0])
            reference = -features[0]["reference"].to(replay)
            replay_error = float((replay.float()-reference.float()).abs().max())
            torch.testing.assert_close(replay, reference, atol=0, rtol=0)
        result.update(block_index=block_index, replay_max_error=replay_error,
                      trainable_parameters=sum(p.numel() for p in params),
                      train_before=evaluate(train), validation_before=evaluate(validation),
                      history=[], train_samples=len(train), validation_samples=len(validation))
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
            result["history"].append(dict(step=step+1, loss=float(value.detach()), grad_norm=float(norm)))
            if (step+1) % 10 == 0:
                print(f"[train] {step+1}/{args.steps} loss={float(value.detach()):.6f}", flush=True)
                save_result()
        torch.cuda.synchronize(args.device)
        result.update(train_seconds=time.perf_counter()-train_started,
                      train_after=evaluate(train), validation_after=evaluate(validation),
                      allocated_peak_MiB=torch.cuda.max_memory_allocated(args.device)/2**20,
                      parameters_changed=any(not torch.equal(p.detach(), old) for p, old in zip(params, initial)))
        if not result["parameters_changed"]:
            raise RuntimeError("adapter parameters did not change")
        metadata = {k: v for k, v in result.items() if k not in ["history", "status"]}
        save_adapter(args.out_dir / "adapter.pt", adapter, block_index, metadata)
        result.update(status="complete", wall_seconds=time.perf_counter()-started)
        save_result()
        print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
        save_result()
        raise


if __name__ == "__main__":
    main()
