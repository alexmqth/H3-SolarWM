#!/usr/bin/env python3
"""Audit full-attention H3 A/D deltas on one shared generated state.

This is a read-only diagnostic.  It deliberately does not train an adapter or
run a causal student.  A generated A rollout supplies the history/current
latent state; the frozen bidirectional H3 teacher then sees the same state with
counterfactual A and D conditioning.  The result tells us whether the paired
teacher target used by online training contains a usable action direction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/abot"))
sys.path.insert(0, str(ROOT / "code"))
import infer as abot
from causal.train_online_selfrollout import action_condition, full_teacher_forward, move_tree


def parse_sigmas(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values or any(not 0 <= x <= 1 for x in values):
        raise ValueError("sigmas must be comma-separated values in [0,1]")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generated-dir", type=Path, required=True,
                    help="benchmark directory containing cached_latents.pt and conditioning.pt")
    ap.add_argument("--teacher-dir", type=Path, nargs=2, required=True,
                    help="A and D teacher conditioning directories")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk-frames", type=int, default=5)
    ap.add_argument("--sigmas", default="0.9395405,0.6894410,0.4252873,0.2407809,0.0")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    sigmas = parse_sigmas(args.sigmas)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    generated = torch.load(args.generated_dir / "cached_latents.pt", map_location="cpu", weights_only=True)
    if generated.ndim != 5 or generated.shape[2] != 12:
        raise ValueError(f"expected cached 12-frame latent rollout, got {tuple(generated.shape)}")
    conds = []
    for path in args.teacher_dir:
        raw = torch.load(path / "conditioning.pt", map_location="cpu", weights_only=True)
        conds.append(move_tree(raw, args.device))
    if not torch.equal(conds[0]["initial_noise"].cpu(), conds[1]["initial_noise"].cpu()):
        raise ValueError("A/D teacher initial noises differ")
    if not torch.equal(conds[0]["audio_noise"].cpu(), conds[1]["audio_noise"].cpu()):
        raise ValueError("A/D teacher audio noises differ")
    pipe = abot.load_pipeline(args.device)
    pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(
        ROOT / "checkpoints/H3-World/step-10000.safetensors"), hotload=True)
    pipe.dit.requires_grad_(False).eval()
    pipe.load_models_to_device(["dit"])
    out = {
        "status": "running", "generated_dir": str(args.generated_dir),
        "teacher_dirs": [str(x) for x in args.teacher_dir], "chunk_frames": args.chunk_frames,
        "sigmas": sigmas, "seed": args.seed, "state_source": "generated A rollout",
        "teacher": "original H3 bidirectional attention; no causal controller",
        "rows": [],
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    try:
        latent_t = generated.shape[2]
        rows = (generated.shape[-2] // 2) * (generated.shape[-1] // 2)
        # Build the action tensors on the model device. Full teacher receives
        # the complete action rows even when only a prefix/current state is
        # denoised, matching train_online_selfrollout.py.
        action_conds = [action_condition(a, latent_t, args.device, generated.dtype)
                        for a in ("A", "D")]
        generated = generated.to(args.device)
        for chunk, start in enumerate(range(0, latent_t, args.chunk_frames)):
            stop = min(start + args.chunk_frames, latent_t)
            current = generated[:, :, start:stop].contiguous()
            history = generated[:, :, :start].contiguous()
            for sigma in sigmas:
                velocities = []
                with torch.no_grad():
                    for case, action_cond in zip(conds, action_conds):
                        velocity = full_teacher_forward(
                            pipe.dit, current, history=history,
                            full_packed=case["packed"], prompt=case["prompt_embeds"],
                            anchor=case["anchor"], audio=case["audio_noise"],
                            sigma=sigma, action_cond=action_cond)
                        if not torch.isfinite(velocity).all():
                            raise FloatingPointError(f"nonfinite teacher velocity: chunk={chunk}, sigma={sigma}")
                        velocities.append(velocity.float())
                delta = velocities[0] - velocities[1]
                a_flat, d_flat = (x.flatten(1) for x in velocities)
                delta_flat = delta.flatten(1)
                norm_a = float(torch.linalg.vector_norm(a_flat, dim=1).mean())
                norm_d = float(torch.linalg.vector_norm(d_flat, dim=1).mean())
                delta_norm = float(torch.linalg.vector_norm(delta_flat, dim=1).mean())
                relative_delta = delta_norm / (0.5 * (norm_a + norm_d) + 1e-6)
                row = dict(chunk=chunk, start_latent=start, stop_latent=stop,
                           sigma=sigma, teacher_A_norm=norm_a, teacher_D_norm=norm_d,
                           delta_norm=delta_norm,
                           delta_over_mean_velocity_norm=relative_delta)
                out["rows"].append(row)
                print(json.dumps(row), flush=True)
                args.out.write_text(json.dumps(out, indent=2) + "\n")
        out.update({"status": "complete", "finite": True,
                    "interpretation": "A/D delta is measured on the same generated state; this does not measure student alignment."})
        args.out.write_text(json.dumps(out, indent=2) + "\n")
    except Exception as exc:
        out.update(status="failed", error=repr(exc))
        args.out.write_text(json.dumps(out, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
