#!/usr/bin/env python3
"""Probe the causal H3 action-row routing without training.

For one fixed generated latent state, compare A/D velocity responses under the
cached controller with action feedback disabled/enabled and with the three
prefix visibility policies.  The A history cache is shared by the A and D
counterfactual calls within each variant.  This is a topology diagnostic, not
a quality metric or a new training method.
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
from causal.h3_cached import H3ChunkCache, chunk_forward, expand_packed_two_anchors
from causal.pretrained_lora import load_adapter
from causal.train_online_selfrollout import action_condition, move_tree


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generated-dir", type=Path, required=True)
    ap.add_argument("--conditioning", type=Path, nargs=2, required=True,
                    help="A and D conditioning directories")
    ap.add_argument("--causal-adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk-frames", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.6)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    generated = torch.load(args.generated_dir / "cached_latents.pt", map_location="cpu", weights_only=True)
    if generated.ndim != 5 or generated.shape[0] != 1:
        raise ValueError(f"expected [1,C,T,H,W], got {tuple(generated.shape)}")
    conds = [move_tree(torch.load(p / "conditioning.pt", map_location="cpu", weights_only=True), args.device)
             for p in args.conditioning]
    pipe = abot.load_pipeline(args.device)
    pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(
        ROOT / "checkpoints/H3-World/step-10000.safetensors"), hotload=True)
    pipe.dit.requires_grad_(False).eval()
    load_adapter(pipe.dit, args.causal_adapter, args.device)
    pipe.load_models_to_device(["dit"])

    generated = generated.to(args.device)
    rows = (generated.shape[-2] // 2) * (generated.shape[-1] // 2)
    packed = [expand_packed_two_anchors(c["packed"], frame_rows=rows) for c in conds]
    actions = [action_condition(a, generated.shape[2], args.device, generated.dtype)
               for a in ("A", "D")]
    # The original image anchor is duplicated for this routing-only probe. Both
    # actions receive identical anchors, noise, and generated latent state.
    anchor = torch.cat((conds[0]["anchor"], conds[0]["anchor"].clone()), dim=0)
    variants = [("own_fb0", "own", False), ("causal_fb0", "causal", False),
                ("causal_fb1", "causal", True), ("all_fb1", "all", True)]
    output = {"status": "running", "generated_dir": str(args.generated_dir),
              "sigma": args.sigma, "chunk_frames": args.chunk_frames,
              "state_source": "generated A rollout", "variants": []}
    args.out.write_text(json.dumps(output, indent=2) + "\n")

    with torch.no_grad():
        for name, prefix_mode, feedback in variants:
            cache = H3ChunkCache(5, "cpu")
            # Build the same generated A history for every variant.  Commit is
            # a clean sigma=0 pass, exactly as the streaming benchmark does.
            history_current = generated[:, :, :args.chunk_frames].contiguous()
            common_a = dict(full_packed=packed[0], prompt=conds[0]["prompt_embeds"],
                            anchor=anchor, audio=conds[0]["audio_noise"],
                            chunk_frames=args.chunk_frames, action_cond=actions[0][:args.chunk_frames],
                            action_prefix_mode=prefix_mode, action_feedback=feedback)
            chunk_forward(pipe.dit, history_current, index=0, cache=cache, sigma=0.0,
                          commit=True, **common_a)
            start = args.chunk_frames
            current = generated[:, :, start:start + args.chunk_frames].contiguous()
            velocities = []
            for packed_i, cond, action in zip(packed, conds, actions):
                common = dict(full_packed=packed_i, prompt=cond["prompt_embeds"],
                              anchor=anchor, audio=cond["audio_noise"],
                              chunk_frames=args.chunk_frames,
                              action_cond=action[start:start + args.chunk_frames],
                              action_prefix_mode=prefix_mode, action_feedback=feedback)
                velocities.append(chunk_forward(pipe.dit, current, index=1, cache=cache,
                                                sigma=args.sigma, commit=False, **common).float())
            delta = (velocities[0] - velocities[1]).flatten(1)
            norm_a = float(torch.linalg.vector_norm(velocities[0].flatten(1), dim=1).mean())
            norm_d = float(torch.linalg.vector_norm(velocities[1].flatten(1), dim=1).mean())
            delta_norm = float(torch.linalg.vector_norm(delta, dim=1).mean())
            # Compare each route to the explicit no-feedback reference later.
            row = {"variant": name, "action_prefix_mode": prefix_mode,
                   "action_feedback": feedback, "chunk": 1,
                   "A_velocity_norm": norm_a, "D_velocity_norm": norm_d,
                   "AD_delta_norm": delta_norm}
            output["variants"].append(row)
            print(json.dumps(row), flush=True)
            args.out.write_text(json.dumps(output, indent=2) + "\n")
    ref = next(x for x in output["variants"] if x["variant"] == "causal_fb0")
    for row in output["variants"]:
        row["delta_norm_ratio_to_causal_fb0"] = row["AD_delta_norm"] / (ref["AD_delta_norm"] + 1e-8)
    output.update(status="complete", interpretation=(
        "A/D delta is measured on the same generated A history and current noisy latent. "
        "A change when feedback is enabled indicates that the explicit action-row-to-current-video "
        "edge is active; it does not prove correct image-space strafe direction."))
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
