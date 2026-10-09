#!/usr/bin/env python3
"""Compare causal-student and full-teacher A/D deltas on one shared rollout."""
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
from causal.h3_cached import H3ChunkCache, chunk_forward, expand_packed_two_anchors, last_frame_image_anchor
from causal.pretrained_lora import load_adapter, load_action_residual
from causal.train_online_selfrollout import action_condition, full_teacher_forward, move_tree


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generated-dir", type=Path, required=True)
    ap.add_argument("--teacher-dir", type=Path, nargs=2, required=True)
    ap.add_argument("--causal-adapter", type=Path, required=True)
    ap.add_argument("--student-action-adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk-frames", type=int, default=5)
    ap.add_argument("--sigmas", default="0.9395405,0.6894410,0.4252873,0.2407809,0.0")
    args = ap.parse_args()
    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    generated = torch.load(args.generated_dir / "cached_latents.pt", map_location="cpu", weights_only=True)
    if generated.shape != (1, 24, 12, 30, 52):
        raise ValueError(f"expected [1,24,12,30,52], got {tuple(generated.shape)}")
    conds = []
    for path in args.teacher_dir:
        conds.append(move_tree(torch.load(path / "conditioning.pt", map_location="cpu", weights_only=True), args.device))

    pipe = abot.load_pipeline(args.device)
    pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(
        ROOT / "checkpoints/H3-World/step-10000.safetensors"), hotload=True)
    pipe.dit.requires_grad_(False).eval()
    visual = load_adapter(pipe.dit, args.causal_adapter, args.device)
    visual_modules = [pipe.dit.blocks[int(i)].attn.qkv_proj for i in visual["block_indices"]]
    student = load_action_residual(pipe.dit, args.student_action_adapter, args.device)["adapter"]
    student.requires_grad_(False)
    pipe.load_models_to_device(["dit"])

    out = {"status": "running", "generated_dir": str(args.generated_dir),
           "causal_adapter": str(args.causal_adapter),
           "student_action_adapter": str(args.student_action_adapter),
           "teacher_dirs": [str(x) for x in args.teacher_dir], "sigmas": sigmas,
           "state_source": "generated A rollout", "rows": []}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    generated = generated.to(args.device)
    rows = (generated.shape[-2] // 2) * (generated.shape[-1] // 2)
    dual_packed = [expand_packed_two_anchors(c["packed"], frame_rows=rows) for c in conds]
    action_conds = [action_condition(a, generated.shape[2], args.device, generated.dtype)
                    for a in ("A", "D")]
    cache = H3ChunkCache(5, "cpu")

    def set_visual(enabled: bool) -> None:
        for module in visual_modules:
            module.enabled = bool(enabled)

    try:
        for chunk, start in enumerate(range(0, generated.shape[2], args.chunk_frames)):
            stop = min(start + args.chunk_frames, generated.shape[2])
            current = generated[:, :, start:stop].contiguous()
            history = generated[:, :, :start].contiguous()
            if chunk == 0:
                pipe.load_models_to_device(["dit"])
                tail_anchor = None
                anchor = torch.cat((conds[0]["anchor"], conds[0]["anchor"].clone()), dim=0)
                anchor_index, anchor_slot = None, 0
            else:
                pipe.load_models_to_device(["video_vae"])
                tail_anchor = last_frame_image_anchor(pipe.video_vae, history, dtype=pipe.torch_dtype)
                pipe.load_models_to_device(["dit"])
                anchor = torch.cat((conds[0]["anchor"], tail_anchor), dim=0)
                anchor_index, anchor_slot = start - 1, 1
            student_common = []
            for case, packed, action in zip(conds, dual_packed, action_conds):
                student_common.append(dict(
                    full_packed=packed, prompt=case["prompt_embeds"], anchor=anchor,
                    audio=case["audio_noise"], chunk_frames=args.chunk_frames,
                    anchor_frame_index=anchor_index, anchor_slot=anchor_slot,
                    action_cond=action[start:stop], action_prefix_mode="causal",
                    action_feedback=True))
            for sigma in sigmas:
                student_v = []
                set_visual(True)
                student.enabled = True
                with torch.no_grad():
                    for case, common in zip(conds, student_common):
                        student_v.append(chunk_forward(
                            pipe.dit, current, index=chunk, cache=cache, sigma=sigma,
                            action_adapter=student, **common).float())
                set_visual(False)
                student.enabled = False
                with torch.no_grad():
                    teacher_v = [full_teacher_forward(
                        pipe.dit, current, history=history,
                        full_packed=case["packed"], prompt=case["prompt_embeds"],
                        anchor=case["anchor"], audio=case["audio_noise"],
                        sigma=sigma, action_cond=action)
                        .float() for case, action in zip(conds, action_conds)]
                set_visual(True)
                student.enabled = True
                ds = (student_v[0] - student_v[1]).flatten(1)
                dt = (teacher_v[0] - teacher_v[1]).flatten(1)
                s_norm = float(torch.linalg.vector_norm(ds, dim=1).mean())
                t_norm = float(torch.linalg.vector_norm(dt, dim=1).mean())
                cosine = float(F.cosine_similarity(ds, dt, dim=1).mean())
                ratio = s_norm / (t_norm + 1e-6)
                row = dict(chunk=chunk, start_latent=start, stop_latent=stop, sigma=sigma,
                           student_delta_norm=s_norm, teacher_delta_norm=t_norm,
                           student_over_teacher_delta_norm=ratio,
                           student_teacher_delta_cosine=cosine)
                out["rows"].append(row)
                print(json.dumps(row), flush=True)
                args.out.write_text(json.dumps(out, indent=2) + "\n")
            # Commit the generated A state, preserving the same clean cache
            # lifecycle as the benchmark. Only A writes history for this audit.
            with torch.no_grad():
                common = student_common[0]
                chunk_forward(pipe.dit, current, index=chunk, cache=cache, sigma=0.0,
                              commit=True, action_adapter=student, **common)
        out.update(status="complete", finite=True,
                   interpretation="cosine compares causal student and full teacher A/D deltas on the same generated state")
        args.out.write_text(json.dumps(out, indent=2) + "\n")
    except Exception as exc:
        out.update(status="failed", error=repr(exc))
        args.out.write_text(json.dumps(out, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
