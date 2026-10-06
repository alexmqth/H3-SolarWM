#!/usr/bin/env python3
"""Run an actual small H3 DiT + QKV LoRA teacher-forcing optimizer smoke."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DiffSynth-Studio-h3-v2"))
sys.path.insert(0, str(ROOT / "code"))

import torch
from causal.h3_training import make_small_h3, synthetic_h3_batch, h3_teacher_forcing_loss


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="training device, e.g. cpu or cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/h3_training_smoke.json")
    args = ap.parse_args()
    if args.steps < 2:
        ap.error("need at least two optimizer steps")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA requested but torch.cuda.is_available() is false")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = make_small_h3().to(device)

    def move(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    batch = move(synthetic_h3_batch())
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.01)
    before = [p.detach().clone() for p in params]
    losses, grad_norms = [], []
    start = time.perf_counter()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = h3_teacher_forcing_loss(model, **batch, sigma=0.6, chunk_index=1)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError("non-finite H3 loss or LoRA gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
        grad_norms.append(float(norm))
    changed = any(not torch.equal(p.detach(), old) for p, old in zip(params, before))
    assert changed and losses[-1] < losses[0], "fixed synthetic-batch smoke failed to optimize"
    result = dict(
        model="actual MiniMaxH3DiT class, randomly initialized small config",
        data="fixed synthetic latent batch; no evidence of pretrained video quality",
        training="clean-history teacher-forced flow matching; no AnyFlow or Stage2 DMD",
        device=str(device), layers=model.num_layers, hidden_size=model.hidden_size,
        attention_head_dim=model.blocks[0].attn.head_dim,
        lora_rank=4, seed=7,
        trainable_parameters=sum(p.numel() for p in params),
        total_parameters=sum(p.numel() for p in model.parameters()),
        steps=args.steps, losses=losses, grad_norms=grad_norms,
        parameters_changed=changed, wall_seconds=time.perf_counter()-start,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
