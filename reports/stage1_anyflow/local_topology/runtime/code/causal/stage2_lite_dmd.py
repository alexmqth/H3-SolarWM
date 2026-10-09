#!/usr/bin/env python3
"""Minimal SolarWM-style Stage2-lite diagnostic for H3-World.

This script is deliberately smaller than the released SolarWM Stage2 recipe.
It keeps one frozen H3 DiT on one device and swaps three small external
adapters:

* ``student``: the existing causal action residual used for self-rollout;
* ``critic``: a trainable hidden action residual fitted on the student
  generated distribution with flow-matching supervision;
* ``teacher``: frozen H3 weights with the new adapters disabled, evaluated
  through the same causal chunk/cache path rather than native bidirectional H3.

The H3-World pipeline callback used by this repository returns
``noise - clean`` velocity and advances samples with ``x_next = x +
delta_sigma * velocity``.  Consequently the clean estimate used below is
``x0 = noisy - sigma * velocity``.  This is intentionally documented here:
SolarWM's standalone H3 SGF backend uses a data-ward sign convention after
its backend adapter, and copying that sign into this code would reverse the
DMD direction.

The initial implementation updated one generated chunk at one noise level per
action.  The script also supports a wider Stage2-lite v2 diagnostic: multiple
generated chunks and multiple noise levels are scored before the student
update. This is still much smaller than the released SolarWM recipe, but it
covers the point where generated-history drift first appears instead of only
correcting the final chunk.

    student self-rollout -> fake-score fit -> frozen teacher -> DMD surrogate

It is a feasibility diagnostic, not a claim of reproducing SolarWM Stage2.
The differentiable student output is a single x0 replay from a re-noised,
detached generated endpoint. It is not the parameter Jacobian of the actual
multi-step rollout or AnyFlow's three-segment Flow Map Backward Simulation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/abot"))
sys.path.insert(0, str(ROOT / "code"))

import infer as abot  # noqa: E402
from causal.h3_cached import (  # noqa: E402
    H3ChunkCache,
    chunk_forward,
    expand_packed_two_anchors,
    last_frame_anchor,
    last_frame_image_anchor,
)
from causal.pretrained_lora import (  # noqa: E402
    CausalActionResidual,
    CausalQKVLoRA,
    load_adapter,
    load_action_residual,
    save_action_residual,
)


def move_tree(value, device):
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {k: move_tree(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move_tree(v, device) for v in value)
    return value


def action_condition(label: str, latent_t: int, device, dtype) -> torch.Tensor:
    aliases = {"W": "forward", "S": "back", "A": "strafe-left", "D": "strafe-right"}
    preset = aliases.get(label.upper(), label)
    if preset not in abot.ACTION_PRESETS:
        raise ValueError(f"unknown action {label!r}; use W/S/A/D or an H3 preset")
    keys = np.zeros((latent_t, len(abot.S.KEYS9)), dtype=np.float32)
    for key in abot.ACTION_PRESETS[preset]:
        keys[:, abot.S.KEYS9.index(key)] = 1.0
    return torch.from_numpy(keys).to(device=device, dtype=dtype)


@contextmanager
def adapter_enabled(adapter, enabled: bool):
    old = getattr(adapter, "enabled", True) if adapter is not None else None
    if adapter is not None:
        adapter.enabled = bool(enabled)
    try:
        yield
    finally:
        if adapter is not None:
            adapter.enabled = old


@contextmanager
def causal_adapter_enabled(pipe, enabled: bool):
    """Temporarily toggle the fixed causal QKV adapters in a shared backbone."""
    modules = [m for m in pipe.dit.modules() if isinstance(m, CausalQKVLoRA)]
    old = [m.enabled for m in modules]
    for module in modules:
        module.enabled = bool(enabled)
    try:
        yield
    finally:
        for module, value in zip(modules, old):
            module.enabled = value


def parse_tail(spec: str, total_blocks: int) -> list[int]:
    text = str(spec).lower().strip()
    if text.startswith("tail"):
        n = int(text[4:])
        if n < 1 or n > total_blocks:
            raise ValueError(f"invalid critic block spec {spec}")
        return list(range(total_blocks - n, total_blocks))
    out = []
    for item in text.split(","):
        if "-" in item:
            lo, hi = (int(x) for x in item.split("-", 1))
            out.extend(range(lo, hi + 1))
        elif item.strip():
            out.append(int(item))
    out = sorted(set(out))
    if not out or min(out) < 0 or max(out) >= total_blocks:
        raise ValueError(f"invalid critic block spec {spec}")
    return out


def parse_float_list(spec: str) -> list[float]:
    """Parse a comma-separated sigma list while preserving user order."""
    values = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not 0.0 < value < 1.0:
            raise ValueError(f"sigma must lie in (0,1), got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"empty sigma list: {spec!r}")
    return values


def parse_int_list(spec: str) -> list[int]:
    """Parse a comma-separated list of chunk indices."""
    values = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError(f"empty integer list: {spec!r}")
    return sorted(set(values))


def make_hidden_adapter(pipe, block_indices, device, rank_note: int = 8, mode="hidden"):
    """Create a zero-initialized, independent action residual.

    ``rank_note`` is recorded for experiment provenance.  The existing H3
    action residual is a small action-conditioned FiLM-like projection rather
    than a matrix LoRA; using the same injection space makes critic/student
    swapping possible without a second 33B backbone.
    """
    ref = next(pipe.dit.parameters())
    first = pipe.dit.blocks[block_indices[0]].attn
    adapter = CausalActionResidual(
        block_indices,
        num_buttons=len(abot.S.KEYS9),
        num_heads=pipe.dit.num_attention_heads,
        head_dim=first.head_dim,
        device=device,
        dtype=ref.dtype,
        mode=mode,
        hidden_size=pipe.dit.hidden_size,
    )
    adapter.rank_note = int(rank_note)
    return adapter


def make_case(cond_dir: Path, label: str, device: str, chunk_frames: int):
    cond = torch.load(cond_dir / "conditioning.pt", map_location="cpu", weights_only=True)
    common = move_tree(cond, device)
    initial = common["initial_noise"]
    if initial.ndim != 5 or initial.shape[0] != 1 or initial.shape[2] != 12:
        raise ValueError(f"{cond_dir}: expected [1,C,12,H,W] initial noise, got {tuple(initial.shape)}")
    rows = (initial.shape[-2] // 2) * (initial.shape[-1] // 2)
    packed = expand_packed_two_anchors(common["packed"], frame_rows=rows)
    original_anchor = common["anchor"]
    if original_anchor.shape[0] != rows:
        raise ValueError(
            f"{cond_dir}: anchor rows {original_anchor.shape[0]} do not match latent rows {rows}")
    teacher_latents = None
    latent_path = cond_dir / "baseline_latents.pt"
    if latent_path.exists():
        teacher_latents = torch.load(
            latent_path, map_location=device, weights_only=True).to(device=device)
        if teacher_latents.shape != initial.shape:
            raise ValueError(
                f"{cond_dir}: baseline latent shape {tuple(teacher_latents.shape)} "
                f"does not match initial noise {tuple(initial.shape)}")
    return {
        "label": str(label).upper(),
        "packed": packed,
        "prompt": common["prompt_embeds"],
        "audio": common["audio_noise"],
        "initial": initial,
        "original_anchor": original_anchor,
        "dual_anchor": torch.cat((original_anchor, original_anchor.clone()), dim=0),
        "action_cond": action_condition(label, int(initial.shape[2]), device, initial.dtype),
        "rows": rows,
        "chunk_frames": int(chunk_frames),
        "teacher_latents": teacher_latents,
    }


def chunk_common(pipe, case, generated, chunk: int, current, *, anchor_mode="latent",
                 rgb_anchor_cache=None):
    """Build the H3 condition for one causal chunk.

    The first Stage2-lite implementation always converted the previous latent
    tail directly into an anchor.  That is not the protocol used by the visual
    tail16 adapter: H3's image condition is obtained by decoding the generated
    prefix to RGB and re-encoding its last frame through the image branch of
    the VAE.  Keeping the choice here, and passing the same cache through
    rollout/critic/teacher, prevents a train/eval anchor mismatch.
    """
    start = chunk * case["chunk_frames"]
    stop = start + current.shape[2]
    if chunk == 0:
        anchor = case["dual_anchor"]
        anchor_frame_index = None
        anchor_slot = 0
    else:
        tail = generated[chunk - 1][:, :, -1:].detach()
        if anchor_mode == "rgb":
            if rgb_anchor_cache is None:
                rgb_anchor_cache = {}
            if chunk not in rgb_anchor_cache:
                # The VAE and DiT share the device in the inference helper;
                # move only the required module for this expensive conversion
                # and restore the DiT before returning to chunk_forward.
                prefix = torch.cat(generated, dim=2).detach()
                pipe.load_models_to_device(["video_vae"])
                rgb_anchor_cache[chunk] = last_frame_image_anchor(
                    pipe.video_vae, prefix, dtype=pipe.torch_dtype).detach()
                pipe.load_models_to_device(["dit"])
            tail_anchor = rgb_anchor_cache[chunk]
        elif anchor_mode == "latent":
            tail_anchor = last_frame_anchor(tail)
        else:
            raise ValueError(f"unknown Stage2-lite anchor mode: {anchor_mode}")
        anchor = torch.cat((
            case["original_anchor"],
            tail_anchor,
        ), dim=0)
        anchor_frame_index = start - 1
        anchor_slot = 1
    return dict(
        full_packed=case["packed"],
        prompt=case["prompt"],
        anchor=anchor,
        audio=case["audio"],
        chunk_frames=case["chunk_frames"],
        anchor_frame_index=anchor_frame_index,
        anchor_slot=anchor_slot,
        action_cond=case["action_cond"][start:stop],
        action_prefix_mode="causal",
        action_feedback=True,
    )


def rollout_student(pipe, case, student, timesteps, history_chunks: int, anchor_mode: str):
    """Generate a detached 39-frame causal trajectory with persistent KV."""
    cache = H3ChunkCache(history_chunks, "cpu")
    generated = []
    rgb_anchor_cache = {}
    initial = case["initial"]
    chunks = (initial.shape[2] + case["chunk_frames"] - 1) // case["chunk_frames"]
    with adapter_enabled(student, True):
        for chunk in range(chunks):
            start = chunk * case["chunk_frames"]
            stop = min(start + case["chunk_frames"], initial.shape[2])
            current = initial[:, :, start:stop].contiguous()
            common = chunk_common(
                pipe, case, generated, chunk, current,
                anchor_mode=anchor_mode, rgb_anchor_cache=rgb_anchor_cache)
            with torch.no_grad():
                for timestep in timesteps:
                    velocity = chunk_forward(
                        pipe.dit, current, index=chunk, cache=cache,
                        sigma=float(timestep) / 1000.0,
                        action_adapter=student, **common)
                    current = pipe.scheduler.step(velocity, timestep, current).detach()
                chunk_forward(
                    pipe.dit, current, index=chunk, cache=cache, sigma=0.0,
                    commit=True, action_adapter=student, **common)
            generated.append(current.detach())
    return generated, rgb_anchor_cache


def commit_history(pipe, case, generated, adapter, target_chunk, history_chunks,
                   *, anchor_mode: str, rgb_anchor_cache):
    """Build detached K/V for chunks before ``target_chunk``."""
    cache = H3ChunkCache(history_chunks, "cpu")
    with adapter_enabled(adapter, adapter is not None):
        with torch.no_grad():
            for chunk in range(target_chunk):
                current = generated[chunk].detach()
                common = chunk_common(
                    pipe, case, generated, chunk, current,
                    anchor_mode=anchor_mode, rgb_anchor_cache=rgb_anchor_cache)
                chunk_forward(
                    pipe.dit, current, index=chunk, cache=cache, sigma=0.0,
                    commit=True, action_adapter=adapter, **common)
    return cache


def critic_update(pipe, case, generated, critic, optimizer, sigma, history_chunks,
                  target_chunk, *, anchor_mode: str, rgb_anchor_cache):
    """Fit fake velocity to ``noise-clean`` on one generated chunk."""
    clean = generated[target_chunk].detach()
    noise = torch.randn_like(clean)
    noisy = (1.0 - sigma) * clean + sigma * noise
    cache = commit_history(
        pipe, case, generated, critic, target_chunk, history_chunks,
        anchor_mode=anchor_mode, rgb_anchor_cache=rgb_anchor_cache)
    common = chunk_common(
        pipe, case, generated, target_chunk, clean,
        anchor_mode=anchor_mode, rgb_anchor_cache=rgb_anchor_cache)
    optimizer.zero_grad(set_to_none=True)
    with adapter_enabled(critic, True):
        velocity = chunk_forward(
            pipe.dit, noisy, index=target_chunk, cache=cache, sigma=sigma,
            action_adapter=critic, allow_grad_read=True,
            use_gradient_checkpointing=True,
            use_gradient_checkpointing_offload=True, **common)
    target = noise - clean
    loss = F.mse_loss(velocity.float(), target.float())
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(list(critic.parameters()), 1.0)
    optimizer.step()
    metrics = {
        "critic_loss": float(loss.detach()),
        "critic_grad_norm": float(grad_norm.detach()),
        "critic_target_rms": float(target.float().square().mean().sqrt()),
        "critic_velocity_rms": float(velocity.detach().float().square().mean().sqrt()),
        "sigma": float(sigma),
        "target_chunk": int(target_chunk),
        "cache_bytes": int(cache.nbytes),
    }
    return metrics, noise.detach(), noisy.detach()


def dmd_student_update(
    pipe, case, generated, student, critic, optimizer, sigma, noise,
    noisy, history_chunks, target_chunk, *, anchor_mode: str, rgb_anchor_cache,
):
    """Apply a detached DMD gradient to student velocity through x0."""
    clean = generated[target_chunk].detach()
    common = chunk_common(
        pipe, case, generated, target_chunk, clean,
        anchor_mode=anchor_mode, rgb_anchor_cache=rgb_anchor_cache)
    with causal_adapter_enabled(pipe, True):
        fake_cache = commit_history(
            pipe, case, generated, critic, target_chunk, history_chunks,
            anchor_mode=anchor_mode, rgb_anchor_cache=rgb_anchor_cache)
        student_cache = commit_history(
            pipe, case, generated, student, target_chunk, history_chunks,
            anchor_mode=anchor_mode, rgb_anchor_cache=rgb_anchor_cache)
    # Remove the trained visual adapter's W/forward bias from the score role.
    # This restores original weights, not the native bidirectional score:
    # teacher_cache and real_velocity below still use causal chunk_forward.
    # Keep that distinction explicit when comparing this historical smoke
    # with a future bidirectional-teacher on-policy or FMBS implementation.
    with causal_adapter_enabled(pipe, False):
        teacher_cache = commit_history(
            pipe, case, generated, None, target_chunk, history_chunks,
            anchor_mode=anchor_mode, rgb_anchor_cache=rgb_anchor_cache)

    with torch.no_grad():
        with causal_adapter_enabled(pipe, True), adapter_enabled(critic, True):
            fake_velocity = chunk_forward(
                pipe.dit, noisy, index=target_chunk, cache=fake_cache, sigma=sigma,
                action_adapter=critic, **common)
        with causal_adapter_enabled(pipe, False), adapter_enabled(None, False):
            real_velocity = chunk_forward(
                pipe.dit, noisy, index=target_chunk, cache=teacher_cache, sigma=sigma,
                action_adapter=None, **common)

    # H3-World's current DiT path predicts noise-clean.  Both estimates use
    # the same noisy state and sigma, so their difference is a pure score-field
    # signal rather than a solver-step difference.
    fake_x0 = noisy.float() - sigma * fake_velocity.float()
    real_x0 = noisy.float() - sigma * real_velocity.float()
    dmd_grad = torch.nan_to_num(fake_x0 - real_x0)

    optimizer.zero_grad(set_to_none=True)
    with causal_adapter_enabled(pipe, True), adapter_enabled(student, True):
        student_velocity = chunk_forward(
            pipe.dit, noisy, index=target_chunk, cache=student_cache, sigma=sigma,
            action_adapter=student, allow_grad_read=True,
            use_gradient_checkpointing=True,
            use_gradient_checkpointing_offload=True, **common)
    student_x0 = noisy.float() - sigma * student_velocity.float()
    normalizer = (student_x0.detach() - real_x0).abs().mean().clamp_min(1e-8)
    normalized_grad = torch.nan_to_num(dmd_grad / normalizer)
    surrogate_target = (student_x0 - normalized_grad.detach()).detach()
    surrogate = 0.5 * (student_x0 - surrogate_target).square().mean()
    surrogate.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(list(student.parameters()), 1.0)
    optimizer.step()
    metrics = {
        "dmd_surrogate_loss": float(surrogate.detach()),
        "dmd_grad_norm": float(grad_norm.detach()),
        "dmd_gradient_rms": float(normalized_grad.detach().square().mean().sqrt()),
        "fake_real_x0_rms": float((fake_x0 - real_x0).square().mean().sqrt()),
        "student_real_x0_rms": float((student_x0.detach() - real_x0).square().mean().sqrt()),
        "student_fake_x0_rms": float((student_x0.detach() - fake_x0).square().mean().sqrt()),
        "dmd_sigma": float(sigma),
        "target_chunk": int(target_chunk),
        "student_cache_bytes": int(student_cache.nbytes),
        "teacher_cache_bytes": int(teacher_cache.nbytes),
        "critic_cache_bytes": int(fake_cache.nbytes),
    }
    return metrics


def paired_action_delta_update(
    pipe, case_a, generated_a, case_d, student, optimizer, sigma,
    history_chunks, target_chunk, dir_weight, mag_weight, *,
    anchor_mode: str, rgb_anchor_cache, latent_target_weight: float = 0.0,
):
    """Match the student's A/D score-field geometry at one shared state.

    ``generated_a[target_chunk]`` is used as the counterfactual state for
    both actions.  The history cache is also built once from that same
    rollout, so the only intervention is the current action condition.  The
    frozen teacher is evaluated with the fixed visual causal adapter disabled;
    the student keeps it enabled and learns an action-dependent QKV residual.
    This directly targets the failure mode where a W-trained visual adapter
    makes A and D produce the same motion.
    """
    clean = generated_a[target_chunk].detach()
    noise = torch.randn_like(clean)
    noisy = (1.0 - sigma) * clean + sigma * noise
    common_a = chunk_common(
        pipe, case_a, generated_a, target_chunk, clean,
        anchor_mode=anchor_mode, rgb_anchor_cache=rgb_anchor_cache)
    common_d = chunk_common(
        pipe, case_d, generated_a, target_chunk, clean,
        anchor_mode=anchor_mode, rgb_anchor_cache=rgb_anchor_cache)

    with causal_adapter_enabled(pipe, True):
        student_cache = commit_history(
            pipe, case_a, generated_a, student, target_chunk, history_chunks,
            anchor_mode=anchor_mode, rgb_anchor_cache=rgb_anchor_cache)
    with causal_adapter_enabled(pipe, False):
        teacher_cache = commit_history(
            pipe, case_a, generated_a, None, target_chunk, history_chunks,
            anchor_mode=anchor_mode, rgb_anchor_cache=rgb_anchor_cache)

    with torch.no_grad():
        with causal_adapter_enabled(pipe, False):
            teacher_a = chunk_forward(
                pipe.dit, noisy, index=target_chunk, cache=teacher_cache,
                sigma=sigma, action_adapter=None, **common_a)
            teacher_d = chunk_forward(
                pipe.dit, noisy, index=target_chunk, cache=teacher_cache,
                sigma=sigma, action_adapter=None, **common_d)
    optimizer.zero_grad(set_to_none=True)
    with causal_adapter_enabled(pipe, True), adapter_enabled(student, True):
        student_a = chunk_forward(
            pipe.dit, noisy, index=target_chunk, cache=student_cache,
            sigma=sigma, action_adapter=student, allow_grad_read=True,
            use_gradient_checkpointing=True,
            use_gradient_checkpointing_offload=True, **common_a)
        student_d = chunk_forward(
            pipe.dit, noisy, index=target_chunk, cache=student_cache,
            sigma=sigma, action_adapter=student, allow_grad_read=True,
            use_gradient_checkpointing=True,
            use_gradient_checkpointing_offload=True, **common_d)

    delta_teacher = (teacher_a.float() - teacher_d.float()).flatten()
    delta_student = (student_a.float() - student_d.float()).flatten()
    teacher_norm = delta_teacher.norm().clamp_min(1e-6)
    student_norm = delta_student.norm().clamp_min(1e-6)
    direction_loss = 1.0 - F.cosine_similarity(
        delta_student.unsqueeze(0), delta_teacher.unsqueeze(0), dim=1).mean()
    magnitude_loss = (student_norm - teacher_norm).abs() / teacher_norm
    paired_loss = dir_weight * direction_loss + mag_weight * magnitude_loss
    endpoint_loss = torch.zeros((), device=noisy.device)
    if latent_target_weight > 0:
        target_start = target_chunk * case_a["chunk_frames"]
        target_stop = target_start + clean.shape[2]
        target_a = case_a.get("teacher_latents")
        target_d = case_d.get("teacher_latents")
        if target_a is None or target_d is None:
            raise ValueError(
                "--latent-target-weight requires baseline_latents.pt in both A and D cases")
        target_a = target_a[:, :, target_start:target_stop].float()
        target_d = target_d[:, :, target_start:target_stop].float()
        student_x0_a = noisy.float() - sigma * student_a.float()
        student_x0_d = noisy.float() - sigma * student_d.float()
        endpoint_loss = 0.5 * (
            F.mse_loss(student_x0_a, target_a)
            + F.mse_loss(student_x0_d, target_d)
        )
        paired_loss = paired_loss + latent_target_weight * endpoint_loss
    paired_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(list(student.parameters()), 1.0)
    optimizer.step()
    metrics = {
        "paired_loss": float(paired_loss.detach()),
        "paired_direction_loss": float(direction_loss.detach()),
        "paired_magnitude_loss": float(magnitude_loss.detach()),
        "paired_teacher_delta_rms": float(delta_teacher.square().mean().sqrt()),
        "paired_student_delta_rms": float(delta_student.detach().square().mean().sqrt()),
        "paired_teacher_delta_norm": float(teacher_norm.detach()),
        "paired_student_delta_norm": float(student_norm.detach()),
        "paired_endpoint_loss": float(endpoint_loss.detach()),
        "paired_latent_target_weight": float(latent_target_weight),
        "paired_grad_norm": float(grad_norm.detach()),
        "paired_sigma": float(sigma),
        "paired_target_chunk": int(target_chunk),
        "paired_student_cache_bytes": int(student_cache.nbytes),
        "paired_teacher_cache_bytes": int(teacher_cache.nbytes),
    }
    return metrics


def own_endpoint_target_update(
    pipe, case, generated, student, optimizer, sigma, history_chunks, target_chunk,
    *, anchor_mode: str, rgb_anchor_cache, weight: float,
):
    """Match an original-H3 endpoint on the action's own generated history.

    This is deliberately separate from ``paired_action_delta_update``.  A
    paired A/D loss uses one shared counterfactual state; an endpoint from an
    independent D rollout is not a valid target for that state.  This helper
    instead keeps each action's generated history and compares its x0 estimate
    with the same-action original H3 endpoint.
    """
    clean = generated[target_chunk].detach()
    target_latents = case.get("teacher_latents")
    if target_latents is None:
        raise ValueError("own endpoint target requires baseline_latents.pt in the teacher dir")
    start = target_chunk * case["chunk_frames"]
    target = target_latents[:, :, start:start + clean.shape[2]].float()
    noise = torch.randn_like(clean)
    noisy = (1.0 - sigma) * clean + sigma * noise
    cache = commit_history(
        pipe, case, generated, student, target_chunk, history_chunks,
        anchor_mode=anchor_mode, rgb_anchor_cache=rgb_anchor_cache)
    common = chunk_common(
        pipe, case, generated, target_chunk, clean,
        anchor_mode=anchor_mode, rgb_anchor_cache=rgb_anchor_cache)
    optimizer.zero_grad(set_to_none=True)
    with causal_adapter_enabled(pipe, True), adapter_enabled(student, True):
        velocity = chunk_forward(
            pipe.dit, noisy, index=target_chunk, cache=cache, sigma=sigma,
            action_adapter=student, allow_grad_read=True,
            use_gradient_checkpointing=True,
            use_gradient_checkpointing_offload=True, **common)
    student_x0 = noisy.float() - sigma * velocity.float()
    endpoint_loss = F.mse_loss(student_x0, target)
    (weight * endpoint_loss).backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(list(student.parameters()), 1.0)
    optimizer.step()
    return {
        "own_endpoint_loss": float(endpoint_loss.detach()),
        "own_endpoint_weight": float(weight),
        "own_endpoint_grad_norm": float(grad_norm.detach()),
        "own_endpoint_sigma": float(sigma),
        "own_endpoint_target_chunk": int(target_chunk),
        "own_endpoint_action": case["label"],
        "own_endpoint_cache_bytes": int(cache.nbytes),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-dir", type=Path, nargs="+", required=True,
                    help="conditioning dirs, one per action (normally A and D)")
    ap.add_argument("--actions", nargs="+", required=True)
    ap.add_argument("--student-action-adapter", type=Path, required=True)
    ap.add_argument("--student-action-mode", choices=["hidden", "qkv"], default="hidden",
                    help="train the loaded hidden residual or a new action-conditioned QKV residual")
    ap.add_argument("--student-blocks", default="tail8",
                    help="blocks for a new qkv student residual; ignored in hidden mode")
    ap.add_argument("--causal-adapter", type=Path, default=None,
                    help="optional frozen causal tail-QKV adapter used by both student and teacher")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--solver-steps", type=int, default=2,
                    help="student rollout solver steps; use 8 for the fixed benchmark")
    ap.add_argument("--chunk-frames", type=int, default=5)
    ap.add_argument("--history-chunks", type=int, default=5)
    ap.add_argument("--anchor-mode", choices=["latent", "rgb"], default="rgb",
                    help=("previous-chunk anchor protocol. 'rgb' decodes the generated "
                          "prefix and re-encodes its last frame through H3's image branch; "
                          "'latent' is the cheaper legacy patchify path."))
    ap.add_argument("--flow-shift", type=float, default=2.22)
    ap.add_argument("--critic-sigma", type=float, default=0.6,
                    help="legacy single sigma; ignored when --critic-sigmas is set")
    ap.add_argument("--critic-sigmas", default=None,
                    help="comma-separated generated-distribution sigmas, e.g. 0.94,0.79,0.57,0.24")
    ap.add_argument("--critic-blocks", default="tail4")
    ap.add_argument("--critic-lr", type=float, default=1e-4)
    ap.add_argument("--student-lr", type=float, default=1e-5)
    ap.add_argument("--updates", type=int, default=1,
                    help="number of A/D self-rollout -> critic -> DMD rounds")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--target-chunk", type=int, default=-1,
                    help="target generated chunk; -1 means final chunk")
    ap.add_argument("--target-chunks", default=None,
                    help="comma-separated generated chunks; overrides --target-chunk")
    ap.add_argument("--paired-delta", action="store_true",
                    help="add shared-state A/D teacher-delta geometry loss")
    ap.add_argument("--paired-dir-weight", type=float, default=1.0)
    ap.add_argument("--paired-mag-weight", type=float, default=0.25)
    ap.add_argument("--paired-only", action="store_true",
                    help="skip critic/DMD updates and train only the paired A/D loss")
    ap.add_argument("--latent-target-weight", type=float, default=0.0,
                    help=("add an A/D chunk-endpoint x0 MSE to the original H3 "
                          "30-step baseline latents; requires baseline_latents.pt "
                          "in both teacher dirs"))
    ap.add_argument("--own-latent-target-weight", type=float, default=0.0,
                    help=("add same-action endpoint x0 MSE on each action's own "
                          "generated history; avoids mixing independent A/D states"))
    args = ap.parse_args()
    if len(args.teacher_dir) != len(args.actions):
        ap.error("--teacher-dir and --actions must have the same length")
    if args.solver_steps < 1 or args.chunk_frames < 1 or args.history_chunks < 1:
        ap.error("solver/chunk/history values must be positive")
    if args.updates < 1:
        ap.error("--updates must be positive")
    if args.paired_only and not args.paired_delta:
        ap.error("--paired-only requires --paired-delta")
    if args.latent_target_weight < 0:
        ap.error("--latent-target-weight must be non-negative")
    if args.own_latent_target_weight < 0:
        ap.error("--own-latent-target-weight must be non-negative")
    if not 0.0 < args.critic_sigma < 1.0:
        ap.error("--critic-sigma must lie in (0,1)")
    try:
        critic_sigmas = parse_float_list(
            args.critic_sigmas if args.critic_sigmas is not None
            else str(args.critic_sigma))
        target_chunks_arg = parse_int_list(args.target_chunks) if args.target_chunks else None
    except ValueError as exc:
        ap.error(str(exc))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.perf_counter()
    result = {
        "status": "running",
        "kind": "H3-World SolarWM-style Stage2-lite critic plus DMD surrogate",
        "objective": "student self-rollout + fake-score flow matching + frozen teacher + DMD",
        "teacher_attention_contract": "original weights through causal chunk/cache attention",
        "dmd_student_gradient_contract": "single x0 replay on a re-noised detached generated endpoint",
        "flow_map_backward_simulation": False,
        "velocity_convention": "H3-World noise-clean; x0=noisy-sigma*velocity",
        "teacher_dirs": [str(x) for x in args.teacher_dir],
        "actions": [str(x).upper() for x in args.actions],
        "student_action_adapter": str(args.student_action_adapter),
        "student_action_mode": args.student_action_mode,
        "student_blocks": args.student_blocks,
        "causal_adapter": str(args.causal_adapter) if args.causal_adapter else None,
        "solver_steps": args.solver_steps,
        "chunk_frames": args.chunk_frames,
        "history_chunks": args.history_chunks,
        "anchor_mode": args.anchor_mode,
        "flow_shift": args.flow_shift,
        "critic_sigma": args.critic_sigma,
        "critic_sigmas": critic_sigmas,
        "critic_blocks": args.critic_blocks,
        "critic_lr": args.critic_lr,
        "student_lr": args.student_lr,
        "updates": args.updates,
        "seed": args.seed,
        "target_chunks_requested": target_chunks_arg,
        "paired_delta": bool(args.paired_delta),
        "paired_dir_weight": args.paired_dir_weight,
        "paired_mag_weight": args.paired_mag_weight,
        "paired_only": bool(args.paired_only),
        "latent_target_weight": args.latent_target_weight,
        "own_latent_target_weight": args.own_latent_target_weight,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "action_metrics": [],
        "update_checkpoints": [],
    }

    def save_json():
        tmp = args.out_dir / "stage2_lite.tmp.json"
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(args.out_dir / "stage2_lite.json")

    save_json()
    try:
        pipe = abot.load_pipeline(args.device)
        pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(
            ROOT / "checkpoints/H3-World/step-10000.safetensors"), hotload=True)
        pipe.dit.requires_grad_(False).eval()
        pipe.load_models_to_device(["dit"])
        if args.causal_adapter:
            result["causal_adapter_info"] = load_adapter(
                pipe.dit, args.causal_adapter, args.device)
        if args.student_action_mode == "hidden":
            loaded = load_action_residual(pipe.dit, args.student_action_adapter, args.device)
            student = loaded["adapter"]
            student_blocks = loaded.get("block_indices")
            student_metadata = loaded.get("metadata", {})
        else:
            # Continue from a previous QKV residual when the supplied path
            # is one; otherwise start from the intended zero initialization.
            state = torch.load(args.student_action_adapter, map_location="cpu", weights_only=True)
            if (state.get("format") == "h3_causal_action_residual_v1"
                    and state.get("mode", "qkv") == "qkv"):
                loaded = load_action_residual(pipe.dit, args.student_action_adapter, args.device)
                student = loaded["adapter"]
                student_blocks = loaded.get("block_indices")
                student_metadata = loaded.get("metadata", {})
                student_metadata = dict(student_metadata, initialized="checkpoint")
            else:
                student_blocks = parse_tail(args.student_blocks, len(pipe.dit.blocks))
                student = make_hidden_adapter(pipe, student_blocks, args.device, mode="qkv")
                student_metadata = {"mode": "qkv", "initialized": "zero"}
        student.requires_grad_(True)
        critic_blocks = parse_tail(args.critic_blocks, len(pipe.dit.blocks))
        critic = make_hidden_adapter(pipe, critic_blocks, args.device)
        critic.requires_grad_(True)
        result.update({
            "student_blocks": student_blocks,
            "student_mode": student_metadata.get("mode", args.student_action_mode),
            "critic_block_indices": critic_blocks,
            "critic_trainable_parameters": sum(p.numel() for p in critic.parameters()),
            "student_trainable_parameters": sum(p.numel() for p in student.parameters()),
        })
        pipe.scheduler.set_timesteps(args.solver_steps, shift=args.flow_shift)
        timesteps = list(pipe.scheduler.timesteps)
        result["scheduler_timesteps"] = [float(x) for x in timesteps]
        cases = [make_case(d, a, args.device, args.chunk_frames)
                 for d, a in zip(args.teacher_dir, args.actions)]
        latent_t = int(cases[0]["initial"].shape[2])
        chunks = (latent_t + args.chunk_frames - 1) // args.chunk_frames
        target_chunk = chunks - 1 if args.target_chunk < 0 else int(args.target_chunk)
        if not 0 <= target_chunk < chunks:
            raise ValueError(f"target chunk {target_chunk} outside 0..{chunks-1}")
        target_chunks = ([target_chunk] if target_chunks_arg is None
                         else target_chunks_arg)
        if any(not 0 <= x < chunks for x in target_chunks):
            raise ValueError(f"target chunks {target_chunks} outside 0..{chunks-1}")
        target_chunks = sorted(set(target_chunks))
        result.update({"latent_frames": latent_t, "num_chunks": chunks,
                       "target_chunk": target_chunk,
                       "target_chunks": target_chunks})
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr)
        student_optimizer = torch.optim.AdamW(student.parameters(), lr=args.student_lr)
        torch.cuda.reset_peak_memory_stats(args.device)

        for update_index in range(args.updates):
            generated_by_action = {}
            for case in cases:
                action_started = time.perf_counter()
                print(
                    f"[stage2-lite] update={update_index + 1}/{args.updates} "
                    f"rollout action={case['label']} steps={args.solver_steps}",
                    flush=True)
                generated, rgb_anchor_cache = rollout_student(
                    pipe, case, student, timesteps, args.history_chunks, args.anchor_mode)
                generated_by_action[case["label"]] = (generated, rgb_anchor_cache)
                samples = []
                if not args.paired_only:
                    # Fit the fake score over all requested history
                    # chunks/sigmas first. Student updates then see the same
                    # generated rollout, making the comparison attributable
                    # to coverage rather than a changing rollout midway.
                    for target in target_chunks:
                        for sigma in critic_sigmas:
                            critic_metrics, noise, noisy = critic_update(
                                pipe, case, generated, critic, critic_optimizer,
                                sigma, args.history_chunks, target,
                                anchor_mode=args.anchor_mode,
                                rgb_anchor_cache=rgb_anchor_cache)
                            samples.append((target, sigma, noise, noisy, critic_metrics))
                            print(json.dumps({
                                "update": update_index + 1,
                                "action": case["label"],
                                "phase": "critic",
                                **critic_metrics,
                            }, indent=2), flush=True)

                    for target, sigma, noise, noisy, critic_metrics in samples:
                        dmd_metrics = dmd_student_update(
                            pipe, case, generated, student, critic, student_optimizer,
                            sigma, noise, noisy, args.history_chunks, target,
                            anchor_mode=args.anchor_mode,
                            rgb_anchor_cache=rgb_anchor_cache)
                        metrics = {
                            "update": update_index + 1,
                            "action": case["label"],
                            "phase": "dmd",
                            "rollout_latents": [list(map(int, x.shape)) for x in generated],
                            "seconds": time.perf_counter() - action_started,
                            **critic_metrics,
                            **dmd_metrics,
                        }
                        result["action_metrics"].append(metrics)
                        print(json.dumps(metrics, indent=2), flush=True)
                        save_json()
                del generated
                if samples:
                    del noise, noisy
                torch.cuda.empty_cache()

            if args.own_latent_target_weight > 0:
                for case in cases:
                    generated, rgb_anchor_cache = generated_by_action[case["label"]]
                    for target in target_chunks:
                        for sigma in critic_sigmas:
                            own_metrics = own_endpoint_target_update(
                                pipe, case, generated, student, student_optimizer,
                                sigma, args.history_chunks, target,
                                anchor_mode=args.anchor_mode,
                                rgb_anchor_cache=rgb_anchor_cache,
                                weight=args.own_latent_target_weight)
                            own_metrics.update({
                                "update": update_index + 1,
                                "phase": "own_endpoint",
                            })
                            result["action_metrics"].append(own_metrics)
                            print(json.dumps(own_metrics, indent=2), flush=True)
                            save_json()

            if args.paired_delta:
                if len(cases) != 2 or {c["label"] for c in cases} != {"A", "D"}:
                    raise ValueError("--paired-delta currently requires exactly A and D cases")
                case_a = next(c for c in cases if c["label"] == "A")
                case_d = next(c for c in cases if c["label"] == "D")
                generated_a, rgb_anchor_cache_a = generated_by_action["A"]
                for target in target_chunks:
                    for sigma in critic_sigmas:
                        paired_metrics = paired_action_delta_update(
                            pipe, case_a, generated_a, case_d, student,
                            student_optimizer, sigma, args.history_chunks,
                            target, args.paired_dir_weight,
                            args.paired_mag_weight,
                            anchor_mode=args.anchor_mode,
                            rgb_anchor_cache=rgb_anchor_cache_a,
                            latent_target_weight=args.latent_target_weight)
                        paired_metrics.update({
                            "update": update_index + 1,
                            "action": "A_vs_D",
                            "phase": "paired_delta",
                        })
                        result["action_metrics"].append(paired_metrics)
                        print(json.dumps(paired_metrics, indent=2), flush=True)
                        save_json()
            # Preserve the action pathway after every self-rollout/alignment
            # round.  A multi-update smoke must expose its learning curve and
            # allow a later evaluator to select a checkpoint without rerunning
            # the expensive training loop.
            update_checkpoint_dir = args.out_dir / f"update_{update_index + 1:02d}"
            update_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_metadata = {
                "stage2_lite_status": "intermediate",
                "update": int(update_index + 1),
                "anchor_mode": args.anchor_mode,
                "paired_delta": bool(args.paired_delta),
                "paired_only": bool(args.paired_only),
                "target_chunks": target_chunks,
                "critic_sigmas": critic_sigmas,
            }
            save_action_residual(
                update_checkpoint_dir / "student_action_adapter.pt",
                student, checkpoint_metadata)
            result["update_checkpoints"].append(
                str(update_checkpoint_dir / "student_action_adapter.pt"))
            save_json()
            del generated_by_action
            torch.cuda.empty_cache()

        metadata = {k: v for k, v in result.items() if k != "action_metrics"}
        save_action_residual(args.out_dir / "student_action_adapter.pt", student, metadata)
        save_action_residual(args.out_dir / "critic_action_adapter.pt", critic, metadata)
        result.update({
            "status": "complete",
            "wall_seconds": time.perf_counter() - started,
            "allocated_peak_MiB": torch.cuda.max_memory_allocated(args.device) / 2**20,
            "reserved_peak_MiB": torch.cuda.max_memory_reserved(args.device) / 2**20,
            "completed_at": datetime.now().astimezone().isoformat(),
        })
        save_json()
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        result.update({"status": "failed", "error": repr(exc),
                       "wall_seconds": time.perf_counter() - started})
        save_json()
        raise


if __name__ == "__main__":
    main()
