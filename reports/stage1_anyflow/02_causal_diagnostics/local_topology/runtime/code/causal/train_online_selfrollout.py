#!/usr/bin/env python3
"""Minimal action-aware online self-rollout teacher-replay diagnostic.

This is intentionally smaller than SolarWM Stage2.  The released H3 DiT is
frozen and an already trained causal action residual is optimized on its own
generated history.  Each optimizer step does the following:

1. roll out 3 causal chunks with the student action residual enabled;
2. detach every generated chunk into a persistent CPU raw-KV cache;
3. build a second, frozen-H3 cache from the same generated chunks;
4. replay the current chunk on both caches and match student velocity to the
   frozen teacher velocity.

The action rows are kept in the packed H3 condition for both paths.  This is a
Stage2-inspired diagnostic, not SGF/DMD and not a claim of SolarWM parity.
It deliberately runs on a 39-frame clip before any long-video extension.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/abot"))
sys.path.insert(0, str(ROOT / "code"))
import infer as abot
from causal.h3_cached import (H3ChunkCache, chunk_forward, expand_packed_two_anchors,
                               last_frame_anchor, last_frame_image_anchor, slice_packed)
from causal.pretrained_lora import (load_action_residual, save_action_residual,
                                    save_adapters, CausalActionPrefixResidual,
                                    save_action_prefix_residual,
                                    load_action_prefix_residual,
                                    save_h3_lora_adapter)
from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3


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


def parse_action_gains(spec: str | None) -> dict[str, float]:
    """Parse ``A=64,D=8`` gain initialization for the action residual."""
    if not spec:
        return {}
    result = {}
    for item in str(spec).split(','):
        item = item.strip()
        if not item or '=' not in item:
            raise ValueError(f'invalid action gain item {item!r}; use KEY=VALUE')
        key, value = (x.strip().upper() for x in item.split('=', 1))
        if key not in abot.S.KEYS9:
            raise ValueError(f'unknown action gain key {key!r}; use {abot.S.KEYS9}')
        gain = float(value)
        if not gain > 0:
            raise ValueError(f'action gain for {key} must be positive')
        result[key] = gain
    return result


@contextmanager
def adapter_state(adapter, enabled: bool):
    old = getattr(adapter, "enabled", True) if adapter is not None else None
    if adapter is not None:
        adapter.enabled = bool(enabled)
    try:
        yield
    finally:
        if adapter is not None:
            adapter.enabled = old


@contextmanager
def action_adapters_state(action_adapter, prefix_adapter, enabled: bool):
    """Toggle both current-video and action-token residuals together."""
    with adapter_state(action_adapter, enabled):
        with adapter_state(prefix_adapter, enabled):
            yield


@contextmanager
def qkv_teacher_state(qkv_params, teacher_state, student_state):
    """Temporarily expose a frozen causal-QKV teacher to the same DiT.

    The base H3 weights are already frozen.  When a causal QKV adapter is also
    trained, the teacher must not silently follow the student after every
    optimizer step.  Keeping one frozen snapshot and swapping only the small
    LoRA tensors avoids a second 40-GiB DiT copy.
    """
    if not qkv_params:
        yield
        return
    with torch.no_grad():
        for parameter, value in zip(qkv_params, teacher_state):
            parameter.copy_(value)
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, value in zip(qkv_params, student_state):
                parameter.copy_(value)


def parse_block_indices(spec, total_blocks):
    """Parse ``tail8``, ``42-49`` or a comma-separated block list."""
    text = str(spec).strip().lower()
    if text.startswith("tail"):
        count = int(text[4:])
        if count < 1 or count > total_blocks:
            raise ValueError(f"invalid tail block count: {spec}")
        return list(range(total_blocks - count, total_blocks))
    result = []
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        if '-' in item:
            lo, hi = (int(x) for x in item.split('-', 1))
            result.extend(range(lo, hi + 1))
        else:
            result.append(int(item))
    result = sorted(set(result))
    if not result or min(result) < 0 or max(result) >= total_blocks:
        raise ValueError(f"invalid H3 LoRA block list: {spec}")
    return result


def install_trainable_h3_lora(pipe, block_indices):
    """Turn selected released H3 LoRA tensors into student parameters.

    DiffSynth hot-loads LoRA matrices in Python lists instead of registered
    parameters.  We replace only the selected list entries with Parameters and
    retain the original tensors for the frozen teacher context below.
    """
    params = []
    entries = []
    for index in block_indices:
        block = pipe.dit.blocks[int(index)]
        for projection_name in ("qkv_proj", "out_proj"):
            module = getattr(block.attn, projection_name)
            if len(module.lora_A_weights) != 1 or len(module.lora_B_weights) != 1:
                raise ValueError(f"expected one released H3 LoRA pair at {module.name}")
            old_a = module.lora_A_weights[0]
            old_b = module.lora_B_weights[0]
            parameter_a = torch.nn.Parameter(old_a.detach().clone(), requires_grad=True)
            parameter_b = torch.nn.Parameter(old_b.detach().clone(), requires_grad=True)
            module.lora_A_weights[0] = parameter_a
            module.lora_B_weights[0] = parameter_b
            entries.append({"name": module.name, "weights_a": module.lora_A_weights,
                            "weights_b": module.lora_B_weights,
                            "parameter_a": parameter_a, "parameter_b": parameter_b,
                            "teacher_a": old_a.detach(), "teacher_b": old_b.detach()})
            params.extend((parameter_a, parameter_b))
    return params, entries


@contextmanager
def h3_lora_teacher_state(entries):
    """Expose the original released H3 LoRA while evaluating the teacher."""
    if not entries:
        yield
        return
    for entry in entries:
        entry["weights_a"][0] = entry["teacher_a"]
        entry["weights_b"][0] = entry["teacher_b"]
    try:
        yield
    finally:
        for entry in entries:
            entry["weights_a"][0] = entry["parameter_a"]
            entry["weights_b"][0] = entry["parameter_b"]


@contextmanager
def teacher_model_state(qkv_params, teacher_state, student_state, h3_lora_entries):
    with qkv_teacher_state(qkv_params, teacher_state, student_state):
        with h3_lora_teacher_state(h3_lora_entries):
            yield


def finite_or_raise(name, tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains non-finite values")


def paired_action_losses(student_a, student_d, teacher_a, teacher_d, eps=1e-6):
    """Return teacher replay, direction and magnitude losses for one sigma.

    All four velocity tensors are evaluated on the same latent state, history
    cache and solver sigma.  Flattening only the non-batch dimensions makes the
    cosine term a true action-delta constraint rather than a per-token average
    that can hide a collapsed A/D response.
    """
    replay = 0.5 * (
        F.mse_loss(student_a.float(), teacher_a.float())
        + F.mse_loss(student_d.float(), teacher_d.float())
    )
    delta_student = (student_a - student_d).float().flatten(1)
    delta_teacher = (teacher_a - teacher_d).float().flatten(1)
    cosine = F.cosine_similarity(delta_student, delta_teacher, dim=1, eps=eps).mean()
    direction = 1.0 - cosine
    norm_student = torch.linalg.vector_norm(delta_student, dim=1)
    norm_teacher = torch.linalg.vector_norm(delta_teacher, dim=1)
    magnitude = ((norm_student - norm_teacher).abs() / (norm_teacher + eps)).mean()
    return replay, direction, magnitude


def full_teacher_forward(dit, current, *, history, full_packed, prompt, anchor,
                         audio, sigma, action_cond):
    """Evaluate released bidirectional H3 on one generated prefix/state.

    The ordinary replay teacher also uses the causal controller, so its A/D
    delta can already be collapsed by the topology we are trying to repair.
    This helper keeps the original full-attention path and freezes the
    generated prefix with a denoise mask, giving paired training the actual
    H3 action geometry as its target.
    """
    start = int(history.shape[2])
    whole = torch.cat((history, current), dim=2)
    rows = (current.shape[-2] // 2) * (current.shape[-1] // 2)
    packed = slice_packed(full_packed, 0, int(whole.shape[2]), rows)
    mask = torch.ones_like(whole)
    if start:
        mask[:, :, :start] = 0
    return model_fn_minimax_h3(
        dit, whole, audio, packed, prompt,
        timestep_video=torch.tensor(float(sigma) * 1000, device=current.device),
        timestep_audio=torch.tensor(1000., device=current.device),
        keyframe_cond_anchor=anchor,
        input_latents_video=whole,
        denoise_mask_video=mask,
        action_cond=action_cond,
        # No causal_chunk_size/control: this is the released bidirectional
        # H3 attention path on the generated prefix and current chunk.
        fixed_prefix_timesteps=False,
    )[0][:, :, start:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-dir", type=Path, nargs="+", required=True,
                    help="one or more 39-frame conditioning dirs, one per action")
    ap.add_argument("--actions", nargs="+", required=True,
                    help="action label for each --teacher-dir, e.g. A D")
    ap.add_argument("--student-action-adapter", type=Path, required=True,
                    help="initial hidden/QKV causal action residual checkpoint")
    ap.add_argument("--causal-adapter", type=Path, default=None,
                    help="optional pretrained causal tail QKV adapter")
    ap.add_argument("--causal-adapter-scope",
                    choices=["all", "commit", "last_step_commit"], default="all",
                    help=("when the pretrained causal visual adapter is enabled: all "
                          "for every denoising/commit forward, commit only for clean "
                          "KV commits, or last_step_commit for the final denoising "
                          "step plus clean commit"))
    ap.add_argument("--action-gain-init", default=None,
                    help="initialize trainable action residual gains, e.g. A=64,D=8")
    ap.add_argument("--action-gain-lr", type=float, default=None,
                    help="learning rate for FP32 action gains (defaults to --lr)")
    ap.add_argument("--action-gain-compute-dtype", choices=["float32", "projection"],
                    default="float32", help="FP32 gain projection; projection reproduces legacy BF16 rounding")
    ap.add_argument("--anchor-mode", choices=["latent", "rgb"], default="latent",
                    help=("previous-chunk anchor used during online rollout: latent is the "
                          "cheap legacy path; rgb matches the RGB dual visual-adapter training "
                          "protocol"))
    ap.add_argument("--train-causal-adapter", action="store_true",
                    help="train the supplied causal QKV adapter with the student")
    ap.add_argument("--action-prefix-adapter", type=Path, default=None,
                    help="optional causal action-token prefix residual checkpoint")
    ap.add_argument("--train-action-prefix", action="store_true",
                    help="train a zero-initialized action-token prefix residual on the causal tail")
    ap.add_argument("--freeze-action-residual", action="store_true",
                    help="freeze the existing video Q/K/V action residual while adapting the prefix")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--optimizer-steps", type=int, default=1)
    ap.add_argument("--solver-steps", type=int, default=4)
    ap.add_argument("--history-chunks", type=int, default=5)
    ap.add_argument("--chunk-frames", type=int, default=5)
    ap.add_argument("--flow-shift", type=float, default=2.22)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--teacher-weight", type=float, default=1.0)
    ap.add_argument("--sigma-replay-weight", type=float, default=1.0,
                    help=("weight for matching the frozen teacher at every actual "
                          "solver sigma; the old diagnostic only matched sigma=0"))
    ap.add_argument("--boundary-weight", type=float, default=0.1)
    ap.add_argument("--paired-action-loss", action="store_true",
                    help=("match A/D teacher velocities from the same generated "
                          "latent/history at every solver sigma"))
    ap.add_argument("--paired-full-teacher", action="store_true",
                    help=("for paired loss, use original bidirectional H3 teacher "
                          "on the same generated state instead of a causal replay "
                          "teacher; this supplies non-collapsed action geometry"))
    ap.add_argument("--pair-actions", nargs=2, default=["A", "D"],
                    metavar=("ACTION_1", "ACTION_2"),
                    help="counterfactual pair used by --paired-action-loss (default: A D)")
    ap.add_argument("--action-dir-weight", type=float, default=0.1,
                    help="weight for cosine direction loss on the paired action delta")
    ap.add_argument("--action-mag-weight", type=float, default=0.1,
                    help="weight for relative action-delta magnitude loss")
    ap.add_argument("--pair-eps", type=float, default=1e-6,
                    help="epsilon used by the paired action magnitude/cosine losses")
    ap.add_argument("--latent-target-weight", type=float, default=0.0,
                    help=("optional endpoint MSE to the same-action H3 30-step latent "
                          "at the final solver step of each chunk"))
    ap.add_argument("--final-replay-weight", type=float, default=0.0,
                    help=("optional second-pass replay loss on the final generated chunk; "
                          "a minimal Stage2-inspired rollout-distribution diagnostic"))
    ap.add_argument("--final-replay-sigma", type=float, default=0.6,
                    help="noise level for --final-replay-weight (0..1)" )
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="save action_adapter.pt under step_XX every N optimizer steps; 0 disables")
    ap.add_argument("--train-h3-lora", action="store_true",
                    help="train selected released H3 action-LoRA matrices with the student")
    ap.add_argument("--h3-lora-blocks", default="tail8",
                    help="H3 LoRA blocks to train: tail8, 42-49, or a comma list")
    ap.add_argument("--h3-lora-lr", type=float, default=None,
                    help="learning rate for released H3 LoRA; defaults to --lr")
    ap.add_argument("--action-prefix-mode", choices=["own", "causal", "all"],
                    default="causal",
                    help=("action rows visible to current video queries; causal exposes "
                          "known past/current controls"))
    ap.add_argument("--action-feedback", action="store_true",
                    help=("restore the causal-safe H3 action-row to current-video "
                          "feedback edge"))
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    if len(args.teacher_dir) != len(args.actions):
        ap.error("--teacher-dir and --actions must have the same length")
    if args.optimizer_steps < 1 or args.solver_steps < 1 or args.chunk_frames < 1:
        ap.error("optimizer/solver/chunk steps must be positive")
    if args.action_dir_weight < 0 or args.action_mag_weight < 0 or args.pair_eps <= 0:
        ap.error("paired-action weights must be non-negative and pair-eps must be positive")
    if args.checkpoint_every < 0:
        ap.error("checkpoint-every must be non-negative")
    if args.h3_lora_lr is not None and args.h3_lora_lr <= 0:
        ap.error("h3-lora-lr must be positive")
    if args.action_gain_lr is not None and args.action_gain_lr <= 0:
        ap.error("action-gain-lr must be positive")
    if args.latent_target_weight < 0:
        ap.error("latent-target-weight must be non-negative")
    if args.final_replay_weight < 0 or not 0 <= args.final_replay_sigma <= 1:
        ap.error("final-replay-weight must be non-negative and sigma must be in [0,1]")
    if args.paired_action_loss and len(args.pair_actions) != 2:
        ap.error("--pair-actions requires exactly two action labels")
    if args.paired_full_teacher and not args.paired_action_loss:
        ap.error("--paired-full-teacher requires --paired-action-loss")
    try:
        action_gain_init = parse_action_gains(args.action_gain_init)
    except ValueError as exc:
        ap.error(str(exc))
    if args.train_causal_adapter and args.causal_adapter is None:
        ap.error("--train-causal-adapter requires --causal-adapter")
    if args.train_action_prefix and args.action_prefix_adapter is not None:
        ap.error("--train-action-prefix cannot be combined with --action-prefix-adapter")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    started = time.perf_counter()

    result = {
        "status": "running",
        "kind": "action-aware online self-rollout teacher-replay diagnostic",
        "objective": "student generated-history replay against frozen H3 teacher; no SGF/DMD",
        "teacher_dirs": [str(x) for x in args.teacher_dir],
        "actions": list(args.actions),
        "student_action_adapter": str(args.student_action_adapter),
        "causal_adapter": str(args.causal_adapter) if args.causal_adapter else None,
        "causal_adapter_scope": args.causal_adapter_scope,
        "action_gain_init": action_gain_init,
        "action_gain_lr": args.action_gain_lr,
        "action_gain_compute_dtype": args.action_gain_compute_dtype,
        "anchor_mode": args.anchor_mode,
        "train_causal_adapter": args.train_causal_adapter,
        "action_prefix_adapter": str(args.action_prefix_adapter) if args.action_prefix_adapter else None,
        "train_action_prefix": args.train_action_prefix,
        "freeze_action_residual": args.freeze_action_residual,
        "train_h3_lora": args.train_h3_lora,
        "h3_lora_blocks": args.h3_lora_blocks,
        "h3_lora_lr": args.h3_lora_lr,
        "latent_target_weight": args.latent_target_weight,
        "final_replay_weight": args.final_replay_weight,
        "final_replay_sigma": args.final_replay_sigma,
        "optimizer_steps": args.optimizer_steps,
        "solver_steps": args.solver_steps,
        "chunk_frames": args.chunk_frames,
        "history_chunks": args.history_chunks,
        "flow_shift": args.flow_shift,
        "lr": args.lr,
        "sigma_replay_weight": args.sigma_replay_weight,
        "teacher_weight": args.teacher_weight,
        "boundary_weight": args.boundary_weight,
        "paired_action_loss": args.paired_action_loss,
        "paired_full_teacher": args.paired_full_teacher,
        "pair_actions": [str(x).upper() for x in args.pair_actions],
        "action_dir_weight": args.action_dir_weight,
        "action_mag_weight": args.action_mag_weight,
        "pair_eps": args.pair_eps,
        "checkpoint_every": args.checkpoint_every,
        "action_prefix_mode": args.action_prefix_mode,
        "action_feedback": args.action_feedback,
        "seed": args.seed,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "rollout_trace": [],
        "loss_history": [],
    }

    def save_result():
        tmp = args.out_dir / "training.tmp.json"
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(args.out_dir / "training.json")

    save_result()
    try:
        # Load the released H3 model once.  The action residual is a separate
        # module, so toggling it off gives a frozen teacher without duplicating
        # the roughly 40-GiB DiT on another device.
        pipe = abot.load_pipeline(args.device)
        pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(
            ROOT / "checkpoints/H3-World/step-10000.safetensors"), hotload=True)
        pipe.dit.requires_grad_(False).eval()
        pipe.load_models_to_device(["dit"])
        h3_lora_params = []
        h3_lora_entries = []
        h3_lora_block_indices = []
        if args.train_h3_lora:
            h3_lora_block_indices = parse_block_indices(
                args.h3_lora_blocks, len(pipe.dit.blocks))
            h3_lora_params, h3_lora_entries = install_trainable_h3_lora(
                pipe, h3_lora_block_indices)
        causal_loaded = None
        qkv_modules = []
        qkv_params = []
        if args.causal_adapter is not None:
            from causal.pretrained_lora import load_adapter
            causal_loaded = load_adapter(pipe.dit, args.causal_adapter, args.device)
            qkv_modules = [pipe.dit.blocks[int(index)].attn.qkv_proj
                           for index in causal_loaded["block_indices"]]
            if args.train_causal_adapter:
                for module in qkv_modules:
                    module.lora_A.requires_grad_(True)
                    module.lora_B.requires_grad_(True)
                qkv_params = [parameter for module in qkv_modules
                              for parameter in (module.lora_A, module.lora_B)]

        def set_causal_adapter_enabled(enabled):
            for module in qkv_modules:
                module.enabled = bool(enabled)
        loaded = load_action_residual(pipe.dit, args.student_action_adapter, args.device)
        student_adapter = loaded["adapter"]
        student_adapter.gain_compute_dtype = args.action_gain_compute_dtype
        if action_gain_init:
            with torch.no_grad():
                for key, gain in action_gain_init.items():
                    student_adapter.action_gain[abot.S.KEYS9.index(key)] = gain
        student_adapter.requires_grad_(not args.freeze_action_residual)
        student_prefix_adapter = None
        if args.action_prefix_adapter is not None:
            loaded_prefix = load_action_prefix_residual(
                pipe.dit, args.action_prefix_adapter, args.device)
            student_prefix_adapter = loaded_prefix["adapter"]
            student_prefix_adapter.requires_grad_(True)
        elif args.train_action_prefix:
            prefix_indices = list(range(max(0, len(pipe.dit.blocks) - 8), len(pipe.dit.blocks)))
            student_prefix_adapter = CausalActionPrefixResidual(
                prefix_indices, len(abot.S.KEYS9), pipe.dit.hidden_size,
                device=args.device, dtype=next(pipe.dit.parameters()).dtype)
            student_prefix_adapter.requires_grad_(True)
        params = (list(qkv_params)
                  + [p for p in student_adapter.parameters() if p.requires_grad]
                  + ([p for p in student_prefix_adapter.parameters() if p.requires_grad]
                     if student_prefix_adapter is not None else [])
                  + list(h3_lora_params))
        if not params:
            raise RuntimeError("student action adapter has no trainable parameters")
        teacher_qkv_state = [p.detach().clone() for p in qkv_params]
        base_params = params[:-len(h3_lora_params)] if h3_lora_params else params
        optimizer_groups = []
        gain_params = ([student_adapter.action_gain]
                       if student_adapter.action_gain.requires_grad else [])
        gain_ids = {id(p) for p in gain_params}
        if gain_params:
            base_params = [p for p in base_params if id(p) not in gain_ids]
        if base_params:
            optimizer_groups.append({"params": base_params, "lr": args.lr})
        if gain_params:
            optimizer_groups.append({"params": gain_params,
                                     "lr": args.action_gain_lr or args.lr})
        if h3_lora_params:
            optimizer_groups.append({"params": h3_lora_params,
                                     "lr": args.h3_lora_lr or args.lr})
        optimizer = torch.optim.AdamW(optimizer_groups)
        pipe.scheduler.set_timesteps(args.solver_steps, shift=args.flow_shift)
        timesteps = list(pipe.scheduler.timesteps)
        sigmas = [float(x) / 1000.0 for x in timesteps]
        result["adapter_format"] = loaded.get("metadata", {}).get("format", "hidden_or_qkv")
        result["adapter_block_indices"] = loaded.get("block_indices")
        result["action_prefix_adapter_block_indices"] = (
            list(student_prefix_adapter.block_indices) if student_prefix_adapter is not None else None)
        result["action_prefix_trainable_parameters"] = (
            sum(p.numel() for p in student_prefix_adapter.parameters() if p.requires_grad)
            if student_prefix_adapter is not None else 0)
        result["causal_adapter_block_indices"] = (
            causal_loaded.get("block_indices") if causal_loaded else None)
        result["qkv_trainable_parameters"] = sum(p.numel() for p in qkv_params)
        result["h3_lora_trainable_parameters"] = sum(p.numel() for p in h3_lora_params)
        result["h3_lora_block_indices"] = h3_lora_block_indices
        result["h3_lora_module_names"] = [x["name"] for x in h3_lora_entries]
        result["trainable_parameters"] = sum(p.numel() for p in params)
        result["scheduler_timesteps"] = [float(x) for x in timesteps]
        result["scheduler_sigmas"] = sigmas

        cases = []
        for label, teacher_dir in zip(args.actions, args.teacher_dir):
            cond = torch.load(teacher_dir / "conditioning.pt", map_location="cpu", weights_only=True)
            common = move_tree(cond, args.device)
            initial = common["initial_noise"]
            if initial.ndim != 5 or initial.shape[0] != 1 or initial.shape[2] < args.chunk_frames:
                raise ValueError(f"{teacher_dir}: expected [1,C,T,H,W] 39-frame initial noise")
            latent_t = int(initial.shape[2])
            if latent_t != 12:
                raise ValueError(f"first prototype expects 12 latent frames, got {latent_t}")
            rows = (initial.shape[-2] // 2) * (initial.shape[-1] // 2)
            packed = expand_packed_two_anchors(common["packed"], frame_rows=rows)
            original_anchor = common["anchor"]
            dual_anchor = torch.cat((original_anchor, original_anchor.clone()), dim=0)
            action_cond = action_condition(label, latent_t, args.device, initial.dtype)
            teacher_latents = None
            if args.latent_target_weight > 0:
                latent_path = teacher_dir / "baseline_latents.pt"
                if not latent_path.exists():
                    raise ValueError(
                        f"{teacher_dir}: --latent-target-weight requires baseline_latents.pt")
                teacher_latents = torch.load(
                    latent_path, map_location=args.device, weights_only=True)
                if teacher_latents.shape != initial.shape:
                    raise ValueError(
                        f"{teacher_dir}: teacher latent shape {tuple(teacher_latents.shape)} "
                        f"does not match initial noise {tuple(initial.shape)}")
            cases.append({
                "label": label,
                "cond": common,
                "packed": packed,
                "original_anchor": original_anchor,
                "dual_anchor": dual_anchor,
                "action_cond": action_cond,
                "initial": initial,
                "audio": common["audio_noise"],
                "prompt": common["prompt_embeds"],
                "rows": rows,
                "latent_t": latent_t,
                "teacher_latents": teacher_latents,
            })
        result["latent_frames"] = cases[0]["latent_t"]
        result["num_chunks"] = (cases[0]["latent_t"] + args.chunk_frames - 1) // args.chunk_frames
        if any(c["latent_t"] != cases[0]["latent_t"] or c["initial"].shape != cases[0]["initial"].shape
               for c in cases):
            raise ValueError("all action conditioning clips must have the same latent shape")
        cases_by_action = {str(c["label"]).upper(): c for c in cases}
        if args.paired_action_loss:
            pair_labels = [str(x).upper() for x in args.pair_actions]
            if any(label not in cases_by_action for label in pair_labels):
                raise ValueError(
                    "--paired-action-loss requires conditioning cases for both "
                    f"{pair_labels}; got {sorted(cases_by_action)}")
            if pair_labels[0] == pair_labels[1]:
                raise ValueError("paired action labels must be different")
            # The pair is counterfactual only if it starts from the same world
            # state.  Conditioning files generated by the benchmark reuse the
            # same noise/prompt/image; reject accidental mismatches early.
            left, right = (cases_by_action[x] for x in pair_labels)
            # Prompt embeddings intentionally differ in the action sentence
            # rows; the scene/image part and all noise sources are shared. The
            # per-action prompt is what makes this a real counterfactual
            # rather than two identical forwards.
            for key in ("initial", "audio"):
                a, b = left[key], right[key]
                if a.shape != b.shape or not torch.equal(a, b):
                    raise ValueError(
                        f"paired cases {pair_labels} do not share identical {key}; "
                        "paired loss would not be counterfactual")
            if left["prompt"].shape != right["prompt"].shape:
                raise ValueError("paired action prompt embeddings have different shapes")
            result["pair_actions"] = pair_labels
            result["pair_conditioning_verified"] = True
        save_result()

        torch.cuda.reset_peak_memory_stats(args.device)
        for opt_step in range(args.optimizer_steps):
            optimizer.zero_grad(set_to_none=True)
            # Snapshot the current student adapter once for this optimizer
            # step. Teacher forwards then swap in the fixed initial causal-QKV
            # state and restore this student state without duplicating the DiT.
            student_qkv_state = [p.detach().clone() for p in qkv_params]
            action_case = cases[opt_step % len(cases)]
            label = action_case["label"]
            cache_student = H3ChunkCache(args.history_chunks, "cpu")
            cache_teacher = H3ChunkCache(args.history_chunks, "cpu")
            generated = []
            step_losses = []
            step_started = time.perf_counter()
            initial = action_case["initial"]
            for chunk in range(result["num_chunks"]):
                start = chunk * args.chunk_frames
                stop = min(start + args.chunk_frames, initial.shape[2])
                current = initial[:, :, start:stop].contiguous()
                if chunk == 0:
                    anchor = action_case["dual_anchor"]
                    anchor_frame_index = None
                    anchor_slot = 0
                else:
                    tail = generated[-1][:, :, -1:].detach()
                    if args.anchor_mode == "rgb":
                        # Match the visual tail16 training protocol: decode
                        # the generated prefix, select its last RGB frame,
                        # and run H3's image-conditioning encoder before
                        # returning to the DiT.  This is intentionally
                        # expensive but prevents a latent/RGB train-test
                        # anchor mismatch.
                        pipe.load_models_to_device(["video_vae"])
                        rgb_history = torch.cat(generated, dim=2).detach()
                        tail_anchor = last_frame_image_anchor(
                            pipe.video_vae, rgb_history, dtype=pipe.torch_dtype)
                        pipe.load_models_to_device(["dit"])
                    else:
                        tail_anchor = last_frame_anchor(tail)
                    anchor = torch.cat((action_case["original_anchor"][:action_case["rows"]],
                                        tail_anchor), dim=0)
                    anchor_frame_index = start - 1
                    anchor_slot = 1
                common = dict(
                    full_packed=action_case["packed"], prompt=action_case["prompt"],
                    anchor=anchor, audio=action_case["audio"], chunk_frames=args.chunk_frames,
                    anchor_frame_index=anchor_frame_index, anchor_slot=anchor_slot,
                    action_cond=action_case["action_cond"][start:stop],
                    action_prefix_mode=args.action_prefix_mode,
                    action_feedback=args.action_feedback,
                )
                for solver_index, timestep in enumerate(timesteps):
                    sigma = float(timestep) / 1000.0
                    # Keep the student and frozen teacher on the same visual
                    # adapter path for this sigma.  A commit-only run trains
                    # the action residual against the path used at rollout,
                    # instead of reusing an adapter trained with a different
                    # noisy score field.
                    set_causal_adapter_enabled(
                        args.causal_adapter_scope == "all"
                        or (args.causal_adapter_scope == "last_step_commit"
                            and solver_index == len(timesteps) - 1))
                    pair_outputs = {}
                    pair_loss_values = None
                    if args.paired_action_loss:
                        # Counterfactual pair: both actions see exactly this
                        # generated latent, the same detached student/teacher
                        # history cache and this solver sigma.  Only the action
                        # rows and action-specific packed metadata differ.
                        for pair_label in (str(x).upper() for x in args.pair_actions):
                            pair_case = cases_by_action[pair_label]
                            pair_common = dict(
                                full_packed=pair_case["packed"],
                                prompt=pair_case["prompt"], anchor=anchor,
                                audio=pair_case["audio"], chunk_frames=args.chunk_frames,
                                anchor_frame_index=anchor_frame_index,
                                anchor_slot=anchor_slot,
                                action_cond=pair_case["action_cond"][start:stop],
                                action_prefix_mode=args.action_prefix_mode,
                                action_feedback=args.action_feedback,
                            )
                            with action_adapters_state(student_adapter, student_prefix_adapter, True):
                                student_velocity = chunk_forward(
                                    pipe.dit, current, index=chunk, cache=cache_student,
                                    sigma=sigma, action_adapter=student_adapter,
                                    action_prefix_adapter=student_prefix_adapter,
                                    allow_grad_read=True, use_gradient_checkpointing=True,
                                    use_gradient_checkpointing_offload=True, **pair_common)
                            with teacher_model_state(qkv_params, teacher_qkv_state,
                                                     student_qkv_state, h3_lora_entries):
                                with action_adapters_state(student_adapter, student_prefix_adapter, False):
                                    with torch.no_grad():
                                        if args.paired_full_teacher:
                                            # The full teacher must use the
                                            # released bidirectional H3 path;
                                            # disable the causal visual adapter
                                            # while retaining the frozen H3
                                            # action LoRA and original anchor.
                                            set_causal_adapter_enabled(False)
                                            teacher_history = (torch.cat(generated, dim=2)
                                                               if generated else
                                                               current[:, :, :0])
                                            teacher_velocity = full_teacher_forward(
                                                pipe.dit, current.detach(),
                                                history=teacher_history,
                                                full_packed=pair_case["cond"]["packed"],
                                                prompt=pair_case["prompt"],
                                                anchor=pair_case["original_anchor"],
                                                audio=pair_case["audio"], sigma=sigma,
                                                action_cond=pair_case["action_cond"])
                                            # Restore the student path before
                                            # the next counterfactual branch.
                                            set_causal_adapter_enabled(
                                                args.causal_adapter_scope == "all"
                                                or (args.causal_adapter_scope == "last_step_commit"
                                                    and solver_index == len(timesteps) - 1))
                                        else:
                                            teacher_velocity = chunk_forward(
                                                pipe.dit, current.detach(), index=chunk,
                                                cache=cache_teacher, sigma=sigma,
                                                action_adapter=student_adapter, **pair_common)
                            finite_or_raise(
                                f"teacher {pair_label} sigma {chunk} step {solver_index}",
                                teacher_velocity)
                            pair_outputs[pair_label] = (student_velocity, teacher_velocity)
                        pair_labels = [str(x).upper() for x in args.pair_actions]
                        student_a, teacher_a = pair_outputs[pair_labels[0]]
                        student_d, teacher_d = pair_outputs[pair_labels[1]]
                        replay_sigma, loss_dir, loss_mag = paired_action_losses(
                            student_a, student_d, teacher_a, teacher_d, eps=args.pair_eps)
                        loss_sigma = (
                            args.sigma_replay_weight * replay_sigma
                            + args.action_dir_weight * loss_dir
                            + args.action_mag_weight * loss_mag
                        )
                        finite_or_raise(
                            f"student pair loss {chunk} step {solver_index}", loss_sigma)
                        # Use a local loss for every actual solver sigma. The
                        # generated state is detached after the Euler update,
                        # while the adapter gradients from both counterfactuals
                        # are retained in this optimizer step.
                        loss_sigma.backward(
                            retain_graph=(args.latent_target_weight > 0
                                          and solver_index == len(timesteps) - 1))
                        if label.upper() in pair_outputs:
                            velocity = pair_outputs[label.upper()][0]
                        else:
                            # This branch keeps the script well-defined for a
                            # non A/D rollout action while still training the
                            # requested pair objective.
                            with action_adapters_state(student_adapter, student_prefix_adapter, True):
                                velocity = chunk_forward(
                                    pipe.dit, current, index=chunk, cache=cache_student,
                                    sigma=sigma, action_adapter=student_adapter,
                                    action_prefix_adapter=student_prefix_adapter,
                                    allow_grad_read=True, use_gradient_checkpointing=True,
                                    use_gradient_checkpointing_offload=True, **common)
                        pair_loss_values = {
                            "sigma_teacher_loss": float(replay_sigma.detach()),
                            "action_dir_loss": float(loss_dir.detach()),
                            "action_mag_loss": float(loss_mag.detach()),
                            "sigma_total_loss": float(loss_sigma.detach()),
                        }
                        step_losses.append({
                            "chunk": chunk, "solver_index": solver_index,
                            "sigma": sigma, **pair_loss_values,
                            "history_source": "online_student_shared_pair_state",
                            "history_detached": True, "student_forward_grad": True,
                            "action": label,
                        })
                        del replay_sigma, loss_dir, loss_mag, loss_sigma
                    else:
                        with action_adapters_state(student_adapter, student_prefix_adapter, True):
                            velocity = chunk_forward(
                                pipe.dit, current, index=chunk, cache=cache_student,
                                sigma=sigma, action_adapter=student_adapter,
                                    action_prefix_adapter=student_prefix_adapter,
                                allow_grad_read=True, use_gradient_checkpointing=True,
                                use_gradient_checkpointing_offload=True, **common)
                        # Match the frozen H3 score field at the same sigma that
                        # the student solver actually uses. Matching only a final
                        # clean replay leaves a four-step solver unconstrained at
                        # its noisy evaluation points and is not few-step
                        # distillation.
                        with teacher_model_state(qkv_params, teacher_qkv_state,
                                                 student_qkv_state, h3_lora_entries):
                            with action_adapters_state(student_adapter, student_prefix_adapter, False):
                                with torch.no_grad():
                                    teacher_sigma_velocity = chunk_forward(
                                        pipe.dit, current.detach(), index=chunk,
                                        cache=cache_teacher, sigma=sigma,
                                        action_adapter=student_adapter, **common)
                        loss_sigma = F.mse_loss(
                            velocity.float(), teacher_sigma_velocity.float())
                        finite_or_raise(
                            f"teacher sigma {chunk} step {solver_index}",
                            teacher_sigma_velocity)
                        (args.sigma_replay_weight * loss_sigma).backward(
                            retain_graph=(args.latent_target_weight > 0
                                          and solver_index == len(timesteps) - 1))
                        step_losses.append({
                            "chunk": chunk,
                            "solver_index": solver_index,
                            "sigma": sigma,
                            "sigma_teacher_loss": float(loss_sigma.detach()),
                            "history_source": "online_student",
                            "history_detached": True,
                            "student_forward_grad": True,
                            "action": label,
                        })
                        del teacher_sigma_velocity, loss_sigma
                    next_current = pipe.scheduler.step(velocity, timestep, current)
                    if (args.latent_target_weight > 0
                            and solver_index == len(timesteps) - 1):
                        target = action_case["teacher_latents"][:, :, start:stop]
                        latent_target_loss = F.mse_loss(
                            next_current.float(), target.float())
                        (args.latent_target_weight * latent_target_loss).backward()
                        step_losses.append({
                            "chunk": chunk,
                            "latent_target_loss": float(latent_target_loss.detach()),
                            "loss": float((args.latent_target_weight
                                           * latent_target_loss).detach()),
                            "action": label,
                        })
                        del latent_target_loss, target
                    current = next_current.detach()
                    finite_or_raise(f"student chunk {chunk} step {solver_index}", current)
                    if args.paired_action_loss:
                        # Explicitly drop the counterfactual graphs after the
                        # local backward before advancing to the next sigma.
                        del pair_outputs
                generated.append(current)
                set_causal_adapter_enabled(
                    args.causal_adapter_scope in ("all", "commit", "last_step_commit"))
                # Teacher replay is frozen H3 on the same generated current
                # chunk and its own detached raw-KV history.  Student replay
                # retains the graph through current and reads detached KV.
                with teacher_model_state(qkv_params, teacher_qkv_state,
                                         student_qkv_state, h3_lora_entries):
                    with action_adapters_state(student_adapter, student_prefix_adapter, False):
                        with torch.no_grad():
                            teacher_velocity = chunk_forward(
                                pipe.dit, current.detach(), index=chunk, cache=cache_teacher,
                                sigma=0.0, action_adapter=student_adapter, **common)
                with action_adapters_state(student_adapter, student_prefix_adapter, True):
                    student_velocity = chunk_forward(
                        pipe.dit, current.detach(), index=chunk, cache=cache_student,
                        sigma=0.0, action_adapter=student_adapter,
                                    action_prefix_adapter=student_prefix_adapter,
                        allow_grad_read=True, use_gradient_checkpointing=True,
                        use_gradient_checkpointing_offload=True, **common)
                teacher_velocity = teacher_velocity.detach()
                loss_teacher = F.mse_loss(student_velocity.float(), teacher_velocity.float())
                # Keep a small first-frame term so the online update does not
                # improve only the interior while worsening chunk boundaries.
                boundary = min(1, current.shape[2])
                loss_boundary = F.mse_loss(
                    student_velocity[:, :, :boundary].float(),
                    teacher_velocity[:, :, :boundary].float())
                loss = args.teacher_weight * loss_teacher + args.boundary_weight * loss_boundary
                loss.backward()
                # Commit only after replay: the current chunk must not read
                # its own just-generated K/V. Both caches are built from the
                # student's generated latent, and all committed tensors are
                # detached before the next chunk.
                with torch.no_grad():
                    with action_adapters_state(student_adapter, student_prefix_adapter, True):
                        chunk_forward(pipe.dit, current.detach(), index=chunk,
                                      cache=cache_student, sigma=0.0, commit=True,
                                      action_adapter=student_adapter, **common)
                    with teacher_model_state(qkv_params, teacher_qkv_state,
                                             student_qkv_state, h3_lora_entries):
                        with action_adapters_state(student_adapter, student_prefix_adapter, False):
                            chunk_forward(pipe.dit, current.detach(), index=chunk,
                                          cache=cache_teacher, sigma=0.0, commit=True,
                                          action_adapter=student_adapter, **common)
                step_losses.append({
                    "chunk": chunk,
                    "teacher_loss": float(loss_teacher.detach()),
                    "boundary_loss": float(loss_boundary.detach()),
                    "loss": float(loss.detach()),
                    "history_source": "online_student",
                    "history_detached": True,
                    "student_forward_grad": True,
                    "commit_no_grad": True,
                    "action": label,
                })
                result["rollout_trace"].append({
                    "chunk": chunk,
                    "action": label,
                    "sigma_teacher_loss_mean": float(np.mean([
                        x["sigma_teacher_loss"] for x in step_losses
                        if x.get("chunk") == chunk and "sigma_teacher_loss" in x] or [0.0])),
                    "clean_teacher_loss": float(loss_teacher.detach()),
                    "boundary_loss": float(loss_boundary.detach()),
                })
                del teacher_velocity, student_velocity, loss
            if args.final_replay_weight > 0 and generated:
                # Minimal two-pass Stage2-style diagnostic: the rollout above
                # is detached, then its final generated chunk is noised and
                # replayed with the same generated history.  This targets the
                # student's own state distribution rather than only the noisy
                # states encountered while solving the chunk.
                replay_student = H3ChunkCache(args.history_chunks, "cpu")
                replay_teacher = H3ChunkCache(args.history_chunks, "cpu")
                for history_index, history_latent in enumerate(generated[:-1]):
                    history_start = history_index * args.chunk_frames
                    history_stop = min(history_start + args.chunk_frames, initial.shape[2])
                    if history_index == 0:
                        history_anchor = action_case["dual_anchor"]
                        history_anchor_index = None
                        history_anchor_slot = 0
                    else:
                        history_anchor = torch.cat((
                            action_case["original_anchor"][:action_case["rows"]],
                            last_frame_anchor(generated[history_index - 1][:, :, -1:])), dim=0)
                        history_anchor_index = history_start - 1
                        history_anchor_slot = 1
                    history_common = dict(
                        full_packed=action_case["packed"],
                        prompt=action_case["prompt"],
                        anchor=history_anchor,
                        audio=action_case["audio"],
                        chunk_frames=args.chunk_frames,
                        anchor_frame_index=history_anchor_index,
                        anchor_slot=history_anchor_slot,
                        action_cond=action_case["action_cond"][history_start:history_stop],
                        action_prefix_mode=args.action_prefix_mode,
                        action_feedback=args.action_feedback,
                    )
                    with torch.no_grad():
                        with action_adapters_state(student_adapter, student_prefix_adapter, True):
                            chunk_forward(pipe.dit, history_latent, index=history_index,
                                          cache=replay_student, sigma=0.0, commit=True,
                                          action_adapter=student_adapter, **history_common)
                        with teacher_model_state(qkv_params, teacher_qkv_state,
                                                 student_qkv_state, h3_lora_entries):
                            with action_adapters_state(student_adapter, student_prefix_adapter, False):
                                chunk_forward(pipe.dit, history_latent, index=history_index,
                                              cache=replay_teacher, sigma=0.0, commit=True,
                                              action_adapter=student_adapter, **history_common)
                final_index = len(generated) - 1
                final_start = final_index * args.chunk_frames
                final_stop = min(final_start + args.chunk_frames, initial.shape[2])
                if final_index == 0:
                    final_anchor = action_case["dual_anchor"]
                    final_anchor_index = None
                    final_anchor_slot = 0
                else:
                    final_anchor = torch.cat((
                        action_case["original_anchor"][:action_case["rows"]],
                        last_frame_anchor(generated[-2][:, :, -1:])), dim=0)
                    final_anchor_index = final_start - 1
                    final_anchor_slot = 1
                final_common = dict(
                    full_packed=action_case["packed"], prompt=action_case["prompt"],
                    anchor=final_anchor, audio=action_case["audio"],
                    chunk_frames=args.chunk_frames,
                    anchor_frame_index=final_anchor_index,
                    anchor_slot=final_anchor_slot,
                    action_cond=action_case["action_cond"][final_start:final_stop],
                    action_prefix_mode=args.action_prefix_mode,
                    action_feedback=args.action_feedback,
                )
                final_clean = generated[-1]
                final_noise = torch.randn_like(final_clean)
                final_noisy = ((1.0 - args.final_replay_sigma) * final_clean
                               + args.final_replay_sigma * final_noise)
                with action_adapters_state(student_adapter, student_prefix_adapter, True):
                    final_student_velocity = chunk_forward(
                        pipe.dit, final_noisy, index=final_index,
                        cache=replay_student, sigma=args.final_replay_sigma,
                        action_adapter=student_adapter,
                        action_prefix_adapter=student_prefix_adapter,
                        allow_grad_read=True, use_gradient_checkpointing=True,
                        use_gradient_checkpointing_offload=True, **final_common)
                with teacher_model_state(qkv_params, teacher_qkv_state,
                                         student_qkv_state, h3_lora_entries):
                    with action_adapters_state(student_adapter, student_prefix_adapter, False):
                        with torch.no_grad():
                            final_teacher_velocity = chunk_forward(
                                pipe.dit, final_noisy.detach(), index=final_index,
                                cache=replay_teacher, sigma=args.final_replay_sigma,
                                action_adapter=student_adapter, **final_common)
                final_replay_loss = F.mse_loss(
                    final_student_velocity.float(), final_teacher_velocity.float())
                (args.final_replay_weight * final_replay_loss).backward()
                step_losses.append({
                    "chunk": final_index,
                    "final_replay_loss": float(final_replay_loss.detach()),
                    "loss": float((args.final_replay_weight * final_replay_loss).detach()),
                    "action": label,
                })
                del (replay_student, replay_teacher, final_noise, final_noisy,
                     final_student_velocity, final_teacher_velocity, final_replay_loss)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite adapter gradient")
            optimizer.step()
            mean_loss = float(np.mean([
                (x["loss"] if "loss" in x else x.get("sigma_total_loss",
                 args.sigma_replay_weight * x["sigma_teacher_loss"]))
                for x in step_losses
            ]))
            sigma_mean = float(np.mean([
                x["sigma_teacher_loss"] for x in step_losses
                if "sigma_teacher_loss" in x
            ] or [0.0]))
            dir_mean = float(np.mean([
                x["action_dir_loss"] for x in step_losses
                if "action_dir_loss" in x
            ] or [0.0]))
            mag_mean = float(np.mean([
                x["action_mag_loss"] for x in step_losses
                if "action_mag_loss" in x
            ] or [0.0]))
            boundary_mean = float(np.mean([
                x["boundary_loss"] for x in step_losses
                if "boundary_loss" in x
            ] or [0.0]))
            latent_target_mean = float(np.mean([
                x["latent_target_loss"] for x in step_losses
                if "latent_target_loss" in x
            ] or [0.0]))
            final_replay_mean = float(np.mean([
                x["final_replay_loss"] for x in step_losses
                if "final_replay_loss" in x
            ] or [0.0]))
            result["loss_history"].append({
                "optimizer_step": opt_step + 1,
                "action": label,
                "mean_loss": mean_loss,
                "sigma_teacher_loss_mean": sigma_mean,
                "action_dir_loss_mean": dir_mean,
                "action_mag_loss_mean": mag_mean,
                "boundary_loss_mean": boundary_mean,
                "latent_target_loss_mean": latent_target_mean,
                "final_replay_loss_mean": final_replay_mean,
                "grad_norm": float(grad_norm),
                "action_gains": dict(zip(abot.S.KEYS9, student_adapter.action_gain.detach().cpu().tolist())),
                "seconds": time.perf_counter() - step_started,
                "student_cache_bytes": cache_student.nbytes,
                "teacher_cache_bytes": cache_teacher.nbytes,
            })
            print(f"[online] step={opt_step + 1}/{args.optimizer_steps} action={label} "
                  f"loss={mean_loss:.6f} replay={sigma_mean:.6f} "
                  f"dir={dir_mean:.6f} mag={mag_mean:.6f} "
                  f"target={latent_target_mean:.6f} "
                  f"final={final_replay_mean:.6f} "
                  f"grad={float(grad_norm):.4f}", flush=True)
            save_result()
            if args.checkpoint_every and ((opt_step + 1) % args.checkpoint_every == 0):
                checkpoint_dir = args.out_dir / f"step_{opt_step + 1:02d}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_metadata = dict(result)
                checkpoint_metadata["checkpoint_optimizer_step"] = opt_step + 1
                save_action_residual(
                    checkpoint_dir / "action_adapter.pt", student_adapter,
                    checkpoint_metadata)
                if student_prefix_adapter is not None:
                    save_action_prefix_residual(
                        checkpoint_dir / "action_prefix_adapter.pt",
                        student_prefix_adapter, checkpoint_metadata)
                if h3_lora_entries:
                    save_h3_lora_adapter(
                        checkpoint_dir / "h3_lora_adapter.pt",
                        [(x["name"], x["parameter_a"], x["parameter_b"])
                         for x in h3_lora_entries], checkpoint_metadata)
            cache_student.clear()
            cache_teacher.clear()

        metadata = {k: v for k, v in result.items() if k not in ("rollout_trace", "loss_history")}
        save_action_residual(args.out_dir / "action_adapter.pt", student_adapter, metadata)
        if student_prefix_adapter is not None:
            save_action_prefix_residual(
                args.out_dir / "action_prefix_adapter.pt", student_prefix_adapter, metadata)
        if h3_lora_entries:
            save_h3_lora_adapter(
                args.out_dir / "h3_lora_adapter.pt",
                [(x["name"], x["parameter_a"], x["parameter_b"])
                 for x in h3_lora_entries], metadata)
        if qkv_modules:
            save_adapters(args.out_dir / "causal_adapter.pt", qkv_modules,
                          causal_loaded["block_indices"], metadata)
        result.update({
            "status": "complete",
            "wall_seconds": time.perf_counter() - started,
            "allocated_peak_MiB": torch.cuda.max_memory_allocated(args.device) / 2**20,
            "reserved_peak_MiB": torch.cuda.max_memory_reserved(args.device) / 2**20,
            "student_cache_peak_bytes": max(
                (x["student_cache_bytes"] for x in result["loss_history"]), default=0),
            "teacher_cache_peak_bytes": max(
                (x["teacher_cache_bytes"] for x in result["loss_history"]), default=0),
            "completed_at": datetime.now().astimezone().isoformat(),
        })
        save_result()
        print(json.dumps({k: v for k, v in result.items() if k not in ("rollout_trace",)}, indent=2))
    except KeyboardInterrupt:
        result.update(status="interrupted", wall_seconds=time.perf_counter() - started)
        save_result()
        raise
    except Exception as exc:
        result.update(status="failed", error=repr(exc), wall_seconds=time.perf_counter() - started)
        save_result()
        raise


if __name__ == "__main__":
    main()
