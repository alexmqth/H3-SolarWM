"""EXP-024 CPU-auditable C1/C2 ordinary-FM coverage over frozen EXP-023 code.

This file defines the prospective GPU profile step but performs no GPU work on
import. The released H3/action LoRA and V3 topology are not changed here.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
EXP23 = ROOT / "submission/experiments/EXP-023_v3_fm_small_data"
RUNTIME = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime"
PREFIX = ROOT / "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate"
for path in (EXP23, RUNTIME / "code", RUNTIME / "code/causal",
             RUNTIME / "DiffSynth-Studio-h3-v2", PREFIX):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from causal.h3_cached import H3ChunkCache  # noqa: E402
from causal.local_topology import visible_inputs  # noqa: E402
from fm_run import cache_signature, verified_attention_scope  # noqa: E402
from interval_train import interval_train  # noqa: E402

BLOCKS = {"C1": (0, 0, 12), "C2": (1, 12, 17)}


def cache_integrity(cache: H3ChunkCache) -> tuple:
    """EXP-023 structural signature plus full CPU K/V/RoPE byte digest.

    The inherited signature checks entry identities/shapes and the first eight
    K/V values. The digest closes its blind spot for later K/V and RoPE values;
    it hashes the raw tensor payload without duplicating the whole ~6.3 GiB KV.
    """
    structural = cache_signature(cache)
    digest = hashlib.sha256()
    for layer in sorted(cache.layers):
        for entry in cache.layers[layer]:
            digest.update(f"{layer}:{entry.index}".encode())
            for item in (entry.key, entry.value, entry.rope):
                value = item.detach().cpu().contiguous()
                digest.update(str((tuple(value.shape), str(value.dtype))).encode())
                digest.update(memoryview(value.view(torch.uint8).numpy()))
    return structural, digest.hexdigest()


def balanced_block_order(clip_ids: list[str], *, seed: int = 13,
                         epochs: int = 1) -> list[tuple[str, str]]:
    """Each clip contributes one C1 and one C2 update per epoch.

    The loss is mean per latent element, so this 50:50 update schedule avoids
    silently weighting the 12-frame C1 2.4x more than the 5-frame C2.
    """
    if not clip_ids or len(set(clip_ids)) != len(clip_ids) or epochs < 1:
        raise ValueError("unique nonempty clips and positive epochs required")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    order = []
    for _ in range(epochs):
        for position in torch.randperm(len(clip_ids), generator=gen).tolist():
            pair = ("C1", "C2") if bool(torch.randint(2, (), generator=gen)) else ("C2", "C1")
            order.extend((clip_ids[position], block) for block in pair)
    return order


def prepare_visible(data: dict, block: str) -> tuple[dict, torch.Tensor]:
    """Physically remove future action/video before either text or DiT call."""
    if block not in BLOCKS:
        raise ValueError(f"unsupported block {block}")
    _, _, stop = BLOCKS[block]
    z = data["z"]
    if (z.ndim != 5 or z.shape[2] != 17
            or (z.shape[-2:] != (30, 52) and not data.get("cpu_toy"))):
        raise ValueError("expected 17 native temporal latents")
    rows = (z.shape[-2] // 2) * (z.shape[-1] // 2)
    packed, prompt = visible_inputs(data["full_packed"], data["prompt"], stop, rows)
    if len(packed["action_text_rows"]) != stop:
        raise RuntimeError("future action rows were not removed")
    if int(packed["seq_len"]) != int(packed["action_video_start"]) + stop * rows:
        raise RuntimeError("future video rows were not removed")
    return packed, prompt


def fm_step(model, data: dict, optimizer: torch.optim.Optimizer, *, block: str,
            noise: torch.Tensor, sigma: float, checkpoint: bool = True,
            offload: bool = True, audit=None, phase_hook=None,
            stage_callback=None, pre_optimizer_gate=None) -> dict:
    """One mean-per-element FM update; C1/C2 are sampled with equal frequency.

    C1 has an empty history cache. C2 commits GT C1 at sigma=0 under no_grad.
    The attention context encloses both the forward and checkpoint backward.
    An evaluation/profiling caller must construct the optimizer over *only*
    the new causal adapter; no base/released-action parameters may be passed.
    """
    if block not in BLOCKS:
        raise ValueError(f"unsupported block {block}")
    if not 0.0 < float(sigma) < 1.0:
        raise ValueError("training sigma must be inside (0,1)")
    if offload and not checkpoint:
        raise ValueError("offload requires checkpoint")
    index, start, stop = BLOCKS[block]
    clean = data["z"][:, :, start:stop]
    if noise.shape != clean.shape or noise.dtype != torch.float32:
        raise ValueError("noise must be independent FP32 tensor shaped like clean block")
    packed, prompt = prepare_visible(data, block)
    cache = H3ChunkCache(5, "cpu")
    optimizer.zero_grad(set_to_none=True)
    if phase_hook is not None:
        phase_hook("commit" if block == "C2" else "forward")
    with verified_attention_scope(audit):
        if block == "C2":
            history_packed, history_prompt = prepare_visible(data, "C1")
            if stage_callback is not None:
                stage_callback("commit", "before")
            with torch.no_grad():
                interval_train(model, data["z"][:, :, :12], start=0, index=0,
                               cache=cache, full_packed=history_packed,
                               prompt=history_prompt, anchor=data["anchor"],
                               audio=data["audio"], sigma=0.0, commit=True)
            if stage_callback is not None:
                stage_callback("commit", "after")
            if cache.nbytes == 0:
                raise RuntimeError("C2 GT C1 clean commit produced no raw KV")
        history_bytes = cache.nbytes
        history_integrity = cache_integrity(cache)
        if phase_hook is not None:
            phase_hook("forward")
        z_sigma = (1.0 - sigma) * clean + sigma * noise
        if stage_callback is not None:
            stage_callback("forward", "before")
        prediction = interval_train(model, z_sigma, start=start, index=index,
                                    cache=cache, full_packed=packed,
                                    prompt=prompt, anchor=data["anchor"],
                                    audio=data["audio"], sigma=sigma,
                                    checkpoint=checkpoint, offload=offload)
        if stage_callback is not None:
            stage_callback("forward", "after")
        target = noise - clean
        loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite FM loss")
        if phase_hook is not None:
            phase_hook("backward")
        if stage_callback is not None:
            stage_callback("backward", "before")
        loss.backward()
        if stage_callback is not None:
            stage_callback("backward", "after")
    if cache_integrity(cache) != history_integrity:
        raise RuntimeError("training forward/backward mutated history KV content or structure")
    params = [p for group in optimizer.param_groups for p in group["params"]]
    if not params or any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
        raise FloatingPointError("missing/nonfinite adapter gradient")
    grad_norm = float(torch.sqrt(sum(p.grad.float().square().sum() for p in params)))
    if grad_norm <= 0:
        raise FloatingPointError("zero adapter gradient")
    if pre_optimizer_gate is not None:
        pre_optimizer_gate(block, cache)
    if phase_hook is not None:
        phase_hook("optimizer")
    if stage_callback is not None:
        stage_callback("optimizer", "before")
    optimizer.step()
    if stage_callback is not None:
        stage_callback("optimizer", "after")
    return {"block": block, "sigma": float(sigma), "loss": float(loss.detach()),
            "grad_l2": grad_norm, "cpu_kv_bytes": history_bytes,
            "loss_normalization": "mean_per_latent_element_equal_block_sampling"}
