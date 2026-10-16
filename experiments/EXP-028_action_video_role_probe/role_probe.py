"""EXP-028 exact LoRA-B role contributions; no dense base-weight gradients.

The hook is on the FP32 residual before the production cast to x.dtype.
Checkpoint replay is counted explicitly: only hooks receiving a cotangent
contribute to the final role sum. The caller sets one immutable role layout
per differentiable DiT call and clears it after backward.
"""
from __future__ import annotations

from collections import Counter
import torch


ROLE_NAMES = ("action_history", "action_current", "video_history",
              "video_current", "non_action_prefix", "other")


def role_layout(packed: dict, *, start: int, visible: int, rows: int, total: int | None = None):
    prefix = int(packed["action_video_start"])
    n = prefix + visible * rows
    if total is not None and total < n:
        raise ValueError("total is shorter than packed visible sequence")
    labels = torch.full((n if total is None else total,), 5, dtype=torch.int8)
    labels[:prefix] = 4
    spans = packed["action_text_rows"]
    if len(spans) < visible or start > visible:
        raise ValueError("missing action spans")
    for i, (lo, hi) in enumerate(spans.tolist()):
        if lo < 0 or hi > prefix or lo >= hi:
            raise ValueError("action span outside prefix")
        labels[lo:hi] = 0 if i < start else 1
    if start:
        labels[prefix:prefix + start * rows] = 2
    labels[prefix + start * rows:prefix + visible * rows] = 3
    if (labels < 0).any() or (labels > 5).any():
        raise AssertionError("non-disjoint role labels")
    return labels


def original_layout(packed: dict, rows: int):
    return role_layout(packed, start=12, visible=17, rows=rows)


def v3_layout(packed: dict, rows: int):
    # V3's visible layout contains 17 action spans but only 5 video frames.
    prefix = int(packed["action_video_start"])
    labels = torch.full((prefix + 5 * rows,), 5, dtype=torch.int8)
    labels[:prefix] = 4
    spans = packed["action_text_rows"]
    if len(spans) != 17:
        raise ValueError("expected 17 action spans")
    for i, (lo, hi) in enumerate(spans.tolist()):
        if lo < 0 or hi > prefix or lo >= hi:
            raise ValueError("bad action span")
        labels[lo:hi] = 0 if i < 12 else 1
    labels[prefix:] = 3
    return labels


class RoleAccumulator:
    def __init__(self, labels: torch.Tensor, out_features: int, rank: int, scale: float,
                 tile: int = 256):
        self.labels = labels.cpu()
        self.out_features = out_features
        self.rank = rank
        self.scale = scale
        self.tile = tile
        self.values = {r: torch.zeros(out_features, rank, dtype=torch.float32)
                       for r in ROLE_NAMES}
        self.counts = Counter(ROLE_NAMES[int(x)] for x in self.labels.tolist())
        self.forward_invocations = 0
        self.cotangent_invocations = 0

    def add(self, u: torch.Tensor, d: torch.Tensor):
        if u.shape != (len(self.labels), self.rank) or d.shape != (len(self.labels), self.out_features):
            raise ValueError(f"role/cotangent shape mismatch: {u.shape}, {d.shape}, {len(self.labels)}")
        self.cotangent_invocations += 1
        for lo in range(0, len(self.labels), self.tile):
            hi = min(lo + self.tile, len(self.labels))
            roles = self.labels[lo:hi].to(u.device)
            for i, role in enumerate(ROLE_NAMES):
                use = roles == i
                if bool(use.any()):
                    product = d[lo:hi][use].float().T @ u[lo:hi][use].float()
                    self.values[role] += (self.scale * product).detach().cpu()

    def check(self, reference: torch.Tensor):
        result = sum(self.values.values())
        ref = reference.detach().float().cpu()
        diff = result - ref
        rms = float(diff.square().mean().sqrt())
        den = max(float(ref.square().mean().sqrt()), 1e-12)
        rel = rms / den
        maxerr = float(diff.abs().max())
        maxref = float(ref.abs().max())
        passed = rel <= 1e-3 and maxerr <= 1e-3 * max(1.0, maxref)
        return dict(pass_role_sum=passed, relative_rms=rel, max_abs=maxerr,
                    reference_norm=float(ref.norm()), role_sum_norm=float(result.norm()),
                    forward_invocations=self.forward_invocations,
                    cotangent_invocations=self.cotangent_invocations,
                    role_counts=dict(self.counts))


def diagnostic_forward(self, x):
    if not self.enabled:
        return self.base(x)
    u = x.float() @ self.lora_A.T
    residual = (u @ self.lora_B.T) * self.scale
    acc = self.role_accumulator
    if acc is not None:
        acc.forward_invocations += 1
        # In no-grad checkpoint prepass, no autograd hook can be registered.
        if residual.requires_grad:
            residual.register_hook(lambda d, u=u.detach(), acc=acc: (acc.add(u, d), d)[1])
    return self.base(x) + residual.to(x.dtype)
