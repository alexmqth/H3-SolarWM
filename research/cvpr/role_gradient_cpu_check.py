"""Tiny CPU-only occurrence-role gradient and LoRA chain-rule checks.

No H3 model, dataset, checkpoint, optimizer, or CUDA is loaded.
"""
from __future__ import annotations
import json
import hashlib
import os
import resource
import time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
START = time.perf_counter()
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
torch.set_num_threads(4)
torch.manual_seed(17)
DTYPE = torch.float64
IN, OUT, RANK = 5, 6, 2
SCALE = 1.0 / RANK
ROLES = ("action", "video", "prefix", "other")
X1 = torch.randn(12, IN, dtype=DTYPE)
X2 = torch.randn(9, IN, dtype=DTYPE)
HISTORY_X = torch.randn(7, IN, dtype=DTYPE)
LABELS1 = ("action", "video", "video", "prefix", "other", "action",
           "video", "prefix", "other", "video", "other", "other")
LABELS2 = ("other", "video", "action", "prefix", "video", "other",
           "prefix", "video", "action")
assert len(LABELS1) == len(X1) and len(LABELS2) == len(X2)
assert set(LABELS1) == set(LABELS2) == set(ROLES)
FROZEN = torch.randn(OUT, IN, dtype=DTYPE)
A0 = torch.randn(RANK, IN, dtype=DTYPE) * .2
B1 = torch.randn(OUT, RANK, dtype=DTYPE) * .2


def loss_for(W):
    # A historical writer runs, but its output is detached before the C2 loss.
    history = (HISTORY_X @ W.T).detach()
    y1 = X1 @ W.T
    y2 = X2 @ W.T
    h = history.mean(dim=0)
    loss = ((y1 + .07*h)**2).mean() + .4*(y2**2).mean() + .03*y1.mean()*y2.mean()
    return loss, (y1, y2)


def role_dense(deltas):
    contributions = {}
    for role in ROLES:
        g = torch.zeros(OUT, IN, dtype=DTYPE)
        for delta, x, labels in zip(deltas, (X1, X2), (LABELS1, LABELS2)):
            mask = torch.tensor([label == role for label in labels], dtype=torch.bool)
            g += delta[mask].T @ x[mask]
        contributions[role] = g
    return contributions


def max_abs(x):
    return float(x.detach().abs().max())


def case(name, B0):
    A = A0.clone().requires_grad_(True)
    B = B0.clone().requires_grad_(True)
    effective = FROZEN + SCALE*(B @ A)
    dense = effective.detach().clone().requires_grad_(True)
    dense_loss, dense_outputs = loss_for(dense)
    dense_deltas = torch.autograd.grad(dense_loss, dense_outputs, retain_graph=True)
    (dense_grad,) = torch.autograd.grad(dense_loss, dense)
    roles = role_dense(dense_deltas)
    role_sum = sum(roles.values())
    assert torch.allclose(role_sum, dense_grad, rtol=1e-12, atol=1e-12)

    lora_loss, lora_outputs = loss_for(effective)
    lora_deltas = torch.autograd.grad(lora_loss, lora_outputs, retain_graph=True)
    lora_grad_A, lora_grad_B = torch.autograd.grad(lora_loss, (A, B))
    lora_roles = role_dense(lora_deltas)
    mapped_A = {role: SCALE*(B.detach().T @ g) for role, g in lora_roles.items()}
    mapped_B = {role: SCALE*(g @ A.detach().T) for role, g in lora_roles.items()}
    assert torch.allclose(sum(mapped_A.values()), lora_grad_A, rtol=1e-12, atol=1e-12)
    assert torch.allclose(sum(mapped_B.values()), lora_grad_B, rtol=1e-12, atol=1e-12)
    assert torch.allclose(role_sum, sum(lora_roles.values()), rtol=1e-12, atol=1e-12)
    if name == "B_zero":
        assert max_abs(lora_grad_A) == 0.0
        assert float(lora_grad_B.norm()) > 0.0
    else:
        assert float(lora_grad_A.norm()) > 0.0
        assert float(lora_grad_B.norm()) > 0.0
    return {
        "name": name,
        "loss": float(lora_loss.detach()),
        "dense_autograd_vs_role_sum_max_abs": max_abs(dense_grad-role_sum),
        "lora_A_autograd_vs_role_sum_max_abs": max_abs(lora_grad_A-sum(mapped_A.values())),
        "lora_B_autograd_vs_role_sum_max_abs": max_abs(lora_grad_B-sum(mapped_B.values())),
        "dense_vs_lora_dense_role_sum_max_abs": max_abs(role_sum-sum(lora_roles.values())),
        "role_dense_norms": {role: float(g.norm()) for role,g in roles.items()},
        "role_lora_A_norms": {role: float(g.norm()) for role,g in mapped_A.items()},
        "role_lora_B_norms": {role: float(g.norm()) for role,g in mapped_B.items()},
        "autograd_lora_A_norm": float(lora_grad_A.norm()),
        "autograd_lora_B_norm": float(lora_grad_B.norm()),
    }


rows = [case("B_zero", torch.zeros_like(B1)), case("B_nonzero", B1)]
# A detached history writer is absent from the actual derivative path. This
# separate probe makes that fact visible even though the live calls share W.
writer = torch.randn(OUT, IN, dtype=DTYPE, requires_grad=True)
reader = torch.randn(OUT, IN, dtype=DTYPE, requires_grad=True)
h = (HISTORY_X @ writer.T).detach()
loss = ((X1 @ reader.T + .1*h.mean(dim=0))**2).mean()
writer_grad, reader_grad = torch.autograd.grad(loss, (writer, reader), allow_unused=True)
assert writer_grad is None and reader_grad is not None

# A negative role dot product alone cannot prove both role losses conflict
# under the shared update: both local first-order changes below are negative.
ga = torch.tensor([1.0, 0.0], dtype=DTYPE)
gv = torch.tensor([-0.2, 1.0], dtype=DTYPE)
shared_step = -(ga + gv)
assert float(torch.dot(ga, gv)) < 0
assert float(torch.dot(ga, shared_step)) < 0
assert float(torch.dot(gv, shared_step)) < 0

result = {
    "status": "PASS_CPU_ONLY",
    "dtype": str(DTYPE),
    "shape": {"input": IN, "output": OUT, "rank": RANK},
    "role_counts": {role: sum(x == role for x in LABELS1+LABELS2) for role in ROLES},
    "differentiable_projection_invocations": 2,
    "detached_history_writer_invocations": 1,
    "all_role_rows_exhaustive": True,
    "cases": rows,
    "detached_history_writer_grad_is_none": writer_grad is None,
    "reader_grad_norm": float(reader_grad.norm()),
    "negative_dot_counterexample": {
        "role_dot": float(torch.dot(ga,gv)),
        "action_first_order_change_under_shared_sgd": float(torch.dot(ga,shared_step)),
        "video_first_order_change_under_shared_sgd": float(torch.dot(gv,shared_step)),
    },
    "tolerance": {"rtol": 1e-12, "atol": 1e-12},
    "no_optimizer_or_gpu": True,
    "cuda_initialized": torch.cuda.is_initialized(),
    "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
    "elapsed_seconds": time.perf_counter() - START,
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
assert not result["cuda_initialized"] and result["max_rss_gib"] < 8
path = HERE / "ROLE_GRADIENT_CPU_RESULT.json"
path.write_text(json.dumps(result, indent=2) + "\n")
print(path, result["status"])
