"""Exact sequential VJP for a coupled two-action objective; no optimizer."""
import torch


def paired_objective(a, d, teacher_a, teacher_d, *, delta_weight=1., epsilon=1e-8):
    if not (a.shape == d.shape == teacher_a.shape == teacher_d.shape):
        raise ValueError('Counterfactual predictions must have equal shapes')
    if a.ndim < 2 or a.shape[0] != 1:
        raise ValueError('This diagnostic requires physical batch one')
    if delta_weight < 0 or epsilon <= 0:
        raise ValueError('Invalid loss weights')
    a, d = a.float(), d.float()
    ta, td = teacher_a.detach().float(), teacher_d.detach().float()
    if not all(torch.isfinite(v).all() for v in (a, d, ta, td)):
        raise FloatingPointError('Nonfinite prediction in paired objective')
    ds, dt = a - d, ta - td
    replay = .5 * ((a - ta).square().mean() + (d - td).square().mean())
    target_energy = dt.square().mean().detach()
    delta = (ds - dt).square().mean() / (target_energy + epsilon)
    loss = replay + delta_weight * delta
    norms = ds.double().norm(), dt.double().norm()
    stats = dict(replay=float(replay.detach()), relative_delta_mse=float(delta.detach()),
        total=float(loss.detach()), teacher_delta_energy=float(target_energy),
        delta_cosine=float((ds.double()*dt.double()).sum() / (norms[0]*norms[1]).clamp_min(1e-30)),
        delta_norm_ratio=float(norms[0]/norms[1].clamp_min(1e-30)))
    return loss, stats


def paired_cotangents(a, d, teacher_a, teacher_d, **kwargs):
    """Compute dL/da,dL/dd without retaining either expensive model graph.

    Caller replays A then D at exactly the same weights and conditions and
    backpropagates each cotangent separately, summing parameter gradients.
    No parameter update may occur between the two backward passes.
    """
    with torch.enable_grad():
        leaves = [x.detach().float().clone().requires_grad_() for x in (a, d)]
        loss, stats = paired_objective(*leaves, teacher_a, teacher_d, **kwargs)
        gradients = torch.autograd.grad(loss, leaves)
    return tuple(g.detach() for g in gradients), stats


def backward_replay(prediction, reference, cotangent, *, tolerance=0.):
    error = float((prediction.detach().float() - reference.float()).abs().max())
    if error > tolerance:
        raise RuntimeError(f'Paired forward replay differs: max_abs={error}, tolerance={tolerance}')
    if not prediction.requires_grad or cotangent.requires_grad:
        raise ValueError('Replay needs a live prediction and detached cotangent')
    torch.autograd.backward(prediction, cotangent.to(prediction))
    return error
