"""DMD supervision for H3's noise-minus-clean velocity convention.

Use a live student endpoint (for example the complete FMBS chain) and teacher
and independent critic velocities at the SAME detached, re-noised sample.
The caller owns attention, conditioning, history, roles, critic warmup and
optimizers. These functions do not make a causal teacher bidirectional.

Reference: SolarWM training/sgf.py and AnyFlow on-policy generator loss.
For z_sigma=(1-sigma)*x0+sigma*noise, x0_hat=z_sigma-sigma*v.
Thus fake_x0-real_x0 = sigma*(v_real-v_fake); using v_fake-v_real directly
would reverse the descent direction in this velocity convention.
"""
from __future__ import annotations

import torch


def _sigma(value, like):
    if isinstance(value, bool):
        raise ValueError('Sigma must be numeric, not bool')
    sigma=torch.as_tensor(value,device=like.device,dtype=torch.float32).detach()
    if sigma.numel()==1:
        sigma=sigma.reshape(*([1]*like.ndim))
    elif sigma.ndim==1 and sigma.shape[0]==like.shape[0]:
        sigma=sigma.reshape(like.shape[0],*([1]*(like.ndim-1)))
    else:
        raise ValueError('Sigma must be scalar or one value per batch item')
    if not bool(torch.isfinite(sigma).all() and (sigma>0).all() and (sigma<=1).all()):
        raise ValueError('Score sigma must lie in (0,1]')
    return sigma


def _check_shapes(reference,*others):
    if reference.ndim<2 or reference.numel()==0:
        raise ValueError('Expected nonempty batch and feature dimensions')
    if any(x.shape!=reference.shape or x.device!=reference.device for x in others):
        raise ValueError('Endpoint, noise and score tensors must share shape and device')


def _mask(value,mask):
    if mask is None:return torch.ones_like(value,dtype=torch.bool)
    try:return torch.broadcast_to(torch.as_tensor(mask,device=value.device,dtype=torch.bool),value.shape)
    except RuntimeError as e:raise ValueError('Mask cannot broadcast to the latent shape') from e


def _selected_finite(value,selected):
    # Ignore conditioned/masked rows BEFORE a reduction or nonlinearity.
    out=torch.where(selected,value,torch.zeros_like(value))
    if not bool(torch.isfinite(out).all()):
        raise FloatingPointError('Non-finite value in an active DMD/FM row')
    return out


@torch.no_grad()
def score_sample(student_endpoint,noise,sigma):
    """Re-noise detached current student samples; no route back through scores."""
    _check_shapes(student_endpoint,noise)
    s=_sigma(sigma,student_endpoint)
    return (1-s)*student_endpoint.detach().float()+s*noise.detach().float()


def critic_flow_loss(prediction,student_endpoint,noise,*,mask=None):
    """Train fake velocity on detached samples from the student's distribution."""
    _check_shapes(prediction,student_endpoint,noise)
    selected=_mask(prediction,mask)
    target=noise.detach().float()-student_endpoint.detach().float()
    residual=_selected_finite(prediction.float()-target,selected)
    return residual.square().sum()/selected.sum().clamp_min(1)


def distribution_matching_loss(student_endpoint,noisy,sigma,*,real_velocity,
                               fake_velocity,normalize=True,mask=None):
    """Return (surrogate, detached direction, detached diagnostics).

    The endpoint must retain the actual generator graph. With no mask, the
    endpoint gradient is direction/numel, from mean reduction. Differentiating
    the scores, noisy input or normalizer is deliberately excluded.
    """
    _check_shapes(student_endpoint,noisy,real_velocity,fake_velocity)
    if not student_endpoint.requires_grad:
        raise ValueError('DMD requires a live student endpoint graph')
    selected=_mask(student_endpoint,mask)
    s=_sigma(sigma,student_endpoint)
    with torch.no_grad():
        clean=student_endpoint.detach().float()
        real=noisy.detach().float()-s*real_velocity.detach().float()
        fake=noisy.detach().float()-s*fake_velocity.detach().float()
        direction=_selected_finite(fake-real,selected)
        dims=tuple(range(1,clean.ndim))
        counts=selected.sum(dim=dims,keepdim=True).clamp_min(1)
        distance=_selected_finite(clean-real,selected).abs()
        normalizer=distance.sum(dim=dims,keepdim=True)/counts
        if normalize:direction=direction/normalizer.clamp_min(1e-8)
        direction=_selected_finite(direction,selected)
        target=torch.where(selected,clean-direction,torch.zeros_like(clean))
        stats=dict(direction_rms=float((direction.square().sum()/selected.sum().clamp_min(1)).sqrt()),
            per_sample_normalizer=normalizer.reshape(clean.shape[0]).cpu().tolist(),
            active_elements=int(selected.sum()),normalized=bool(normalize),
            convention='fake_x0-real_x0 = sigma*(real_velocity-fake_velocity)')
    output=torch.where(selected,student_endpoint.float(),torch.zeros_like(student_endpoint.float()))
    residual=output-target
    loss=.5*residual.square().sum()/selected.sum().clamp_min(1)
    return loss,direction,stats
