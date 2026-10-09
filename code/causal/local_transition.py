"""Differentiable T2/N local transition, isolated from frozen E1 inference.

Visible raw history is detached and temporarily noised each call. All hidden
states are recomputed; there is no persistent KV or backprop through history.
Single original I0, native text time, ordinary FM, current-window loss only.
"""
import torch

from .h3_cached import slice_packed
from .local_topology import OriginalWindowAttention, sdpa


class DifferentiableWindowAttention(OriginalWindowAttention):
    # Context has no commits/mutable cache. Checkpoint recomputation is safe.
    allow_grad_read = True

    def attend(self, q, k, v, *, rope_freqs, layer, apply_rope, scale):
        if self.mask is None:
            self.mask = self.build_mask(q.device)
        return sdpa(apply_rope(q,rope_freqs),apply_rope(k,rope_freqs),v,self.mask,scale)


def transition_forward(dit,current,*,history,history_noise,full_packed,prompt,
                       anchor,audio,sigma,index,chunk_frames=12,history_chunks=5,
                       checkpoint=False,checkpoint_offload=False):
    """N-only ordinary-FM interface, returns velocity for current rows only.

    The full_packed argument is already visibility-trimmed AFTER applying a
    future-independent global position contract. Input tensors are read-only.
    """
    from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3
    if hasattr(dit,'anyflow_conditioner'):
        raise ValueError('Local capability gate uses ordinary FM before AnyFlow')
    if not 0 <= sigma <= 1 or index < 0 or chunk_frames < 1 or history_chunks < 1:
        raise ValueError('Invalid window or sigma')
    start=index*chunk_frames;stop=start+current.shape[2]
    if history.shape!=history_noise.shape or history.shape[2]!=start:
        raise ValueError('Matching known history/noise ending at current window required')
    if history.dtype!=history_noise.dtype or history.device!=history_noise.device:
        raise ValueError('History and its noise must share dtype/device')
    if not 0 < current.shape[2] <= chunk_frames or len(full_packed['action_text_rows'])!=stop:
        raise ValueError('Remove future rows before transition evaluation')
    rows=(current.shape[-2]//2)*(current.shape[-1]//2)
    if anchor.shape[0]!=rows:
        raise ValueError('Exactly one original image anchor required')
    first=max(0,start-history_chunks*chunk_frames);h=start-first
    dtype=getattr(dit,'_h3_input_dtype',prompt.dtype)
    known=(1-sigma)*history.detach()[:,:,first:]+sigma*history_noise.detach()[:,:,first:]
    whole=torch.cat((known,current),dim=2).to(dtype)
    packed=slice_packed(full_packed,first,stop,rows)
    control=DifferentiableWindowAttention(packed,whole.shape[2],rows,first)
    output=model_fn_minimax_h3(dit,whole,audio.to(dtype),packed,prompt,
        timestep_video=torch.tensor(sigma*1000,device=current.device),
        timestep_audio=torch.tensor(1000.,device=current.device),
        keyframe_cond_anchor=anchor.to(dtype),input_latents_video=None,
        fixed_prefix_timesteps=False,causal_control=control,
        use_gradient_checkpointing=checkpoint,
        use_gradient_checkpointing_offload=checkpoint_offload)[0]
    return output[:,:,h:]


def observed_transition_losses(positive,negative,target,*,margin,action_weight):
    """Observed-positive FM + optional wrong-current-action ranking hinge.

    This is not an estimator of the unobserved negative-action video. Log
    positive/negative errors separately to catch negative-error inflation.
    """
    if positive.shape!=target.shape or margin<0 or action_weight<0:
        raise ValueError('Invalid target shape or loss coefficients')
    ep=(positive.float()-target.detach().float()).square().mean()
    if negative is None:
        if action_weight:
            raise ValueError('An eligible negative is required for action loss')
        return dict(total=ep,positive_FM=ep,negative_FM=None,action_hinge=ep*0)
    if negative.shape!=target.shape:
        raise ValueError('Counterfactual input must predict the same target layout')
    en=(negative.float()-target.detach().float()).square().mean()
    hinge=torch.relu(ep-en+margin)
    return dict(total=ep+action_weight*hinge,positive_FM=ep,negative_FM=en,action_hinge=hinge)
