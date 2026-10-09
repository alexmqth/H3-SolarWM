"""Inference-only routing audit. Explicit intervals; no training or decoding.

The cached branch reuses production ChunkAttention/H3ChunkCache. The full
reference preserves own-action feedback and differs only in declared edges.
Matched-sigma history prefills are diagnostic, not deployment clean commits.
"""
import hashlib
import torch
from causal.h3_cached import H3ChunkCache, ChunkAttention, slice_packed
from causal.local_topology import annotation_ids, sdpa
from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3


def tensor_hash(x):
    x = x.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(x.shape), str(x.dtype))).encode()
                          + x.view(torch.uint8).numpy().tobytes()).hexdigest()


def errors(actual, reference):
    a, b = actual.detach().float(), reference.detach().to(actual.device).float()
    d = a-b
    rms = b.square().mean().sqrt()
    return dict(max_abs=float(d.abs().max()), rms=float(d.square().mean().sqrt()),
                reference_rms=float(rms), relative_rms=float(d.square().mean().sqrt()/rms.clamp_min(1e-12)))


def delta_metrics(pair, reference):
    x = (pair['A']-pair['D']).double().flatten()
    y = (reference['A']-reference['D']).double().flatten()
    xn, yn = x.norm(), y.norm()
    return dict(cosine=float(torch.dot(x,y)/(xn*yn).clamp_min(1e-30)),
                norm_ratio=float(xn/yn.clamp_min(1e-30)),
                relative_l2=float((x-y).norm()/yn.clamp_min(1e-30)),
                delta_rms=float(x.square().mean().sqrt()),
                reference_delta_rms=float(y.square().mean().sqrt()),
                delta_to_velocity_rms=float(x.square().mean().sqrt()/
                    pair['A'].double().square().mean().sqrt().clamp_min(1e-30)))


def mask(packed, frames, rows, split, *, causal, public_feedback, route='own'):
    """Independent full graph, with action feedback in EVERY historical chunk."""
    device=packed['token_tags'].device
    p=int(packed['action_video_start']); n=frames*rows
    ann=torch.cat([annotation_ids(p,packed['action_text_rows'],device),
                   torch.full((n,),-1,device=device,dtype=torch.long)])
    frm=torch.cat([torch.full((p,),-1,device=device,dtype=torch.long),
                   torch.arange(n,device=device)//rows])
    aq,ak,fq,fk=ann[:,None],ann[None,:],frm[:,None],frm[None,:]
    same=(aq>=0)&(ak>=0)&(aq==ak)
    read=(fq>=0)&(ak>=0)&(fq==ak)
    if route=='full': read=(fq>=0)&(ak>=0)&(ak<=fq)
    elif route=='within':
        read=(fq>=0)&(ak>=0)&(ak<=fq)&((ak>=split)==(fq>=split))
    elif route=='cross':
        read=read|((fq>=split)&(ak>=0)&(ak<split))
    elif route!='own': raise ValueError(route)
    result=~(((ak>=0)&~same&~read)|((aq>=0)&(fk>=0)&(aq!=fk)))
    if not public_feedback:
        result &= ~((aq<0)&(fq<0)&(fk>=0))
    if causal:
        if public_feedback: raise ValueError('Shared dynamic prefix is not strict causal')
        result &= ~((fq>=0)&(fq<split)&(fk>=split))
    return result


class FullAttention:
    allow_grad_read=False
    def __init__(self, packed, frames, rows, split, *, causal, public_feedback,
                 compare_cache=None, route='own'):
        self.packed=packed; self.frames=frames; self.rows=rows; self.split=split
        self.causal=causal; self.public_feedback=public_feedback; self.route=route
        self.compare_cache=compare_cache; self.layer_errors=[]; self._mask=None
    def apply_hidden(self,hidden,layer): return hidden
    def attend(self,q,k,v,*,rope_freqs,layer,apply_rope,scale):
        if torch.is_grad_enabled(): raise RuntimeError('Audit is no_grad only')
        if self._mask is None:
            self._mask=mask(self.packed,self.frames,self.rows,self.split,
                causal=self.causal,public_feedback=self.public_feedback,route=self.route)
        if self.compare_cache is not None:
            e=self.compare_cache.history(layer,1)[0]
            p=int(self.packed['action_video_start']); sl=slice(p,p+self.split*self.rows)
            self.layer_errors.append(dict(layer=layer,key=errors(k[sl],e.key),
                value=errors(v[sl],e.value),rope=errors(rope_freqs[sl],e.rope)))
        return sdpa(apply_rope(q,rope_freqs),apply_rope(k,rope_freqs),v,self._mask,scale)


class ScopedActionAttention(ChunkAttention):
    """Only video->action key mask changes; all cache/read/feedback code is shared."""
    route='own'
    chunk_start=0
    def masks(self,current_rows,history_rows,device):
        prefix,video=super().masks(current_rows,history_rows,device)
        if self.route=='own': return prefix,video
        ann=annotation_ids(self.prefix,self.action_rows,device)
        frame=self.frame_start+torch.arange(current_rows,device=device)//self.frame_rows
        own=ann[None,:]==frame[:,None]
        if self.route=='within':
            allowed=(ann[None,:]>=self.chunk_start)&(ann[None,:]<=frame[:,None])
        elif self.route=='cross': allowed=own|((ann[None,:]>=0)&(ann[None,:]<self.chunk_start))
        elif self.route=='full': allowed=ann[None,:]<=frame[:,None]
        else: raise ValueError(self.route)
        video=video.clone(); video[:,:self.prefix]=(ann[None,:]<0)|allowed
        return prefix,video


def forward(model, current, *, history, sigma, packed, prompt, anchor, audio,
            kind, cache=None, route='own', history_noised=False, compare_cache=None,
            prefix_fixed=False):
    """R1/R1P/R2 recompute; cached reads production KV. All share native conditions."""
    start=history.shape[2]; stop=start+current.shape[2]
    rows=(current.shape[-2]//2)*(current.shape[-1]//2)
    assert len(packed['action_text_rows'])==stop and anchor.shape[0]==rows
    whole=torch.cat([history,current],2) if kind!='cached' else current
    whole=whole.to(getattr(model,'_h3_input_dtype',prompt.dtype))
    selected=slice_packed(packed,0 if kind!='cached' else start,stop,rows)
    update=torch.ones_like(whole)
    if kind!='cached': update[:,:,:start]=0
    if kind=='cached':
        ctrl=ScopedActionAttention(cache,1,int(selected['action_video_start']),rows,
            start,selected['action_text_rows'],action_prefix_mode='own',action_feedback=True)
        ctrl.route=route;ctrl.chunk_start=start
    else:
        ctrl=FullAttention(selected,stop,rows,start,causal=kind=='R2',
            public_feedback=kind=='R1',compare_cache=compare_cache,route=route)
    result=model_fn_minimax_h3(model,whole,audio.to(whole.dtype),selected,prompt,
        timestep_video=torch.tensor(float(sigma)*1000,device=whole.device),
        timestep_audio=torch.tensor(1000.,device=whole.device),
        keyframe_cond_anchor=anchor.to(whole.dtype),
        input_latents_video=None if history_noised else whole,
        denoise_mask_video=update,fixed_prefix_timesteps=prefix_fixed,causal_control=ctrl)[0]
    return (result if kind=='cached' else result[:,:,start:]),ctrl


def prefill(model, history, *, sigma, packed, prompt, anchor, audio, route='own',
            history_noised=False, prefix_fixed=False):
    """sigma0=native deployment commit; sigma>0=explicit matching-context control.

    All video rows are clean-time unless history_noised=True. This diagnostic
    bypasses chunk_forward's sigma0 guard deliberately, with no optimizer and
    no reuse across sigmas. The same production attend/commit logic is used.
    """
    rows=(history.shape[-2]//2)*(history.shape[-1]//2)
    assert len(packed['action_text_rows'])==history.shape[2]
    cache=H3ChunkCache(5,'cpu')
    ctrl=ScopedActionAttention(cache,0,int(packed['action_video_start']),rows,0,
        packed['action_text_rows'],action_prefix_mode='own',action_feedback=True,commit=True)
    ctrl.route=route;ctrl.chunk_start=0
    h=history.to(getattr(model,'_h3_input_dtype',prompt.dtype))
    model_fn_minimax_h3(model,h,audio.to(h.dtype),packed,prompt,
        timestep_video=torch.tensor(float(sigma)*1000,device=h.device),
        timestep_audio=torch.tensor(1000.,device=h.device),
        keyframe_cond_anchor=anchor.to(h.dtype),input_latents_video=None if history_noised else h,
        denoise_mask_video=torch.zeros_like(h),fixed_prefix_timesteps=prefix_fixed,causal_control=ctrl)
    return cache


def cache_hash(cache):
    h=hashlib.sha256()
    for layer,entries in sorted(cache.layers.items()):
        for e in entries:
            h.update(str((layer,e.index)).encode())
            for x in (e.key,e.value,e.rope): h.update(tensor_hash(x).encode())
    return h.hexdigest()
