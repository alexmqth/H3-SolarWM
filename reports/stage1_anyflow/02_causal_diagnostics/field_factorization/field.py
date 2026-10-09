"""Experiment A: identical full-prefix tokens and backend, changing roles/masks.

This is an input-layout-controlled field diagnostic, not persistent-cache
inference or a newly proposed inference architecture. History is clean and the
current chunk is explicitly noised. All video ancestors are recomputed.
"""
import importlib.util
from pathlib import Path
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
PREVIOUS=ROOT/'outputs/2026-10-08-17/stage1_generated_action_geometry'
spec=importlib.util.spec_from_file_location('previous_geometry_probe',PREVIOUS/'probe.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def setup(gpu):
    p.setup(gpu,14)
    global torch
    import torch


class FullPrefixAttention:
    """Whole [prefix, history, current] on both branches, identical SDPA backend.

    Original predicate reproduces released directed action binding. Causal
    predicate uses past/current action visibility, chunk video causality and
    own-frame action feedback. Prefix action states can ground in their own
    clean past video here; deployed cache recomputes a shorter prefix and is
    NOT claimed equivalent. This isolates full-token causalization first.
    """
    allow_grad_read=False
    def __init__(self,packed,frames,rows,mode,actions=None,adapter=None):
        self.prefix=int(packed['action_video_start']);self.rows=rows
        self.frames=frames;self.mode=mode;self.actions=actions;self.adapter=adapter
        self.spans=packed['action_text_rows'];self.mask=None

    def build_mask(self,device):
        p0=self.prefix;n=self.frames*self.rows;size=p0+n
        ann=torch.full((size,),-1,device=device,dtype=torch.long)
        for frame,(lo,hi) in enumerate(self.spans.tolist()):ann[lo:hi]=frame
        frm=torch.full((size,),-1,device=device,dtype=torch.long)
        frm[p0:]=torch.arange(n,device=device)//self.rows
        aq,ak=ann[:,None],ann[None,:];fq,fk=frm[:,None],frm[None,:]
        same=(aq>=0)&(ak>=0)&(aq==ak)
        if self.mode=='original':
            read_action=(fq>=0)&(ak>=0)&(fq==ak)
            blocked_out=(ak>=0)&~same&~read_action
            blocked_in=(aq>=0)&(fk>=0)&(aq!=fk)
            mask=~(blocked_out|blocked_in)
        elif self.mode=='causal':
            common=(frm<0)&(ann<0)
            prefix_to_prefix=(fq<0)&(fk<0)&((ak<0)|same)
            action_feedback=(aq>=0)&(fk>=0)&(aq==fk)
            video_to_prefix=(fq>=0)&(fk<0)&((ak<0)|(ak<=fq))
            video_to_video=(fq>=0)&(fk>=0)&(fk//5<=fq//5)&(fk//5>=fq//5-5)
            mask=prefix_to_prefix|action_feedback|video_to_prefix|video_to_video
            assert not bool(mask[common,p0:].any())
            assert bool(mask.diag().all())
        else:raise ValueError(self.mode)
        return mask

    def apply_hidden(self,hidden,layer):
        if self.adapter is None:return hidden
        return self.adapter.apply_hidden(layer,hidden,prefix=self.prefix,
            frame_rows=self.rows,action_cond=self.actions)

    def attend(self,q,k,v,*,rope_freqs,layer,apply_rope,scale):
        if self.adapter is not None:
            q,k,v=self.adapter(layer,q,k,v,prefix=self.prefix,
                frame_rows=self.rows,action_cond=self.actions)
        if self.mask is None:self.mask=self.build_mask(q.device)
        q=apply_rope(q,rope_freqs);k=apply_rope(k,rope_freqs)
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0,1).unsqueeze(0),k.transpose(0,1).unsqueeze(0),
            v.transpose(0,1).unsqueeze(0),attn_mask=self.mask[None,None],scale=scale
            ).squeeze(0).transpose(0,1)


def forward(model,state,history,cond,prompt,anchor,full_packed,sigma,*,
            mode,actions=None,adapter=None,target=None,return_all=False):
    from causal.h3_cached import slice_packed,retime_anchor_position
    from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3
    whole=torch.cat((history,state),dim=2).to(getattr(model,'_h3_input_dtype',state.dtype))
    rows=(state.shape[-2]//2)*(state.shape[-1]//2);start=history.shape[2]
    packed=slice_packed(full_packed,0,whole.shape[2],rows)
    if start:
        packed=retime_anchor_position(packed,full_packed,anchor_rows=anchor.shape[0],
            frame_index=start-1,frame_rows=rows,anchor_slot=1)
    mask=torch.ones_like(whole);mask[:,:,:start]=0
    control=FullPrefixAttention(packed,whole.shape[2],rows,mode,actions,adapter)
    output=model_fn_minimax_h3(model,whole,cond['audio_noise'].to(whole.dtype),packed,prompt,
        timestep_video=torch.tensor(sigma*1000,device=whole.device),
        timestep_audio=torch.tensor(1000.,device=whole.device),
        keyframe_cond_anchor=anchor.to(whole.dtype),input_latents_video=whole,
        denoise_mask_video=mask,fixed_prefix_timesteps=True,causal_control=control,
        target_timestep_video=None if target is None else torch.tensor(target*1000,device=whole.device))[0].float()
    return output if return_all else output[:,:,start:]
