"""Independent native-mask comparison and future-video/action noninterference."""
import json
import field as f
f.setup('')
import torch
from causal.h3_training import make_small_h3,synthetic_h3_batch
from causal.h3_cached import expand_packed_two_anchors,slice_packed,H3ChunkCache,chunk_forward
from diffsynth.models.minimax_h3_dit import _build_action_block_masks

torch.set_num_threads(2);torch.manual_seed(13)
model=make_small_h3().eval().requires_grad_(False)
batch=synthetic_h3_batch(frames=12,seed=13)
cond=dict(audio_noise=batch['audio_latents'],packed=batch['packed'])
packed=expand_packed_two_anchors(batch['packed'],frame_rows=4)
anchor=torch.cat((batch['anchor_rows'],batch['anchor_rows']))
prompt=batch['prompt_embeds'];history=batch['clean_video'][:,:,:5]
state=.4*batch['clean_video'][:,:,5:10]+.6*batch['noise'][:,:,5:10]
local=slice_packed(packed,0,10,4)
original=f.FullPrefixAttention(local,10,4,'original').build_mask('cpu')
native=_build_action_block_masks(local['action_text_rows'],local['action_video_start'],4,12,
    local['cu_seqlens'],local['seq_len'],'cpu',n_real=local['seq_len'])[0]
q=torch.arange(local['seq_len'])[:,None];kv=torch.arange(local['seq_len'])[None,:]
gold=native.mask_mod(torch.tensor(0),torch.tensor(0),q,kv)
assert torch.equal(original,gold),'Original action predicate changed'
with torch.no_grad():
    reference=f.p.teacher_forward(model,state,history,cond,prompt,anchor,packed,.6,matched=True)
    custom=f.forward(model,state,history,cond,prompt,anchor,packed,.6,mode='original')
    error=float((custom-reference).abs().max());assert error<2e-6,error
    # A single chunk is the deployed cached controller with no historical KV.
    first=batch['clean_video'][:,:,:5]
    native_chunk=chunk_forward(model,first,full_packed=packed,prompt=prompt,anchor=anchor,
        audio=cond['audio_noise'],sigma=.6,index=0,chunk_frames=5,cache=H3ChunkCache(5,'cpu'),
        action_prefix_mode='causal',action_feedback=True)
    full_chunk=f.forward(model,first,first[:,:,:0],cond,prompt,anchor,packed,.6,mode='causal')
    single_error=float((native_chunk-full_chunk).abs().max());assert single_error<2e-6,single_error
    # All12 frames current/noised so history clamping cannot hide a future leak.
    entire=batch['clean_video'].clone();changed=entire.clone();changed[:,:,10:]+=5
    baseline=f.forward(model,entire,entire[:,:,:0],cond,prompt,anchor,packed,.6,mode='causal')
    future=f.forward(model,changed,entire[:,:,:0],cond,prompt,anchor,packed,.6,mode='causal')
    future_error=float((baseline[:,:,:10]-future[:,:,:10]).abs().max())
    assert future_error==0,future_error
    donor=prompt+torch.randn_like(prompt)
    fp=f.p.replace_current(prompt,donor,packed,10,12)
    va=f.forward(model,entire,entire[:,:,:0],cond,fp,anchor,packed,.6,mode='causal')
    assert torch.equal(baseline[:,:,:10],va[:,:,:10])
    currentp=f.p.replace_current(prompt,donor,packed,5,10)
    vc=f.forward(model,entire,entire[:,:,:0],cond,currentp,anchor,packed,.6,mode='causal')
    assert f.p.rms(baseline[:,:,5:10]-vc[:,:,5:10])>1e-6
result=dict(status='passed',original_mask_exact_native_predicate=True,
    original_output_max_abs_vs_native=error,single_chunk_max_abs_vs_cached=single_error,
    future_video_max_abs=future_error,future_action_max_abs=0.,current_action_effect_nonzero=True,
    scope='random tiny H3, full-token diagnostic; not persistent-cache equivalence for history')
(f.OUT/'cpu_field.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
