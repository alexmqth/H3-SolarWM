"""Tiny real-H3 integration: adapter isolation, interventions, masks, KV."""
import json
from pathlib import Path
import probe as p
p.setup('')
import torch
from causal.h3_training import make_small_h3,synthetic_h3_batch
from causal.h3_cached import H3ChunkCache,chunk_forward,expand_packed_two_anchors
from causal.anyflow import install_anyflow
from causal.stage1_lora import install_stage1_lora
from causal.pretrained_lora import install_adapters

torch.set_num_threads(2);torch.manual_seed(13)
model=make_small_h3().eval()
# Remove tiny helper's unconditional test LoRA to get an unadapted reference.
for block in model.blocks: block.attn.qkv_proj=block.attn.qkv_proj.base
b=synthetic_h3_batch(frames=12,seed=13)
cond=dict(prompt_embeds=b['prompt_embeds'],anchor=b['anchor_rows'],
          audio_noise=b['audio_latents'],packed=b['packed'])
packed=expand_packed_two_anchors(cond['packed'],frame_rows=4)
anchor=torch.cat((cond['anchor'],cond['anchor']))
history=b['clean_video'][:,:,:5]
z=.4*b['clean_video'][:,:,5:10]+.6*b['noise'][:,:,5:10]
with torch.no_grad():
    reference={m:p.teacher_forward(model,z,history,cond,cond['prompt_embeds'],
        anchor if m else cond['anchor'],packed if m else cond['packed'],.6,matched=m)
        for m in (False,True)}
visual,_=install_adapters(model,rank=2,block_indices=(0,1))
time=install_anyflow(model)
bank=install_stage1_lora(model,rank=2,alpha=2)
with torch.no_grad():
    for v in visual: v.lora_B.normal_(0,.1)
    for m in bank.modules:
        for x in m.lora_B: x.normal_(0,.1)
    for x in time.parameters(): x.add_(torch.randn_like(x)*.03)
model.requires_grad_(False)
versions=[(x,x._version) for x in model.parameters()]
cache=H3ChunkCache(5,'cpu')
common=dict(full_packed=packed,anchor=anchor,audio=cond['audio_noise'],chunk_frames=5,
    action_prefix_mode='causal',action_feedback=True)
donor=cond['prompt_embeds']+torch.randn_like(cond['prompt_embeds'])
with torch.no_grad(),p.RoutingAudit() as audit:
    chunk_forward(model,history,index=0,cache=cache,sigma=0,target_sigma=0,commit=True,
                  prompt=cond['prompt_embeds'],**common)
    before=p.cache_digest(cache);commits=cache.commits
    switched=p.replace_current(cond['prompt_embeds'],donor,packed,5,10)
    audit.active=True
    a=chunk_forward(model,z,index=1,cache=cache,sigma=.6,target_sigma=.6,
                    prompt=cond['prompt_embeds'],**common)
    audit.active=False
    d=chunk_forward(model,z,index=1,cache=cache,sigma=.6,target_sigma=.6,
                    prompt=switched,**common)
    assert p.rms(a-d)>1e-5
    future=p.replace_current(cond['prompt_embeds'],donor,packed,10,12)
    vf=chunk_forward(model,z,index=1,cache=cache,sigma=.6,target_sigma=.6,
                     prompt=future,**common)
    assert torch.equal(a,vf),'Future action leakage'
    with p.original_weights(model,visual,bank,None):
        for matched in (False,True):
            restored=p.teacher_forward(model,z,history,cond,cond['prompt_embeds'],
                anchor if matched else cond['anchor'],packed if matched else cond['packed'],
                .6,matched=matched)
            assert torch.equal(restored,reference[matched]),'Teacher retained an adaptation'
    assert all(m.enabled for m in [*visual,*bank.modules])
    assert model.anyflow_conditioner is time
    a2=chunk_forward(model,z,index=1,cache=cache,sigma=.6,target_sigma=.6,
                     prompt=cond['prompt_embeds'],**common)
    assert torch.equal(a,a2),'Student not restored after teacher switch'
    assert p.cache_digest(cache)==before and cache.commits==commits
assert all(x._version==v for x,v in versions)
# Real stored conditioning: span pairing/head/noise equality is also checked on CPU.
real={a:torch.load(p.GENERATED/a/'conditioning.pt',map_location='cpu',weights_only=True)
      for a in ('A','D')}
real_checks=[]
for label in ('A','D'):
    for chunk in (1,2):
        start=chunk*5;stop=min(start+5,12)
        x=p.replace_current(real[label]['prompt_embeds'],real['D' if label=='A' else 'A']['prompt_embeds'],
                            real[label]['packed'],start,stop)
        changed=(x!=real[label]['prompt_embeds']).any(-1).nonzero().flatten().tolist()
        assert changed
        real_checks.append(dict(history=label,chunk=chunk,changed_rows=changed))
result=dict(status='passed',scope='random tiny H3 + real conditioning layout; not 33B geometry/quality',
    exact_original_teacher_restoration=True,exact_student_restoration=True,
    future_only_rmse=p.rms(a-vf),current_only_action_response_rms=p.rms(a-d),
    actual_SDPA_mask_audits=audit.records,real_conditioning_interventions=real_checks,
    KV_content_and_commits_unchanged=True,parameter_versions_unchanged=True)
(p.OUT/'cpu_integration.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in
                  ('actual_SDPA_mask_audits','real_conditioning_interventions')}))
