"""Actual tiny-H3 execution tests for the new conditioning protocol."""
from pathlib import Path
import sys
import pytest
import torch

BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE/'runtime/code'),str(BASE/'runtime/DiffSynth-Studio-h3-v2')]
from causal.h3_training import make_small_h3,synthetic_h3_batch
from causal.h3_precision import configure_precision
from causal.local_topology import visible_inputs,window_forward
from single_anchor import original_image_window_variant
from diffsynth.pipelines.minimax_h3_audio_video import patchify_video
from history_conditioning import history_forward


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_real_inputs_times_identity_and_immutable_history(dtype):
    torch.set_num_threads(2);torch.manual_seed(911)
    model=make_small_h3().to(dtype).eval().requires_grad_(False)
    configure_precision(model,'h3_fp32')
    b=synthetic_h3_batch(frames=37,seed=321)
    captured=[]
    def hook(module,args,kw):captured.append(kw)
    handle=model.register_forward_pre_hook(hook,with_kwargs=True)
    original_image_pos=b['packed']['img_position_ids'][0,b['packed']['img_pos'][:4]]
    try:
        with torch.no_grad():
            for i,start in enumerate((0,12,24)):
                stop=start+12
                packed,prompt=visible_inputs(b['packed'],b['prompt_embeds'].to(dtype),stop,4)
                history=b['clean_video'][:,:,:start].float()
                noise=b['noise'][:,:,:start].float()
                current=b['noise'][:,:,start:stop].float()
                snapshots=[x.clone() for x in (history,noise,current)]
                for sigma in (1.,.5,.1,0.):
                    kw=dict(history=history,history_noise=noise,full_packed=packed,prompt=prompt,
                            anchor=b['anchor_rows'].to(dtype),audio=b['audio_latents'],
                            sigma=sigma,index=i,chunk_frames=12,history_chunks=5)
                    c=history_forward(model,current,mode='C',**kw)
                    n=history_forward(model,current,mode='N',**kw)
                    call_c,call_n=captured[-2:];captured.clear()
                    for mode,call in [('C',call_c),('N',call_n)]:
                        pos=call['img_pos_info']['position_ids'].flatten()
                        vid=pos[4:]
                        times=call['unique_timesteps'][call['inverse_indices']]
                        torch.testing.assert_close(call['img_position_ids'][0,pos[:4]],original_image_pos,atol=0,rtol=0)
                        expected_history=history if mode=='C' else (1-sigma)*history+sigma*noise
                        expected=patchify_video(torch.cat([expected_history,current],2))
                        torch.testing.assert_close(call['x'][0,vid],expected,atol=0,rtol=0)
                        expected_time=torch.full_like(times[vid],1-sigma)
                        if mode=='C':expected_time[:start*4]=1.
                        torch.testing.assert_close(times[vid],expected_time,atol=1e-7,rtol=0)
                        tp=call['text_pos_info']['position_ids'].flatten()
                        torch.testing.assert_close(times[tp],torch.full_like(times[tp],1-sigma),atol=1e-7,rtol=0)
                        assert len(packed['action_text_rows'])==stop
                    if start==0 or sigma==0:
                        torch.testing.assert_close(c,n,atol=0,rtol=0)
                    if start>0 and sigma==.5:
                        assert float((c-n).abs().max())>1e-6
                    for value,snapshot in zip((history,noise,current),snapshots):
                        torch.testing.assert_close(value,snapshot,atol=0,rtol=0)
    finally:handle.remove()


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_future_isolation_current_action_effect_and_actual_history_use(dtype):
    torch.set_num_threads(2);torch.manual_seed(911)
    model=make_small_h3().to(dtype).eval().requires_grad_(False);configure_precision(model,'h3_fp32')
    b=synthetic_h3_batch(frames=37,seed=321);text=b['prompt_embeds'].to(dtype)
    with torch.no_grad():
        for i,start in enumerate((0,12,24)):
            stop=start+12;layout,prompt=visible_inputs(b['packed'],text,stop,4)
            state=b['noise'][:,:,start:stop].float()
            kw=dict(history=b['clean_video'][:,:,:start].float(),history_noise=b['noise'][:,:,:start].float(),
                    full_packed=layout,prompt=prompt,anchor=b['anchor_rows'].to(dtype),audio=b['audio_latents'],
                    sigma=.5,index=i,chunk_frames=12,history_chunks=5)
            original=history_forward(model,state,mode='N',**kw)
            future=text.clone()
            for lo,hi in b['packed']['action_text_spans_local'][stop:]:future[lo:hi]+=10
            layout2,prompt2=visible_inputs(b['packed'],future,stop,4)
            noise2=b['noise'].clone();noise2[:,:,stop:]+=10
            alt=history_forward(model,noise2[:,:,start:stop].float(),mode='N',**dict(kw,full_packed=layout2,prompt=prompt2,history_noise=noise2[:,:,:start].float()))
            torch.testing.assert_close(original,alt,atol=0,rtol=0)
            action=text.clone()
            for lo,hi in b['packed']['action_text_spans_local'][start:stop]:action[lo:hi]=3*action[lo:hi].flip(-1)+1
            lp,pt=visible_inputs(b['packed'],action,stop,4)
            intervention=history_forward(model,state,mode='N',**dict(kw,full_packed=lp,prompt=pt))
            assert float((original-intervention).abs().max())>1e-6
            if start:
                changed=history_forward(model,state,mode='N',**dict(kw,history=kw['history']+2))
                assert float((original-changed).abs().max())>1e-6
            clean=history_forward(model,state,mode='C',**kw)
            oldkw={k:v for k,v in kw.items() if k!='history_noise'}
            reference=original_image_window_variant(window_forward)(model,state,**oldkw)
            torch.testing.assert_close(clean,reference,atol=0,rtol=0)
