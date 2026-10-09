"""Tiny-H3 execution on actual 37-frame layout, three 12-latent windows."""
from pathlib import Path
import sys
import pytest
import torch

BASE=Path(__file__).resolve().parent;RT=BASE/'runtime'
sys.path[:0]=[str(RT/'code'),str(RT/'DiffSynth-Studio-h3-v2')]
from causal import h3_cached as hc
from causal.h3_training import make_small_h3,synthetic_h3_batch
from causal.h3_precision import configure_precision
from causal.local_topology import window_forward,visible_inputs,grounded_prefix
from native_prefix import native_prefix_variant
from single_anchor import original_image_window_variant

single=original_image_window_variant(window_forward)
cached=native_prefix_variant(hc.chunk_forward)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_coarse_visible_paths_and_current_action_forks(dtype):
    torch.manual_seed(911);torch.set_num_threads(2)
    model=make_small_h3().eval().to(dtype).requires_grad_(False)
    configure_precision(model,'h3_fp32')
    b=synthetic_h3_batch(frames=37,seed=321)
    prompt=b['prompt_embeds'].to(dtype);anchor=b['anchor_rows'].to(dtype)
    anchor_pos=b['packed']['img_position_ids'][0,b['packed']['img_pos'][:4]]
    seen=[]
    def capture(module,args,kwargs):
        rows=kwargs['img_pos_info']['position_ids'].view(-1)
        seen.append(kwargs['img_position_ids'][0,rows[:4]].clone())
    hook=model.register_forward_pre_hook(capture,with_kwargs=True)
    with torch.no_grad():
        for i,start in enumerate((0,12,24)):
            stop=start+12
            packed,text=visible_inputs(b['packed'],prompt,stop,4)
            history=b['clean_video'][:,:,:start];saved=history.clone()
            state=b['noise'][:,:,start:stop]
            cond=dict(full_packed=packed,prompt=text,anchor=anchor,audio=b['audio_latents'],chunk_frames=12)
            kw=dict(history=history,sigma=.6,index=i,history_chunks=5,**cond)
            value=single(model,state,**kw)
            torch.testing.assert_close(seen[-1],anchor_pos,atol=0,rtol=0)
            repeat=single(model,state,**kw)
            torch.testing.assert_close(value,repeat,atol=0,rtol=0)
            torch.testing.assert_close(history,saved,atol=0,rtol=0)
            future=prompt.clone()
            for lo,hi in b['packed']['action_text_spans_local'][stop:]:future[lo:hi]+=10
            layout2,text2=visible_inputs(b['packed'],future,stop,4)
            noise2=b['noise'].clone();noise2[:,:,stop:]+=10
            alt=single(model,noise2[:,:,start:stop],**dict(kw,full_packed=layout2,prompt=text2))
            torch.testing.assert_close(value,alt,atol=0,rtol=0)
            action=prompt.clone()
            for lo,hi in b['packed']['action_text_spans_local'][start:stop]:action[lo:hi]=3*action[lo:hi].flip(-1)+1
            layout3,text3=visible_inputs(b['packed'],action,stop,4)
            delta=single(model,state,**dict(kw,full_packed=layout3,prompt=text3))-value
            assert float(delta.abs().max())>1e-6
            assert len(packed['action_text_rows'])==stop
            if i==0:
                with grounded_prefix(chunk_frames=12):
                    ref=cached(model,state,index=0,sigma=.6,cache=hc.H3ChunkCache(5,'cpu'),
                        action_prefix_mode='own',action_feedback=True,**cond)
                torch.testing.assert_close(value,ref,atol=2e-6,rtol=2e-6)
    hook.remove()
