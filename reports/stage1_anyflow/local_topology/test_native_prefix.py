"""Actual H3 time preprocessing and cache semantics for the isolated policy."""
from pathlib import Path
import sys
import pytest
import torch

BASE=Path(__file__).resolve().parent
RT=BASE/'runtime'
sys.path[:0]=[str(RT/'code'),str(RT/'DiffSynth-Studio-h3-v2'),str(RT/'tests')]
from causal import h3_cached as hc
from causal.local_topology import window_forward,grounded_prefix
from test_local_topology import setup,conditions,cache_snapshot,equal_cache
from native_prefix import native_prefix_variant

native_window=native_prefix_variant(window_forward)
native_chunk=native_prefix_variant(hc.chunk_forward)


@pytest.mark.parametrize('sigma',[1.,.6,.07110826873779297])
def test_actual_preprocessing_only_text_time_changes_on_production_grid(sigma):
    model,b=setup();records=[]
    def capture(module,args,kwargs):
        t=kwargs['unique_timesteps'][kwargs['inverse_indices']]
        records.append((t.detach().clone(),kwargs['text_pos_info']['position_ids'].view(-1).clone()))
    hook=model.register_forward_pre_hook(capture,with_kwargs=True)
    kw=dict(history=b['clean_video'][:,:,:5],sigma=sigma,index=1,**conditions(b,1))
    with torch.no_grad():
        a=window_forward(model,b['noise'][:,:,5:10],**kw)
        d=native_window(model,b['noise'][:,:,5:10],**kw)
    hook.remove();(fixed,text),(native,text2)=records
    assert torch.equal(text,text2)
    assert torch.all(fixed[text]==1.)
    torch.testing.assert_close(native[text],torch.full_like(native[text],1-sigma),atol=1e-7,rtol=0)
    other=torch.ones_like(fixed,dtype=torch.bool);other[text]=False
    torch.testing.assert_close(fixed[other],native[other],atol=0,rtol=0)
    assert float((a-d).abs().max())>1e-6


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_native_prefix_first_identity_and_committed_video_replay(dtype):
    model,b=setup(dtype);cache=hc.H3ChunkCache(2,'cpu')
    def call(state,i,cache,sigma=.6,commit=False):
        return native_chunk(model,state,index=i,cache=cache,sigma=sigma,commit=commit,
            action_prefix_mode='own',action_feedback=True,
            anchor_frame_index=i*5-1 if i else None,**conditions(b,i))
    with torch.no_grad(),grounded_prefix():
        for i in range(4):
            state=b['noise'][:,:,i*5:min(i*5+5,17)]
            before=cache_snapshot(cache);out=call(state,i,cache)
            if i==0:
                ref=native_window(model,state,history=state[:,:,:0],sigma=.6,index=i,**conditions(b,i))
                torch.testing.assert_close(out,ref,atol=2e-6,rtol=2e-6)
            rebuild=hc.H3ChunkCache(2,'cpu')
            for j in range(i):call(b['clean_video'][:,:,j*5:j*5+5],j,rebuild,0.,True)
            repeat=call(state,i,rebuild)
            torch.testing.assert_close(out,repeat,atol=0,rtol=0)
            equal_cache(before,cache_snapshot(cache));equal_cache(before,cache_snapshot(rebuild))
            call(b['clean_video'][:,:,i*5:min(i*5+5,17)],i,cache,0.,True)
