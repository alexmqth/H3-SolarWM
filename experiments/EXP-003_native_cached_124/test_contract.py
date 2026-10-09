"""CPU preflight for the newly extended global intervals and timing-only router."""
from pathlib import Path
import sys

import pytest
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FROZEN=ROOT/"H3-World/outputs/2026-10-09-22/chunk_partition_cb"
ROUTER=ROOT/"submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate"
sys.path[:0]=[str(FROZEN/"runtime/code"),str(FROZEN/"runtime/DiffSynth-Studio-h3-v2"),
              str(ROOT/"submission/experiments/EXP-001_v2b_124"),str(ROUTER),str(HERE)]
from causal.h3_cached import ChunkAttention,H3ChunkCache,slice_packed
from causal.local_topology import visible_inputs
from rollout_contract import prompt_for_path
import current_prefix as accepted
import metered_prefix as metered


@pytest.fixture(scope="module")
def inputs():
    return {a:torch.load(FROZEN/f"source_coarse/inputs/parking_{a}.pt",
                         map_location="cpu",weights_only=True) for a in "AD"}


@pytest.mark.parametrize("start,stop,index",[(17,22,2),(22,27,3),(27,32,4),(32,37,5)])
@pytest.mark.parametrize("path",["AA","AD"])
def test_full_visible_global_layout(inputs,start,stop,index,path):
    full=inputs["A"]["packed"]
    prompt=prompt_for_path(inputs,path,stop)
    layout,text=visible_inputs(full,prompt,stop,390)
    assert len(layout["action_text_rows"])==stop
    assert len(text)==len(layout["text_pos"])
    assert layout["img_pos"].numel()==(1+stop)*390
    assert layout["seq_len"]==layout["action_video_start"]+stop*390
    chunk=slice_packed(layout,start,stop,390)
    assert chunk["seq_len"]==chunk["action_video_start"]+5*390
    assert torch.equal(chunk["img_position_ids"][:,-5*390:],
                       full["img_position_ids"][:,int(full["action_video_start"])+start*390:
                                                        int(full["action_video_start"])+stop*390])
    assert index=={17:2,22:3,27:4,32:5}[start]
    p=int(chunk["action_video_start"])
    ctl=ChunkAttention(H3ChunkCache(5,"cpu"),index,p,390,start,
                       chunk["action_text_rows"],"own",True)
    _,video_mask=ctl.masks(5*390,start*390,"cpu")
    ann=torch.full((p,),-1,dtype=torch.long)
    for frame,(lo,hi) in enumerate(chunk["action_text_rows"].tolist()):ann[lo:hi]=frame
    frames=start+torch.arange(5*390)//390
    assert torch.equal(video_mask[:,:p],(ann[None,:]<0)|(ann[None,:]==frames[:,None]))
    assert not (ann>=stop).any()


def test_cache_index_five_uses_all_five_ancestors():
    cache=H3ChunkCache(max_history=5,storage_device="cpu")
    with torch.no_grad():
        for i in range(5):
            cache.commit(0,i,torch.ones(2,1,2)*i,torch.ones(2,1,2),torch.zeros(2,1))
    assert [e.index for e in cache.history(0,5)]==[0,1,2,3,4]
    assert cache.nbytes>0


def test_metered_router_preserves_cpu_attention_output():
    cache=H3ChunkCache(max_history=5,storage_device="cpu")
    with torch.no_grad():
        cache.commit(0,0,torch.randn(2,4,16),torch.randn(2,4,16),torch.zeros(2,1))
    action=torch.tensor([[2,3],[3,4]],dtype=torch.long)
    q,k,v=[torch.randn(8,4,16) for _ in range(3)]
    def call():
        ctl=ChunkAttention(cache,1,6,2,1,action,"own",True)
        return ctl.attend(q,k,v,rope_freqs=torch.zeros(8,1),layer=0,
                          apply_rope=lambda x,r:x,scale=.25)
    with torch.no_grad(),accepted.current_prefix_feedback():
        expected=call()
    metered.METER.reset()
    with torch.no_grad(),metered.current_prefix_feedback():
        actual=call()
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    counts=metered.METER.finish()
    assert counts["cache_transfer_bytes"]==sum(x.numel()*x.element_size()
        for e in cache.layers[0] for x in (e.key,e.value,e.rope))


def test_interval_is_native_and_full_length():
    from interval_cached import interval_cached
    import inspect
    source=inspect.getsource(interval_cached)
    assert "fixed_prefix_timesteps=False" in source
    assert "(32,37,5)" in source
    assert "start = index*chunk_frames" not in source
