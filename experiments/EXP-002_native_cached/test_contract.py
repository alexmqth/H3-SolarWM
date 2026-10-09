"""CPU-only checks against the frozen real parking input layout."""
import hashlib
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb"
sys.path[:0] = [str(FROZEN/"runtime/code"), str(FROZEN/"runtime/DiffSynth-Studio-h3-v2"),
                str(ROOT/"submission/experiments/EXP-001_v2b_124"), str(HERE)]
from causal.local_topology import visible_inputs
from causal.h3_cached import H3ChunkCache, ChunkAttention, slice_packed
from rollout_contract import prompt_for_path


@pytest.fixture(scope="module")
def inputs():
    return {a: torch.load(FROZEN/f"source_coarse/inputs/parking_{a}.pt",
                          map_location="cpu", weights_only=True) for a in "AD"}


@pytest.mark.parametrize("stop,start,index", [(12,0,0),(17,12,1),(22,17,2)])
@pytest.mark.parametrize("path", ["AA","AD"])
def test_native_visible_rows_and_global_positions(inputs, stop, start, index, path):
    source=inputs["A"]["packed"]
    prompt=prompt_for_path(inputs,path,stop)
    packed,text=visible_inputs(source,prompt,stop,390)
    assert len(packed["action_text_rows"])==stop
    assert len(packed["action_text_spans_local"])==stop
    assert packed["seq_len"]==packed["action_video_start"]+stop*390
    assert len(text)==len(packed["text_pos"])
    assert packed["img_pos"].numel()==(stop+1)*390  # one native I0
    assert torch.equal(packed["img_position_ids"][:,-stop*390:],
                       source["img_position_ids"][:,int(source["action_video_start"]):
                                                   int(source["action_video_start"])+stop*390])
    current=slice_packed(packed,start,stop,390)
    assert current["seq_len"]==current["action_video_start"]+(stop-start)*390
    assert torch.equal(current["img_position_ids"][:,-(stop-start)*390:],
                       packed["img_position_ids"][:,-(stop-start)*390:])
    assert torch.equal(current["action_text_rows"],packed["action_text_rows"])
    assert index=={0:0,12:1,17:2}[start]
    for frame,(lo,hi) in enumerate(current["action_text_rows"].tolist()):
        assert 0<=lo<hi<=current["action_video_start"]
        assert (hi-lo)==int(source["action_text_rows"][frame,1]-source["action_text_rows"][frame,0])


def test_current_action_and_future_isolation(inputs):
    packed_a,prompt_a=visible_inputs(inputs["A"]["packed"],prompt_for_path(inputs,"AA",17),17,390)
    packed_d,prompt_d=visible_inputs(inputs["A"]["packed"],prompt_for_path(inputs,"AD",17),17,390)
    assert torch.equal(packed_a["img_position_ids"],packed_d["img_position_ids"])
    assert torch.equal(prompt_a[:int(packed_a["action_text_spans_local"][12][0])],
                       prompt_d[:int(packed_d["action_text_spans_local"][12][0])])
    assert not torch.equal(prompt_a,prompt_d)
    p=int(packed_a["action_video_start"])
    ctl=ChunkAttention(H3ChunkCache(5,"cpu"),1,p,390,12,
                       packed_a["action_text_rows"],"own",True)
    prefix,video=ctl.masks(5*390,12*390,"cpu")
    ann=torch.full((p,),-1,dtype=torch.long)
    for f,(lo,hi) in enumerate(packed_a["action_text_rows"].tolist()): ann[lo:hi]=f
    frames=12+torch.arange(5*390)//390
    # Current video can read its own action but no other action directly.
    assert torch.equal(video[:,:p],(ann[None,:]<0)|(ann[None,:]==frames[:,None]))
    assert not (ann>=17).any()  # future action rows are physically absent
    assert all(not video[frames==f,:p][:,ann==g].any()
               for f in range(12,17) for g in range(12,17) if f!=g)
    assert prefix.shape==(p,p) and video.shape==(5*390,p+12*390+5*390)


def test_cpu_cache_append_and_read_only():
    cache=H3ChunkCache(max_history=5,storage_device="cpu")
    with torch.no_grad():
        for layer in (0,1):
            cache.commit(layer,0,torch.ones(12,2,3),torch.ones(12,2,3),torch.zeros(12,1))
    digest=lambda: hashlib.sha256(b"".join(
        x.detach().numpy().tobytes() for layer in sorted(cache.layers)
        for e in cache.layers[layer] for x in (e.key,e.value,e.rope))).hexdigest()
    before=digest()
    for layer in (0,1):
        e=cache.history(layer,1)[0]
        assert e.index==0 and e.key.shape[0]==12
    assert digest()==before and cache.commits==2
    with torch.no_grad():
        for layer in (0,1):
            cache.commit(layer,1,torch.ones(5,2,3)*2,torch.ones(5,2,3)*2,torch.zeros(5,1))
    assert [e.index for e in cache.history(0,2)]==[0,1]
    assert cache.nbytes>0 and cache.commits==4


def test_no_fixed_interval_formula():
    from interval_cached import interval_cached
    import inspect
    source=inspect.getsource(interval_cached)
    assert "start = index*chunk_frames" not in source
    assert "fixed_prefix_timesteps=False" in source
    assert "(12, 17, 1)" in source
