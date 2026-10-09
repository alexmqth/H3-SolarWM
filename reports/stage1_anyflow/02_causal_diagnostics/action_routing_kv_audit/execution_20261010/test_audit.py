from pathlib import Path
import sys
import copy
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'DiffSynth-Studio-h3-v2')]
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.h3_precision import configure_precision
from causal.local_topology import visible_inputs, OriginalWindowAttention
from causal.h3_cached import ChunkAttention
from audit_core import mask,forward,prefill,ScopedActionAttention,cache_hash,errors


def setup(dtype=torch.float32):
    torch.set_num_threads(2);torch.manual_seed(781)
    model=make_small_h3().eval().to(dtype).requires_grad_(False)
    configure_precision(model,'h3_fp32')
    b=synthetic_h3_batch(frames=22,seed=141)
    b['prompt_embeds']=b['prompt_embeds'].to(dtype)
    return model,b


def cond(b,stop,prompt=None):
    packed,text=visible_inputs(b['packed'],b['prompt_embeds'] if prompt is None else prompt,stop,4)
    return dict(packed=packed,prompt=text,anchor=b['anchor_rows'],audio=b['audio_latents'])


def test_mask_has_exactly_declared_edges():
    _,b=setup();c=cond(b,17);p=c['packed']['action_video_start']
    r1=mask(c['packed'],17,4,12,causal=False,public_feedback=True)
    assert torch.equal(r1,OriginalWindowAttention(c['packed'],17,4).build_mask('cpu'))
    rp=mask(c['packed'],17,4,12,causal=False,public_feedback=False)
    r2=mask(c['packed'],17,4,12,causal=True,public_feedback=False)
    assert torch.equal(rp[p+12*4:],r2[p+12*4:])
    changed=rp & ~r2;expect=torch.zeros_like(changed);expect[p:p+48,p+48:]=True
    assert torch.equal(changed,expect) and not bool((r2 & ~rp).any())
    for i,(lo,hi) in enumerate(c['packed']['action_text_rows'].tolist()):
        assert bool(r2[lo:hi,p+i*4:p+(i+1)*4].all())
        assert bool(r2[p+i*4:p+(i+1)*4,lo:hi].all())
    # Actual multi-layer transitive reachability: no current nodes -> historical V.
    reach=r2.clone()
    for _ in range(6): reach=reach | ((reach.float()@reach.float())>0)
    assert not bool(reach[p:p+48,p+48:].any())


def test_route_split_matches_production_and_union():
    _,b=setup();c=cond(b,17);p=c['packed']['action_video_start']
    from causal.h3_cached import H3ChunkCache
    common=dict(cache=H3ChunkCache(),index=1,prefix=p,frame_rows=4,
                frame_start=12,action_rows=c['packed']['action_text_rows'],action_feedback=True)
    actual=ChunkAttention(**common,action_prefix_mode='causal').masks(20,48,'cpu')[1]
    masks={}
    for route in ('own','within','cross','full'):
        ctrl=ScopedActionAttention(**common);ctrl.route=route;ctrl.chunk_start=12
        masks[route]=ctrl.masks(20,48,'cpu')[1]
    assert torch.equal(masks['full'],actual)
    assert torch.equal(masks['full'],masks['within']|masks['cross'])
    assert torch.equal(masks['own'],masks['within']&masks['cross'])


@pytest.mark.parametrize('sigma',[.939540,.689441,.240781])
def test_matched_context_kv_and_velocity(sigma):
    model,b=setup();h=b['clean_video'][:,:,:12];state=b['noise'][:,:,12:17]
    with torch.no_grad():
        cache=prefill(model,h,sigma=sigma,**cond(b,12));digest=cache_hash(cache)
        ref,ctrl=forward(model,state,history=h,sigma=sigma,kind='R2',compare_cache=cache,**cond(b,17))
        actual,_=forward(model,state,history=h,sigma=sigma,kind='cached',cache=cache,**cond(b,17))
        torch.testing.assert_close(actual,ref,atol=2e-6,rtol=2e-5)
        assert all(e[k]['relative_rms']<2e-6 for e in ctrl.layer_errors for k in ('key','value'))
        assert all(e['rope']['max_abs']==0 for e in ctrl.layer_errors)
        altered=b['prompt_embeds'].clone()
        for lo,hi in b['packed']['action_text_spans_local'][12:17]:altered[lo:hi]=altered[lo:hi].flip(-1)+3
        other,c2=forward(model,state,history=h,sigma=sigma,kind='R2',compare_cache=cache,**cond(b,17,altered))
        cached,_=forward(model,state,history=h,sigma=sigma,kind='cached',cache=cache,**cond(b,17,altered))
        torch.testing.assert_close(cached,other,atol=2e-6,rtol=2e-5)
        assert float((other-ref).abs().max())>1e-6
        assert all(e[k]['relative_rms']<2e-6 for e in c2.layer_errors for k in ('key','value'))
        assert cache_hash(cache)==digest


def test_future_isolation_and_past_indirect_route():
    model,b=setup();h=b['clean_video'][:,:,:12];state=b['noise'][:,:,12:17]
    changed=b['prompt_embeds'].clone()
    for lo,hi in b['packed']['action_text_spans_local'][:12]: changed[lo:hi]=changed[lo:hi].flip(-1)+3
    with torch.no_grad():
        cache=prefill(model,h,sigma=.7,**cond(b,12))
        base,_=forward(model,state,history=h,sigma=.7,kind='cached',cache=cache,**cond(b,17))
        bypass,_=forward(model,state,history=h,sigma=.7,kind='cached',cache=cache,**cond(b,17,changed))
        torch.testing.assert_close(base,bypass,atol=0,rtol=0)
        other_cache=prefill(model,h,sigma=.7,**cond(b,12,changed))
        indirect,_=forward(model,state,history=h,sigma=.7,kind='cached',cache=other_cache,**cond(b,17))
        assert float((indirect-base).abs().max())>1e-6
        future=b['prompt_embeds'].clone()
        for lo,hi in b['packed']['action_text_spans_local'][17:]:future[lo:hi]+=99
        out,_=forward(model,state,history=h,sigma=.7,kind='cached',cache=cache,**cond(b,17,future))
        torch.testing.assert_close(base,out,atol=0,rtol=0)


def test_clean_commit_is_a_different_context_not_the_p0_reference():
    model,b=setup();h=b['clean_video'][:,:,:12];state=b['noise'][:,:,12:17]
    with torch.no_grad():
        native=prefill(model,h,sigma=.7,**cond(b,12))
        frozen=prefill(model,h,sigma=0,**cond(b,12))
        assert cache_hash(native)!=cache_hash(frozen)
        a,_=forward(model,state,history=h,sigma=.7,kind='cached',cache=native,**cond(b,17))
        f,_=forward(model,state,history=h,sigma=.7,kind='cached',cache=frozen,**cond(b,17))
        assert float((a-f).abs().max())>1e-6
