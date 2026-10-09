"""CPU preflight of an isolated candidate, using the actual tiny H3 code path.

No pretrained weights or quality claim. Full-prefix reference is only valid
for chunk zero. Later replay reconstructs each ancestor under its own original
conditions, because a single shared mutable prefix would change cache semantics.
"""
from contextlib import contextmanager
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'DiffSynth-Studio-h3-v2'),
               str(ROOT / 'outputs/2026-10-08-18/stage1_real_abot_fm')]
from causal.h3_cached import ChunkAttention, H3ChunkCache, chunk_forward
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.h3_precision import configure_precision
import geometry_primitives as g
from current_prefix import current_prefix_feedback, verify_one_edge_change


def setup(dtype=torch.float32):
    torch.set_num_threads(2); torch.manual_seed(318)
    model = make_small_h3().eval().to(dtype).requires_grad_(False)
    configure_precision(model, 'h3_fp32')
    batch = synthetic_h3_batch(frames=17, seed=712)
    common = dict(full_packed=batch['packed'], prompt=batch['prompt_embeds'].to(dtype),
                  anchor=batch['anchor_rows'].to(dtype), audio=batch['audio_latents'].to(dtype),
                  chunk_frames=5, action_prefix_mode='own', action_feedback=True)
    return model, batch, common


def perturb_actions(prompt, packed, first, stop):
    changed = prompt.clone()
    for lo, hi in packed['action_text_spans_local'][first:stop]:
        changed[int(lo):int(hi)] = changed[int(lo):int(hi)].flip(-1) * 3 + 2
    assert not torch.equal(changed, prompt)
    return changed


def conditions(common, batch, index):
    # Separate immutable per-chunk conditions, including a moving anchor.
    c = dict(common)
    if index:
        from causal.h3_cached import last_frame_anchor
        c['anchor'] = last_frame_anchor(batch['clean_video'][:, :, index*5-1:index*5]).to(c['anchor'].dtype)
        c['anchor_frame_index'] = index*5-1
    return c


@contextmanager
def record_sdpa_masks():
    original = F.scaled_dot_product_attention
    records = []
    def checked(q, k, v, *args, **kwargs):
        records.append(kwargs['attn_mask'][0, 0].detach().clone())
        return original(q, k, v, *args, **kwargs)
    F.scaled_dot_product_attention = checked
    try:
        yield records
    finally:
        F.scaled_dot_product_attention = original


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_first_chunk_matches_original_directed_attention(dtype):
    model, b, c = setup(dtype)
    state = b['noise'][:, :, :5] * .7 + b['clean_video'][:, :, :5] * .3
    with torch.no_grad():
        ref = g.forward(model, state, state[:, :, :0], {'audio_noise': c['audio']},
            c['prompt'], c['anchor'], c['full_packed'], .7, mode='original')
        with current_prefix_feedback():
            actual = chunk_forward(model, state, sigma=.7, index=0, cache=H3ChunkCache(2,'cpu'), **c)
    torch.testing.assert_close(actual.float(), ref, atol=2e-6, rtol=2e-6)
    print(f'first_chunk_{dtype}_max_abs={float((actual.float()-ref).abs().max()):.9g}')


@pytest.mark.parametrize('index', [0, 1, 2])
def test_actual_attention_masks_match_declared_edges_and_no_future_paths(index):
    _, b, c = setup()
    p = int(c['full_packed']['action_video_start']); rows = 4
    spans = c['full_packed']['action_text_rows']; n = 5*rows
    cache = H3ChunkCache(2, 'cpu')
    with torch.no_grad():
        for old in range(index):
            cache.commit(0, old, torch.randn(n,4,16), torch.randn(n,4,16), torch.zeros(n,1))
        ctl = ChunkAttention(cache,index,p,rows,index*5,spans,'own',True)
        with current_prefix_feedback(), record_sdpa_masks() as masks:
            ctl.attend(*(torch.randn(p+n,4,16) for _ in range(3)),
                rope_freqs=torch.zeros(p+n,1), layer=0, apply_rope=lambda x,r:x, scale=.25)
    prefix, video = masks
    ann = torch.full((p,),-1,dtype=torch.long)
    for f,(lo,hi) in enumerate(spans.tolist()): ann[lo:hi] = f
    frame = index*5+torch.arange(n)//rows
    h = index*n
    expected_prefix = torch.cat(((ann[None,:]<0)|(ann[:,None]==ann[None,:]),
        torch.zeros(p,h,dtype=torch.bool),
        (ann[:,None]<0)|(ann[:,None]==frame[None,:])),1)
    expected_video = torch.cat(((ann[None,:]<0)|(ann[None,:]==frame[:,None]),
        torch.ones(n,h+n,dtype=torch.bool)),1)
    assert torch.equal(prefix,expected_prefix) and torch.equal(video,expected_video)
    assert not prefix[:,p:p+h].any()  # Prefix never reads historical video KV.
    # Ignore already-frozen history nodes. Check every possible multi-layer
    # current dependency, including common-prefix mediation.
    active = torch.cat((torch.arange(p),torch.arange(p+h,p+h+n)))
    graph = torch.cat((prefix[:,active],video[:,active]),0)
    reach = graph.clone()
    for _ in range(graph.shape[0]):
        nxt = reach | (reach.float() @ graph.float() > 0)
        if torch.equal(nxt,reach): break
        reach = nxt
    future = torch.nonzero(ann >= (index+1)*5).flatten()
    assert future.numel() and not reach[p:,future].any()


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_future_actions_do_not_change_earlier_outputs_or_committed_kv(dtype):
    model,b,c = setup(dtype)
    with current_prefix_feedback(), torch.no_grad():
        for index in range(3):
            past_a, past_b = H3ChunkCache(2,'cpu'), H3ChunkCache(2,'cpu')
            changed = dict(c, prompt=perturb_actions(c['prompt'],c['full_packed'],(index+1)*5,17))
            for old in range(index):
                for cache, common in [(past_a,c),(past_b,changed)]:
                    chunk_forward(model,b['clean_video'][:,:,old*5:old*5+5],sigma=0,
                        index=old,cache=cache,commit=True,**conditions(common,b,old))
            assert g.cache_digest(past_a) == g.cache_digest(past_b)
            state=b['noise'][:,:,index*5:index*5+5]
            outs = [chunk_forward(model,state,sigma=.69,index=index,cache=cache,
                     **conditions(common,b,index)) for cache,common in [(past_a,c),(past_b,changed)]]
            torch.testing.assert_close(*outs,atol=0,rtol=0)
            for cache,common in [(past_a,c),(past_b,changed)]:
                chunk_forward(model,b['clean_video'][:,:,index*5:index*5+5],sigma=0,
                    index=index,cache=cache,commit=True,**conditions(common,b,index))
            assert g.cache_digest(past_a) == g.cache_digest(past_b)


def test_current_action_and_past_kv_affect_output_without_mutating_read_cache():
    model,b,c=setup(); cache=H3ChunkCache(2,'cpu')
    with current_prefix_feedback(),torch.no_grad():
        for i in range(2):
            chunk_forward(model,b['clean_video'][:,:,i*5:i*5+5],sigma=0,index=i,
                cache=cache,commit=True,**conditions(c,b,i))
        before=g.cache_digest(cache); versions=g.cache_versions(cache); commits=cache.commits
        common=conditions(c,b,2);state=b['noise'][:,:,10:15]
        base=chunk_forward(model,state,sigma=.69,index=2,cache=cache,**common)
        alt=chunk_forward(model,state,sigma=.69,index=2,cache=cache,
            **dict(common,prompt=perturb_actions(c['prompt'],c['full_packed'],10,15)))
        assert g.cache_digest(cache)==before and cache.commits==commits
        g.assert_versions(versions)
        effect=float((base-alt).float().square().mean().sqrt())
        assert effect>1e-6
        import copy
        changed=copy.deepcopy(cache)
        for es in changed.layers.values():
            for e in es: e.value.add_(.25)
        history_out=chunk_forward(model,state,sigma=.69,index=2,cache=changed,**common)
        history_effect=float((base-history_out).float().square().mean().sqrt())
        assert history_effect>1e-6
        assert g.cache_digest(cache)==before
        print(f'current_action_rms={effect:.9g} history_value_intervention_rms={history_effect:.9g}')


def test_full_ancestry_replay_preserves_per_chunk_conditions_and_eviction():
    model,b,c=setup(); cache=H3ChunkCache(2,'cpu')
    with current_prefix_feedback(),torch.no_grad():
        for index in range(4):
            state=b['noise'][:,:,index*5:min(index*5+5,17)]
            before=g.cache_digest(cache); commits=cache.commits
            persistent=chunk_forward(model,state,sigma=.24,index=index,cache=cache,**conditions(c,b,index))
            fresh=H3ChunkCache(2,'cpu')
            # Recompute ALL ancestors (not only the sliding window) with each
            # ancestor's own anchor, timestamp, action schedule and clean state.
            for old in range(index):
                chunk_forward(model,b['clean_video'][:,:,old*5:old*5+5],sigma=0,
                    index=old,cache=fresh,commit=True,**conditions(c,b,old))
            replay=chunk_forward(model,state,sigma=.24,index=index,cache=fresh,**conditions(c,b,index))
            torch.testing.assert_close(persistent,replay,atol=0,rtol=0)
            assert g.cache_digest(cache)==before==g.cache_digest(fresh) and commits==cache.commits
            chunk_forward(model,b['clean_video'][:,:,index*5:min(index*5+5,17)],sigma=0,
                index=index,cache=cache,commit=True,**conditions(c,b,index))
            assert all([e.index for e in es]==list(range(max(0,index-1),index+1))
                       for es in cache.layers.values())


def test_patch_declared_edge_only_and_restores_after_exception():
    original=ChunkAttention.attend
    verify_one_edge_change(original)
    with pytest.raises(RuntimeError,match='sentinel'):
        with current_prefix_feedback():
            assert ChunkAttention.attend is not original
            raise RuntimeError('sentinel')
    assert ChunkAttention.attend is original
