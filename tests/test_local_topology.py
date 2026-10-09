"""E1 real tiny-H3 execution: leakage, Original identity, cache and window.

These checks establish inference semantics, not pretrained action quality.
The fixed layout contract is independent of future action content/row counts.
"""
from pathlib import Path
import copy
import sys

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'DiffSynth-Studio-h3-v2')]
from causal import h3_cached as hc
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.h3_precision import configure_precision
from causal.local_topology import (visible_inputs, grounded_prefix,
                                  window_forward, OriginalWindowAttention)
from diffsynth.models.minimax_h3_dit import _build_action_block_masks


def setup(dtype=torch.float32):
    torch.set_num_threads(2)
    torch.manual_seed(911)
    model = make_small_h3().eval().to(dtype).requires_grad_(False)
    configure_precision(model, 'h3_fp32')
    b = synthetic_h3_batch(frames=17, seed=321)
    b['prompt_embeds'] = b['prompt_embeds'].to(dtype)
    b['anchor_rows'] = torch.cat([b['anchor_rows']]*2).to(dtype)
    b['packed'] = hc.expand_packed_two_anchors(b['packed'], frame_rows=4)
    return model, b


def conditions(b, index, prompt=None, full=None):
    stop = min(index*5+5, b['clean_video'].shape[2])
    packed, trimmed = visible_inputs(full or b['packed'],
        b['prompt_embeds'] if prompt is None else prompt, stop, 4)
    anchor = b['anchor_rows']
    if index:
        anchor = torch.cat([anchor[:4], hc.last_frame_anchor(b['clean_video'][:, :, index*5-1:index*5]).to(anchor)])
    return dict(full_packed=packed, prompt=trimmed, anchor=anchor,
                audio=b['audio_latents'], chunk_frames=5, anchor_slot=1)


def cached(model, b, state, i, cache, *, sigma=.7, commit=False, prompt=None):
    return hc.chunk_forward(model, state, sigma=sigma, index=i, cache=cache,
        commit=commit, action_prefix_mode='own', action_feedback=True,
        anchor_frame_index=i*5-1 if i else None, **conditions(b, i, prompt))


def changed_prompt(b, first, stop):
    prompt = b['prompt_embeds'].clone()
    for lo, hi in b['packed']['action_text_spans_local'][first:stop]:
        prompt[lo:hi] = 3*prompt[lo:hi].flip(-1)+1
    return prompt


def cache_snapshot(cache):
    return [(e.index, *(x.clone() for x in (e.key, e.value, e.rope)))
            for es in cache.layers.values() for e in es]


def equal_cache(a, b):
    assert len(a) == len(b)
    for aa, bb in zip(a, b):
        assert aa[0] == bb[0]
        for x, y in zip(aa[1:], bb[1:]):
            torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_original_predicate_matches_actual_released_mask_on_visible_window():
    _, b = setup()
    for stop in (5, 10, 15, 17):
        packed, prompt = visible_inputs(b['packed'], b['prompt_embeds'], stop, 4)
        p, size = packed['action_video_start'], packed['seq_len']
        masks = _build_action_block_masks(packed['action_text_rows'], p, 4, stop,
            packed['cu_seqlens'], size, 'cpu', n_real=size)
        ref = create_mask(masks[0].mask_mod, 1, 1, size, size, device='cpu')[0, 0]
        actual = OriginalWindowAttention(packed, stop, 4).build_mask('cpu')
        assert torch.equal(ref, actual)
        assert len(packed['action_text_spans_local']) == stop
        assert len(prompt) == int(packed['text_pos'].numel())


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_first_chunk_identity_and_no_future_content(dtype):
    model, b = setup(dtype)
    state = b['noise'][:, :, :5]
    future = changed_prompt(b, 5, 17)
    with torch.no_grad():
        ref = window_forward(model, state, history=state[:, :, :0], sigma=.7,
                             index=0, **conditions(b, 0))
        for history in (False, True):
            with grounded_prefix(history=history):
                cache = hc.H3ChunkCache(2, 'cpu')
                out = cached(model, b, state, 0, cache)
                alternate = cached(model, b, state, 0, cache, prompt=future)
                torch.testing.assert_close(out, alternate, atol=0, rtol=0)
                torch.testing.assert_close(out, ref, atol=2e-6, rtol=2e-6)
    assert cache.nbytes == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_t1_future_isolation_history_use_and_replay_with_eviction(dtype):
    model, b = setup(dtype)
    cache = hc.H3ChunkCache(2, 'cpu')
    with torch.no_grad(), grounded_prefix():
        for i in range(4):
            state = b['noise'][:, :, i*5:min(i*5+5,17)]
            before = cache_snapshot(cache)
            out = cached(model, b, state, i, cache)
            if i < 3:
                future = changed_prompt(b, min(i*5+5,17), 17)
                alt = cached(model, b, state, i, cache, prompt=future)
                torch.testing.assert_close(out, alt, atol=0, rtol=0)
            rebuilt = hc.H3ChunkCache(2, 'cpu')
            for old in range(i):
                cached(model,b,b['clean_video'][:,:,old*5:old*5+5],old,rebuilt,sigma=0,commit=True)
            replay = cached(model,b,state,i,rebuilt)
            torch.testing.assert_close(out,replay,atol=0,rtol=0)
            equal_cache(before,cache_snapshot(cache))
            equal_cache(before,cache_snapshot(rebuilt))
            action = changed_prompt(b,i*5,min(i*5+5,17))
            alt = cached(model,b,state,i,cache,prompt=action)
            assert float((out-alt).float().abs().max()) > 1e-6
            if i:
                edited = copy.deepcopy(cache)
                for es in edited.layers.values():
                    for e in es: e.value.add_(.25)
                alt = cached(model,b,state,i,edited)
                assert float((out-alt).float().abs().max()) > 1e-6
            equal_cache(before,cache_snapshot(cache))
            clean = b['clean_video'][:,:,i*5:min(i*5+5,17)]
            cached(model,b,clean,i,cache,sigma=0,commit=True)
            assert all([e.index for e in es] == list(range(max(0,i-1),i+1)) for es in cache.layers.values())


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_t2_actual_action_sigma_changes_recompute_and_history_immutable(dtype):
    model,b=setup(dtype)
    with torch.no_grad():
        for i in (1,2,3):
            state=b['noise'][:,:,i*5:min(i*5+5,17)]
            hist=b['clean_video'][:,:,:i*5]
            original=hist.clone()
            kw=dict(history=hist,sigma=.7,index=i,history_chunks=2)
            out=window_forward(model,state,**kw,**conditions(b,i))
            repeat=window_forward(model,state,**kw,**conditions(b,i))
            torch.testing.assert_close(out,repeat,atol=0,rtol=0)
            act=changed_prompt(b,i*5,min(i*5+5,17))
            alt=window_forward(model,state,**kw,**conditions(b,i,act))
            assert float((out-alt).float().abs().max())>1e-6
            sigma=window_forward(model,state,**dict(kw,sigma=.3),**conditions(b,i))
            assert float((out-sigma).float().abs().max())>1e-6
            if i<3:
                future=changed_prompt(b,i*5+5,17)
                noleak=window_forward(model,state,**kw,**conditions(b,i,future))
                torch.testing.assert_close(out,noleak,atol=0,rtol=0)
            torch.testing.assert_close(hist,original,atol=0,rtol=0)


def test_unknown_future_rows_can_be_removed_without_changing_visible_contract():
    _,b=setup()
    # Successive physical deletion changes total text length and packed row
    # indices, while preserving the declared GLOBAL position coordinates.
    direct,prompt=visible_inputs(b['packed'],b['prompt_embeds'],5,4)
    for intermediate in (10,15,17):
        middle,text=visible_inputs(b['packed'],b['prompt_embeds'],intermediate,4)
        twice,other=visible_inputs(middle,text,5,4)
        assert direct.keys()==twice.keys()
        for key in direct:
            if torch.is_tensor(direct[key]):
                torch.testing.assert_close(direct[key],twice[key],atol=0,rtol=0)
            else: assert direct[key]==twice[key],key
        torch.testing.assert_close(prompt,other,atol=0,rtol=0)


def test_no_training_or_nested_patch_and_restore_on_exception():
    model,b=setup(); original=hc.ChunkAttention
    with pytest.raises(RuntimeError,match='no_grad'):
        with grounded_prefix():cached(model,b,b['noise'][:,:,:5],0,hc.H3ChunkCache())
    assert hc.ChunkAttention is original
    with grounded_prefix():
        with pytest.raises(RuntimeError,match='already active'):
            with grounded_prefix():pass
    assert hc.ChunkAttention is original
