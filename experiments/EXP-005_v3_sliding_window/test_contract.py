"""CPU-only EXP-005 protocol checks. Never loads 33B weights or raw real KV."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb"
ROUTER = ROOT / "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate"
sys.path[:0] = [str(FROZEN / "runtime/code"),
                str(FROZEN / "runtime/DiffSynth-Studio-h3-v2"),
                str(ROOT / "submission/experiments/EXP-001_v2b_124"),
                str(ROUTER), str(HERE)]
torch.set_num_threads(4)

from causal.h3_cached import ChunkAttention, H3ChunkCache, slice_packed
from causal.local_topology import visible_inputs
from chunk_plan import PLAN, cache_identity, validate_cache
from current_prefix import current_prefix_feedback
from diffsynth.models.minimax_h3_dit import MiniMaxH3Rope, _apply_rope
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder
from interval_sw import interval_sw
from position_sw import (assert_frozen_past_preserved, assert_native_video_grid,
                         localize_window, native_times)
from rollout_contract import prompt_for_path, stitch_immutable


def load_accepted_interval(name: str, filename: str):
    path = ROOT / "submission/experiments" / filename / "interval_cached.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.interval_cached


class TinyDit:
    _h3_input_dtype = torch.float32

    def __init__(self):
        self.rope = MiniMaxH3Rope(16)


def toy_layout(stop: int, *, rows: int = 2, origin: float = 100.):
    """Structural-only fixture; dummy text is not a valid H3 action prompt."""
    text = stop
    anchor = rows
    audio = 2
    prefix = text + anchor + audio
    seq = prefix + stop * rows
    position = torch.zeros((1, seq, 3), dtype=torch.float64)
    position[0, :text, 0] = torch.arange(text)
    position[0, text:text+anchor, 0] = origin
    position[0, text+anchor:prefix, 0] = origin
    grid = native_times(stop, origin)
    position[0, prefix:, 0] = grid.repeat_interleave(rows)
    position[0, prefix:, 2] = torch.arange(rows).repeat(stop)
    position[0, text:text+anchor, 2] = torch.arange(rows)
    position[0, text+anchor:prefix, 2] = torch.arange(audio)
    tags = torch.ones(seq, dtype=torch.long)
    tags[text:text+anchor] = 0
    tags[text+anchor:prefix] = 2
    tags[prefix:] = 0
    packed = dict(img_position_ids=position, token_tags=tags,
        text_pos=torch.arange(text),
        img_pos=torch.cat((torch.arange(text, text+anchor), torch.arange(prefix, seq))),
        audio_pos=torch.arange(text+anchor, prefix),
        action_text_rows=torch.tensor([(i, i+1) for i in range(stop)]),
        action_text_spans_local=[(i, i+1) for i in range(stop)],
        action_video_start=prefix, seq_len=seq,
        cu_seqlens=torch.tensor([0, seq], dtype=torch.int32), action_real_used=seq)
    prompt = torch.arange(text*4, dtype=torch.float32).reshape(text, 4)
    return packed, prompt


def toy_cache(index: int, layout: dict, *, rows: int = 2, layers: int = 2):
    cache = H3ChunkCache(max_history=5, storage_device="cpu")
    rope = MiniMaxH3Rope(16)
    p = int(layout["action_video_start"])
    with torch.no_grad():
        for chunk in range(index):
            start, stop = PLAN.span(chunk)
            n = (stop - start) * rows
            native = rope(layout["img_position_ids"][:, p+start*rows:p+stop*rows])
            for layer in range(layers):
                key = torch.arange(n*128, dtype=torch.float32).reshape(n, 1, 128) * .0001 + chunk + layer
                cache.commit(layer, chunk, key, key + .25, native)
    return cache


def fake_model_fn(dit, current, audio, packed, prompt, *, causal_control, **kwargs):
    p = int(packed["action_video_start"])
    n = p + current.shape[2] * causal_control.frame_rows
    gen = torch.Generator().manual_seed(101)
    q, k, v = [torch.randn(n, 1, 128, generator=gen) for _ in range(3)]
    freqs = dit.rope(packed["img_position_ids"])
    out = causal_control.attend(q, k, v, rope_freqs=freqs, layer=0,
                                apply_rope=_apply_rope, scale=128**-.5)
    if causal_control.commit:
        for layer in range(1, 2):
            causal_control.cache.commit(layer, causal_control.index,
                k[p:], v[p:], freqs[p:])
    return out, None


@pytest.fixture(scope="module")
def parking():
    return {a: torch.load(FROZEN / f"source_coarse/inputs/parking_{a}.pt",
                          map_location="cpu", weights_only=True) for a in "AD"}


def test_chunk_plan_native_intervals_and_ancestors():
    assert [PLAN.span(i) for i in range(10)] == [
        (0,12),(12,17),(17,22),(22,27),(27,32),(32,37),
        (37,42),(42,47),(47,52),(52,57)]
    assert PLAN.ancestors(5) == (0,1,2,3,4)
    assert PLAN.ancestors(6) == (1,2,3,4,5)
    assert PLAN.ancestors(8) == (3,4,5,6,7)
    assert PLAN.history_start(6) == 12
    assert [PLAN.rgb_stop(i) for i in range(8)] == [39,56,73,90,107,124,141,158]
    with pytest.raises(ValueError): PLAN.span(-1)


def test_actual_cache_commit_eviction_and_capacity():
    layout, _ = toy_layout(57)
    cache = toy_cache(5, layout)
    assert validate_cache(cache, plan=PLAN, index=5, frame_rows=2, expected_layers=2)["ancestors"] == [0,1,2,3,4]
    old = {e.index: (e, e.key.data_ptr(), e.value.data_ptr(), e.rope.data_ptr())
           for e in cache.layers[0]}
    with torch.no_grad():
        for layer in range(2):
            cache.commit(layer, 5, torch.ones(10,1,128), torch.ones(10,1,128),
                         MiniMaxH3Rope(16)(layout["img_position_ids"][:,
                             int(layout["action_video_start"])+32*2:
                             int(layout["action_video_start"])+37*2]))
    audit = validate_cache(cache, plan=PLAN, index=6, frame_rows=2, expected_layers=2)
    assert audit["ancestors"] == [1,2,3,4,5] and audit["rows_per_layer"] == 50
    assert 0 not in [e.index for e in cache.layers[0]]
    for e in cache.layers[0]:
        if e.index in old:
            before, kp, vp, rp = old[e.index]
            assert e is before and (e.key.data_ptr(),e.value.data_ptr(),e.rope.data_ptr()) == (kp,vp,rp)
    assert cache.nbytes < cache.peak_bytes or cache.nbytes == cache.peak_bytes
    for index in (6,7,8):
        with torch.no_grad():
            start, stop = PLAN.span(index)
            rope = MiniMaxH3Rope(16)(layout["img_position_ids"][:,
                int(layout["action_video_start"])+start*2:
                int(layout["action_video_start"])+stop*2])
            for layer in range(2):
                cache.commit(layer,index,torch.ones(10,1,128),torch.ones(10,1,128),rope)
        audit = validate_cache(cache, plan=PLAN, index=index+1, frame_rows=2, expected_layers=2)
        assert audit["rows_per_layer"] == 50


def test_empty_missing_wrong_cache_is_rejected():
    layout, _ = toy_layout(42)
    cache = H3ChunkCache(5,"cpu")
    with pytest.raises(RuntimeError): validate_cache(cache,plan=PLAN,index=6,frame_rows=2,expected_layers=2)
    cache = toy_cache(6,layout)
    del cache.layers[1]
    with pytest.raises(RuntimeError): validate_cache(cache,plan=PLAN,index=6,frame_rows=2,expected_layers=2)
    cache = toy_cache(6,layout)
    cache.layers[0][0].index = 0
    with pytest.raises(RuntimeError): validate_cache(cache,plan=PLAN,index=6,frame_rows=2,expected_layers=2)


def test_real_parking_native_grid_and_future_isolation(parking):
    full = parking["A"]["packed"]
    origin = float(full["img_position_ids"][0, int(full["action_video_start"]), 0])
    full_visible, _ = visible_inputs(full,prompt_for_path(parking,"AA",37),37,390)
    assert_native_video_grid(full_visible, frame_rows=390, stop=37, origin=origin)
    for index in range(2,6):
        start, stop = PLAN.span(index)
        prompt = prompt_for_path(parking,"AA",stop)
        visible, text = visible_inputs(full,prompt,stop,390)
        assert len(visible["action_text_rows"]) == stop
        assert len(text) == len(visible["text_pos"])
        assert int(visible["seq_len"]) == int(visible["action_video_start"]) + stop*390
        sliced = slice_packed(visible,start,stop,390)
        assert torch.equal(sliced["img_position_ids"][:, -5*390:],
            full["img_position_ids"][:,int(full["action_video_start"])+start*390:
                                          int(full["action_video_start"])+stop*390])
        assert all(int(hi)<=int(visible["action_video_start"])
                   for lo,hi in visible["action_text_rows"])
        assert len(visible["action_text_rows"]) == stop  # no action rows after visible stop


@pytest.mark.parametrize("index",[2,3,4,5])
def test_global_old_interval_protocol_and_attention_exact(monkeypatch,index):
    from diffsynth.pipelines import minimax_h3_audio_video as pipeline
    calls=[]
    def recording_model_fn(dit,current,audio,packed,prompt,*,causal_control,**kwargs):
        calls.append(dict(current=current.clone(),audio=audio.clone(),
            prompt=prompt.clone(),anchor=kwargs["keyframe_cond_anchor"].clone(),
            positions=packed["img_position_ids"].clone(),
            action_rows=packed["action_text_rows"].clone(),
            video_timestep=kwargs["timestep_video"].clone(),
            audio_timestep=kwargs["timestep_audio"].clone(),
            fixed_prefix_timesteps=kwargs["fixed_prefix_timesteps"],
            action_prefix_mode=causal_control.action_prefix_mode,
            action_feedback=causal_control.action_feedback,
            frame_start=causal_control.frame_start,
            prefix=causal_control.prefix))
        return fake_model_fn(dit,current,audio,packed,prompt,
            causal_control=causal_control,**kwargs)
    monkeypatch.setattr(pipeline,"model_fn_minimax_h3",recording_model_fn)
    old = load_accepted_interval("accepted_exp003", "EXP-003_native_cached_124")
    start, stop = PLAN.span(index)
    layout, prompt = toy_layout(stop)
    cache = toy_cache(index,layout)
    current = torch.zeros(1,24,5,2,4)
    anchor = torch.zeros(2,4)
    audio = torch.zeros(2,4)
    dit = TinyDit()
    with torch.no_grad(),current_prefix_feedback():
        expected = old(dit,current,start=start,index=index,cache=cache,
            full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,sigma=.37)
        actual,audit = interval_sw(dit,current,start=start,index=index,cache=cache,
            full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,sigma=.37,
            expected_layers=2,return_audit=True)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert len(calls)==2
    for field in ("current","audio","prompt","anchor","positions",
                  "action_rows","video_timestep","audio_timestep"):
        torch.testing.assert_close(calls[0][field],calls[1][field],atol=0,rtol=0)
    for field in ("fixed_prefix_timesteps","action_prefix_mode",
                  "action_feedback","frame_start","prefix"):
        assert calls[0][field]==calls[1][field]
    assert float(calls[1]["video_timestep"])==370.
    assert float(calls[1]["audio_timestep"])==1000.
    assert calls[1]["action_prefix_mode"]=="own" and calls[1]["action_feedback"]
    assert not calls[1]["fixed_prefix_timesteps"]
    assert audit["cache_before"]["ancestors"] == list(range(index))
    assert audit["audio_timestep"] == 1000. and audit["video_sigma"] == .37
    assert not audit["fixed_prefix_timesteps"] and audit["action_feedback"]


def test_local_pre_eviction_identity_and_post_eviction_canonical_storage():
    frozen,_ = toy_layout(37)
    layout,_ = toy_layout(42)
    cache = toy_cache(6,layout)
    dit = TinyDit()
    old_pos = layout["img_position_ids"].clone()
    prefix = int(layout["action_video_start"])
    span = PLAN.span(6)
    packed = slice_packed(layout,*span,2)
    localized,view,audit = localize_window(canonical_cache=cache,packed=packed,
        visible_layout=layout,plan=PLAN,index=6,frame_rows=2,rope_fn=dit.rope,origin=100.)
    assert audit["oldest_latent"] == 12 and audit["ancestors"] == [1,2,3,4,5]
    assert torch.equal(localized["img_position_ids"][:,:prefix],packed["img_position_ids"][:,:prefix])
    assert torch.equal(localized["img_position_ids"][0,prefix:,0],
                       native_times(30,100.)[25:30].repeat_interleave(2))
    assert not torch.equal(localized["img_position_ids"][0,prefix:,0],packed["img_position_ids"][0,prefix:,0])
    e0=cache.history(0,6)[0]
    le0=view.history(0,6)[0]
    assert le0.key is e0.key and le0.value is e0.value and le0.rope is not e0.rope
    assert torch.equal(layout["img_position_ids"],old_pos)
    current_rope=dit.rope(localized["img_position_ids"][:,prefix:])
    with torch.no_grad():
        view.commit(0,6,torch.ones(10,1,128),torch.ones(10,1,128),current_rope)
    canonical=dit.rope(layout["img_position_ids"][:,prefix+37*2:prefix+42*2])
    assert torch.equal(cache.layers[0][-1].rope,canonical)
    # This toy commit only used one layer; it deliberately cannot be a valid full cache.
    with pytest.raises(RuntimeError): validate_cache(cache,plan=PLAN,index=7,frame_rows=2,expected_layers=2)
    packed5=slice_packed(frozen,*PLAN.span(5),2)
    c5=toy_cache(5,frozen)
    same, same_cache, note = localize_window(canonical_cache=c5,packed=packed5,
        visible_layout=frozen,plan=PLAN,index=5,frame_rows=2,rope_fn=dit.rope,origin=100.)
    assert same is packed5 and same_cache is c5 and not note["remapped"]


def test_actual_h3_rope_prefix_cross_modal_sensitivity():
    layout,_=toy_layout(42)
    dit=TinyDit()
    p=int(layout["action_video_start"])
    global_pos=layout["img_position_ids"][:,p+37*2:p+42*2]
    local_pos=global_pos.clone()
    local_pos[0,:,0]=native_times(30,100.)[25:30].repeat_interleave(2)
    query=torch.randn(10,1,128,generator=torch.Generator().manual_seed(12))
    key=torch.randn(p,1,128,generator=torch.Generator().manual_seed(13))
    pref=dit.rope(layout["img_position_ids"][:,:p])
    assert torch.equal(pref,dit.rope(layout["img_position_ids"][:,:p]))
    rotated_key=_apply_rope(key,pref)
    global_score=_apply_rope(query,dit.rope(global_pos))[:,0] @ rotated_key[:,0].T
    local_score=_apply_rope(query,dit.rope(local_pos))[:,0] @ rotated_key[:,0].T
    assert not torch.allclose(global_score,local_score)
    # Prefix text/action/I0/audio stays global, yet video-to-prefix logits change.


def test_post_eviction_interval_global_local_and_long_fixture_gate(monkeypatch):
    from diffsynth.pipelines import minimax_h3_audio_video as pipeline
    monkeypatch.setattr(pipeline,"model_fn_minimax_h3",fake_model_fn)
    frozen, old_prompt = toy_layout(37)
    extended, prompt = toy_layout(42)
    cache = toy_cache(6,extended)
    dit = TinyDit()
    current = torch.zeros(1,24,5,2,4)
    inputs = dict(start=37,index=6,cache=cache,full_packed=extended,prompt=prompt,
                  anchor=torch.zeros(2,4),audio=torch.zeros(2,4),sigma=.41,
                  expected_layers=2)
    with torch.no_grad(), current_prefix_feedback():
        with pytest.raises(RuntimeError,match="separately approved native fixture"):
            interval_sw(dit,current,**inputs)
        with pytest.raises(RuntimeError,match="certified long native fixture"):
            interval_sw(dit,current,structural_probe_only=True,**inputs)
        before=cache_identity(cache)
        global_result,g=interval_sw(dit,current,mode="global",return_audit=True,
            frozen_reference=frozen,frozen_reference_prompt=old_prompt,
            structural_probe_only=True,**inputs)
        local_result,l=interval_sw(dit,current,mode="local",return_audit=True,
            frozen_reference=frozen,frozen_reference_prompt=old_prompt,
            structural_probe_only=True,**inputs)
        assert cache_identity(cache)==before
    assert g["cache_before"]["ancestors"]==[1,2,3,4,5]
    assert l["position"]["oldest_latent"]==12 and l["position"]["remapped"]
    assert not torch.allclose(global_result,local_result)
    bad=dict(inputs);bad["full_packed"],bad["prompt"]=toy_layout(47)
    with torch.no_grad(),current_prefix_feedback(),pytest.raises(ValueError,match="visibility-trimmed"):
        interval_sw(dit,current,frozen_reference=frozen,
                    frozen_reference_prompt=old_prompt,structural_probe_only=True,**bad)


def test_extension_guard_and_default_builder_drift():
    frozen,prompt37=toy_layout(37)
    candidate,prompt42=toy_layout(42)
    assert_frozen_past_preserved(candidate,frozen,frame_rows=2)
    assert torch.equal(prompt42[:len(prompt37)],prompt37)
    bad=dict(candidate); bad["img_position_ids"]=candidate["img_position_ids"].clone()
    bad["img_position_ids"][0,bad["img_pos"][2],0] += 1
    with pytest.raises(RuntimeError,match="first37_video"):
        assert_frozen_past_preserved(bad,frozen,frame_rows=2)
    builder=MiniMaxH3Unit_PackedSequenceBuilder()
    old_spans=[(300+i,301+i) for i in range(37)]
    new_spans=old_spans+[(600+i,601+i) for i in range(5)]
    old=builder._build_packed_fl2va(600,37,2,4,1,[0],action_text_spans=old_spans)
    rebuilt=builder._build_packed_fl2va(605,42,2,4,1,[0],action_text_spans=new_spans)
    # The released builder mirrors action positions using the *new* full
    # video span and moves I0/video with text_len. Never call this unchanged
    # to extend a frozen 37-latent rollout.
    with pytest.raises(RuntimeError):
        assert_frozen_past_preserved(rebuilt,old,frame_rows=2)


def test_clean_commit_gate_future_rejection_and_rgb_append(monkeypatch):
    from diffsynth.pipelines import minimax_h3_audio_video as pipeline
    monkeypatch.setattr(pipeline,"model_fn_minimax_h3",fake_model_fn)
    layout,prompt=toy_layout(17)
    cache=toy_cache(1,layout)
    dit=TinyDit(); current=torch.zeros(1,24,5,2,4)
    kw=dict(start=12,index=1,cache=cache,full_packed=layout,prompt=prompt,
            anchor=torch.zeros(2,4),audio=torch.zeros(2,4),expected_layers=2)
    with torch.no_grad(), current_prefix_feedback():
        with pytest.raises(ValueError): interval_sw(dit,current,sigma=.2,commit=True,**kw)
        _,audit=interval_sw(dit,current,sigma=0,commit=True,return_audit=True,**kw)
    assert audit["cache_after"]["ancestors"] == [0,1]
    assert cache.commits==4
    with pytest.raises(ValueError):
        with torch.no_grad(), current_prefix_feedback():interval_sw(dit,current,start=13,index=1,cache=cache,
            full_packed=layout,prompt=prompt,anchor=torch.zeros(2,4),
            audio=torch.zeros(2,4),sigma=.2,expected_layers=2)
    old=torch.zeros(39,2,2,3,dtype=torch.uint8)
    decoded=torch.ones(56,2,2,3,dtype=torch.uint8)
    import numpy as np
    appended=stitch_immutable(old.numpy(),decoded.numpy(),12,17)
    assert np.array_equal(appended[:39],old.numpy()) and len(appended)==56


def test_interval_requires_current_prefix_feedback(monkeypatch):
    from diffsynth.pipelines import minimax_h3_audio_video as pipeline
    monkeypatch.setattr(pipeline,"model_fn_minimax_h3",fake_model_fn)
    layout,prompt=toy_layout(17)
    cache=toy_cache(1,layout)
    with torch.no_grad(), pytest.raises(RuntimeError, match="current_prefix_feedback"):
        interval_sw(TinyDit(),torch.zeros(1,24,5,2,4),start=12,index=1,
                    cache=cache,full_packed=layout,prompt=prompt,
                    anchor=torch.zeros(2,4),audio=torch.zeros(2,4),sigma=.2,
                    expected_layers=2)
