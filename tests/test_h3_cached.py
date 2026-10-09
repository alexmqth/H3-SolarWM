from pathlib import Path
from dataclasses import replace
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'DiffSynth-Studio-h3-v2'))
sys.path.insert(0, str(ROOT/'code'))
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.pretrained_lora import replay_tail
from causal.h3_cached import (H3ChunkCache, chunk_forward, expand_packed_two_anchors,
                              recompute_forward, retime_anchor_position, slice_packed,
                              last_frame_anchor, last_frame_image_anchor)


def test_image_anchor_uses_first_decoded_frame_and_image_reencode():
    class FakeVAE:
        def decode_video(self, latent, **kwargs):
            assert kwargs["process_image"] is False
            # Match the short temporal padding used by the H3 video decoder.
            return torch.stack([latent[:, :3, 0] * 0 + i / 3 for i in range(4)], dim=2)

        def encode_video(self, rgb, **kwargs):
            assert kwargs["process_image"] is True
            assert rgb.shape[2] == 1
            return torch.zeros(1, 24, 1, 2, 2, dtype=rgb.dtype, device=rgb.device)

    latent = torch.ones(1, 24, 2, 4, 4, dtype=torch.bfloat16)
    rows = last_frame_image_anchor(FakeVAE(), latent)
    assert rows.shape == (1, 96)
    torch.testing.assert_close(rows, torch.zeros_like(rows))


def test_retime_anchor_uses_global_video_frame_grid():
    # img_pos is [one condition frame, three video frames]; the condition
    # starts at temporal position 100 while video frame 1 is at 201.
    source = {
        "img_pos": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        "img_position_ids": torch.tensor([[[100., 1., 2.], [100., 3., 4.],
                                            [200., 5., 6.], [200., 7., 8.],
                                            [201., 5., 6.], [201., 7., 8.],
                                            [202., 5., 6.], [202., 7., 8.]]]),
    }
    sliced = {
        "img_pos": torch.tensor([0, 1, 4, 5, 6, 7]),
        "img_position_ids": source["img_position_ids"][:, [0, 1, 4, 5, 6, 7]].clone(),
    }
    retimed = retime_anchor_position(sliced, source, anchor_rows=2,
                                     frame_index=1, frame_rows=2)
    torch.testing.assert_close(retimed["img_position_ids"][0, :2],
                               source["img_position_ids"][0, 4:6])
    # The video rows and source dictionaries remain untouched.
    torch.testing.assert_close(retimed["img_position_ids"][0, 2:], sliced["img_position_ids"][0, 2:])
    torch.testing.assert_close(source["img_position_ids"][0, :2],
                               torch.tensor([[100., 1., 2.], [100., 3., 4.]]))


def test_dual_anchor_expansion_preserves_original_and_remaps_suffix():
    # text(2), one image anchor(2), audio(1), two video frames(4), pad(1).
    full = {
        "img_pos": torch.tensor([2, 3, 5, 6, 7, 8]),
        "audio_pos": torch.tensor([4]),
        "text_pos": torch.tensor([0, 1]),
        "img_position_ids": torch.tensor([[[0., 0., 0.], [0., 1., 1.],
                                             [10., 0., 0.], [10., 1., 1.],
                                             [20., 0., 0.], [20., 1., 1.],
                                             [21., 0., 0.], [21., 1., 1.],
                                             [99., 0., 0.]]]),
        "token_tags": torch.tensor([0, 1, 0, 0, 2, 0, 0, 0, -1]),
        "cu_seqlens": torch.tensor([0, 8, 9], dtype=torch.int32),
        "seq_len": 9,
        "action_video_start": 5,
        "action_real_used": 8,
        "action_text_rows": torch.tensor([[1, 2]]),
        "action_text_spans_local": [(1, 2)],
    }
    dual = expand_packed_two_anchors(full, frame_rows=2)
    assert dual["seq_len"] == 11
    assert dual["action_video_start"] == 7
    assert dual["anchor_count"] == 2
    # Original anchor remains at its old positions; inserted anchor is before
    # audio, and all suffix positions are shifted by exactly one frame.
    assert dual["img_pos"][:2].tolist() == [2, 3]
    assert dual["img_pos"][2:4].tolist() == [4, 5]
    assert dual["audio_pos"].tolist() == [6]
    assert dual["img_pos"][4:].tolist() == [7, 8, 9, 10]
    torch.testing.assert_close(dual["img_position_ids"][0, 4:6],
                               full["img_position_ids"][0, 5:7])
    # The action text span is before the insertion point and is unchanged.
    assert dual["action_text_rows"].tolist() == [[1, 2]]
    sliced = slice_packed(dual, 1, 2, 2)
    retimed = retime_anchor_position(sliced, dual, anchor_rows=4,
                                     frame_index=1, frame_rows=2, anchor_slot=1)
    target = retimed["img_pos"][2:4]
    expected = dual["img_position_ids"][0, dual["img_pos"][4 + 1 * 2:4 + 1 * 2 + 2]]
    torch.testing.assert_close(retimed["img_position_ids"][0, target], expected)
    # The first anchor slot is still the original scene grid.
    torch.testing.assert_close(retimed["img_position_ids"][0, retimed["img_pos"][:2]],
                               dual["img_position_ids"][0, dual["img_pos"][:2]])


def test_actual_multilayer_h3_cached_equals_recomputed_after_eviction_and_partial_tail():
    torch.set_num_threads(2)
    torch.manual_seed(8)
    model = make_small_h3().eval()
    batch = synthetic_h3_batch(frames=22)
    clean = batch['clean_video']
    common = dict(full_packed=batch['packed'], prompt=batch['prompt_embeds'],
                  anchor=batch['anchor_rows'], audio=batch['audio_latents'], chunk_frames=5)
    cache = H3ChunkCache(max_history=2, storage_device='cpu')
    with torch.no_grad():
        for i, start in enumerate(range(0, clean.shape[2], 5)):
            current = batch['noise'][:, :, start:start+5]
            for sigma in [.8, .3]:
                cached = chunk_forward(model, current, index=i, cache=cache, sigma=sigma, **common)
                reference = recompute_forward(model, current, history=clean[:, :, :start],
                                               sigma=sigma, window_chunks=3, **common)
                torch.testing.assert_close(cached, reference, atol=2e-6, rtol=2e-5)
            chunk_forward(model, clean[:, :, start:start+5], index=i, cache=cache,
                          sigma=0., commit=True, **common)
    assert cache.commits == 10  # 5 chunks * 2 layers, not denoising forwards
    assert all([e.index for e in entries] == [3, 4] for entries in cache.layers.values())
    assert cache.peak_bytes > 0
    cache.clear()
    assert cache.nbytes == 0


def test_cache_rejects_grad_duplicate_skip_and_noisy_commit():
    cache = H3ChunkCache()
    k = torch.randn(3, 2, 12)
    r = torch.randn(3, 12)
    with pytest.raises(RuntimeError, match='no_grad'):
        cache.commit(0, 0, k, k, r)
    with torch.no_grad():
        cache.commit(0, 0, k, k, r)
        with pytest.raises(RuntimeError, match='history mismatch'):
            cache.commit(0, 0, k, k, r)
        with pytest.raises(RuntimeError, match='history mismatch'):
            cache.commit(0, 2, k, k, r)


def test_dynamic_anchor_tail_replay_preserves_history_and_allows_current_gradients():
    """Regression: reusing today's anchor to recompute yesterday's K/V is wrong."""
    torch.set_num_threads(2)
    torch.manual_seed(8)
    model = make_small_h3().eval()
    batch = synthetic_h3_batch(frames=10)
    clean = batch['clean_video']
    full = expand_packed_two_anchors(batch['packed'], frame_rows=4)
    common = dict(full_packed=full, prompt=batch['prompt_embeds'],
                  audio=batch['audio_latents'], chunk_frames=5)
    cache = H3ChunkCache(2, 'cpu')
    with torch.no_grad():
        chunk_forward(model, clean[:, :, :5], index=0, cache=cache, sigma=0., commit=True,
                      anchor=torch.cat([batch['anchor_rows']]*2), **common)
    # Capture the final trainable block, retaining the version of history
    # that was produced with chunk 0's own anchor.
    captured = {}
    def hook(module, positional, keyword):
        captured['x'] = positional[0].detach().clone()
        ctrl = keyword['causal_control']
        captured['kw'] = dict(keyword, causal_control=replace(
            ctrl, cache=cache.snapshot([0, 1]), allow_grad_read=True, _masks={}))
    def after(module, positional, output):
        captured['reference'] = output.detach().clone()
    h = model.blocks[0].register_forward_pre_hook(hook, with_kwargs=True)
    h2 = model.blocks[-1].register_forward_hook(after)
    anchor = torch.cat([batch['anchor_rows'], last_frame_anchor(clean[:, :, 4:5])])
    dynamic = dict(common, anchor=anchor, anchor_frame_index=4, anchor_slot=1)
    with torch.no_grad():
        cached = chunk_forward(model, batch['noise'][:, :, 5:], index=1,
                               cache=cache, sigma=.6, **dynamic)
    h.remove(); h2.remove()
    # Discarding or advancing the live cache must not change captured features.
    cache.clear()
    control = captured['kw']['causal_control']
    assert [e.index for e in control.cache.history(1, 1)] == [0]
    replay = replay_tail(model.blocks, [0, 1], captured['x'], captured['kw'])
    torch.testing.assert_close(replay, captured['reference'], atol=0, rtol=0)
    # Regression: passing the captured first layer_index to every block reads
    # that layer's K/V in all tail layers, even with zero adapter weights.
    with torch.no_grad():
        wrong_tail = captured['x']
        for block in model.blocks:
            wrong_tail = block(wrong_tail, **captured['kw'])
        assert float((wrong_tail - captured['reference']).abs().max()) > 1e-5
    replay.square().mean().backward()
    assert model.blocks[-1].attn.qkv_proj.lora_B.grad.abs().sum() > 0
    assert all(not e.key.requires_grad and not e.value.requires_grad
               for e in control.cache.history(1, 1))
    with torch.no_grad():
        wrong = recompute_forward(model, batch['noise'][:, :, 5:],
                                  history=clean[:, :, :5], sigma=.6, window_chunks=3, **dynamic)
    assert float((cached-wrong).abs().max()) > 1e-5
