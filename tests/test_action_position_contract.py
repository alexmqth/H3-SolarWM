"""Future action LENGTH must not alter past outputs through RoPE metadata."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'DiffSynth-Studio-h3-v2')]
from causal.anyflow import install_anyflow
from causal.h3_cached import H3ChunkCache, chunk_forward
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.position_contract import fixed_action_position_origin, initial_action_text_reference
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder


@pytest.mark.parametrize('anyflow', [False, True])
def test_future_content_and_clause_length_invariance_with_fixed_origin(anyflow):
    torch.set_num_threads(2)
    torch.manual_seed(49)
    model = make_small_h3().eval()
    if anyflow:
        install_anyflow(model)
    batch = synthetic_h3_batch(frames=12, seed=31)
    common = dict(anchor=batch['anchor_rows'], audio=batch['audio_latents'],
        chunk_frames=5, action_prefix_mode='causal', action_feedback=True)
    def predict(packed, prompt):
        return chunk_forward(model, batch['noise'][:, :, :5], sigma=.7,
            target_sigma=.2 if anyflow else None, index=0,
            cache=H3ChunkCache(5, 'cpu'), full_packed=packed, prompt=prompt, **common)
    past_end = 16 + 5 * 8
    prompt = batch['prompt_embeds']
    changed_content = prompt.clone()
    changed_content[past_end:] += 5 * torch.randn_like(changed_content[past_end:])
    spans, pieces, cursor = [], [prompt[:16]], 16
    for i in range(12):
        length = 8 if i < 5 else 10
        spans.append((cursor, cursor + length))
        cursor += length
        pieces.append(prompt[16+i*8:16+(i+1)*8] if i < 5 else torch.randn(length, 32))
    longer = MiniMaxH3Unit_PackedSequenceBuilder()._build_packed_fl2va(
        cursor, 12, 4, 4, 2, [0], action_text_spans=spans)
    reference = initial_action_text_reference(batch['packed'])
    assert reference == len(prompt) == initial_action_text_reference(longer)
    fixed = fixed_action_position_origin(longer, reference_text_length=reference)
    unchanged = fixed_action_position_origin(batch['packed'], reference_text_length=reference)
    torch.testing.assert_close(unchanged['img_position_ids'], batch['packed']['img_position_ids'], atol=0, rtol=0)
    with torch.no_grad():
        original = predict(batch['packed'], prompt)
        torch.testing.assert_close(predict(batch['packed'], changed_content), original, atol=0, rtol=0)
        # Existing/native packing leaks future length through global origin.
        assert float((predict(longer, torch.cat(pieces)) - original).abs().max()) > 1e-4
        torch.testing.assert_close(predict(fixed, torch.cat(pieces)), original, atol=2e-6, rtol=2e-5)
    # No mutation of the caller's native packing; repeated use is idempotent.
    assert float(longer['img_position_ids'][0, int(longer['action_video_start']), 0]) == cursor
    torch.testing.assert_close(fixed_action_position_origin(fixed, reference_text_length=reference)['img_position_ids'],
                               fixed['img_position_ids'], atol=0, rtol=0)
    with pytest.raises(ValueError, match='established'):
        fixed_action_position_origin(fixed, reference_text_length=reference+1)
