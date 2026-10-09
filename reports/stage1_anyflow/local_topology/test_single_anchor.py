"""Real tiny-H3 checks for native single-I0 visible-window conditioning."""
from pathlib import Path
import sys
import pytest
import torch

BASE = Path(__file__).resolve().parent
RT = BASE / 'runtime'
sys.path[:0] = [str(RT/'code'), str(RT/'DiffSynth-Studio-h3-v2'), str(RT/'tests')]
from causal import h3_cached as hc
from causal.h3_training import synthetic_h3_batch
from causal.local_topology import window_forward, visible_inputs, grounded_prefix
from test_local_topology import setup
from native_prefix import native_prefix_variant
from single_anchor import original_image_window_variant

single = original_image_window_variant(window_forward)
native = native_prefix_variant(window_forward)
cached = native_prefix_variant(hc.chunk_forward)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_original_image_never_retimed_future_isolation_and_history_readonly(dtype):
    model, _ = setup(dtype)
    b = synthetic_h3_batch(frames=17, seed=321)
    b['prompt_embeds'] = b['prompt_embeds'].to(dtype)
    b['anchor_rows'] = b['anchor_rows'].to(dtype)
    captured = []
    def capture(module, args, kwargs):
        ids = kwargs['img_pos_info']['position_ids'].view(-1)
        captured.append((kwargs['img_position_ids'][0, ids[:4]].clone(),
                         kwargs['unique_timesteps'][kwargs['inverse_indices']].clone(),
                         kwargs['text_pos_info']['position_ids'].view(-1).clone()))
    hook = model.register_forward_pre_hook(capture, with_kwargs=True)
    image_pos = b['packed']['img_position_ids'][0, b['packed']['img_pos'][:4]]
    with torch.no_grad():
        for i in range(4):
            start, stop = i*5, min(i*5+5,17)
            layout, prompt = visible_inputs(b['packed'], b['prompt_embeds'], stop, 4)
            state = b['noise'][:,:,start:stop]
            history = b['clean_video'][:,:,:start]
            snapshot = history.clone()
            kw = dict(history=history, full_packed=layout, prompt=prompt,
                      anchor=b['anchor_rows'], audio=b['audio_latents'], sigma=.6,
                      index=i, chunk_frames=5, history_chunks=2)
            out = single(model, state, **kw)
            pos, time, text = captured[-1]
            torch.testing.assert_close(pos, image_pos, atol=0, rtol=0)
            torch.testing.assert_close(time[text], torch.full_like(time[text], .4), atol=1e-7, rtol=0)
            repeat = single(model, state, **kw)
            torch.testing.assert_close(out, repeat, atol=0, rtol=0)
            torch.testing.assert_close(history, snapshot, atol=0, rtol=0)
            if i == 0:
                ref = native(model, state, **kw)
                torch.testing.assert_close(out, ref, atol=0, rtol=0)
                with grounded_prefix():
                    cp = {k:v for k,v in kw.items() if k not in ('history','history_chunks')}
                    value = cached(model, state, cache=hc.H3ChunkCache(2,'cpu'),
                                   action_prefix_mode='own', action_feedback=True, **cp)
                torch.testing.assert_close(out, value, atol=2e-6, rtol=2e-6)
            future = b['prompt_embeds'].clone()
            for lo, hi in b['packed']['action_text_spans_local'][stop:]:
                future[lo:hi] += 3
            packed2, prompt2 = visible_inputs(b['packed'], future, stop, 4)
            alternate = single(model, state, **dict(kw, full_packed=packed2, prompt=prompt2))
            torch.testing.assert_close(out, alternate, atol=0, rtol=0)
    hook.remove()


def test_refuse_duplicate_anchor():
    model, b = setup()
    with pytest.raises(ValueError, match='Exactly one'):
        single(model, b['noise'][:,:,:5], anchor=b['anchor_rows'])
