"""CPU boolean-mask inventory on real packed layouts, no model predictions.

Intercepts the deployed ChunkAttention SDPA masks using zero Q/K/V placeholders.
This is a structural comparison, not an effect-size or trained quality test.
"""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'runtime/code'))
from causal.h3_cached import ChunkAttention, H3ChunkCache
from geometry_primitives import FullPrefixAttention


def main():
    torch.set_num_threads(4)
    prep = json.loads((BASE / 'counterfactual/preparation.json').read_text())
    result = dict(scope='CPU masks from actual packed layouts; no H3 prediction or numerical effect attribution', records=[])
    with torch.no_grad():
        for row in prep['records']:
            state = torch.load(row['file'], map_location='cpu', weights_only=True)
            packed = state['packed']; start, stop = state['start'], state['stop']
            p = int(packed['action_video_start']); rows = 390; n = (stop - start) * rows
            spans = packed['action_text_rows']; chunk = row['chunk']
            cache = H3ChunkCache(5, 'cpu')
            for k in range(chunk):
                z = torch.zeros(5 * rows, 1, 2)
                cache.commit(0, k, z, z, z)
            control = ChunkAttention(cache, chunk, p, rows, start, spans,
                                     action_prefix_mode='causal', action_feedback=True)
            captures = []
            def capture(q, k, v, *, attn_mask, **kwargs):
                captures.append(attn_mask.squeeze(0).squeeze(0).clone())
                return torch.zeros_like(q)
            q = torch.zeros(p + n, 1, 2)
            with patch('torch.nn.functional.scaled_dot_product_attention', capture):
                control.attend(q, q, q, rope_freqs=q, layer=0, apply_rope=lambda x, _: x, scale=1.)
            assert len(captures) == 2
            prefix_mask, video_mask = captures
            original = FullPrefixAttention(packed, stop, rows, 'original').build_mask('cpu')
            # Query a later frame inside the chunk so past actions exist even
            # for chunk0, independently of a historical KV cache.
            current_frame = start + 1
            current_video = p + current_frame * rows
            current_action = int(spans[current_frame, 0])
            previous_action = int(spans[current_frame - 1, 0])
            action_indices = {i for lo, hi in spans.tolist() for i in range(lo, hi)}
            common = next(i for i in range(p) if i not in action_indices)
            def pair(qi, ki):
                causal = prefix_mask[qi, ki] if qi < p else video_mask[qi - p - start * rows, ki]
                return dict(original=bool(original[qi, ki]), causal_cached=bool(causal))
            routes = dict(current_video_reads_own_action=pair(current_video, current_action),
                          own_action_reads_current_video=pair(current_action, current_video),
                          current_video_reads_earlier_action=pair(current_video, previous_action),
                          common_prefix_reads_current_video=pair(common, current_video))
            if start:
                past_action = int(spans[0, 0]); past_video = p
                routes['past_action_reads_own_historical_video'] = pair(past_action, past_video)
                routes['current_video_reads_historical_video_KV'] = pair(current_video, past_video)
            assert routes['current_video_reads_own_action'] == dict(original=True, causal_cached=True)
            assert routes['own_action_reads_current_video'] == dict(original=True, causal_cached=True)
            assert routes['current_video_reads_earlier_action'] == dict(original=False, causal_cached=True)
            assert routes['common_prefix_reads_current_video'] == dict(original=True, causal_cached=False)
            result['records'].append(dict(clip=row['clip'], chunk=chunk, layout_sha256=row['sha256'],
                                          prefix_rows=p, video_rows_per_latent=rows, routes=routes))
    result['source_hashes'] = {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__), BASE / 'runtime/code/causal/h3_cached.py', BASE / 'geometry_primitives.py']}
    result['status'] = 'passed'
    (BASE / 'attention_route_inventory.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
