#!/usr/bin/env python3
"""Check the packaged legacy RGB adapters and a complete 39-frame acceptance run."""
import argparse
import json
from pathlib import Path
import sys
import av

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from causal.evaluate_videos import evaluate
from causal.evaluate_action_control import evaluate as action_flow

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
args = p.parse_args()
setup = json.loads((args.run/'setup.json').read_text())
run = json.loads((args.run/'cached.json').read_text())
manifest = json.loads((ROOT/'meeting/DEMO_PROVENANCE.json').read_text())
assert run['status'] == 'complete' and run['num_frames'] == 39
assert run['steps'] == 8 and run['denoiser_forwards'] == 24 and run['commit_forwards'] == 3
assert run['history_source'] == 'generated' and run['cache_device'] == 'cpu'
assert run['anchor_mode'] == 'dynamic_last_frame_rgb_dual'
assert run['action_prefix_mode'] == 'causal' and run['action_feedback']
assert setup['released_lora_pairs_loaded'] == manifest['released_h3']['lora_pairs']
assert setup['input_artifacts']['h3_checkpoint']['sha256'] == manifest['released_h3']['sha256']
assert setup['causal_adapter']['block_indices'] == list(range(34, 50))
assert setup['causal_action_adapter']['block_indices'] == list(range(42, 50))
assert run['seed'] == 13 and run['flow_shift'] == 2.22
assert run['causal_adapter_scope'] == 'all'
assert setup['config']['anyflow_adapter'] is None and setup['config']['stage1_lora'] is None
for field, asset in [('causal_adapter','checkpoints/visual_rgb_tail16/causal_adapter.pt'),
                     ('causal_action_adapter','checkpoints/visual_rgb_tail16/action_adapter.pt')]:
    assert setup['input_artifacts'][field]['sha256'] == manifest['assets'][asset]['sha256']
video = args.run/'cached.mp4'
metrics = evaluate(video)
assert metrics['frames'] == 39 and metrics['fps'] == 24 and metrics['codec'] == 'h264'
assert (metrics['width'], metrics['height']) == (832, 480)
with av.open(str(video)) as c:
    assert c.streams.video[0].codec_context.format.name == 'yuv420p'
flow = action_flow(video)
# Approximate RGB chunk boundaries in this legacy 5-latent protocol.
# These are descriptive gray MAD values, not a quality score or the RGB metric in the meeting table.
boundaries = [17, 34]
result = {'status':'pass', 'meaning':'Execution, correct artifact loading, cache lifecycle and measurement output; not a quality gate',
          'video':metrics, 'action_flow':flow, 'boundary_gray_MAD':
          {str(b):metrics['adjacent_frame_mad'][b-1] for b in boundaries},
          'end_to_end_seconds':run['wall_including_shared_setup_seconds'],
          'first_internal_chunk_seconds':run['chunk_seconds'][0],
          'chunk_seconds':run['chunk_seconds'],
          'GPU_peak_allocated_MiB':run['memory_entire_run_peak_MiB'],
          'CPU_raw_KV_peak_MiB':run['kv_cache_peak_MiB'],
          'noisy_forwards':run['denoiser_forwards'], 'clean_commits':run['commit_forwards'],
          'warmup':False, 'repeats':1, 'component_memory_breakdown':'not measured'}
(args.run/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('video','action_flow')},indent=2))
