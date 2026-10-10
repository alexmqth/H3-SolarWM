"""Independent CPU audit of one completed G1; generation quality is separate."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
from common import CFG, OUT, ROOT, setup_paths, verify_code_manifest, verify_sources

setup_paths()
import av
import numpy as np
import torch
from causal.anyflow_sampling import configure_video_schedule
from causal.local_topology import visible_inputs
from diffsynth.diffusion.flow_match import FlowMatchScheduler


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 << 20), b''):
            h.update(part)
    return h.hexdigest()


def tensor_sha(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(t.shape), str(t.dtype))).encode() +
                          t.view(torch.uint8).numpy().tobytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--scene', choices=list(CFG['scenes']), required=True)
parser.add_argument('--method', choices=['FM30'], required=True)
args = parser.parse_args()
verify_sources()
verify_code_manifest()
directory = OUT / 'G1' / args.scene / args.method
row = json.loads((directory / 'result.json').read_text())
assert row['status'] == 'complete_pending_visual_review'
assert row['scene'] == args.scene and row['method'] == args.method
assert row['runner_sha256'] == sha(EXP / 'run_teacher.py')
steps = 30 if args.method == 'FM30' else 8
assert [row[k] for k in ('sampling_forwards', 'commit_forwards', 'vae_decodes')] == [steps, 0, 1]
assert row['precision']['native_fp32_weights_restored']
assert row['peak_allocated_gib'] <= 44
entry = json.loads((OUT / 'P1_result.json').read_text())['fixtures'][args.scene]
fixture_path = ROOT / entry['path']
assert sha(fixture_path) == entry['sha256'] == row['fixture_sha256']
fixture = torch.load(fixture_path, map_location='cpu', weights_only=True)
assert tensor_sha(fixture['initial_noise'][:, :, :12].float()) == row['initial_noise_sha256']
layout, prompt = visible_inputs(fixture['packed'], fixture['prompts']['A'], 12, 390)
assert tensor_sha(prompt) == row['prompt_sha256']
assert tensor_sha(layout['img_position_ids']) == row['position_sha256']
expected_sigmas = configure_video_schedule(FlowMatchScheduler('MiniMax-H3'), steps=steps,
                                         grid='native', flow_shift=2.22)
assert row['sigmas'] == expected_sigmas
endpoint = directory / 'first12.pt'
assert sha(endpoint) == row['endpoint_sha256']
z = torch.load(endpoint, map_location='cpu', weights_only=True)
assert z.shape == (1, 24, 12, 30, 52) and torch.isfinite(z).all()
assert tensor_sha(z) == row['endpoint_tensor_sha256']
rgb_path = directory / 'published_39.npy'
assert sha(rgb_path) == row['published_sha256']
rgb = np.load(rgb_path)
assert rgb.shape == (39, 480, 832, 3) and rgb.dtype == np.uint8
video = directory / 'first39.mp4'
assert sha(video) == row['video_sha256']
with av.open(str(video)) as c:
    frames = list(c.decode(video=0))
    assert len(frames) == 39 and c.streams.video[0].average_rate == 24
    assert all((f.width, f.height) == (832, 480) for f in frames)
    assert all(b.pts > a.pts for a, b in zip(frames, frames[1:]))
result = {'task': 'EXP-014/v1', 'scene': args.scene, 'method': args.method,
          'independent_G1_execution_audit': 'PASS', 'visual_assessment': 'SEPARATE',
          'row_sha256': sha(directory / 'result.json'), 'endpoint_sha256': row['endpoint_sha256'],
          'published_sha256': row['published_sha256'], 'video_sha256': row['video_sha256'],
          'sampling_forwards': steps, 'commit_forwards': 0, 'vae_decodes': 1,
          'sampling_seconds': row['sampling_seconds'], 'decode_seconds': row['decode_seconds'],
          'wall_seconds': row['wall_seconds'], 'peak_allocated_gib': row['peak_allocated_gib']}
(HERE / f'G1_{args.scene}_{args.method}_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
