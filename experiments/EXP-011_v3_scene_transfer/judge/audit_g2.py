"""Independent CPU audit of one completed scene/method G2, including real KV."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
from common import OUT, ROOT, setup_paths, verify_code_manifest, verify_sources

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
parser.add_argument('--scene', choices=['industrial', 'village'], required=True)
parser.add_argument('--method', choices=['FM30', 'FM8'], required=True)
args = parser.parse_args()
verify_sources()
verify_code_manifest()
directory = OUT / 'G2' / args.scene / args.method
row = json.loads((directory / 'result.json').read_text())
assert row['status'] == 'complete_pending_visual_review'
assert row['scene'] == args.scene and row['method'] == args.method
assert row['runner_sha256'] == sha(EXP / 'run_exp011.py')
steps = 30 if args.method == 'FM30' else 8
assert [row[k] for k in ('sampling_forwards', 'commit_forwards', 'vae_decodes')] == [2*steps, 1, 2]
assert row['precision']['native_fp32_weights_restored'] and row['peak_allocated_gib'] <= 44
first = OUT / 'G1' / args.scene / args.method
g1_audit = json.loads((HERE / f'G1_{args.scene}_{args.method}_audit.json').read_text())
assert row['first_endpoint_sha256'] == sha(first / 'first12.pt') == g1_audit['endpoint_sha256']
assert row['first_published_sha256'] == sha(first / 'published_39.npy') == g1_audit['published_sha256']
old_rgb = np.load(first / 'published_39.npy')
entry = json.loads((OUT / 'P1_result.json').read_text())['fixtures'][args.scene]
assert sha(ROOT / entry['path']) == entry['sha256'] == row['fixture_sha256']
fixture = torch.load(ROOT / entry['path'], map_location='cpu', weights_only=True)
cache_path = directory / 'cache_through12.pt'
assert sha(cache_path) == row['cache_sha256']
cache = torch.load(cache_path, map_location='cpu', weights_only=False)
assert set(cache.layers) == set(range(50))
assert cache.commits == 50 and cache.max_history == 5 and cache.storage_device == 'cpu'
for entries in cache.layers.values():
    assert len(entries) == 1 and entries[0].index == 0
    e = entries[0]
    assert e.key.shape == e.value.shape and e.key.shape[0] == e.rope.shape[0] == 12*390
    assert all(not x.requires_grad for x in (e.key, e.value, e.rope))
cache_bytes = cache.nbytes
assert cache_bytes == row['cpu_raw_kv_bytes'] == 6799104000
del cache
sigmas = configure_video_schedule(FlowMatchScheduler('MiniMax-H3'), steps=steps,
                                 grid='native', flow_shift=2.22)
branches = {}
for branch in ('AA', 'AD'):
    base = directory / branch
    part = json.loads((base / 'result.json').read_text())
    assert part['status'] == 'complete_pending_visual_review'
    assert part['sampling_forwards'] == steps and part['vae_decodes'] == 1
    assert part['source_cache_sha256'] == row['cache_sha256']
    assert part['first_endpoint_sha256'] == row['first_endpoint_sha256']
    assert part['sigmas'] == sigmas and part['cpu_raw_kv_bytes'] == cache_bytes
    assert part['peak_allocated_gib'] <= 44
    assert part['initial_noise_sha256'] == tensor_sha(fixture['initial_noise'][:, :, 12:17].float())
    prompt = fixture['prompts']['A'].clone()
    if branch == 'AD':
        for lo, hi in fixture['packed']['action_text_spans_local'][12:17]:
            prompt[lo:hi] = fixture['prompts']['D'][lo:hi]
    layout, prompt = visible_inputs(fixture['packed'], prompt, 17, 390)
    assert tensor_sha(prompt) == part['prompt_sha256']
    assert tensor_sha(layout['img_position_ids']) == part['position_sha256']
    zpath = base / 'chunk_12_17.pt'
    assert sha(zpath) == part['endpoint_sha256']
    z = torch.load(zpath, map_location='cpu', weights_only=True)
    assert z.shape == (1, 24, 5, 30, 52) and torch.isfinite(z).all()
    assert tensor_sha(z) == part['endpoint_tensor_sha256']
    rgb_path = base / 'published_56.npy'
    assert sha(rgb_path) == part['published_sha256']
    rgb = np.load(rgb_path)
    assert rgb.shape == (56, 480, 832, 3) and np.array_equal(rgb[:39], old_rgb)
    video = base / 'rollout_56.mp4'
    assert sha(video) == part['video_sha256'] == row['branches'][branch]['video_sha256']
    with av.open(str(video)) as c:
        frames = list(c.decode(video=0))
        assert len(frames) == 56 and c.streams.video[0].average_rate == 24
        assert all((f.width, f.height) == (832, 480) for f in frames)
        assert all(b.pts > a.pts for a, b in zip(frames, frames[1:]))
    branches[branch] = {k: part[k] for k in ('sampling_seconds', 'decode_seconds', 'video_sha256',
                                             'initial_noise_sha256', 'prompt_sha256', 'position_sha256')}
assert branches['AA']['position_sha256'] == branches['AD']['position_sha256']
assert branches['AA']['prompt_sha256'] != branches['AD']['prompt_sha256']
result = {'task': 'EXP-011/v1', 'scene': args.scene, 'method': args.method,
          'independent_G2_execution_audit': 'PASS', 'visual_assessment': 'SEPARATE',
          'row_sha256': sha(directory / 'result.json'), 'own_C1_source_sha_verified': True,
          'real_cache_all_50_layers_index0': True, 'cpu_raw_kv_bytes': cache_bytes,
          'old39_RGB_unchanged_both_branches': True, 'forwards': 2*steps+1, 'vae_decodes': 2,
          'commit_seconds': row['commit_seconds'], 'wall_seconds': row['wall_seconds'],
          'peak_allocated_gib': row['peak_allocated_gib'], 'branches': branches}
(HERE / f'G2_{args.scene}_{args.method}_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
