"""Independent CPU evidence check for a completed EXP-012 scene."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
from common import ROOT, OUT, sha, setup_paths, verify_sources, verify_code_manifest
setup_paths()
import av
import numpy as np
import torch
from causal.local_topology import visible_inputs
from causal.anyflow_sampling import configure_video_schedule
from diffsynth.diffusion.flow_match import FlowMatchScheduler


def tensor_sha(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(t.shape), str(t.dtype))).encode() +
                          t.view(torch.uint8).numpy().tobytes()).hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument('--scene', choices=['industrial', 'village'], required=True)
args = ap.parse_args()
verify_sources()
verify_code_manifest()
sources = json.loads((EXP / 'source_manifest.json').read_text())['sources']
review = json.loads((HERE / 'P0_REVIEW.json').read_text())
for path, digest in review['additional_protocol_dependencies'].items():
    assert sha(path) == digest
directory = OUT / 'G1' / args.scene
row_path = directory / 'result.json'
row = json.loads(row_path.read_text())
assert row['status'] == 'complete_pending_visual_review' and row['scene'] == args.scene
assert [row[k] for k in ('sampling_forwards', 'commit_forwards', 'vae_decodes')] == [16, 1, 2]
assert row['runner_sha256'] == sha(EXP / 'run_exp012.py')
assert row['precision']['native_fp32_weights_restored'] and row['peak_allocated_gib'] <= 44
assert row['checkpoint']['step'] == 32
assert row['checkpoint']['qkv_sha256'] == sources['af2_qkv']['sha256']
assert row['checkpoint']['target_sha256'] == sources['af2_target']['sha256']
first_path = ROOT / sources[f'g1_{args.scene}_first12']['path']
rgb39_path = ROOT / sources[f'g1_{args.scene}_rgb39']['path']
assert sha(first_path) == row['first_endpoint_sha256']
assert sha(rgb39_path) == row['first_published_sha256']
first = torch.load(first_path, map_location='cpu', weights_only=True)
assert tensor_sha(first) == row['first_tensor_sha256']
old_rgb = np.load(rgb39_path)
fixture = torch.load(ROOT / sources[f'fixture_{args.scene}']['path'], map_location='cpu', weights_only=True)
cache_path = directory / 'af_cache_through12.pt'
assert sha(cache_path) == row['cache_sha256']
cache = torch.load(cache_path, map_location='cpu', weights_only=False)
assert cache.max_history == 5 and cache.commits == 50 and cache.storage_device == 'cpu'
assert set(cache.layers) == set(range(50))
assert cache.nbytes == row['cpu_raw_kv_bytes'] == 6799104000
for entries in cache.layers.values():
    assert len(entries) == 1 and entries[0].index == 0
    e = entries[0]
    assert e.key.shape == e.value.shape and e.key.shape[0] == e.rope.shape[0] == 4680
    assert all(not t.requires_grad and t.device.type == 'cpu' for t in (e.key, e.value, e.rope))
# Compare actual tensor contents with the FM8 own-weight cache for the same C1.
parent_cache_path = ROOT / 'H3-World/outputs/EXP-011_v3_scene_transfer/G2' / args.scene / 'FM8/cache_through12.pt'
parent_row = json.loads((parent_cache_path.parent / 'result.json').read_text())
assert sha(parent_cache_path) == parent_row['cache_sha256']
parent = torch.load(parent_cache_path, map_location='cpu', weights_only=False)
different_layers = []
for layer in range(50):
    a, b = cache.layers[layer][0], parent.layers[layer][0]
    assert torch.equal(a.rope, b.rope)
    if not (torch.equal(a.key, b.key) and torch.equal(a.value, b.value)):
        different_layers.append(layer)
del cache, parent
sigmas = configure_video_schedule(FlowMatchScheduler('MiniMax-H3'), steps=8, grid='native', flow_shift=2.22)
branches = {}
for branch in ('AA', 'AD'):
    base = directory / branch
    part = json.loads((base / 'result.json').read_text())
    assert part['status'] == 'complete_pending_visual_review'
    assert part['sampling_forwards'] == 8 and part['vae_decodes'] == 1
    assert part['source_cache_sha256'] == row['cache_sha256']
    assert part['first_endpoint_sha256'] == row['first_endpoint_sha256']
    assert part['sigmas'] == sigmas
    assert part['initial_noise_sha256'] == tensor_sha(fixture['initial_noise'][:, :, 12:17].float())
    prompt = fixture['prompts']['A'].clone()
    if branch == 'AD':
        for lo, hi in fixture['packed']['action_text_spans_local'][12:17]:
            prompt[lo:hi] = fixture['prompts']['D'][lo:hi]
    layout, prompt = visible_inputs(fixture['packed'], prompt, 17, 390)
    assert tensor_sha(prompt) == part['prompt_sha256']
    assert tensor_sha(layout['img_position_ids']) == part['position_sha256']
    baseline = json.loads((parent_cache_path.parent / branch / 'result.json').read_text())
    for key in ('sigmas', 'initial_noise_sha256', 'position_sha256', 'prompt_sha256'):
        assert part[key] == baseline[key]
    endpoint = base / 'chunk_12_17.pt'
    assert sha(endpoint) == part['endpoint_sha256']
    z = torch.load(endpoint, map_location='cpu', weights_only=True)
    assert z.shape == (1, 24, 5, 30, 52) and torch.isfinite(z).all()
    assert tensor_sha(z) == part['endpoint_tensor_sha256']
    rgb_path = base / 'published_56.npy'
    assert sha(rgb_path) == part['published_sha256']
    rgb = np.load(rgb_path)
    assert rgb.shape == (56, 480, 832, 3) and np.array_equal(rgb[:39], old_rgb)
    video = base / 'rollout_56.mp4'
    assert sha(video) == part['video_sha256'] == row['branches'][branch]['video_sha256']
    for name, count in [('rollout_56.mp4', 56), ('new_17.mp4', 17)]:
        with av.open(str(base / name)) as c:
            f = list(c.decode(video=0))
            assert len(f) == count and c.streams.video[0].average_rate == 24
            assert all((v.width, v.height) == (832, 480) for v in f)
            assert all(b.pts > a.pts for a, b in zip(f, f[1:]))
    branches[branch] = {k: part[k] for k in ('sampling_seconds', 'decode_seconds', 'video_sha256')}
result = dict(task='EXP-012/v1', scene=args.scene, independent_execution_audit='PASS',
              visual_assessment='SEPARATE', row_sha256=sha(row_path),
              real_AF_cache_all_50_layers=True, cpu_raw_kv_bytes=6799104000,
              AF_vs_FM8_different_raw_KV_layers=different_layers,
              AF_vs_FM8_rope_exactly_equal_all_layers=True,
              shared_C1_noise_actions_sigmas_positions_match_FM8=True,
              old39_RGB_unchanged=True, forwards=17, vae_decodes=2,
              peak_allocated_gib=row['peak_allocated_gib'], wall_seconds=row['wall_seconds'],
              load_model_seconds=row['load_model_seconds'], commit_seconds=row['commit_seconds'], branches=branches)
(HERE / f'G1_{args.scene}_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
