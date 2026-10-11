"""Independent CPU verification of a completed AD recovery and preserved parent."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import av
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
import recover_ad as r
r.setup_paths()
from causal.anyflow_sampling import configure_video_schedule
from diffsynth.diffusion.flow_match import FlowMatchScheduler


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--scene', choices=r.CONFIG['scenes'], required=True)
args = parser.parse_args()
r.verify_sources()
r.verify_code_manifest()
r.verify_recovery_code()
r.verify_snapshot()
r.require_recovery_marker(args.scene)
p = r.RECOVERY / args.scene
row = json.loads((p / 'AD/result.json').read_text())
budget = json.loads((p / 'budget.json').read_text())
assert row['status'] == 'complete_pending_visual_review'
assert row['task'] == budget['task'] == 'EXP-014/v2'
assert row['scene'] == budget['scene'] == args.scene
assert row['runner_sha256'] == sha(EXP / 'recover_ad.py')
assert row['precision']['native_fp32_weights_restored']
assert budget['events'][-1]['status'] == 'complete'
assert [budget[k] for k in ['sampling_forwards', 'vae_decodes', 'commit_forwards', 'training_updates']] == [30, 1, 0, 0]
assert 0 < budget['gpu_seconds'] <= 600 and row['peak_allocated_gib'] <= 44
pre = next(x for x in json.loads((HERE / 'RECOVERY_CPU_INDEPENDENT_AUDIT.json').read_text())['scenes'] if x['scene'] == args.scene)
assert row['preflight'] == pre
old = r.OUT / 'G2' / args.scene / 'FM30'
old_ad = json.loads((old / 'AD/result.json').read_text())
for field in ['prompt_sha256', 'position_sha256', 'initial_noise_sha256', 'sigmas']:
    assert row[field] == old_ad[field]
assert row['sigmas'] == configure_video_schedule(FlowMatchScheduler('MiniMax-H3'), steps=30, grid='native', flow_shift=2.22)
assert row['cpu_raw_kv_bytes'] == pre['cache_bytes'] == 6799104000
endpoint = p / 'AD/chunk_12_17.pt'
assert sha(endpoint) == row['endpoint_sha256']
z = torch.load(endpoint, map_location='cpu', weights_only=True)
assert z.shape == (1, 24, 5, 30, 52) and torch.isfinite(z).all()
assert r.tensor_sha(z) == row['endpoint_tensor_sha256']
assert sha(p / 'AD/published_56.npy') == row['published_sha256']
rgb = np.load(p / 'AD/published_56.npy')
old_rgb = np.load(r.OUT / 'G1' / args.scene / 'FM30/published_39.npy')
assert rgb.shape == (56, 480, 832, 3) and np.array_equal(rgb[:39], old_rgb)
assert sha(p / 'AD/rollout_56.mp4') == row['video_sha256']
for name, count in [('rollout_56.mp4', 56), ('new_17.mp4', 17)]:
    with av.open(str(p / 'AD' / name)) as c:
        fs = list(c.decode(video=0))
        assert len(fs) == count and c.streams.video[0].average_rate == 24
        assert all((f.width, f.height) == (832, 480) for f in fs)
        assert all(b.pts > a.pts for a, b in zip(fs, fs[1:]))
target = json.loads((p / 'target_manifest.json').read_text())
for item in [target['teacher_C1_latent'], *target['teacher_C2_endpoint'].values()]:
    assert sha(r.ROOT / item['path']) == item['sha256']
assert target['teacher_C2_endpoint']['AD']['sha256'] == row['endpoint_sha256']
result = dict(task='EXP-014/v2', scene=args.scene, status='PASS_PROTOCOL_ONLY',
              gpu_calls_by_audit=0, original_sources_unchanged=True,
              cache_matches_independently_verified_original=True,
              original_AD_inputs_match=True, old39_RGB_unchanged=True,
              sampling_forwards=30, commit_forwards=0, vae_decodes=1,
              gpu_seconds=budget['gpu_seconds'], peak_allocated_gib=row['peak_allocated_gib'],
              sampling_seconds=row['sampling_seconds'], decode_seconds=row['decode_seconds'],
              video_sha256=row['video_sha256'], endpoint_sha256=row['endpoint_sha256'],
              visual_assessment='SEPARATE')
(HERE / f'RECOVERY_{args.scene}_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
