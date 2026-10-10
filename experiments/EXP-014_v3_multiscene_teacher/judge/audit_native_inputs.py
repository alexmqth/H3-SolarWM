"""Independent CPU check of completed native fixtures, noise, positions and budget."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
from common import CFG, OUT, ROOT, setup_paths, verify_code_manifest, verify_sources

setup_paths()
import torch
from causal.local_topology import visible_inputs
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, MiniMaxH3Unit_NoiseInitializer


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 << 20), b''):
            h.update(part)
    return h.hexdigest()


verify_sources()
verify_code_manifest()
row = json.loads((OUT / 'P1_result.json').read_text())
ledger = json.loads((OUT / 'P1/budget.json').read_text())
assert row['status'] == 'complete_pending_cpu_audit'
assert row['source_manifest_sha256'] == sha(EXP / 'source_manifest.json')
assert row['runner_sha256'] == sha(EXP / 'encode_native.py')
assert row['marker_sha256'] == sha(HERE / 'P1_APPROVED.json')
assert ledger['events'][-1]['event'] == 'stage_stop' and ledger['events'][-1]['status'] == 'complete'
assert 0 < ledger['gpu_seconds'] <= 600
assert [ledger[k] for k in ('text_encoder_calls', 'image_vae_encodes', 'video_vae_encodes',
                           'sampling_forwards', 'commit_forwards', 'vae_decodes')] == [12, 4, 0, 0, 0, 0]
assert row['peak_allocated_gib'] <= 44
assert set(row['fixtures']) == set(CFG['scenes'])
source = json.loads((EXP / 'source_manifest.json').read_text())
assert len(source['scenes']) == 4
assert set(source['train_episode_ids']).isdisjoint(source['excluded_validation_episode_ids'])
assert source['candidate_manifest_sha256'] == 'b6af2b0208a5dcf83c02f76fa826e5a03ba52fa9bd83de357f9474b5efb257b6'
cpu_pipe = MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
native = MiniMaxH3Unit_NoiseInitializer().process(cpu_pipe, 13, 124, 480, 832, 'cpu')
results = {}
for name, item in row['fixtures'].items():
    path = ROOT / item['path']
    assert sha(path) == item['sha256']
    data = torch.load(path, map_location='cpu', weights_only=True)
    assert source['scenes'][name]['split'] == 'train'
    assert data['scene_static'] == source['scenes'][name]['scene_static']
    assert data['source_png_sha256'] == sha(ROOT / source['scenes'][name]['png'])
    assert torch.equal(data['initial_noise'], native['video_latents'])
    assert torch.equal(data['audio_noise'], native['audio_latents'])
    assert data['anchor'].shape == (390, 96) and torch.isfinite(data['anchor']).all()
    packed = data['packed']
    spans = packed['action_text_spans_local']
    a, d = data['prompts']['A'], data['prompts']['D']
    assert torch.isfinite(a).all() and torch.isfinite(d).all()
    action = torch.zeros(len(a), dtype=torch.bool)
    for lo, hi in spans:
        action[lo:hi] = True
        assert not torch.equal(a[lo:hi], d[lo:hi])
    assert torch.equal(a[~action], d[~action])
    for stop in (12, 17):
        layout, prompt = visible_inputs(packed, a, stop, 390)
        keep = torch.zeros(int(packed['seq_len']), dtype=torch.bool)
        keep[:int(packed['action_video_start']) + stop * 390] = True
        for lo, hi in spans[stop:]:
            keep[packed['text_pos'][lo:hi]] = False
        # All retained coordinates, including prefix, keep exact full37 values.
        assert torch.equal(layout['img_position_ids'], packed['img_position_ids'][:, keep])
        assert len(layout['action_text_rows']) == stop
        assert layout['img_pos'].numel() == (stop + 1) * 390
        for field in packed:
            if 'camera' in field.lower():
                x, y = packed[field], layout[field]
                assert torch.equal(x, y) if torch.is_tensor(x) else x == y
    results[name] = {'fixture_sha256': item['sha256'], 'native_cpu_noise_reconstruction': 'EXACT',
                     'single_i0_anchor_rows': 390, 'global_positions_preserved': True,
                     'nonaction_prompt_unchanged_A_D': True}
result = {'task': 'EXP-014/v1', 'independent_P1_audit': 'PASS', 'gpu_calls_by_audit': 0,
          'p1_gpu_seconds_including_failed_starts': ledger['gpu_seconds'],
          'actual_text_calls': 12, 'actual_image_vae_encodes': 4,
          'denoiser_forwards': 0, 'vae_decodes': 0, 'fixtures': results}
(HERE / 'P1_INDEPENDENT_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
