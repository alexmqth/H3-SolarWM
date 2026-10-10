"""Independent CPU audit of the DMD8 C2 evaluation stopped for visual collapse."""
from pathlib import Path
import hashlib
import json
import av
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[2]
OUT = ROOT / 'H3-World/outputs/EXP-008_v3_dmd_eval'
FM8 = ROOT / 'H3-World/outputs/EXP-006_v3_fm8_full'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()

ledger = json.loads((OUT / 'budget.json').read_text())
assert 'active_start' not in ledger
assert tuple(ledger[k] for k in ('forwards', 'sampling', 'commits', 'vae')) == (16, 14, 2, 1)
assert ledger['gpu_seconds'] <= .35 * 3600
assert len(ledger['runs']) == 3
assert all(r['status'] == 'complete_pending_visual_review' for r in ledger['runs'][:2])
assert ledger['runs'][2]['stage'] == 'second' and ledger['runs'][2]['path'] == 'AD' and ledger['runs'][2]['status'] == 'failed'
ad = json.loads((OUT/'AD/second_12_17.json').read_text())
assert ad['status'] == 'failed' and ad['step_completed'] == 5 and 'KeyboardInterrupt' in ad['error']
assert not (OUT/'AD/rollout_56.mp4').exists()
assert not any((OUT/b/'third_17_22.json').exists() for b in ('AA','AD'))
manifest_path = EXP / 'dmd_eval/source_manifest_eval.json'
manifest = json.loads(manifest_path.read_text())
train = json.loads((ROOT / 'H3-World/outputs/EXP-008_v3_dmd_train_v2/result.json').read_text())
cycle = train['last_complete_cycle']
assert manifest['checkpoint_cycle'] == cycle and 2 <= cycle <= 8
for name, entry in manifest['sources'].items():
    assert sha(ROOT / entry['path']) == entry['sha256'], name
rows = {}
cache_hashes = {}
for branch, stage, start, stop, index in [(None, 'prefill', 0, 12, 0)] + [
    (b, s, a, z, i) for b in ('AA',)
    for s, a, z, i in [('second', 12, 17, 1)]
]:
    folder = OUT if branch is None else OUT / branch
    row = json.loads((folder / f'{stage}_{start}_{stop}.json').read_text())
    label = 'C1' if branch is None else f'{branch}_C{index+1}'
    assert row['status'] == 'complete_pending_visual_review'
    assert row['checkpoint_cycle'] == cycle
    assert row['checkpoint_manifest_sha256'] == sha(manifest_path)
    assert row['runner_sha256'] == manifest['sources']['runner']['sha256']
    assert (row['sampling_forwards'], row['commit_forwards'], row['vae_calls']) == ((0, 1, 0) if index == 0 else (8, int(index == 1), 1))
    assert row['peak_allocated_gib'] <= 44
    assert row['fm8_first_sha256'] == manifest['sources']['fm8_first']['sha256']
    if index < 2:
        cache = folder / ('cache_through12.pt' if index == 0 else 'cache_through17.pt')
        cache_hashes[label] = sha(cache)
        assert cache_hashes[label] == row['cache_after_sha256']
        assert row['cache_after']['entries'] == [[0, 4680]] + ([[1, 1950]] if index else [])
        assert row['cache_after']['layer_count'] == 50
    if index:
        ref = json.loads((FM8 / branch / f'chunk_{start}_{stop}.json').read_text())
        assert row['noise_sha256'] == ref['current_noise_sha256']
        assert row['prompt_sha256'] == ref['prompt_sha256']
        assert row['sigmas'] == ref['sigmas']
        assert row['history_cache_read_unchanged'] and row['source_cache_file_unchanged'] and row['old_RGB_unchanged']
        expected = [[0, 4680]] + ([[1, 1950]] if index == 2 else [])
        assert row['cache_before']['entries'] == expected and row['cache_before']['layer_count'] == 50
        parent = 'C1' if index == 1 else f'{branch}_C2'
        assert row['source_cache_sha256'] == cache_hashes[parent]
        if index == 1:
            assert row['history_tensor_sha256'] == ref['history_tensor_sha256']
            assert row['source_cache_sha256'] != ref['cache_file_sha256']
        assert sha(folder / f'chunk_{start}_{stop}.pt') == row['endpoint_sha256']
        n = 39 + 17 * index
        rgb_path = folder / f'published_{n}.npy'
        assert sha(rgb_path) == row['published_file_sha256']
        rgb = np.load(rgb_path)
        assert rgb.shape == (n, 480, 832, 3)
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == row['published_RGB_sha256']
        previous = np.load(FM8 / 'published_39.npy' if index == 1 else folder / 'published_56.npy')
        assert np.array_equal(rgb[:len(previous)], previous)
        video = folder / f'rollout_{n}.mp4'
        assert sha(video) == row['video_sha256']
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            assert stream.average_rate == 24
            frames = list(container.decode(video=0))
            pts = [f.pts for f in frames]
            assert len(pts) == n and all(b > a for a, b in zip(pts, pts[1:]))
            assert all((f.width, f.height) == (832, 480) for f in frames)
    rows[label] = row
result = dict(task='EXP-008/v2-DMD8-EVAL', checkpoint_cycle=cycle, independent_audit_passed=True,
    forwards_started_or_reserved=16, sampling_reserved=14, sampling_completed=13, interrupted_sampling=1, commits=2, vae=1, updates=0,
    cancelled_stages=['AA third', 'AD third'], generation_verdict='FAIL_AA_C2_SUSTAINED_COLLAPSE', AD_verdict='INTERRUPTED_NO_COMPLETE_VIDEO',
    gpu_seconds=ledger['gpu_seconds'], gpu_hours=ledger['gpu_seconds']/3600,
    interrupted_AD_cost_included=True, max_allocated_gib_completed_stages=max(r['peak_allocated_gib'] for r in rows.values()),
    rows={name: {k: row.get(k) for k in ('sampling_seconds', 'commit_seconds', 'decode_seconds', 'seconds', 'flow', 'boundary_gray_MAD', 'inside_gray_MAD', 'cache_before', 'peak_allocated_gib')} for name, row in rows.items()},
    scope='Shared FM8 C1; matched history in AA C2 only. AD interrupted, not evaluated. C3 cancelled after sustained visual collapse. Audit verifies provenance and execution, not generation success.')
(HERE / 'eval_c2_stop_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
