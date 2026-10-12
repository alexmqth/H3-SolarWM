"""Independent CPU audit of EXP-018 completed-row evidence (no model imports).
Checks fresh outputs, full-length decoding, shared inputs, accounting, and measured GPU peaks.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import math
import os
import av

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[2]
EXPECTED = {'v0': (30, 0), 'v1': (64, 8), 'v2a': (64, 8), 'v2b': (180, 0), 'v3_30': (180, 5), 'v3_8': (48, 5), 'v4': (48, 5)}


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def video_info(p, expected_frames):
    with av.open(str(p)) as c:
        s = c.streams.video[0]
        s.codec_context.thread_count = 2
        assert (s.width, s.height) == (832, 480)
        assert str(s.average_rate) == '24'
        pts = [f.pts for f in c.decode(video=0)]
        assert len(pts) == expected_frames
        assert all(a < b for a, b in zip(pts, pts[1:]))
        return {'frames': len(pts), 'fps': 24, 'seconds': len(pts) / 24, 'full_decode': True, 'monotonic_pts': True}


def main():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    index = json.loads((EXP / 'results_index.json').read_text())
    assert set(index['rows']) == set(EXPECTED)
    audit = {'task': 'EXP-018/v1', 'type': 'Independent CPU evidence audit; quality assessed separately', 'rows': {}}
    noise = set()
    devices = set()
    total_forward = 0
    total_wall = 0
    for row_id, (sampling, commits) in EXPECTED.items():
        row = index['rows'][row_id]
        assert row['status'] == 'complete_pending_judge', (row_id, row['status'])
        assert row['sampling_nfe_total'] == sampling
        assert row['commit_forward_total'] == commits
        assert row.get('prefill_forward_total', 0) == 0
        assert row['total_forward_count'] == sampling + commits
        wall = row['cold_total_seconds']
        peak = row['peak_allocated_gib']
        assert math.isfinite(wall) and 0 < wall <= 2700
        assert math.isfinite(peak) and 0 < peak < 44
        total_wall += wall
        total_forward += sampling + commits
        devices.add(row['gpu_uuid'])
        full = ROOT / row['full_video']
        display = ROOT / row['display_video']
        assert 'EXP-018_unified_5s_benchmark' in full.parts
        assert 'EXP-018_unified_5s_benchmark' in display.parts
        launch = json.loads((full.parent / 'launch.json').read_text())
        started = datetime.fromisoformat(launch['started_at']).timestamp()
        assert full.stat().st_mtime >= started and display.stat().st_mtime >= started
        assert sha(full) == row['full_sha256']
        assert sha(display) == row['display_sha256']
        if row_id in ('v0', 'v1', 'v2a'):
            setup = json.loads((full.parent / 'setup.json').read_text())
            assert setup['vram_reserve_gib'] == 18
            assert setup['input_fingerprints']['video_noise_shape'] == [1, 24, 37, 30, 52]
            noise_sha = setup['input_fingerprints']['video_noise_sha256']
            assert peak + 1e-6 >= setup['memory']['allocated_peak_MiB'] / 1024
            fresh = {'condition_seconds': setup['conditioning_seconds'], 'input_sha256': setup['input_fingerprints']['initial_image_sha256']}
        else:
            original = json.loads((full.parent / 'native_metrics.json').read_text())
            assert original['fresh_first_chunk']
            assert original['historical_endpoint_reads'] == original['historical_kv_reads'] == 0
            assert len(original['chunks']) == 6
            assert [c['latent_interval'] for c in original['chunks']] == [[0,12],[12,17],[17,22],[22,27],[27,32],[32,37]]
            assert original['conditioning_seconds'] > 0
            noise_sha = original['video_noise_sha256']
            fresh = {'condition_seconds': original['conditioning_seconds'], 'fresh_first_chunk': True, 'no_existing_generated_history': True}
        assert noise_sha
        noise.add(noise_sha)
        audit['rows'][row_id] = {'status': 'PASS', 'sampling_nfe': sampling, 'commit_forwards': commits, 'cold_wall_seconds': wall, 'peak_allocated_gib': peak, 'freshness': fresh, 'full124': video_info(full,124), 'display120': video_info(display,120), 'metrics_sha256': sha(full.parent / 'metrics.json')}
    assert len(noise) == 1, 'Input noise differs: investigate protocol before claiming same-noise comparison'
    assert len(devices) == 1, 'Record separate GPUs if serial-device plan changed'
    attempts = [a for values in index.get('attempts', {}).values() for a in values]
    assert len(attempts) <= 1, 'Only one diagnosed GPU retry authorized'
    failed_wall = 0
    failed_forward = 0
    for attempt in attempts:
        folder = ROOT / attempt['artifact_dir']
        original = json.loads((folder / 'native_metrics.json').read_text())
        assert original['status'] == 'failed'
        assert (folder / 'run.log').is_file()
        failed_wall += attempt['cold_total_seconds']
        failed_forward += attempt['total_forward_count']
    assert total_forward + failed_forward <= 850 and total_wall + failed_wall <= 14400
    audit.update(status='PASS', matched_noise_sha256=noise.pop(), gpu_uuid=devices.pop(),
        completed_row_forwards=total_forward, completed_row_wall_seconds=total_wall,
        failed_attempts=len(attempts), failed_attempt_wall_seconds=failed_wall,
        failed_attempt_forwards=failed_forward,
        total_budget_wall_seconds=total_wall+failed_wall,
        total_budget_gpu_hours=(total_wall+failed_wall)/3600,
        note='V0 wall is reconstructed launch-to-MP4 time after CPU postprocess assertion; later walls include process exit. Single-run observed costs; execution audit does not certify visual quality.')
    (HERE / 'results_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': audit['status'], 'rows': len(audit['rows']), 'forwards': total_forward, 'wall_seconds': total_wall}))


if __name__ == '__main__':
    main()
