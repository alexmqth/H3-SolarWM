"""Archive terminal E2 evidence, excluding checkpoints, raw data and runtime weights."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import av

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
DEST = ROOT/'submission/reports/stage1_anyflow/real_transition_windows'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''): h.update(chunk)
    return h.hexdigest()


def terminal(receipt):
    proc = Path(f'/proc/{receipt["pid"]}/stat')
    if proc.exists():
        fields = proc.read_text().rsplit(')', 1)[1].split()
        assert fields[19] != receipt['start_ticks'] or fields[0] in ('Z', 'X'), 'Exact process still running'


def main():
    review = json.loads((BASE/'review_step4/visual_review.json').read_text())
    assert review['status'] == 'complete_static_review'
    metrics = json.loads((BASE/'review_step4/metrics.json').read_text())
    assert not metrics['pending'] and len(metrics['metrics']) == 18
    protocol = json.loads((BASE/'evaluation_protocol.json').read_text())
    for name, digest in protocol['sources'].items(): assert sha(BASE/name) == digest
    for name, digest in json.loads((BASE/'runtime_manifest.json').read_text()).items():
        assert sha(BASE/'runtime'/name) == digest
    folders = ['review_step4', 'heldout_fit', 'train_fm_only', 'train_fm_action', 'coverage_candidate_review']
    for arm in ['fm_only', 'fm_action']:
        train = json.loads((BASE/f'train_{arm}/training.json').read_text())
        assert train['optimizer_updates'] == 4 and train['base_parameters_unchanged']; terminal(train)
        queue = json.loads((BASE/f'eval_queue_{arm}.json').read_text())
        assert queue['status'] == 'complete_all_preselected_suites'; terminal(queue)
        for suite in ['parking_A', 'parking_D', 'gt']:
            folder = f'eval_{arm}_{suite}'; folders.append(folder)
            receipt = json.loads((BASE/folder/'evaluation.json').read_text())
            assert receipt['status'] == 'complete_pending_visual_review'; terminal(receipt)
            assert receipt['sampling_forwards'] == 60 and receipt['diagnostic_forwards'] == 2
            assert receipt['zero_checkpoint_identity_max_abs'] == 0
            assert receipt['frozen_parameters_unchanged']
            if suite.startswith('parking_'): assert receipt['archived_N_field_replay_max_abs'] == 0
            for row in receipt['records']: assert sha(row['current_video']) == row['video_sha256']
    fit = json.loads((BASE/'heldout_fit/fit.json').read_text())
    assert fit['status'] == 'complete' and fit['forwards'] == 48; terminal(fit)
    files = [p for p in BASE.iterdir() if p.is_file() and not p.is_symlink() and p.suffix in ('.py', '.md', '.json', '.log') and '.tmp.' not in p.name and p.name != 'archive_evaluation_audit.json']
    for folder in folders:
        files.extend(p for p in (BASE/folder).rglob('*') if p.is_file() and not p.is_symlink()
                     and p.suffix in ('.md', '.json', '.jpg', '.png', '.mp4') and '.tmp.' not in p.name)
    videos = []; copied = {}
    for path in files:
        rel = path.relative_to(BASE)
        if path.suffix == '.mp4':
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                count = sum(1 for _ in container.decode(video=0))
                assert stream.codec_context.name == 'h264' and stream.codec_context.pix_fmt == 'yuv420p'
                assert float(stream.average_rate) == 24 and count in (39, 42, 47, 50)
                videos.append(dict(path=str(rel), frames=count, width=stream.width, height=stream.height, codec='h264', pix_fmt='yuv420p', fps=24))
        target = DEST/rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target); assert sha(path) == sha(target)
        copied[str(rel)] = sha(path)
    audit = dict(at=datetime.now().astimezone().isoformat(), status='archived_verified',
                 runtime_source_files_unchanged=307, files=copied, videos=videos,
                 optimizer_updates=8, evaluation_sampling_forwards=360, evaluation_identity_forwards=12,
                 heldout_fit_forwards=48, stage1_accepted=False,
                 exclusions=['weights', 'optimizer', 'latent/conditioning tensors', 'raw RGB/action/pose NPZ', 'runtime'])
    for parent in [BASE, DEST]: (parent/'archive_evaluation_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({'copied': len(copied), 'videos_decoded': len(videos), 'runtime_unchanged': True}))


if __name__ == '__main__': main()
