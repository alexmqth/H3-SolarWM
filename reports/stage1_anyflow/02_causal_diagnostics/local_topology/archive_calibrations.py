"""Audit and archive completed E1 conditioning calibrations, never tensor caches.

No automatic visual acceptance; per-group manual review is separate.
The source output directory is passed explicitly for reproducible re-audits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import av


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(base, dest, groups):
    audits = []
    for name in groups:
        source = base / name
        receipt = json.loads((source / 'evaluation.json').read_text())
        assert receipt['status'] == 'complete_pending_visual_review', name
        assert receipt['optimizer_updates'] == 0
        assert receipt['parameter_versions_unchanged']
        proc = Path(f'/proc/{receipt["pid"]}/stat')
        if proc.exists():
            state = proc.read_text().rsplit(')', 1)[1].split()
            assert state[19] != receipt['start_ticks'] or state[0] in ('Z', 'X'), 'Job still active'
        records = receipt['records']
        if 'history_reference' in receipt:
            assert receipt['denoiser_forwards'] == 180 and len(records) == 6
            assert receipt['history_latents_unchanged']
            for i in range(3):
                a, d = [next(r for r in records if r['chunk'] == i and r['action'] == act) for act in 'AD']
                for field in ('initial_noise_sha256', 'history_sha256', 'anchor_sha256'):
                    assert a[field] == d[field], (name, i, field)
        else:
            assert receipt['model_forwards'] == 60 and len(records) == 2
            for field in ('noise_sha256', 'anchor_sha256'):
                assert records[0][field] == records[1][field]
        for r in records:
            video = source / Path(r['flow']['path']).name
            assert sha(video) == r['video_sha256']
        videos = []
        for video in sorted(source.glob('*.mp4')):
            if video.stem.startswith('chunk'):
                expected = (17, 17, 5)[int(video.stem[5])]
            elif video.stem[0] in 'AD' and len(video.stem.split('_')) == 3 and video.stem.split('_')[1].isdigit():
                _, start, stop = video.stem.split('_'); expected = int(stop) - int(start)
            else:
                expected = 39
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                assert stream.codec_context.name == 'h264'
                assert stream.average_rate == 24
                frames = list(container.decode(video=0))
                assert len(frames) == expected
                assert all(f.format.name == 'yuv420p' for f in frames)
            videos.append(dict(file=video.name, frames=expected, fps=24,
                               codec='h264', pix_fmt='yuv420p', sha256=sha(video)))
        audit = dict(status='complete_full_decode', videos=videos,
                     paired_inputs_verified=True, scope='Technical validity, not visual/action acceptance')
        (source / 'video_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        target = dest / name; target.mkdir(parents=True, exist_ok=True)
        for file in source.iterdir():
            if file.suffix in ('.json', '.jpg', '.mp4', '.md'):
                shutil.copy2(file, target / file.name)
                assert sha(file) == sha(target / file.name)
        audits.append(dict(group=name, video_count=len(videos), receipt_sha256=sha(source/'evaluation.json')))
    print(json.dumps(audits, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--dest', type=Path, required=True)
    p.add_argument('groups', nargs='+')
    a = p.parse_args(); archive(a.base, a.dest, a.groups)
