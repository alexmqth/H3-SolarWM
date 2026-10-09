"""Read-only run audit plus archival of completed parking reports; no GPU jobs."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import av

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
DEST = ROOT / 'submission/reports/stage1_anyflow/fm_density_control'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process(pid, expected_ticks=None):
    path = Path('/proc') / str(pid)
    if not path.exists():
        return dict(pid=pid, expected_start_ticks=expected_ticks, same_process_live=False, observed='absent')
    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    ticks, state = fields[19], fields[0]
    same = expected_ticks is None or ticks == str(expected_ticks)
    return dict(pid=pid, expected_start_ticks=expected_ticks, start_ticks=ticks,
                state=state, same_process_live=same and state not in ('Z', 'X'))


def video_audit(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(video=0))
        assert len(frames) == 39 and stream.average_rate == 24
        assert stream.codec_context.name == 'h264'
        assert all(frame.format.name == 'yuv420p' for frame in frames)
        probe = dict(codec_name='h264', pix_fmt='yuv420p', avg_frame_rate='24/1',
                     nb_read_frames=len(frames), width=stream.width, height=stream.height)
    data = path.read_bytes()
    pos, boxes = 0, []
    while pos + 8 <= len(data):
        size, kind = int.from_bytes(data[pos:pos+4], 'big'), data[pos+4:pos+8].decode('ascii')
        if size == 1:
            size = int.from_bytes(data[pos+8:pos+16], 'big')
        elif size == 0:
            size = len(data) - pos
        assert size >= 8
        boxes.append(kind)
        pos += size
    assert boxes.index('moov') < boxes.index('mdat')
    return dict(file=path.name, sha256=sha(path), stream=probe, full_decode=True, faststart=True)


def main():
    queue = json.loads((BASE / 'post48_queue.json').read_text())
    reports = json.loads((BASE / 'post48_reports.json').read_text())
    parking = json.loads((BASE / 'parking_queue_step48.json').read_text())
    assert queue['status'] == 'complete' and not queue['active']
    assert len(queue['completed']) == 10 and all(x['exit_code'] == 0 for x in queue['completed'])
    assert len(reports['completed']) == 8 and not reports['pending'] and not reports['active']
    assert all(x['exit_code'] == 0 for x in reports['completed'])
    assert parking['status'] == 'complete' and len(parking['completed']) == 4
    processes = [process(pid, ticks) for pid, ticks in [
        (1563900, '255332670'), (1615283, '255366821'), (1898518, '255496357'),
        (3933368, '256246132'), (3606445, '256119956'), (278980, None)]]
    assert not any(x['same_process_live'] for x in processes)
    frozen = json.loads((BASE / 'runtime_manifest.json').read_text())
    for rel, expected in frozen.items():
        assert sha(BASE / 'runtime' / rel) == expected, rel
    for path, expected in queue['source_sha256'].items():
        assert sha(Path(path)) == expected, path
    for rel, expected in reports['source_sha256'].items():
        assert sha(BASE / rel) == expected, rel
    files = []
    for group in ('parking_30', 'parking_8'):
        folder = BASE / 'report' / group
        videos = [video_audit(folder / f'{a}_comparison.mp4') for a in ('A', 'D')]
        (folder / 'video_archive_audit.json').write_text(json.dumps(videos, indent=2) + '\n')
        shutil.copytree(folder, DEST / 'report' / group, dirs_exist_ok=True)
        for src in folder.iterdir():
            if src.is_file():
                dst = DEST / 'report' / group / src.name
                assert sha(src) == sha(dst)
                files.append(dict(file=str(dst.relative_to(DEST)), sha256=sha(dst)))
    for row in parking['completed']:
        src = Path(row['path']).parent / 'evaluation.json'
        dst = DEST / src.relative_to(BASE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        assert sha(src) == sha(dst)
    result = dict(at=datetime.datetime.now().astimezone().isoformat(),
                  status='complete_evaluations_local_action_gate_failed',
                  optimizer_updates=48, evaluation_jobs=10, report_groups=8,
                  processes=processes, frozen_runtime_files=len(frozen),
                  frozen_sources_unchanged=True, parking_videos_archived=files,
                  new_training_started=False, old_workflow_superseded=True,
                  review_scope='all39-frame static contact sheets, not real-time playback',
                  stage1_accepted=False)
    (BASE / 'closeout.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for name in ('closeout.py', 'closeout.json', 'post48_queue.json', 'post48_reports.json',
                 'parking_queue_step48.json'):
        shutil.copy2(BASE / name, DEST / name)
    print(json.dumps({k: v for k, v in result.items() if k not in ('parking_videos_archived', 'processes')}, indent=2))


if __name__ == '__main__':
    main()
