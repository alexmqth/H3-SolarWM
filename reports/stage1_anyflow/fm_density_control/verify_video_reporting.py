"""Exercise the frozen video reporter using old completed controls only.

All rendered identity fixtures are temporary and deleted. They are not new
candidate outputs or quality evidence. No model forward or GPU is used.
"""
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time

import av


BASE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    receipt = BASE / 'video_report_validation.json'
    if receipt.exists():
        raise FileExistsError('Existing validation receipt; inspect instead of repeating')
    start = time.perf_counter()
    source = BASE / 'report_density.py'
    fingerprint = sha(source)
    result = dict(status='running', at=datetime.now().astimezone().isoformat(),
        scope='CPU full-video reporter integration using OLD control in both FM48 columns; no candidate results',
        report_source_sha256=fingerprint, checks=[], GPU_forwards=0,
        fixture_artifacts_retained=False)
    try:
        spec = importlib.util.spec_from_file_location('density_report_fixture', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Deliberate identity fixture: both FM48 columns read the old branch.
        # Do not write into the live report directory or alter any source.
        module.BASE = module.CONTROL
        with tempfile.TemporaryDirectory(prefix='density_video_fixture_') as temporary:
            root = Path(temporary)
            jobs = (
                ('natural_gt_30', lambda out: module.report_natural(out, 'gt', 30)),
                ('parking_8', lambda out: module.report_parking(out, 8)),
            )
            for name, run in jobs:
                out = root / name
                run(out)
                report = json.loads((out / 'metrics.json').read_text())
                assert report['status'] == 'complete_artifacts'
                assert report['stage1_accepted'] is False
                assert report['visual_review'] == 'pending full39-frame manual inspection'
                rows = report['rows']
                # Identical sources must produce identical recorded statistics.
                keys = ('seconds', 'noisy_forwards', 'clean_commits', 'gpu_peak_MiB',
                        'cpu_kv_MiB', 'frame_gray_MAD', 'boundary_rgb_MAD', 'horizontal_flow')
                group_field = 'clip' if name.startswith('natural') else 'action'
                for group in sorted({r[group_field] for r in rows}):
                    pair = [r for r in rows if r[group_field] == group and r['method'].startswith('FM48')]
                    assert len(pair) == 2
                    assert all(pair[0][key] == pair[1][key] for key in keys)
                decoded = []
                for video in report['artifacts']:
                    path = out / video
                    with av.open(str(path)) as container:
                        stream = container.streams.video[0]
                        frames = list(container.decode(video=0))
                        assert len(frames) == 39 and stream.average_rate == 24
                        assert stream.codec_context.name == 'h264'
                        assert all(frame.format.name == 'yuv420p' for frame in frames)
                    content = path.read_bytes()
                    # Parse top-level ISO BMFF atoms to check faststart.
                    atoms = []
                    offset = 0
                    while offset < len(content):
                        size = int.from_bytes(content[offset:offset+4], 'big')
                        kind = content[offset+4:offset+8].decode('ascii')
                        if size == 1:
                            size = int.from_bytes(content[offset+8:offset+16], 'big')
                        if size == 0:
                            size = len(content)-offset
                        assert size >= 8
                        atoms.append(kind)
                        offset += size
                    assert offset == len(content)
                    assert atoms.index('moov') < atoms.index('mdat')
                    decoded.append(dict(name=video, frames=39, fps=24,
                        codec='h264', pixel_format='yuv420p', faststart=True))
                assert len(decoded) == 2
                assert len(list(out.glob('*_all39.jpg'))) == len(rows)
                assert len(list(out.glob('*_matched.jpg'))) == 2
                result['checks'].append(dict(group=name, identity_metrics=True,
                    no_automatic_visual_PASS=True, full_contact_sheets=len(rows), videos=decoded))
        assert sha(source) == fingerprint
        result['status'] = 'passed'
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['wall_seconds'] = time.perf_counter()-start
        receipt.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
