"""Independent CPU check of inventory, deterministic selection and raw key alignment."""
from pathlib import Path
import hashlib
import json
import tarfile
import av
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = HERE.parents[3]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


rows = [json.loads(x) for x in (EXP / 'inventory.jsonl').read_text().splitlines() if x]
assert len(rows) == 24 and len({r['clip_id'] for r in rows}) == 24
keys = ['W', 'A', 'S', 'D', 'Q', 'E', 'I', 'J', 'K', 'L', 'Space']
raw = {}
splits = {'train': set(), 'validation': set()}
video_hashes = {'train': set(), 'validation': set()}
total_frames = 0
for row in rows:
    sid = row['sample_id']
    splits[row['split']].add(sid)
    video_hashes[row['split']].add(row['source_video_sha256'])
    if sid not in raw:
        assert sha(row['source_video']) == row['source_video_sha256']
        assert sha(row['source_annotations']) == row['source_annotations_sha256']
        with tarfile.open(row['source_annotations']) as t:
            raw[sid] = json.load(t.extractfile('action.json'))
    for name in ('video', 'first_frame', 'action'):
        assert sha(row[f'{name}_path']) == row[f'{name}_sha256']
    indices = [row['src_start'] + 5 * (j // 4) + j % 4 for j in range(39)]
    assert row['source_frame_indices'] == indices
    a = np.load(row['action_path'], allow_pickle=False)
    assert a.shape == (39, 17) and np.isfinite(a).all()
    expected = np.array([[int(raw[sid]['frames'][i]['keys'][k]) for k in keys] for i in indices])
    assert np.array_equal(a[:, :11], expected)
    assert row['action_recomputed_from_source_max_abs'] == 0
    with av.open(row['video_path']) as c:
        f = list(c.decode(video=0))
        assert len(f) == 39 and c.streams.video[0].average_rate == 24
        assert all((v.width, v.height) == (832, 480) for v in f)
        assert all(b.pts > a.pts for a, b in zip(f, f[1:]))
        assert np.array_equal(f[0].to_ndarray(format='rgb24'), np.asarray(Image.open(row['first_frame_path']).convert('RGB')))
        total_frames += len(f)
assert len(splits['train']) == 4 and len(splits['validation']) == 2
assert splits['train'].isdisjoint(splits['validation'])
assert video_hashes['train'].isdisjoint(video_hashes['validation'])
assert splits['validation'] == {'118eb5d8b75e1b8ac23a4e9ae77af9a9', 'dfec8ed3237860eba14d67c089ecd041'}
expected = json.loads((HERE / 'EXPECTED_SELECTION.json').read_text())['selected']
selected = json.loads((EXP / 'candidate_manifest.json').read_text())['selected']
assert len(selected) == len(expected) == 4
for a, b in zip(selected, expected):
    assert a['episode_id'] == b['sample_id'] and a['clip_id'] == b['clip_id']
    assert a['first_frame_sha256'] == b['image_sha256'] == sha(a['first_frame'])
    assert a['split'] == 'train' and a['counterfactual_is_not_recorded_ground_truth'] is True
    group = [r for r in rows if r['sample_id'] == a['episode_id'] and r['target_filename_label'] == 'A']
    assert a['clip_id'] == min(group, key=lambda r: (r['src_start'], r['clip_id']))['clip_id']
worker = json.loads((EXP / 'source_split_action_audit.json').read_text())
assert not worker['clip_check_failures']
assert worker['old_encoded_suitable_for_V3_Single_I0'] is False
rebuilds = worker['ambiguous_pixel_shift_exact_ffmpeg_rebuilds']
assert all(x['exact_byte_match'] and x['rebuild_sha256'] == x['existing_sha256'] for x in rebuilds)
result = dict(status='PASS', task='EXP-013/v1', gpu_calls=0,
              clips=24, complete_decoded_frames=total_frames,
              raw_binary_actions_checked=24*39, key_columns=11,
              continuous_columns='Worker recomputation from source helper exactly matches all24; independent raw spot checks focus on binary keys',
              deterministic_candidates_match=True, source_and_episode_split_disjoint=True,
              png_exactly_matches_decoded_first_frame_all24=True,
              ambiguous_pixel_proxy='Not evidence of misalignment: producer rebuilds match existing bytes',
              inventory_sha256=sha(EXP / 'inventory.jsonl'), candidate_manifest_sha256=sha(EXP / 'candidate_manifest.json'))
(HERE / 'INDEPENDENT_DATA_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
