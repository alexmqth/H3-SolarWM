"""CPU-only label coverage audit, no data selection from model outputs.

The optional source scan uses training episodes ONLY and creates no training
cache. Label purity is not proof of causal correctness or action-effect delay.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE/'runtime/code/abot'))
import abot_action as A

prep = json.loads((BASE/'preparation.json').read_text())
protocol = json.loads((BASE/'training_protocol.json').read_text())
clips = {r['clip_id']: r for r in prep['clips']}
used = []
for update, samples in enumerate(protocol['samples'], start=1):
    for sample in samples:
        row = clips[sample['clip_id']]
        window = next(w for w in row['windows'] if w['latent_start'] == sample['window']*12)
        active = {k: v for k, v in window['key_counts'].items() if v}
        used.append(dict(update=update, clip_id=sample['clip_id'], window=sample['window'], sigma=sample['sigma'],
                         active_keys=active, pure_lateral=set(active).issubset({'A', 'D'}),
                         camera_present=bool(set(active).intersection({'I', 'J', 'K', 'L'}))))

episodes = {r['sample_id']: r for r in prep['clips'] if r['split'] == 'train'}
assert not set(episodes).intersection(r['sample_id'] for r in prep['clips'] if r['split'] == 'validation')
offsets = np.asarray(A.window_offsets(124)); spans = A.frame_spans(37)
rows = []; inventory = []
for sid, row in sorted(episodes.items()):
    archive = BASE.parents[2]/'data/abot_bridge/raw/data'/sid[:2]/sid/'annotations.tar'
    digest = hashlib.sha256(archive.read_bytes()).hexdigest(); assert digest == row['source_annotations_sha256']
    with tarfile.open(archive) as handle:
        source = json.load(handle.extractfile('action.json'))
    assert source['fps'] == 30 and source['sample_stride'] == 1
    keys = np.array([[bool(f['keys'].get(k, False)) for k in A.KEY_COLS] for f in source['frames']])
    candidates = []
    for start in range(0, len(keys)-int(offsets[-1]), 15):
        for index in (1, 2):
            lo, hi = spans[index*12][0], spans[index*12+11][1]
            current = keys[start+offsets[lo:hi]]
            for action in ('A', 'D'):
                col = A.KEY_COLS.index(action)
                others = [i for i in range(len(A.KEY_COLS)) if i != col]
                if current[:, col].mean() >= .8 and not current[:, others].any():
                    candidates.append(dict(sample_id=sid, src_start=start, window=index, action=action,
                                           source_first=int(start+offsets[lo]), source_last=int(start+offsets[hi-1]),
                                           active_fraction=float(current[:, col].mean()), annotation_sha256=digest))
    rows.extend(candidates)
    # Distinct duration coverage: overlapping starts are NOT independent examples.
    intervals = sorted({(r['src_start'], r['src_start']+int(offsets[-1])) for r in candidates})
    independent = []
    for lo, hi in intervals:
        if not independent or lo > independent[-1][1]: independent.append([lo, hi])
    inventory.append(dict(sample_id=sid, overlapping_candidate_count=len(candidates),
                          greedy_nonoverlapping_full124_count=len(independent), full_source_spans=independent))

result = dict(at=datetime.now().astimezone().isoformat(), status='CPU_only_coverage_audit',
              GPU_calls=0, new_optimizer_updates=0, source_scan_split='train_only',
              scan_rule='Every15 source frames; existing12latent later windows; >=80% A-only or D-only, remaining neutral; no other key anywhere in current window. No outcome or validation criterion.',
              trained_microbatches=used, pure_lateral_microbatches=sum(r['pure_lateral'] for r in used),
              camera_mixed_microbatches=sum(r['camera_present'] for r in used),
              training_episode_inventory=inventory, overlapping_candidates=rows,
              limitations='Label purity is not a verified motion consequence. No new videos decoded, captions/scene compatibility/effect lag not audited. Candidates are NOT approved training data. No same-state counterfactual pair is created.')
(BASE/'supervision_coverage.json').write_text(json.dumps(result, indent=2)+'\n')
lines = ['# Supervision coverage audit\n\n',
         f"The frozen four-update schedule uses {len(used)} microbatches: **{result['pure_lateral_microbatches']} pure A/D**, {result['camera_mixed_microbatches']} mixed with camera keys. Other examples include W/S alongside A/D. This is a coverage limitation, not proof of the cause of ghosting.\n\n",
         '| Update | Window | Sigma | Active keys (RGB count) |\n|---:|---:|---:|---|\n']
for row in used:
    lines.append(f"| {row['update']} | {row['window']} | {row['sigma']} | {row['active_keys']} |\n")
lines += ['\n## Training-only availability scan\n\n',
          'No GPU, downloads, encoding, optimizer updates or validation tuning. These are label candidates only; reliability needs source RGB/action-delay review before any new experiment. Overlapping windows are not independent samples.\n\n',
          '| Training episode | Overlapping candidates | Greedy nonoverlapping 124RGB spans |\n|---|---:|---:|\n']
for row in inventory:
    lines.append(f"| {row['sample_id']} | {row['overlapping_candidate_count']} | {row['greedy_nonoverlapping_full124_count']} |\n")
lines += ['\nThe action-swap ranking objective explains one observed noisy transition. It does not provide an observed alternate-action video, likelihood ratio, or direct motion-direction target. Thus even successful ranking would still require independent video action/structure validation.\n',
          '\nThe GT generation cases also use dataset-native joint controls and measured current-window F. This is oracle conditioning for a local structure check, not evidence that the measured speed is available to an interactive user. Pure parking A/D controls have no such F input.\n']
(BASE/'SUPERVISION_COVERAGE.md').write_text(''.join(lines))
print(json.dumps({k: result[k] for k in ['pure_lateral_microbatches', 'camera_mixed_microbatches', 'training_episode_inventory']}))
