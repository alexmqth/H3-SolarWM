"""Reuse the existing Farneback metric on completed videos, split by time.

This is descriptive evidence, not a new action classifier or pass criterion.
Boundaries use the existing 17/34 RGB convention; temporal VAE receptive fields
mean these ranges are not independent chunk-isolation experiments.
"""
from pathlib import Path
import hashlib
import json
import sys

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'code/causal'))
from evaluate_action_control import _frames

sources = []
for a in 'AD':
    sources.append(dict(method='Original', action=a, steps=30,
        video=str(ROOT / f'outputs/2026-10-02-03/action_{a}_teacher_39/baseline.mp4')))
for folder in (OUT, ROOT / 'outputs/2026-10-08-10/stage1_shift12_duration64',
               ROOT / 'outputs/2026-10-08-11/stage1_shift12_step32_4step'):
    for e in json.loads((folder / 'run.json').read_text())['evaluations']:
        if e['step'] == 32:
            sources.append(dict(method=e['objective']+'32', action=e['action'],
                steps=e['steps_per_chunk'], video=e['video'],
                recorded_horizontal_mean=e['flow']['horizontal_flow_px']['mean']))
assert len(sources) == 10
for row in sources:
    frames = list(_frames(Path(row['video'])))
    assert len(frames) == 39
    values = []
    for previous, current in zip(frames, frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(previous, current, None,
            pyr_scale=.5, levels=3, winsize=21, iterations=3, poly_n=5,
            poly_sigma=1.2, flags=0)
        h, w = current.shape
        values.append(float(np.mean(flow[12:h-12, 21:w-21, 0])))
    row['horizontal_mean'] = float(np.mean(values))
    if 'recorded_horizontal_mean' in row:
        assert abs(row['horizontal_mean'] - row['recorded_horizontal_mean']) < 1e-12
    row['per_transition'] = values
    row['segments'] = [dict(rgb_range=[start, stop], transition_count=stop-start-1,
        horizontal_mean=float(np.mean(values[start:stop-1])))
        for start, stop in ((0, 17), (17, 34), (34, 39))]
    row['boundary_transitions'] = {str(i): values[i-1] for i in (17, 34)}
    row['video_sha256'] = hashlib.sha256(Path(row['video']).read_bytes()).hexdigest()
report = dict(scope='Existing Farneback metric split by RGB time; no new metric or quality gate',
    caveat='Ranges follow 17/34 RGB boundaries; not independent latent chunk isolation due to temporal VAE. Short final range has only four transitions.',
    rows=sources)
(OUT / 'report/segment_response.json').write_text(json.dumps(report, indent=2)+'\n')
for r in sources:
    print(r['method'], r['action'], r['steps'], [round(x['horizontal_mean'],6) for x in r['segments']])
