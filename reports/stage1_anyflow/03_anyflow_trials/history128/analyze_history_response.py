"""Completed128 history comparison: existing flow metric and actual latent equality."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
GENERATED = ROOT / 'outputs/2026-10-08-13/stage1_parallel_resume68_to128'
sys.path.insert(0, str(ROOT / 'code/causal'))
from evaluate_action_control import _frames

torch.set_num_threads(4)
rows, audits = [], []
for action in 'AD':
    sources = []
    for history, folder, subdir in [('generated', GENERATED, 'generated_8step_native'),
                                    ('clean', OUT, 'clean_8step_native')]:
        directory = folder / f'eval/anyflow/step_128/{subdir}/{action}'
        e = json.loads((directory / 'evaluation.json').read_text())
        assert e['runtime']['status'] == 'complete'
        sources.append((history, directory, e))
        video = directory / 'cached.mp4'
        frames = list(_frames(video))
        assert len(frames) == 39
        values = []
        for previous, current in zip(frames, frames[1:]):
            flow = cv2.calcOpticalFlowFarneback(previous, current, None,
                pyr_scale=.5, levels=3, winsize=21, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0)
            h, w = current.shape
            values.append(float(np.mean(flow[12:h-12, 21:w-21, 0])))
        mean = float(np.mean(values))
        assert abs(mean - e['flow']['horizontal_flow_px']['mean']) < 1e-12
        rows.append(dict(action=action, history=history, step=128, steps_per_chunk=8,
            video=str(video), video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
            horizontal_mean=mean, per_transition=values,
            segments=[dict(rgb_range=[start, stop], transitions=stop-start-1,
                horizontal_mean=float(np.mean(values[start:stop-1])))
                for start, stop in [(0,17),(17,34),(34,39)]],
            boundary_transitions={str(i):values[i-1] for i in [17,34]}))
    gen, clean = [torch.load(d / 'cached_latents.pt', map_location='cpu', weights_only=True)
                  for _, d, _ in sources]
    assert gen.shape == clean.shape == (1,24,12,30,52)
    first = dict(exactly_equal=torch.equal(gen[:,:,:5],clean[:,:,:5]),
        max_abs=float((gen[:,:,:5]-clean[:,:,:5]).abs().max()))
    assert first['exactly_equal'], 'First chunk changed although it has no prior history'
    conditioning = [torch.load(d / 'conditioning.pt', map_location='cpu', weights_only=True)
                    for _, d, _ in sources]
    checks = {k:torch.equal(conditioning[0][k],conditioning[1][k])
              for k in ['initial_noise','audio_noise','prompt_embeds','anchor']}
    checks.update({'packed:'+k:torch.equal(conditioning[0]['packed'][k],conditioning[1]['packed'][k])
                   for k,v in conditioning[0]['packed'].items() if isinstance(v,torch.Tensor)})
    assert all(checks.values()), 'History experiment altered common conditioning'
    audits.append(dict(action=action, first_chunk_latents=first, conditioning=checks,
        chunk_latent_max_abs=[float((gen[:,:,s:t]-clean[:,:,s:t]).abs().max()) for s,t in [(0,5),(5,10),(10,12)]]))
report = dict(at=datetime.now().astimezone().isoformat(), step=128,
    scope='Same checkpoint generated/teacher history; actual tensor checks and unchanged Farneback metric',
    caveats=['Oracle history contains action outcomes; this is not a fixed-history current-action intervention.',
             'RGB ranges are not independent latent chunks due to temporal VAE; final range has four transitions.',
             'First-chunk latent equality is a history-isolation check, not proof of action fidelity.'],
    rows=rows, audits=audits)
(OUT / 'report').mkdir(exist_ok=True)
(OUT / 'report/history_response.json').write_text(json.dumps(report,indent=2)+'\n')
for row in rows:
    print(row['action'], row['history'], row['horizontal_mean'],
          [s['horizontal_mean'] for s in row['segments']])
print('First chunk and conditioning checks:', audits)
