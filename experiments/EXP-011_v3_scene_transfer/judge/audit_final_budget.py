"""CPU-only final accounting; per-case tensor audits are separate frozen evidence."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / 'H3-World/outputs/EXP-011_v3_scene_transfer'
budget_path = OUT / 'budget.json'
budget = json.loads(budget_path.read_text())
expected = dict(sampling_forwards=228, commit_forwards=4, vae_decodes=12,
                text_encoder_calls=6, image_vae_encodes=2, video_vae_encodes=0)
assert {k: budget[k] for k in expected} == expected
assert budget['p1_gpu_seconds'] <= 900 and budget['inference_gpu_seconds'] <= 4500
counts = Counter()
peak = 0.
for scene in ('industrial', 'village'):
    for method in ('FM30', 'FM8'):
        for stage in ('G1', 'G2'):
            row = json.loads((OUT / stage / scene / method / 'result.json').read_text())
            audit = json.loads((HERE / f'{stage}_{scene}_{method}_audit.json').read_text())
            assert audit[f'independent_{stage}_execution_audit'] == 'PASS'
            assert row['status'] == 'complete_pending_visual_review'
            for key in ('sampling_forwards', 'commit_forwards', 'vae_decodes'):
                counts[key] += row[key]
            peak = max(peak, row['peak_allocated_gib'])
for key in counts:
    assert counts[key] == expected[key]
starts = []
elapsed = 0.
stages = []
for event in budget['events']:
    if not event.get('stage', '').startswith('G'):
        continue
    if event['event'] == 'stage_start':
        assert not starts, 'Overlapping inference stages not covered by single-GPU accounting'
        starts.append(event)
    elif event['event'] == 'stage_stop':
        start = starts.pop()
        assert start['stage'] == event['stage'] and event['status'] == 'complete'
        elapsed += (datetime.fromisoformat(event['at']) - datetime.fromisoformat(start['at'])).total_seconds()
        stages.append(event['stage'])
assert not starts and len(stages) == 8
assert abs(elapsed - budget['inference_gpu_seconds']) < .2
result = dict(status='PASS', task='EXP-011/v1', actual_counts=expected,
              denoiser_forwards=counts['sampling_forwards'] + counts['commit_forwards'],
              inference_gpu_seconds=budget['inference_gpu_seconds'],
              p1_gpu_seconds_including_failed_startups=budget['p1_gpu_seconds'],
              total_gpu_hours=(budget['p1_gpu_seconds'] + budget['inference_gpu_seconds']) / 3600,
              inference_peak_allocated_gib=peak,
              completed_inference_stages=stages,
              budget_sha256=hashlib.sha256(budget_path.read_bytes()).hexdigest(),
              scope='Execution accounting only; visual capability reviewed separately')
(HERE / 'FINAL_BUDGET_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
(HERE / 'budget_final_frozen.json').write_bytes(budget_path.read_bytes())
print(json.dumps(result, indent=2))
