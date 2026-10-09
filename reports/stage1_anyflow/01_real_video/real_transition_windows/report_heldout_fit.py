"""Summarize the preregistered 48-forward diagnostic without fitting coefficients."""
import hashlib
import json
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parent
source = BASE/'heldout_fit/fit.json'
data = json.loads(source.read_text())
assert data['status'] == 'complete' and data['forwards'] == 48
assert data['state_pairing_verified'] and len(data['records']) == 24
roles = ['zero', 'fm_only', 'fm_action']
fields = ['positive_FM', 'negative_FM', 'gap', 'same_state_action_delta_rms']
summary = []
for sigma in [None, .85, .55, .25, .10]:
    for role in roles:
        rows = [r for r in data['records'] if r['role'] == role and (sigma is None or r['sigma'] == sigma)]
        summary.append(dict(role=role, sigma=sigma, n=len(rows),
                            correct_order=sum(r['gap'] > 0 for r in rows),
                            **{key: mean(r[key] for r in rows) for key in fields}))
pairs = []
for zero in [r for r in data['records'] if r['role'] == 'zero']:
    for role in roles[1:]:
        row = next(r for r in data['records'] if r['role'] == role and r['clip_id'] == zero['clip_id'] and r['sigma'] == zero['sigma'])
        for key in ['history_sha256', 'noise_sha256', 'state_sha256', 'target_sha256']:
            assert row[key] == zero[key]
        pairs.append(dict(role=role, clip_id=row['clip_id'], sigma=row['sigma'],
                          **{key+'_change': row[key]-zero[key] for key in fields}))
result = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), summary=summary,
              paired_changes=pairs, validation_used_for_tuning=False,
              caveat='Two episodes/states at four correlated sigmas, not eight independent video trials. Negative action has no observed counterfactual video.')
(BASE/'heldout_fit/summary.json').write_text(json.dumps(result, indent=2)+'\n')
lines = ['# Held-out observed-transition fit: 0 vs 4 updates\n\n',
         '48 matched-state forwards; two validation transitions, four sigmas, positive/swapped action, three parameter banks. No coefficient selection or retraining uses these values.\n\n',
         '| Sigma | Bank | Positive FM | Swapped-action FM | Gap (neg − pos) | Action delta RMS | Correct order |\n',
         '|---|---|---:|---:|---:|---:|---:|\n']
for row in summary:
    sigma = 'all' if row['sigma'] is None else str(row['sigma'])
    lines.append(f"| {sigma} | {row['role']} | {row['positive_FM']:.8f} | {row['negative_FM']:.8f} | {row['gap']:+.8f} | {row['same_state_action_delta_rms']:.8f} | {row['correct_order']}/{row['n']} |\n")
lines += ['\n## Interpretation\n\n',
          'Both arms reduce mean positive FM by less than 0.03% after four updates. FM+action does not improve the mean action ranking over FM-only: correct ordering is 4/8 in both, versus 5/8 at initialization. Its mean action delta RMS also does not increase. This is no evidence of added benefit from this short ranking-loss trial.\n\n',
          'The high-noise sigma0.85 case dominates the negative average gap, while signs vary across state and sigma. Larger low-noise absolute FM alone is not a controlled demonstration that the training noise distribution caused the video artifacts. Do not change sigma weighting or lambda based on this validation set.\n\n',
          'A lower swapped-action error does not prove a swapped-action video is correct: the target is the one actually observed transition. The two scenes contain joint character and camera controls, and there are no observed alternate-action videos. This diagnostic must be considered together with pure-action parking generation and full-frame structure review.\n\n',
          'Four optimizer updates and two held-out states cannot establish that action-consequence supervision is impossible or that more training would necessarily help. They do not justify promoting this checkpoint to AnyFlow or Stage2.\n']
(BASE/'heldout_fit/RESULTS.md').write_text(''.join(lines))
print(json.dumps(result['summary'][:3]))
