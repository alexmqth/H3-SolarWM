# Full-teacher A/D delta audit

The frozen original bidirectional H3 teacher was evaluated on the same generated A rollout state. For each latent chunk and sigma, only the action conditioning was changed from A to D. This audit does not update a student.

| chunk | sigma | ||vA|| | ||vD|| | ||vA-vD|| | delta / mean velocity |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.939541 | 825.54 | 830.59 | 17.09 | 0.0206 |
| 0 | 0.689441 | 657.10 | 660.12 | 9.79 | 0.0149 |
| 0 | 0.425287 | 535.42 | 535.50 | 8.19 | 0.0153 |
| 0 | 0.240781 | 473.79 | 474.02 | 12.36 | 0.0261 |
| 0 | 0.000000 | 493.98 | 494.97 | 16.63 | 0.0336 |
| 1 | 0.939541 | 805.84 | 807.13 | 12.16 | 0.0151 |
| 1 | 0.689441 | 690.46 | 694.79 | 13.95 | 0.0201 |
| 1 | 0.425287 | 589.36 | 589.75 | 7.56 | 0.0128 |
| 1 | 0.240781 | 532.56 | 534.38 | 10.93 | 0.0205 |
| 1 | 0.000000 | 518.55 | 518.69 | 10.46 | 0.0202 |
| 2 | 0.939541 | 481.42 | 486.02 | 11.83 | 0.0245 |
| 2 | 0.689441 | 411.97 | 409.44 | 8.68 | 0.0211 |
| 2 | 0.425287 | 336.49 | 337.98 | 5.43 | 0.0161 |
| 2 | 0.240781 | 305.47 | 303.69 | 7.64 | 0.0251 |
| 2 | 0.000000 | 308.28 | 308.28 | 7.48 | 0.0243 |

Mean delta/velocity ratio: 0.0207; range: 0.0128–0.0336.

Interpretation: A/D teacher signal is finite and nonzero on this generated state, but small relative to the full velocity field (roughly 1–3%). This explains why a few million-parameter residual can show a valid gradient while changing the rollout only weakly. It does not prove that the target direction is correctly aligned with image-space strafe motion, because this report does not backpropagate through the student or measure optical flow.

Limits: the state comes from one A rollout, one scene and one seed; the teacher uses the generated prefix/current chunk, not a full original 39-frame trajectory. A second audit on a D-generated state or multiple seeds would be needed before claiming a general target property.

- Raw data: `audit.json`.
- Script: `H3-World/code/causal/audit_teacher_action_delta.py`.
