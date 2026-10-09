# Full-teacher A/D delta audit on D-generated state

The frozen original bidirectional H3 teacher was evaluated on the same generated D rollout state. For each latent chunk and sigma, only the action conditioning was changed from A to D.

| chunk | sigma | ||vA|| | ||vD|| | ||vA-vD|| | delta / mean velocity |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.939541 | 802.18 | 803.03 | 12.47 | 0.0155 |
| 0 | 0.689441 | 658.22 | 658.15 | 10.87 | 0.0165 |
| 0 | 0.425287 | 499.50 | 499.96 | 9.42 | 0.0188 |
| 0 | 0.240781 | 444.88 | 444.83 | 10.96 | 0.0246 |
| 0 | 0.000000 | 475.67 | 475.57 | 20.18 | 0.0424 |
| 1 | 0.939541 | 771.00 | 776.79 | 17.50 | 0.0226 |
| 1 | 0.689441 | 674.35 | 670.29 | 14.21 | 0.0211 |
| 1 | 0.425287 | 528.78 | 528.56 | 8.65 | 0.0164 |
| 1 | 0.240781 | 469.64 | 471.42 | 10.24 | 0.0218 |
| 1 | 0.000000 | 463.08 | 462.89 | 9.26 | 0.0200 |
| 2 | 0.939541 | 483.85 | 484.69 | 9.23 | 0.0191 |
| 2 | 0.689441 | 421.21 | 419.73 | 6.30 | 0.0150 |
| 2 | 0.425287 | 326.13 | 327.09 | 4.97 | 0.0152 |
| 2 | 0.240781 | 280.44 | 279.42 | 5.39 | 0.0192 |
| 2 | 0.000000 | 280.45 | 280.38 | 5.59 | 0.0199 |

Mean delta/velocity ratio: 0.0206; range: 0.0150–0.0424.

The delta remains finite and nonzero on this independent generated state, but is a small fraction of the full score field. This is evidence about target magnitude only; it does not measure student alignment or image-space motion.

- Raw data: `audit.json`.
- Script: `H3-World/code/causal/audit_teacher_action_delta.py`.
