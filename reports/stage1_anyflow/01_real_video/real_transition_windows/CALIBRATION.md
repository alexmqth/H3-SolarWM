# Train-only action loss calibration

No optimizer updates.2 training clips,4 sigmas each, same actual history/state for positive vs current A/D-swapped condition.

| Clip | sigma | e_positive | e_negative | e_negative−e_positive | FM/pair grad cosine |
|---|---:|---:|---:|---:|---:|
| 438661 / A | 0.85 | 0.12943675 | 0.12916636 | -0.00027038 | 0.2499 |
| 438661 / A | 0.55 | 0.15152352 | 0.15147768 | -0.00004584 | 0.0205 |
| 438661 / A | 0.25 | 0.24875256 | 0.24884164 | +0.00008908 | 0.0525 |
| 438661 / A | 0.1 | 0.40077603 | 0.40066457 | -0.00011146 | 0.0453 |
| 438661 / D | 0.85 | 0.09219752 | 0.09210046 | -0.00009706 | 0.0834 |
| 438661 / D | 0.55 | 0.11845757 | 0.11855961 | +0.00010204 | 0.0280 |
| 438661 / D | 0.25 | 0.21787453 | 0.21795274 | +0.00007822 | 0.0206 |
| 438661 / D | 0.1 | 0.37459934 | 0.37456524 | -0.00003409 | 0.0405 |

Wrong-current-action has smaller FM error in **5/8** states. This is a conditional fit diagnostic, not a motion-quality or action-direction score.

Pre-registered rule yields margin=9.306892753e-05, lambda=2.485890831. Active states=7/8; action gradient initially targets25% of FM magnitude (cap10 was inactive). This is a limited train-only scale choice, not tuned on validation.

Wall=267.62s; peak allocated=25.97GiB.16forward/backward cases plus checkpoint re-execution. Parameters unchanged.
