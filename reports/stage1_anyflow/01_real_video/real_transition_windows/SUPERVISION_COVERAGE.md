# Supervision coverage audit

The frozen four-update schedule uses 8 microbatches: **0 pure A/D**, 3 mixed with camera keys. Other examples include W/S alongside A/D. This is a coverage limitation, not proof of the cause of ghosting.

| Update | Window | Sigma | Active keys (RGB count) |
|---:|---:|---:|---|
| 1 | 1 | 0.85 | {'A': 41, 'S': 41, 'L': 42} |
| 1 | 1 | 0.55 | {'W': 36, 'D': 42} |
| 2 | 1 | 0.25 | {'S': 40, 'D': 41, 'J': 1} |
| 2 | 2 | 0.1 | {'W': 39, 'A': 39} |
| 3 | 1 | 0.55 | {'A': 42, 'S': 42, 'J': 42} |
| 3 | 1 | 0.85 | {'W': 34, 'D': 42} |
| 4 | 1 | 0.1 | {'W': 33, 'A': 26} |
| 4 | 0 | 0.25 | {'W': 39, 'D': 39} |

## Training-only availability scan

No GPU, downloads, encoding, optimizer updates or validation tuning. These are label candidates only; reliability needs source RGB/action-delay review before any new experiment. Overlapping windows are not independent samples.

| Training episode | Overlapping candidates | Greedy nonoverlapping 124RGB spans |
|---|---:|---:|
| 43866101158a538d22c52ac5b2fa606b | 0 | 0 |
| 7199292ca430b478bfd0d83cf2e519d8 | 0 | 0 |
| 9dc2e5882a294b8d68d6b2b31ff3ef65 | 5 | 1 |
| b784d995827f53977a8067e5b9dc4b5e | 10 | 1 |

The action-swap ranking objective explains one observed noisy transition. It does not provide an observed alternate-action video, likelihood ratio, or direct motion-direction target. Thus even successful ranking would still require independent video action/structure validation.

The GT generation cases also use dataset-native joint controls and measured current-window F. This is oracle conditioning for a local structure check, not evidence that the measured speed is available to an interactive user. Pure parking A/D controls have no such F input.

## Source visual follow-up

Two train-only candidates were extracted and checked without a GPU. The D road clip contains clear rightward walking and intact structure. The A outdoor clip is dark and initially nearly stationary despite A becoming active at RGB85; strong visible walking begins later. This is an unresolved timing/control-effect question, not proof of a fixed dataset-wide label offset. No label shift was applied. The scenes/states differ, so they are not a counterfactual pair. Neither was encoded or admitted into a new training run.

[Source review and exact indices](coverage_candidate_review/review.json) · [A source sheet](coverage_candidate_review/train_pure_A_source.jpg) · [D source sheet](coverage_candidate_review/train_pure_D_source.jpg).
