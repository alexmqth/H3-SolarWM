# QKV tail4 paired action alignment: 4-update rollout curve

Protocol: 39 RGB frames / 3 chunks / 8 steps per chunk / RGB dual anchor / generated history / seed 13.

| Update | flow(A) | flow(D) | A-D | paired cosine proxy (1-dir loss) | norm ratio | A/D gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | -0.8563 | -0.7790 | -0.0774 | +0.3901 | 1.018 | FAIL |
| 2 | -0.8886 | -0.7536 | -0.1350 | +0.3474 | 0.973 | FAIL |
| 3 | -0.8703 | -0.7785 | -0.0918 | +0.3448 | 1.019 | FAIL |
| 4 | -0.8506 | -0.7684 | -0.0822 | +0.4242 | 1.045 | FAIL |

The optical-flow values are a signed image-motion proxy. Positive/negative signs are the frozen protocol gate; they are not a complete action-accuracy metric.

## Per-video stability

| Update | Action | RGB boundary MAD | Frame MAD | Sampling s | GPU GiB | CPU KV GiB |
|---:|:---:|---:|---:|---:|---:|---:|
| 1 | A | 2.843 | 3.732 | 181.9 | 39.17 | 6.33 |
| 1 | D | 2.796 | 3.666 | 182.8 | 39.17 | 6.33 |
| 2 | A | 2.949 | 3.780 | 174.7 | 39.17 | 6.33 |
| 2 | D | 2.855 | 3.523 | 171.1 | 39.17 | 6.33 |
| 3 | A | 2.918 | 3.760 | 163.6 | 39.17 | 6.33 |
| 3 | D | 2.838 | 3.690 | 181.8 | 39.17 | 6.33 |
| 4 | A | 2.870 | 3.742 | 182.3 | 39.17 | 6.33 |
| 4 | D | 2.896 | 3.609 | 172.5 | 39.17 | 6.33 |

## Interpretation

- RGB-consistent anchor keeps the person and parking-garage structure stable through frame 38 in all evaluated checkpoints.
- Four QKV paired updates are not a successful action-control recovery if the flow gate remains failed.
- A lower direction loss or near-one score-field norm ratio is an internal alignment diagnostic; it does not imply that free generated-history rollout has the correct image-space action sign.
