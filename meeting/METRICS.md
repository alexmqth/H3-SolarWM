# Meeting metrics

The formal 124-frame fixed-mix grid is the main fair comparison. Each action uses the same initial RGB frame, scene prompt, action sequence held for all 8 chunks, seed 13, initial video/audio noise, resolution, and frame count. The values below are one recorded run per action, not a warmup-then-multiple-run mean; this is stated explicitly so the table is not overclaimed.

## End-to-end and memory

| Action | Original 30-step e2e (s) | Causal 8-step/chunk e2e (s) | Original peak GPU (MiB) | Causal peak GPU (MiB) | Causal CPU raw KV (MiB) | Causal first chunk (s) | Causal mean chunk (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| W | 441.5 | 450.2 | 39925 | 39927 | 13509 | 41.5 | 47.4 |
| S | 444.8 | 451.9 | 39925 | 39927 | 13509 | 41.1 | 47.5 |
| A | 454.2 | 438.8 | 39925 | 39948 | 13509 | 41.3 | 45.6 |
| D | 450.4 | 383.4 | 39925 | 39948 | 13509 | 36.7 | 39.0 |

## Video continuity and control

| Action | Original mean MAD | Causal mean MAD | Original boundary MAD | Causal boundary MAD | Original horizontal flow | Causal horizontal flow |
|---|---:|---:|---:|---:|---:|---:|
| W | 4.32 | 4.60 | 4.97 | 6.29 | -1.111 | -0.624 |
| S | 4.13 | 4.62 | 4.70 | 7.76 | -1.084 | -0.096 |
| A | 4.52 | 7.10 | 5.14 | 13.68 | +1.077 | +0.143 |
| D | 4.46 | 5.39 | 5.39 | 9.47 | -1.602 | -0.311 |

## Interpretation

- The causal branch uses 64 noisy denoiser forwards plus 8 clean KV commits, while the original uses 30 full-horizon forwards. The current prototype is therefore not a wall-clock speedup; its contribution is causal execution, history reuse, and a path toward interactive chunking.
- A/D separation in this formal fixed-mix grid is 2.679 for the original teacher and 0.453 for causal. The causal video is action-dependent, but direction magnitude and temporal continuity degrade.
- A later RGB-consistent visual adapter repairs person/garage decomposition in 39/124-frame clips. That visual-stability result is shown separately and must not be read as action-geometry recovery.
- FVD/LPIPS/PSNR/VBench are not reported because this held-out demo has no frame-aligned ground-truth future for the generated rollout. The submitted quality proxies are RGB MAD, chunk-boundary MAD, signed horizontal Farneback flow, and visual contact sheets.
