# 39-frame FP32 action-gain learning curve

No checkpoint passed the action gate. RGB dual improves visual stability over the earlier latent-anchor failure, but A still has weak motion.

| Model | flow(A) | flow(D) | A-D | Numeric gate |
|---|---:|---:|---:|---|
| teacher | +1.1813 | -0.8421 | 2.0234 | PASS |
| inference_A64_D8 | +0.0474 | -0.7812 | 0.8286 | FAIL |
| step_01 | +0.0489 | -0.7782 | 0.8272 | FAIL |
| step_02 | +0.0445 | -0.7629 | 0.8075 | FAIL |
| step_03 | +0.0431 | -0.7782 | 0.8213 | FAIL |
| step_04 | +0.0350 | -0.7652 | 0.8002 | FAIL |

## Training

Completed four A/D alternating optimizer updates in 2265.4 s (37.8 min); peak allocated VRAM 40.06 GiB; student and teacher each use 6.33 GiB CPU raw KV.

| Step | Rollout action | Replay | Direction | Magnitude | Boundary | Gain A | Gain D |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | A | 0.03111 | 1.04804 | 8.11859 | 0.06588 | 63.98360 | 8.00920 |
| 2 | D | 0.03567 | 0.94514 | 8.20047 | 0.00086 | 63.96719 | 8.01841 |
| 3 | A | 0.02673 | 1.02404 | 6.86086 | 0.05609 | 63.95080 | 8.02762 |
| 4 | D | 0.03127 | 0.95503 | 7.33404 | 0.00093 | 63.93443 | 8.03687 |

Only the existing action residual projections and their gains are optimized. This is paired teacher replay, not critic/DMD. Gains start at manually selected A=64/D=8; this training does not learn the amplitude difference from unity.

Direction loss remains near 1 (near-orthogonal teacher/student deltas); relative magnitude error remains about 7. Losses should be compared on the same rollout action (1 vs 3, 2 vs 4), not treated as monotonic across different states.

The FP32 fix removes early rounding of gains in the small projection. The final residual still enters a BF16 model. Legacy checkpoints retain the old precision mode for exact archived-path compatibility.

## Inference and continuity

| Model/action | Sampling s | After-cond s | Total s | GPU GiB | CPU KV GiB | Noisy + commit calls | RGB MAD | Boundary RGB MAD |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| teacher/A | 151.4 | 197.7 | 212.4 | 38.99 | 0.00 | 30 + 0 | 5.149 | 4.125 |
| teacher/D | 151.6 | 203.3 | 218.0 | 38.99 | 0.00 | 30 + 0 | 3.777 | 4.399 |
| inference_A64_D8/A | 160.4 | 204.3 | 216.6 | 39.17 | 6.33 | 24 + 3 | 1.596 | 2.547 |
| inference_A64_D8/D | 166.1 | 211.1 | 223.3 | 39.17 | 6.33 | 24 + 3 | 3.860 | 4.373 |
| step_01/A | 186.7 | 232.6 | 246.0 | 39.17 | 6.33 | 24 + 3 | 1.809 | 2.379 |
| step_01/D | 164.9 | 211.8 | 224.7 | 39.17 | 6.33 | 24 + 3 | 3.872 | 4.813 |
| step_02/A | 168.2 | 216.0 | 230.9 | 39.17 | 6.33 | 24 + 3 | 1.768 | 2.140 |
| step_02/D | 147.6 | 190.1 | 203.2 | 39.17 | 6.33 | 24 + 3 | 3.547 | 3.939 |
| step_03/A | 170.0 | 215.9 | 228.8 | 39.17 | 6.33 | 24 + 3 | 1.837 | 1.957 |
| step_03/D | 190.4 | 236.9 | 249.9 | 39.17 | 6.33 | 24 + 3 | 3.749 | 4.510 |
| step_04/A | 156.8 | 202.4 | 215.3 | 39.17 | 6.33 | 24 + 3 | 1.827 | 1.875 |
| step_04/D | 159.1 | 205.9 | 218.9 | 39.17 | 6.33 | 24 + 3 | 3.513 | 3.795 |

- One scene and seed; no significance claim from small differences.
- 39 frames at 24 fps is 1.625 seconds, not the final 124-frame demo.
- Both action residual projection weights and nine gains were trained; visual tail16 QKV and released H3 LoRA remained frozen.
- RGB/frame and boundary MAD describe motion and discontinuity, not quality. Boundary frames are 17 and 34.
- Timing is observational: training on GPU0 overlapped evaluation on GPUs1/4; shared CPU/I/O load differs from archived teacher timing.
- New cache replay equality was not rerun; replay error=0 refers to earlier cache correctness tests.
- Original H3 reference here is the archived 30-step benchmark with flow shift 2.22.
- Four updates cannot establish an architectural ceiling or prove Stage2 is necessary/sufficient.

## Review files

- [Contact sheet](contact_sheet.jpg): teacher, earlier latent-anchor Stage2-lite, current RGB-dual step 1, current step 4; A/D at frames 0, 12, 25, 38.
- [Review video](../h3world_rgb_gain_diagnostic_AD_39.mp4): A/D rows; teacher / older latent-anchor failure / current RGB-dual step 1 columns. Diagnostic only; 39 frames, 24 fps.
- [Machine-readable report](gain_curve_report.json) includes exact equality checks against saved same-action teacher conditioning.
- [Training record](trainable_gain_fp32_full_teacher_rgb_8x4/training.json); individual videos under `eval_step01` through `eval_step04`.

## Visual review

The contact sheet shows the person surviving through frame 38 in step 1 and step 4, unlike the earlier latent-anchor clip. However, A appears to recede into the scene rather than reproduce teacher lateral motion; D retains leg blur. A tiny positive global flow does not establish correct strafe-left behavior. The review columns differ in adapters as well as anchor and are not an anchor-only ablation.

## Decision

Do not promote a new 124-frame demo or extend gain/anchor sweeps. Best separation in this curve is 0.8272, below manual initialization 0.8286 and the 1.0 gate; differences at this scale are not evidence of improvement. Short-video signs alone do not establish temporally grounded control or generalization.
If further work is pursued, first audit the full-teacher target on identical states: the online teacher sees a generated prefix/current chunk, not the complete original 39-frame denoising trajectory. Verify teacher A/D delta and attainable student gradient before choosing larger action-path adaptation or RGB-consistent multi-sigma critic/DMD. The present four-step result does not settle that choice.
