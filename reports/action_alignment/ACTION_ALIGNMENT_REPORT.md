# RGB-consistent action-pathway alignment smoke

本轮固定 RGB dual anchor、causal KV、generated history、8 steps/chunk 和 39-frame protocol，只训练 full-teacher A/D counterfactual delta alignment。hidden residual 与 tail4 action-QKV 各做 1 个 update；训练只用于判断 action pathway 是否有可学习方向。

| Variant | paired cosine | norm ratio | flow(A) | flow(D) | A−D | Gate | Person stable |
|---|---:|---:|---:|---:|---:|---|---|
| hidden | 0.139 | 1.974 | -1.2499 | -1.4293 | 0.1794 | FAIL | yes |
| qkv_tail4 | 0.390 | 1.018 | -0.8359 | -0.7685 | -0.0675 | FAIL | yes |

## Interpretation

The tail4 QKV refiner is the better alignment parameterization at the score-field level: its mean paired cosine is higher and its student/teacher delta norm ratio is closer to one. However, this does not translate to the free-running image-space action gate after one update. Both variants retain the person and garage structure through frame 38 under RGB dual anchors, so the remaining failure is action geometry rather than the old latent-anchor decomposition.

## Resource and protocol

Both runs used one shared 33B backbone, CPU raw KV, four sigmas, generated history, and 8 solver steps per chunk. The hidden and QKV runs completed without NaN/OOM. Each resulting A/D clip is 39 frames at 24 fps (1.625 s), H.264 Constrained Baseline/YUV420P.

## Decision

Do not promote either checkpoint to the 124-frame demo. The QKV direction signal justifies a longer low-learning-rate QKV alignment run only if time permits; it does not justify adding SGF/DMD yet. If more updates still improve paired cosine but not rollout flow, report a causal-topology/action-representation mismatch and stop scaling this path.

## Files

- [Hidden A/D video](../../archive/referenced_assets/H3-World/outputs/2026-10-06-08/action_align_hidden_rgb_39_8step_pair1_final/AD.mp4)
- [Tail4 QKV A/D video](../../archive/referenced_assets/H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair1_final/AD.mp4)
- [Contact sheet](../../archive/referenced_assets/H3-World/outputs/2026-10-06-08/action_alignment_contact_sheet.jpg)
- [Machine-readable report](../../archive/referenced_assets/H3-World/outputs/2026-10-06-08/ACTION_ALIGNMENT_REPORT.json)
- [Flow metrics](../../archive/referenced_assets/H3-World/outputs/2026-10-06-08/action_alignment_flow.json)
