# Stage2-lite v2: RGB-consistent multi-chunk/multi-sigma report

本轮修复了 Stage2-lite 与 visual tail16 训练之间的 anchor 协议不一致：student self-rollout、critic fake-score、teacher cache 全部使用 generated prefix 的 RGB decode → H3 image encode → dual anchor。39 帧实验覆盖 chunk 1/2 和四个 solver sigma，仍是共享 33B backbone 的 feasibility diagnostic。

## Protocol

{
  "frames_rgb": 39,
  "latent_frames": 12,
  "chunks": 3,
  "chunk_latents": 5,
  "solver_steps_per_chunk": 8,
  "history_chunks": 5,
  "sigmas": [
    0.94,
    0.79,
    0.57,
    0.24
  ],
  "target_chunks": [
    1,
    2
  ],
  "anchor_mode": "rgb",
  "history": "generated",
  "cache": "persistent raw KV on CPU",
  "seed": 13
}

## Result

| Action | Horizontal flow | Vertical flow | Frame RGB MAD | Boundary RGB MAD |
|---|---:|---:|---:|---:|
| A | -1.1369 | +0.3237 | 3.643 | 4.132 |
| D | -1.4556 | +0.2595 | 3.721 | 4.443 |

A−D horizontal-flow separation = **0.3187**. The strict short-clip gate requires A>0, D<0, and A−D>1.0; this checkpoint **does not pass** that action gate. Both clips retain a visible person and stable parking-garage structure through frame 38, so the RGB anchor fixes the main late decomposition seen in the previous latent-only Stage2-lite videos. The remaining failure is action geometry: the common scene/forward motion dominates the horizontal proxy, and A/D are not separated like the original H3 teacher.

## Resource use

Training completed in 872.9 s (14.5 min), with 39.37 GiB allocated peak. The largest persistent student/teacher/critic raw-KV cache was about 5.28 GiB on CPU. The 39-frame A/D evaluation used 24 noisy denoiser forwards plus 3 clean commits per action; the RGB anchor adds a VAE conversion at chunk boundaries.

## Files

- [A/D compatible side-by-side](AD_rgb.mp4)
- [A/D contact sheet](contact_sheet_rgb.jpg)
- [A flow](eval_rgb/A/cached.mp4) and [D flow](eval_rgb/D/cached.mp4)
- [Flow JSON](eval_rgb/AD_flow.json)
- [Training JSON](stage2_lite.json)

## Decision

RGB-consistent anchors are now implemented and validated. They improve visual stability but do not recover H3 action geometry by themselves; this is evidence that the remaining issue is in the causal action pathway/score-field alignment, not only in temporal conditioning. Do not extend this checkpoint to 124 frames as the final action grid. The next controlled experiment should adapt the action pathway with the RGB protocol fixed, using the full-teacher counterfactual delta as an alignment target; further gain scaling or another anchor sweep is not justified.
