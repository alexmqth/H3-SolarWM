# 8-page meeting slide outline

## Slide 1 — Question and answer

**Question:** Can SolarWM-style causalization be added to H3-World without losing H3 action control?

**Answer:** causal chunk/KV execution is feasible. The later RGB-consistent anchor and visual adapter remove the most severe person/garage decomposition. The current generated-history model still fails the strict A/D direction gate, so it is a causal feasibility prototype rather than a fully action-preserving Stage2 model.

Show: `annotated/h3world_rgb_stable_action_grid_124_timed.mp4`.

## Slide 2 — Why the first meeting video looked broken

The first package selected `final_fixed_mix124_8step` as the main grid. That run had no visual tail16 adapter, used latent-only dual anchoring, used `action_prefix_mode=own`, and disabled action feedback. It was decodable but accumulated temporal-VAE ghosting. The selection was my packaging mistake; the old result is now under `diagnostics/legacy_fixed_mix/`.

The current main uses RGB decode/re-encode dual anchoring plus tail16 visual QKV adaptation. It is visibly more coherent, but residual blur/ghosting and action-direction errors remain.

## Slide 3 — What H3-World already provides

- initial frame + scene prompt;
- per-latent W/S/A/D language action rows;
- directed H3 action-to-video routing;
- original 30-step full-horizon generation.

The migration preserves the action rows and H3 condition path.

## Slide 4 — SolarWM idea mapped to H3

```text
H3 action rows + causal chunk attention
             + persistent raw KV
             + clean history commit
             + generated-history rollout
             + RGB-consistent anchor
```

Stage0.5/Stage1 make causal execution and teacher replay executable. Stage2 is separate rollout-distribution matching: student self-rollout, frozen teacher and trainable fake-score critic.

Implementation locations: `code/causal/h3_cached.py`, `benchmark.py`, `train_online_selfrollout.py`, `stage2_lite_dmd.py`, `evaluate_action_control.py`, and `code/diffsynth_h3_action.patch`.

## Slide 5 — Current 124-frame demo

Show: `annotated/h3world_rgb_stable_W_original_vs_causal_timed.mp4` or the four-action grid.

- left: original H3, 30 full-horizon steps;
- right: RGB visual-main causal, 8 steps/chunk × 8 chunks = 64 noisy forwards + 8 clean commits;
- same initial image, prompt, action, seed 13, initial noise, resolution and 124 frames;
- both are 24 fps / 5.17 s.

The video is the repaired visual-main result, not the old fixed-mix grid.

## Slide 6 — Efficiency and continuity

From `METRICS.md`:

- original e2e: 441.5–454.2 s;
- current RGB-main causal e2e: 673.4–767.9 s;
- causal first chunk: 41.8–57.4 s; mean chunk: 75.1–88.6 s;
- causal peak GPU: 30.8–39.0 GiB (31,570–39,940 MiB);
- CPU raw KV: 13.19 GiB (13,509 MiB).

These are single runs without warmup/repeat averaging. The current prototype is slower overall because it performs 64 noisy forwards versus 30 full-horizon forwards and pays RGB anchor decode/re-encode cost. KV reuse establishes the causal interface; it does not by itself guarantee wall-clock speedup.

## Slide 7 — Visual repair and Stage2-lite

Show `visual_stability/stage2_rgb_anchor_endpoint_visual_stability_comparison_39.mp4` and, if time permits, `stage2_lite/stage2_lite_rgb_endpoint_integrated_AD_39.mp4`.

- old latent-only anchor caused person fragmentation;
- RGB decode/re-encode makes training and inference anchor semantics consistent;
- 124-frame RGB-main clips keep person/garage recognizable to the end;
- Stage2-lite student/critic/teacher chain runs without NaN/OOM;
- visual repair is not action-geometry recovery.

## Slide 8 — Action result and next step

| Quantity | Original H3 | Current RGB-main causal |
|---|---:|---:|
| A horizontal flow | +1.077 | -0.784 |
| D horizontal flow | -1.602 | -1.007 |
| A-D separation | 2.679 | 0.223 |

The strict gate `flow(A)>0, flow(D)<0, A-D>1.0` fails. Do not present the visual repair as preserved four-direction control.

Final sentence:

> H3-World causalization and persistent KV are mechanically feasible. RGB-consistent anchoring with visual adaptation improves coherence at 124 frames, but the same checkpoint collapses visually in the 20-second rollout and fails the A/D action gate. Both action control and long-horizon generated-history drift remain unresolved.

## Optional slide — Action response versus visual stability

Show `action_vs_stability/original_action_stronger_visual_stable_AD_124.mp4`: Original H3 / old fixed-mix / RGB visual adapter. Old causal: A=+0.143, D=-0.311, separation=0.453 with visible drift. RGB causal: A=-0.784, D=-1.007, separation=0.223 with more coherent structure. Neither is full action preservation.

Then show the 10.125 s (243f) and 20.042 s (481f) W pairs from `long_horizon/`. These are independently sampled real long rollouts using the same RGB visual checkpoint. The 10-second tail becomes blurred/ghosted; the 20-second causal sample is a visual failure, with severe fog around 10 s and barely recognizable person/scene after 15 s. Same-length original/causal noise hashes match. KV holds five history chunks, but RGB-prefix decoding still grows with the generated prefix. Complete execution and bounded KV do not imply visual stability.
