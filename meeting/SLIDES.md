# 8-page meeting slide outline

## Slide 1 — Question and answer

**Question:** Can SolarWM-style causalization be added to H3-World without losing H3 action control?

**Answer:** causal chunk rollout and persistent KV are feasible; RGB anchor recovers long-horizon visual stability; generated-history A/D action geometry is not yet fully preserved.

Show: `annotated/h3world_final_action_grid_124_timed.mp4`.

## Slide 2 — What H3-World already provides

- initial frame + scene prompt;
- per-latent W/S/A/D language action rows;
- directed action-to-video attention routing;
- original 30-step full-horizon generation.

Key point: the migration must preserve the action rows and routing, not replace H3 with an unconditional video model.

## Slide 3 — SolarWM idea mapped to H3

```text
H3 action rows + causal chunk attention
             + persistent raw KV
             + clean history commit
             + generated-history rollout
```

Stage0.5/Stage1 make the causal interface executable. Stage2 is the separate generated-distribution matching problem for a robust few-step student.

## Slide 4 — Implementation locations

| Component | File |
|---|---|
| raw KV and clean commit | `code/causal/h3_cached.py` |
| original/cached benchmark | `code/causal/benchmark.py` |
| online teacher replay | `code/causal/train_online_selfrollout.py` |
| Stage2-lite critic/DMD | `code/causal/stage2_lite_dmd.py` |
| action flow metric | `code/causal/evaluate_action_control.py` |
| H3 directed attention | `code/diffsynth_h3_action.patch` |

## Slide 5 — Main 124-frame demo

Left: original H3, 30 full-horizon steps.

Right: causal, 8 steps/chunk x 8 chunks = 64 noisy forwards + 8 clean commits.

Both sides: same initial image, prompt, action, seed 13, initial noise, resolution and 124 frames.

Show: `annotated/h3world_final_W_original_vs_causal_timed.mp4` or the timed four-action grid.

## Slide 6 — Efficiency and continuity

Use the first table in `METRICS.md`:

- original e2e: approximately 441.5–454.2 s;
- causal e2e: approximately 383.4–451.9 s in the formal fixed-mix runs;
- causal peak GPU: approximately 39.9 GiB;
- causal CPU raw KV: 13.51 GiB for the 5-chunk history window;
- first causal chunk: approximately 36.7–41.5 s;
- current timing is one recorded run, not a warmup mean.

Explain that KV reuse gives the causal interface and incremental chunk execution; the present prototype is not yet a total wall-clock speedup because it uses more denoiser forwards than the 30-step baseline.

## Slide 7 — Visual stability and Stage2-lite

Show the RGB anchor old-versus-new comparison and the integrated Stage2-lite A/D clip.

- latent-only anchor caused person fragmentation;
- RGB decode/re-encode dual anchor fixes the protocol mismatch;
- 39/124-frame person and garage structure remain coherent;
- Stage2-lite student/critic/teacher chain runs without NaN/OOM;
- neither visual repair nor one-update DMD restores A/D geometry.

## Slide 8 — Action result and next step

| Quantity | Original H3 | Formal causal fixed-mix |
|---|---:|---:|
| A-D horizontal-flow separation | 2.679 | 0.453 |
| A sign | +1.077 | +0.143 |
| D sign | -1.602 | -0.311 |

The causal branch is action-dependent but weaker and less continuous at chunk boundaries. The stricter RGB generated-history gate `flow(A)>0, flow(D)<0, A-D>1.0` remains failed.

Final sentence:

> Stage1 causalization is feasible and visual stability is recoverable. The remaining problem is generated-history action-conditioned score geometry; a full Stage2-style multi-state rollout-distribution objective or a stronger causal action pathway is the next step.
