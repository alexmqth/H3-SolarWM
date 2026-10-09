# Same-generated-state current-chunk action geometry: measured results

No training or video acceptance. All 12 local cases completed at AnyFlow128.
These are explicitly noised generated endpoint interpolants, not saved solver intermediates.
The main comparison is instantaneous student `r=t` versus Original H3 instantaneous velocity.

| History | Chunk | sigma | Cosine, matched teacher | Cosine, native teacher | Student/teacher norm, matched | Student delta RMS | Teacher delta RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 1 | 0.939540 | 0.032473 | 0.057322 | 2.242882 | 0.051898 | 0.023139 |
| A | 1 | 0.689441 | 0.003561 | 0.027674 | 1.054124 | 0.020783 | 0.019715 |
| A | 1 | 0.240781 | 0.019047 | 0.013619 | 1.208545 | 0.018497 | 0.015305 |
| A | 2 | 0.939540 | 0.083599 | -0.350064 | 1.555469 | 0.024214 | 0.015567 |
| A | 2 | 0.689441 | 0.118810 | 0.072599 | 0.793495 | 0.016560 | 0.020870 |
| A | 2 | 0.240781 | -0.005467 | -0.000289 | 1.026491 | 0.017115 | 0.016673 |
| D | 1 | 0.939540 | 0.205050 | 0.044158 | 1.190873 | 0.046001 | 0.038628 |
| D | 1 | 0.689441 | 0.082955 | 0.124187 | 1.195461 | 0.025576 | 0.021395 |
| D | 1 | 0.240781 | 0.031851 | 0.039622 | 1.152590 | 0.021523 | 0.018674 |
| D | 2 | 0.939540 | -0.102335 | 0.277378 | 3.556306 | 0.042442 | 0.011934 |
| D | 2 | 0.689441 | -0.058812 | 0.033135 | 0.916378 | 0.016948 | 0.018495 |
| D | 2 | 0.240781 | 0.047904 | 0.016771 | 1.017520 | 0.016998 | 0.016705 |

## Controls and provenance

- Current spans only: latent rows 5–9 / 10–11. Head/past/future rows remain fixed.
- Actual SDPA masks checked at layers 0/25/49: current/past action visibility, own-frame feedback, historical video reads.
- Full KV hashes and commit counts unchanged across the counterfactual branches; parameter versions unchanged.
- Released H3 action LoRA retained; our visual/Stage1 bank/AnyFlow/residual adaptations disabled only for teacher forwards.
- Teacher uses the same generated prefix/noisy chunk, without future video. Matched profile uses identical dual anchors/global positions/fixed prefix times; native profile uses single initial anchor/native times.
- Architectural differences remain: recomputed bidirectional teacher history versus frozen student KV; teacher own-action direct binding versus student past/current action visibility.
- A/D spans have identical saved lengths. The future-action control tests embedding-content leakage at FIXED layout, not arbitrary changes in future sentence lengths.
- Latent rows 5–9 map to RGB [17,34); 10–11 to RGB [34,39). Global current-video positions and the tail-anchor position are preserved (see temporal_layout.json). This does not imply strict frame causality inside a chunk or hard RGB boundaries through the temporal VAE.

```json
{
  "student_repeat_rmse": [
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "teacher_repeat_rmse": [
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "future_only_rmse": [
    0.0,
    0.0
  ]
}
```

## Middle-sigma pathway intervention

Each entry isolates one input while holding the other fixed. Cosine refers to the matched teacher delta.

| History | Chunk | Isolated path | Effect RMS | Effect/teacher norm | Cosine |
|---|---:|---|---:|---:|---:|
| A | 1 | text_effect_residual_A | 0.014686 | 0.744895 | -0.005038 |
| A | 1 | text_effect_residual_D | 0.014821 | 0.751723 | -0.000933 |
| A | 1 | residual_effect_text_A | 0.014456 | 0.733247 | 0.006076 |
| A | 1 | residual_effect_text_D | 0.014336 | 0.727130 | 0.010323 |
| A | 2 | text_effect_residual_A | 0.008468 | 0.405761 | -0.001354 |
| A | 2 | text_effect_residual_D | 0.008823 | 0.422787 | -0.004885 |
| A | 2 | residual_effect_text_A | 0.014859 | 0.711973 | 0.135316 |
| A | 2 | residual_effect_text_D | 0.014577 | 0.698464 | 0.135762 |
| D | 1 | text_effect_residual_A | 0.020531 | 0.959646 | 0.078901 |
| D | 1 | text_effect_residual_D | 0.020667 | 0.965973 | 0.080385 |
| D | 1 | residual_effect_text_A | 0.015722 | 0.734865 | 0.029283 |
| D | 1 | residual_effect_text_D | 0.015826 | 0.739711 | 0.031704 |
| D | 2 | text_effect_residual_A | 0.009696 | 0.524266 | -0.099920 |
| D | 2 | text_effect_residual_D | 0.009709 | 0.524975 | -0.096006 |
| D | 2 | residual_effect_text_A | 0.015065 | 0.814583 | -0.004288 |
| D | 2 | residual_effect_text_D | 0.015108 | 0.816882 | -0.001848 |

## Execution and limits

```json
{
  "A": {
    "noisy": 57,
    "clean": 2,
    "peak_GPU_MiB": 28763.3896484375,
    "peak_CPU_KV_MiB": 5403.4423828125,
    "wall_seconds": 473.86556740803644
  },
  "D": {
    "noisy": 57,
    "clean": 2,
    "peak_GPU_MiB": 27626.27880859375,
    "peak_CPU_KV_MiB": 5403.4423828125,
    "wall_seconds": 421.9231374193914
  }
}
```

One GPU at a time, CPU raw KV and CPU weight offload. These diagnostic timings are not an inference benchmark.
Finite interval deltas and per-latent-frame metrics are preserved in the receipts. Finite delta versus instantaneous teacher is not a like-for-like velocity comparison.
One seed and scene. A nonzero local action response does not prove correct image-space strafe direction, video stability or that Stage2 will fix the issue. No new demo was generated or promoted.
