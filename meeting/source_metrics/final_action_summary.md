# Final fixed-mix causal W/S/A/D action grid (124 frames)

Checkpoint: `H3-World/outputs/2026-10-02-scheduled-sampling/fixed_mix_0.5/action_adapter.pt`

All four actions share seed 13, first frame, prompt, initial noise, 124 frames (5.17 s), 5-latent-frame chunks, 8 steps/chunk, generated history, dynamic latent dual anchor, and CPU raw-KV caching. The causal branch therefore performs 64 noisy denoiser forwards plus 8 clean commits per video.

| Action | Original H3 time (s) | Causal time (s) | Causal GPU peak (MiB) | Causal CPU KV (MiB) | Original MAD | Causal MAD | Original boundary | Causal boundary | Original h-flow | Causal h-flow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W | 441.5 | 450.2 | 39927 | 13509 | 4.32 | 4.60 | 4.97 | 6.29 | -1.111 | -0.624 |
| S | 444.8 | 451.9 | 39927 | 13509 | 4.13 | 4.62 | 4.70 | 7.76 | -1.084 | -0.096 |
| A | 454.2 | 438.8 | 39948 | 13509 | 4.52 | 7.10 | 5.14 | 13.68 | 1.077 | 0.143 |
| D | 450.4 | 383.4 | 39948 | 13509 | 4.46 | 5.39 | 5.39 | 9.47 | -1.602 | -0.311 |

## Action response

The original H3 A-D horizontal-flow separation is `2.679`; the causal fixed-mix checkpoint retains the correct sign but only `0.453`. W/S horizontal flow is not interpreted as a strict forward/backward score for this scene. The grid shows action-conditioned differences, but causal generated-history drift is visible, especially in A/D after later chunks; this is evidence of partial action preservation, not four-direction fidelity. Pairwise causal RGB differences are W-vs-S `29.92` mean and A-vs-D `8.75` mean (see `action_pair_differences.json`); these confirm the interventions are not identical clips, but do not establish direction accuracy.

## Files

- Review grid: `H3-World/outputs/h3world_final_fixed_mix_action_grid_124.mp4`
- Per-action side-by-side videos: `H3-World/outputs/h3world_final_{W,S,A,D}_original_vs_causal.mp4`
- Raw per-action benchmark directories: this directory’s `W/`, `S/`, `A/`, and `D/` subdirectories
- Metrics: `action_flow.json`, `original_action_flow.json`, `compact_continuity_metrics.json`, `action_pair_differences.json`, and `final_action_summary.json`
