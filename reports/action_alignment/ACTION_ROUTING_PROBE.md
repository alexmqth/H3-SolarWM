# Causal action-routing probe (2026-10-06)

This probe does not train a model. It loads the same RGB-consistent visual causal adapter and
runs one fixed chunk on the same generated A history and current noisy latent for four routing
policies. A and D use the same history, anchor, prompt, noise, sigma (`0.6`) and current latent;
only the action prefix visibility and the explicit action-row feedback edge change.

| Variant | Prefix policy | Feedback | A/D score-delta norm | Relative to causal/no-feedback |
|---|---|---:|---:|---:|
| `own_fb0` | current action row only | off | 5.502 | 0.748× |
| `causal_fb0` | past + current action rows | off | 7.352 | 1.000× |
| `causal_fb1` | past + current action rows | on | 7.642 | 1.039× |
| `all_fb1` | all action rows | on | 8.338 | 1.134× |

The probe uses the generated A rollout from
`eval_update01/A`, a clean A history commit for chunk 0, and a chunk 1 A/D counterfactual. The
exact values are in [`action_routing_probe.json`](../../archive/referenced_assets/H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair4_final/action_routing_probe.json).

`causal_fb1` is measurably different from `causal_fb0`, so the explicit action-row-to-current-video
feedback edge is active. Exposing all known action rows increases the delta further. This rules out
a simple “the causal mask completely disconnected the action rows” explanation. The effect is still
small relative to the absolute velocity norm (about 643 at this state), which is consistent with the
weak teacher action delta measured in the earlier audit. The remaining failure is therefore more likely
an action-conditioned score-field representation/distribution problem than a missing edge alone:
feedback exists, but it does not produce the correct image-space A/D geometry after generated-history
rollout.

This is a topology diagnostic, not an action-accuracy metric. It does not change the frozen inference
protocol and does not justify a new 124-frame grid.
