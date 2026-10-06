# Causal versus original H3 action geometry probe (2026-10-06)

This is a frozen, single-state comparison. The same generated A history (chunk 0), current noisy
latent (chunk 1), RGB dual anchor, prompt, audio noise and sigma `0.6` are used for both actions.
The causal branch uses the visual RGB-consistent causal adapter, persistent CPU raw KV, causal prefix
visibility and action feedback. The teacher branch uses the released bidirectional H3 path on the
same generated history; no student action-QKV adapter is installed.

| Quantity | Frozen causal | Original H3 teacher |
|---|---:|---:|
| A/D delta norm | 7.642 | 8.704 |
| Ratio causal / teacher | 0.878 | — |
| A velocity norm | 642.857 | 599.640 |
| D velocity norm | 643.615 | 598.936 |
| Causal-vs-teacher delta cosine | **−0.015** | — |

The causal A/D delta has nearly the teacher's magnitude, but its direction is effectively orthogonal
to the teacher's action delta. This rules out both of the simplest explanations: the action signal is
not completely absent, and merely increasing its gain cannot rotate it into the original H3 geometry.
It supports the conclusion that causal chunk attention plus generated-history state changes the
action-conditioned score field itself. The tail4 QKV paired loss improved an internal student/teacher
delta cosine on some checkpoints, but did not fix this free-running geometry.

This is a diagnostic on one scene/seed/state, not an action-accuracy claim. Full 39-frame flow gates
remain the acceptance criterion, and no 124-frame grid is produced from this probe.
