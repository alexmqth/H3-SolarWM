# Full-teacher counterfactual delta audit: A and D generated states

| generated state | states | min ratio | max ratio | mean ratio |
|---|---:|---:|---:|---:|
| A rollout | 15 | 0.0128 | 0.0336 | 0.0207 |
| D rollout | 15 | 0.0150 | 0.0424 | 0.0206 |

The frozen bidirectional H3 teacher produces finite, nonzero A/D counterfactual deltas on both generated states, but the deltas remain small relative to the full velocity field. This supports the interpretation that the paired target is weak, while leaving its image-space direction and student alignment unresolved.

Raw reports: [A state](REPORT.md) and [D state](../../2026-10-06-11/teacher_delta_audit_Dstate/REPORT.md).
