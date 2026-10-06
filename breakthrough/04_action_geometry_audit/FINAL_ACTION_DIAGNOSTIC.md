# Final action-geometry diagnostic (2026-10-06)

## Fixed acceptance protocol

All new action experiments use the same 39-RGB-frame / 12-latent-frame / 3-chunk protocol, 5 latent
frames per chunk, history window 5, 8 solver steps per chunk, flow shift 2.22, RGB dual anchor,
generated history, persistent CPU raw KV, causal action prefix, action feedback, and seed 13. The gate is
`flow(A)>0`, `flow(D)<0`, `A−D>1.0`, with a visible person and stable garage structure through frame 38.

## Rollout results

| Variant | Training change | flow(A) | flow(D) | A−D | Gate |
|---|---|---:|---:|---:|---|
| RGB Stage2-lite v2 | critic/DMD, one update | −1.1369 | −1.4556 | +0.3187 | FAIL |
| Tail4 action-QKV, update 1 | paired full-teacher delta | −0.8563 | −0.7790 | −0.0774 | FAIL |
| Tail4 action-QKV, update 2 | paired full-teacher delta | −0.8886 | −0.7536 | −0.1350 | FAIL |
| Tail4 action-QKV, update 3 | paired full-teacher delta | −0.8703 | −0.7785 | −0.0918 | FAIL |
| Tail4 action-QKV, update 4 | paired full-teacher delta | −0.8506 | −0.7684 | −0.0822 | FAIL |
| Tail8 action-prefix, update 1 | prefix hidden residual | −1.1091 | −1.4603 | +0.3513 | FAIL |
| Released H3 action LoRA tail8, update 1 | QKV/out LoRA | −1.1249 | −1.4151 | +0.2902 | FAIL |

The RGB-consistent anchor removes the earlier person decomposition: all listed RGB runs keep the person
and garage visible through frame 38. The remaining failure is action direction, not video decodability
or the anchor protocol.

## Geometry and routing evidence

On one shared generated state, frozen causal versus original bidirectional H3 produced A/D delta norms
7.642 and 8.704 (ratio 0.878), but delta cosine −0.015. A routing probe showed that action feedback is
not disconnected: enabling the causal-safe feedback edge changes A/D delta norm from 7.352 to 7.642,
and exposing all known action rows gives 8.338. The edge is active, but it does not rotate the causal
score field into the original image-space action direction.

## Decision

The prototype has demonstrated causal chunk attention, persistent raw KV caching, RGB-consistent
history anchoring, and stable 39/124-frame causal execution. It has not demonstrated preservation of
H3's A/D action geometry under generated-history rollout. More single-state optimizer updates, gain
sweeps, anchor changes, or a new 124-frame grid would not be an attributable next step. A successful
follow-up requires multi-state/multi-seed action supervision or full SolarWM Stage2-style rollout
training with distribution matching; that is outside this minimal prototype and current acceptance gate.
