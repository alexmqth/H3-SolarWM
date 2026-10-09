# Stage2-lite 39-frame result

This is a minimal SolarWM-style diagnostic on H3-World. It uses one shared
frozen H3 backbone and swaps a student causal action residual, a trainable
critic residual, and the frozen teacher state. Each update runs a detached
student self-rollout, fits the critic on one generated final chunk at
`sigma=0.6`, and applies one DMD surrogate update to the student.

The run used 39 RGB frames / 12 latent frames / 3 chunks, 5 latent frames per
chunk, 8 solver steps per chunk, causal action rows, action feedback, dynamic
dual latent anchor, CPU raw-KV, seed 13, and generated history. Four A/D
rounds were run with `critic tail4`, critic LR `1e-4`, and student LR `1e-4`.

| Method | A flow | D flow | A-D | Status |
|---|---:|---:|---:|---|
| Fixed-mix Stage1 baseline | -0.01325 | -0.76891 | 0.75566 | baseline |
| Stage2-lite, 1 A/D round | -0.01380 | -0.77279 | 0.75900 | no gate |
| Stage2-lite, 4 A/D rounds | +0.00477 | -0.76839 | 0.77316 | sign gate only |

The four-round run therefore restores the A sign by a small margin and keeps D
negative, but it does not reach the short-clip separation gate `A-D > 1.0`
or the original H3 teacher separation (about 2.02). It is a valid feasibility
result for the training chain, not a successful Stage2 checkpoint.

The critic and DMD chain was numerically valid on the shared 33B backbone:
peak allocated GPU memory was about 39.4 GiB, each raw-KV cache was about 5.28
GiB for the 3-chunk target, and no second or third 33B model was loaded. The
critic loss remained around 0.09--0.11 and all DMD gradients were finite.

The final side-by-side video is
`H3-World/outputs/stage2_lite_39_8step_4updates_AD.mp4`. Raw metrics are in
`eval/AD_flow.json`, training diagnostics in `stage2_lite.json`, and the two
individual videos are under `eval/A/cached.mp4` and `eval/D/cached.mp4`.

The contact sheet confirms that the generated-history visual failure remains:
the parking-garage structure is mostly retained, but the character becomes
transparent/ghosted around frames 15--30 and is largely absent near frame 38.
The DMD prototype therefore changes the action proxy slightly without solving
the long-horizon visual drift.

For playback compatibility, the final `eval/A/cached.mp4`, `eval/D/cached.mp4`,
and root side-by-side video use H.264 Constrained Baseline with YUV420P. The
original D High-Profile file is retained as `eval/D/cached_highprofile.mp4`.
