# Prior visual reference for EXP-029

Judge, 2026-10-17 HKT, recorded before EXP-029 GPU launch. This is a reinspection of existing EXP-014 media, not new generation or a revision of EXP-014 acceptance.

## Sources inspected

All 17 generated frames in each of four existing contact sheets:

- `H3-World/outputs/EXP-014_v3_multiscene_teacher/G2/s0_43866101/FM30/AA/new_17_frames.jpg`
- `H3-World/outputs/EXP-014_v3_multiscene_teacher/G2/s0_43866101/FM30/AD/new_17_frames.jpg`
- `H3-World/outputs/EXP-014_v3_multiscene_teacher/G2/s3_b784d995/FM30/AA/new_17_frames.jpg`
- `H3-World/outputs/EXP-014_v3_multiscene_teacher/recovery_v2/s3_b784d995/AD/new_17_frames.jpg`

The s3 original G2 AD directory contains interrupted-run records; its complete AD endpoint/video is under recovery_v2. Historical provenance must preserve that distinction. Final prior review is [EXP-014 Judge review](../EXP-014_v3_multiscene_teacher/judge/FINAL_REVIEW.md).

## Observations and review focus

**s0, mountain/trees:** both branches retain a recognizable person and show pose/leg movement; AA and AD visibly diverge in their later poses and background alignment. Camera/background motion contributes to the difference. This supports using s0 as a responsive paired reference; it does not independently measure correct character-relative-to-world strafing or response latency. Inspect actor feet/torso against nearby terrain and background separately in the new ON/OFF comparison.

**s3, dark grass:** AA and AD remain visually very similar across the 17 frames, with a small dark subject and background motion. Occlusion, low contrast and the 0.71-second generated window limit character-direction assessment. This is the weak-fork reference; a weak new fork is not proof that causal control is impossible or that a longer response could not occur.

New EXP-029 results will be judged within its matched runtime and same-history comparison. Repeating the prior phenotype is useful context, but exact equality to prior outputs is not required across the explicitly changed offload policy. A general quality change, reduced motion, or pixel difference alone will not establish an action-specific mechanism.
