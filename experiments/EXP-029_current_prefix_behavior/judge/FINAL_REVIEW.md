# EXP-029 final Judge review

2026-10-17 HKT. **Accept the bounded intervention and its evidence; no model upgrade and no further GPU authorization.** Eight continuations are now complete. This is a study of an existing lawful baseline edge, not a newly validated method.

## Execution, recovery and independent audit

The overnight run completed seven branches and stopped on its one-hour wall cap during s3 OFF_D. It was not an OOM or a failed quality result for the missing branch. The interrupted23 sampling calls count toward cost. Its raw failed status remains frozen under `original_attempt/`.

The user subsequently authorized trying GPU1/6. The dedicated recovery used GPU1 for one ON clean-history commit and one30NFE OFF_D continuation, plus one decode. It did not regenerate the other seven branches. Reconstructed s3 KV digest exactly matched the original; prompt/noise/position/sigma checks passed. Judge independently matched27 recovery source/input hashes, verified all42 frozen files for the seven original branches unchanged, checked raw39-frame history preservation, endpoint SHA and unchanged cache digest. All registered recovery processes have exited; GPU1 no longer lists the recovery process.

Judge also independently verified all21 published source/copy pairs (10videos,8contact sheets and3JSON records) and that all four ON raw56-frame arrays exactly reproduce their complete EXP-014 reference arrays, including s3 AD from recovery_v2. Each C1 raw KV occupies6,799,104,000bytes (6.332GiB) in CPU memory.

Judge inspected all17 generated frames plus the prior boundary frame for each of the eight outputs. This verifies the limited visual observations below; it is not a quantitative character/camera tracking benchmark. Worker media/provenance delivery is linked from the [showcase](../../../report/v3/fm2anyflow/current_prefix_behavior/README.md).

## Behavior and scientific decision

- **s0 mountain/trees:** ON and OFF both preserve a recognizable person and visible A/D branching. Disabling non-action prefix feedback changes poses, gait and background trajectory; it does not abolish the action-conditioned fork. Actor/world direction and camera-follow contributions are not fully separated. Ordinary softness/pose artifacts remain within a feasibility result.
- **s3 dark grass:** ON_A/ON_D remain close. The recovered OFF_D is also close to OFF_A. Removing this feedback edge does not visibly solve the weak-fork scene. Low subject contrast and only17 new frames (0.71seconds) limit directional and latency interpretation.
- There is an output effect of this edge, but no evidence here that it is the dominant reason for V3's weak action switching. Do not label visual change as control accuracy, or claim that this ablation explains the multifactor V1-to-V3 change.

Auxiliary raw RGB MAD, not action accuracy: s0 A/D is28.307 with ON versus20.925 with OFF; s3 A/D is3.188 with ON versus3.156 with OFF. Same-action ON/OFF differences also exist (s0 A22.917/D10.581, s3 A4.792/D5.568). Background/camera changes contribute heavily, so these numbers do not establish a percentage improvement in control.

**ROI decision:** close the current-prefix ablation and the earlier limited role-gradient survey. Do not add scenes/seeds/layers merely to strengthen a preferred hypothesis. Neither these videos nor EXP-028 justify dualAdapter training. Any next GPU task should distinguish a new behavioral hypothesis (for example delayed action response versus persistent historical motion) with a specific matched comparison; a new task/budget is required. The present experiment is not sufficient evidence for that historical-motion hypothesis either.

## Cost and scope

| Item | Original attempt | Missing-branch recovery |
|---|---:|---:|
| Complete process wall seconds | 3597.815735 | 321.699993 |
| In-process ledger seconds | 3587.433901 | 312.335611 |
| DiT attempts |235 (2commit+233sampling)|31 (1commit+30sampling)|
| Completed decodes |7|1|
| Peak allocated / reserved GiB |25.441 /26.531|16.679 /19.174|
| Process maximum RSS GiB |83.536|76.981|

Combined **3919.515728 process seconds / 65.325 minutes / 1.088754 single-GPU hours**,266 DiT attempts,8decodes,0training updates. Each completed continuation uses30sampling NFE, with commit calls reported separately. Recovery sampling249.183seconds and decode8.205seconds. CPU raw KV for each C1 is reported separately in Worker metrics. Shared-GPU utilization and adaptive CPU weight residence differ across runs; these times are not a controlled ON/OFF speedup comparison or proof of a fixed minimum VRAM requirement.

Each output contains39 frozen generated-history frames plus17 new frames at832×480/24fps. This is not eight new full5-second generations, does not establish long-horizon stability, and does not generalize beyond the two selected states/one saved noise fixture. Official V3 remains frozen.

## Overnight reporting limitation

The original experiment stopped before09:00 and its deadline guard reached cutoff. Judge's final audit/report and publish were delayed until the user's daytime follow-up; monitoring events then arrived late. This delay is recorded in the [night report](../../../research/overnight_20261017/FINAL_REPORT.md). Do not claim that the09:00 final handoff was timely. The daytime recovery is separately authorized and separately accounted.
