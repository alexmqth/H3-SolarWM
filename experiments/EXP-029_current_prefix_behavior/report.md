# EXP-029 Worker report: current-prefix feedback behavior

**Status: complete eight-branch dataset; [Judge accepted the bounded evidence](judge/FINAL_REVIEW.md), with no model upgrade or further GPU task.** The original overnight process timed out after seven complete branches. The authorized `recovery01` process generated only the missing `s3/OFF_D` branch; the seven original outputs and the failed-run evidence were preserved. No training, text/image encoding, download, AnyFlow or DMD was performed.

## Question and fixed protocol

We tested whether **non-action current-prefix queries reading current video keys** changes a V3 continuation's motion or visual structure. The same generated C1, native Single I0/audio, historical A actions, global RoPE positions, C2 noisy latent and 31-point native sigma schedule / 30 sampling updates were held fixed within each scene. C1 has 12 video latents and 39 published RGB frames; C2 adds 5 latents and 17 RGB frames. All displayed 56-frame clips reuse the exact 39 published C1 RGB frames. Thus they are short continuations, **not 56 newly generated frames**.

The four C2 conditions were `ON_A`, `ON_D`, `OFF_A`, `OFF_D`. A/D replaces only current action spans 12:17. ON/OFF changes exactly one mask edge: non-action prefix queries may/may not read current C2 video keys. Current video own-action routing, action feedback, past raw KV, sigma and other visible inputs stay the same. C1 is committed once per scene with the frozen ON topology; all four branches read the same detached cache. CPU [preflight](CPU_PREFLIGHT.json) and [input audit](EXP029_INPUT_AUDIT.json) verified the mask/condition restrictions. The recovery [preflight](recovery01/CPU_PREFLIGHT.json) checked the same current prompt/noise/positions/schedule against the interrupted original row.

The model is Original H3 with released Action LoRA, no new adapter. Precision is `h3_fp32`, offload reserve 18 GiB, CPU raw KV, single GPU per process, `30 NFE/chunk`, flow shift 2.22, 832×480 at 24 fps. Saved EXP-014 generated `first12.pt` and `published_39.npy` were reused; no GT reset. The 50-layer C1 raw KV occupies **6,799,104,000 bytes = 6.332 GiB CPU**. Recovery rebuilt s3 C1 with one ON clean commit and matched its original cache SHA exactly (`2cd1f8e6d4659e74fa70618c0b0317272c18c575244bd55f1c2df14d2cfd8ca9`) before any OFF_D sampling.

## Deliverables

| Scene | Four original/recovered branch videos | ON/OFF × A/D comparison |
| --- | --- | --- |
| `s0_43866101` | [ON_A](../../report/v3/fm2anyflow/current_prefix_behavior/videos/s0_ON_A_rollout56.mp4), [ON_D](../../report/v3/fm2anyflow/current_prefix_behavior/videos/s0_ON_D_rollout56.mp4), [OFF_A](../../report/v3/fm2anyflow/current_prefix_behavior/videos/s0_OFF_A_rollout56.mp4), [OFF_D](../../report/v3/fm2anyflow/current_prefix_behavior/videos/s0_OFF_D_rollout56.mp4) | [2×2 video](../../report/v3/fm2anyflow/current_prefix_behavior/videos/s0_ON_OFF_AD_2x2_56.mp4) |
| `s3_b784d995` | [ON_A](../../report/v3/fm2anyflow/current_prefix_behavior/videos/s3_ON_A_rollout56.mp4), [ON_D](../../report/v3/fm2anyflow/current_prefix_behavior/videos/s3_ON_D_rollout56.mp4), [OFF_A](../../report/v3/fm2anyflow/current_prefix_behavior/videos/s3_OFF_A_rollout56.mp4), [OFF_D recovery](../../report/v3/fm2anyflow/current_prefix_behavior/videos/s3_OFF_D_rollout56.mp4) | [2×2 video](../../report/v3/fm2anyflow/current_prefix_behavior/videos/s3_ON_OFF_AD_2x2_56.mp4) |

The portable meeting copy is [current_prefix_behavior](../../report/v3/fm2anyflow/current_prefix_behavior/README.md). Source identity, full-path hashes, timings, boundaries and video SHA are in [all-eight manifest](recovery01/ALL_EIGHT_MANIFEST.json); the original seven are frozen by [ORIGINAL_FREEZE.json](recovery01/ORIGINAL_FREEZE.json). PyAV decoded all eight 56-frame clips, eight 17-frame clips and both 2×2 clips at 24 fps. All eight raw `published56.npy` arrays have first39 frames exactly equal to their saved C1 source. All four ON raw56 arrays also reproduce the historical EXP-014 complete AA/AD arrays **elementwise exactly**; s3 historical AD comes from the complete `recovery_v2` path, not the interrupted G2/AD record. See [replay audit](recovery01/ON_HISTORICAL_REPLAY_CPU.json).

## Results: branch differentiation and continuity

The table uses raw RGB, so H.264 re-encoding is excluded from the measurements. Boundary MAD compares frames 38→39; within-C2 MAD averages consecutive new frames 39→55. Horizontal flow is the EXP-014 Farneback central-ROI median summed over new-frame transitions. It is an **auxiliary screen/camera motion proxy**, not a character-action correctness score.

| Scene | Branch | Horizontal flow sum | Boundary MAD | Within-C2 MAD | Sampling s | Decode s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| s0 | ON_A | +42.15 | 7.450 | 10.063 | 368.41 | 9.53 |
| s0 | ON_D | −27.46 | 5.494 | 9.818 | 363.41 | 10.89 |
| s0 | OFF_A | +16.25 | 6.074 | 6.683 | 447.92 | 10.19 |
| s0 | OFF_D | −26.86 | 6.386 | 9.313 | 423.17 | 10.97 |
| s3 | ON_A | +66.18 | 4.068 | 5.495 | 493.46 | 11.54 |
| s3 | ON_D | +58.66 | 4.077 | 5.364 | 489.69 | 11.56 |
| s3 | OFF_A | +77.95 | 4.254 | 6.261 | 365.40 | 9.33 |
| s3 | OFF_D, recovery | +78.11 | 4.373 | 6.333 | 249.18 | 8.20 |

For the 17 new frames, s0's A/D cross-branch RGB MAD is **28.31 ON vs 20.93 OFF**. The ON→OFF fixed-action change is 22.92 for A and 10.58 for D. In s3, A/D MAD is **3.19 ON vs 3.16 OFF**; ON→OFF is 4.79 for A and 5.57 for D. Complete per-transition flow and pair MAD are in [ACTION_DIAGNOSTICS_CPU.json](recovery01/ACTION_DIAGNOSTICS_CPU.json).

Visual review of all 18-frame contact sheets (prior frame38 + 17 new frames) finds recognizable actors and continuous scene layouts in both scenes, with no immediate scene switch in the measured C2. In s0, ON_A produces a visibly different lateral trajectory from ON_D; OFF_A also differs from OFF_D, but its positive horizontal motion is smaller. Thus OFF does **not** erase all action differentiation. The changed prefix edge has a material behavioral effect in s0, strongest for A, and also changes general appearance/motion. In s3, all four continuations look similar, the actor stays small in grass, and central flow is positive for A and D under both topologies. This scene does **not** demonstrate reliable A/D semantic direction. Screen-space flow can include camera motion; neither its sign nor RGB MAD alone proves actor-relative strafe correctness.

These are two saved scenes and one seed. The result supports **a scene-dependent contribution from current non-action prefix feedback**, not a universal action-control fix or a new trained model. It does not establish 124-frame stability, long-horizon behavior, or inference speedup.

## Resource accounting and failure record

| Process | GPU/time (HKT) | Scheduler launch→exit | Ledger process-internal | DiT calls | VAE decode | Peak allocated/reserved | Max RSS | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Original attempt | GPU0, 03:10:16–04:10:14 | 3597.816 s | 3587.434 s | 235 = 2 commit + 233 sampling | 7 | 25.441 / 26.531 GiB | 83.536 GiB | Wall-cap stop; s3 OFF_D has 23 attempted steps, no endpoint/video |
| `recovery01` | GPU1, 11:51:01–11:56:23 | 321.700 s | 312.336 s | 31 = 1 commit + 30 sampling | 1 | 16.679 / 19.174 GiB | 76.981 GiB | Complete, exit code 0 |
| Combined actual spend | Two serial one-GPU processes | **3919.516 GPU s** | 3899.770 s | **266 = 3 commit + 263 sampling** | **8** | Separate, not added | Separate | Eight branch videos obtained |

The original 233 sampling attempts **include** the unusable 23-step OFF_D partial trajectory. Recovery could not resume it because no intermediate latent was saved; its 30 steps are additional. No retry of the seven completed branches occurred. The original [ledger](original_attempt/GPU_LEDGER.json), [scheduler status](original_attempt/SCHEDULER_STATUS.json) and [failure log](original_attempt/GPU_RUN.log) remain unchanged. The recovery [ledger](recovery01/GPU_LEDGER.json), [status](recovery01/SCHEDULER_STATUS.json) and [log](recovery01/GPU_RUN.log) are separate. Sampling times above are branch-local and affected by changing shared-GPU load; they are not a controlled ON/OFF speed benchmark.

## Reproducibility and limits

The original attempt used [run_exp029.py](run_exp029.py) through [admit.py](admit.py), authorized by [taskbook v1](taskbook_v1.md) and [memory admission addendum](ADDENDUM_02_MEMORY_ADMISSION.md). Recovery used [run_recovery.py](recovery01/run_recovery.py) through [recovery admit.py](recovery01/admit.py), authorized by [finite recovery addendum](ADDENDUM_03_FINITE_RECOVERY.md). The one-shot commands were:

```bash
CUDA_VISIBLE_DEVICES=-1 .venvs/h3world/bin/python submission/experiments/EXP-029_current_prefix_behavior/recovery01/cpu_preflight.py
.venvs/h3world/bin/python -u submission/experiments/EXP-029_current_prefix_behavior/recovery01/admit.py
CUDA_VISIBLE_DEVICES=-1 .venvs/h3world/bin/python submission/experiments/EXP-029_current_prefix_behavior/recovery01/finalize_cpu.py
```

These commands document the completed run; the admission script intentionally refuses a duplicate attempt in the existing output directory. GPU source/input SHA and original seven-output freeze were checked before recovery; the [final eight-source manifest](recovery01/ALL_EIGHT_MANIFEST.json) verifies unchanged originals afterward. The GPU loader resolved model files via a symlinked runtime path; the inspected released weight shard in the EXP-014 and recovery runtime paths resolves to the same local `models/MiniMax-H3` file. The four exact ON replays are stronger evidence of compatibility than the symlink alone. Any cross-scene generalization or actor-relative action conclusion needs new evidence and remains outside this task.
