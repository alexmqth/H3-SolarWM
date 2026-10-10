# EXP-010 — Full FM8 continuation with SW-G eviction

**Current status (2026-10-11 HKT): CPU preparation and the separately authorized shared core to AA124 are complete; C7/C8 have not run.** The [Judge taskbook](taskbook_v1.md) is the active protocol. [Worker core report](worker_report_core.md) provides the measured result and [124-frame video](artifacts/core/shared/rollout_124.mp4). This is a Mainline experiment, not a new training run. It tests whether the previously usable 8-step ordinary H3 rollout can extend from the published EXP-006 AA73 result to AA124 and then A-continue/D-switch 158 frames while retaining the accepted Global RoPE, five-ancestor raw-KV window.

The input is the **actual EXP-006 FM8 AA73**: first12, C2 and C3 latent endpoints, AA cache through C2 and published 73 RGB frames. The first new forward clean-commits the existing C3; C4–C6 then use 8 native FM steps and clean commits. C6 is the first real eviction of C1, retaining exact ancestor indices `[1,2,3,4,5]`. After Judge reviews all C4–C6 frames at 124 RGB, C7 branches from the shared AA124 state into continued A and switched D, and C8 reads each branch's own C7 history. The long47 fixture was already certified by EXP-005; its 0:37 conditions and noise preserve the old input exactly, while 37:47 uses the saved extension seed. This is an explicit long-fixture extension, not the default H3 47-latent packed builder.

| Phase | Stages | New forwards | VAE | GPU release |
| --- | --- | ---: | ---: | --- |
| Shared core | `commit_c3`, `C4`, `C5`, `C6` | 28 = 24 sampling + 4 commits | 3 | `GPU_AUTHORIZATION_CORE.json` after Judge code/CPU review |
| Branches | `C7 A`, `C7 D`, `C8 A`, `C8 D` | 34 = 32 sampling + 2 commits | 4 | `GPU_AUTHORIZATION_BRANCH.json` **and** hash-bound `CORE_JUDGE_REVIEW.json` |

The [runner](run_fm8_sw.py) locks one shared budget across all stages, reserves each forward/decode before the call, rejects prior stage results, checks the absolute 09:00 HKT cutoff and 44 GiB allocated cap, and saves the source cache/endpoint/RGB hashes. It uses the frozen `interval_stage2` in Global mode and `validate_cache` for exact five-ancestor indices on all 50 layers. It does not install AnyFlow or DMD components. Core outputs, including large tensor/cache files, are under `H3-World/outputs/EXP-010_v3_fm8_sw158/`; no source EXP-006/005 result was overwritten.

The [CPU preflight](cpu_preflight.json) checked **36 frozen source files**, the FM8 AA73 cache/endpoint/RGB chain, the native 8-step sigma grid, old packed layout and prompt equivalence at 12/17/22/27/32/37, A/D branch conditions at 42/47, saved extension noise, and expected eviction ancestry. It made **0 GPU calls**. The frozen [manifest](source_manifest.json) binds the current runner, config, input files and source modules; modifying any of those invalidates GPU release. The unmarked GPU entry was also tested and correctly rejected with `Judge core GPU marker absent` before loading the model. Judge later issued the [core release](GPU_RELEASE_CORE.md), and the core used 28 forwards/3 VAE/0.157 GPU-hours; C6 commit kept exactly C2–C6 in all 50 layers.

From project root, repeat the CPU check with:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python submission/experiments/EXP-010_v3_fm8_sw158/run_fm8_sw.py --preflight
```

After the appropriate Judge marker, each approved GPU call has the form below. Run one stage at a time, keep stdout, inspect each new contact sheet, and stop for Judge review after C6. No stage is automatically retried.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python -u submission/experiments/EXP-010_v3_fm8_sw158/run_fm8_sw.py --stage commit_c3 --gpu 0
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python -u submission/experiments/EXP-010_v3_fm8_sw158/run_fm8_sw.py --stage C4 --gpu 0
# Then C5, C6. A separately released C7/C8 call adds --path A or --path D.
```

Total additional budget: **62 full forwards, 7 VAE decodes, 0 updates, at most 0.75 GPU-hours**, all before 09:00 HKT. GPU3/4 were occupied by others during CPU preparation. More available GPUs are unnecessary for this sequential cache ancestry test; never use them without a revised marker/combined budget. The 30-step SW-G 158-frame videos already exist and will be comparison references with different generated histories. No whole-video speedup or single-variable claim is implied.

After actual C7/C8 videos exist, the CPU-only [comparison script](make_comparisons.py) can place the existing 30-step SW-G and the new FM8+SW-G A/D 158-frame results side by side. It refuses missing files or overwrites, fully decodes both sources and the H.264 output, and labels their different generated histories. No EXP-010 158-frame comparison exists at the current core stop point.
