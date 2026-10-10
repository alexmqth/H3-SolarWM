# EXP-009/v1 Worker CPU preparation

2026-10-11 04:40 HKT. EXP-009 has **not** started GPU inference. The current [taskbook](taskbook_v1.md) authorizes CPU preparation; `GPU_AUTHORIZATION.json` and `C3_GPU_AUTHORIZATION.json` are absent. No training, AnyFlow extension or DMD run was started.

Implemented [run_fm4_af4.py](run_fm4_af4.py) and [config.json](config.json) for matched ordinary FM4 and frozen AF2 step32 AF4 continuation on the common EXP-006 FM8 C1. FM4 calls the accepted `interval_cached` entry and native `scheduler.step`, with no AF modules installed. AF4 loads the paired step32 QKV/target-time weights and uses `interval_student` plus next-sigma `finite_map_step`. Each candidate has its own C1 clean-commit cache and separate C2/C3 branch cache; clean history, Single I0, current-prefix feedback, global positions and generated-history C3 are preserved. The runner gates every GPU entry by frozen manifest, Judge marker, time/forward/VAE budget and actual card free memory; C3 has an additional marker gate.

Executed from project root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python submission/experiments/EXP-009_v3_fm4_af4/run_fm4_af4.py --preflight
```

Result: [cpu_preflight.json](cpu_preflight.json) records `CPU_PASS_NO_GPU_AUTHORIZATION`, 28 frozen source files and 0 GPU calls. Native 4-step sigmas are `1, 0.8694517211914062, 0.6894410400390625, 0.4252873229980469, 0`. The native ordinary scheduler and finite-map scalar update differ at most `1.1920928955078125e-07` from FP32 operation ordering in the CPU probe. AA/AD first12 visible prompt tokens match; at stop17/22 they differ and future rows are physically trimmed. Shared initial/video/audio/anchor inputs and published FM8 first39 shape are verified. The paired step32 state metadata and SHA match the AF2 receipt.

The [source manifest](source_manifest.json) was written only after all assertions passed. Runner/config edits after this point invalidate the manifest and require an explicit re-freeze and Judge re-review before any GPU work. [make_comparisons.py](make_comparisons.py) is prepared for actual 56/73-frame MP4 only; no EXP-009 MP4 currently exists. GPU3/4 are occupied by others; idle GPU0/1/2/5/6/7 were visible at approximately 04:38 HKT, but an idle card is not a release marker. User permits up to eight cards overnight and at most three after 09:00; this single-card sequential protocol fits either limit.

Next action is Judge source review and a manifest-bound GPU marker for prefill/C2. After C2, hand all new frames and raw metrics to Judge before any C3 marker. Existing AF8/FM8 results remain unchanged.
