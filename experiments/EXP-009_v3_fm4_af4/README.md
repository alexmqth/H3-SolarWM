# EXP-009 — Ordinary FM4 vs trained AnyFlow AF4

Status (2026-10-11 05:00 HKT): **the approved FM4/AF4 73-frame experiment is complete; no further EXP-009 GPU work is planned.** Judge issued separate [C2](GPU_RELEASE.md) and [C3](C3_GPU_AUTHORIZATION.json) releases. [Final Worker report](worker_report_final.md) separates engineering completion from video quality. FM4 retains a limited AA/AD direction response with partial visual quality. AF4 has no combined benefit; its AD third block loses direction and the person fragments into translucent pieces.

The experiment asks whether the frozen AF2 step32 target-time/QKV student at 4 NFE better preserves AA/AD motion and structure than ordinary H3 FM4. Both begin with the exact 39 RGB frames and clean 12-latent endpoint from EXP-006 FM8, but each builds its **own** C1 hidden KV. C2 shares the same clean history and initial noise; C3 uses each candidate's own generated C2. This is a comparison of **continuation** at 4 NFE, not a full-video FM4/AF4 generation result or a single-variable training-objective ablation. Existing FM8/AF8 videos are 8-NFE references.

| Candidate | Weights | Per-chunk sampling | Update |
| --- | --- | ---: | --- |
| FM4 | Original H3 + released action LoRA | 4 | Native H3 `scheduler.step` via `interval_cached` |
| AF4 | Same base + frozen AF2 step32 rank-8 QKV and target-time conditioner | 4 | Next-sigma finite map via `interval_student` |

The native shift-2.22 4-step grid is `[1, 0.8694517211914062, 0.6894410400390625, 0.4252873229980469, 0]`. It is the even-index subsequence of the published FM8 grid. The [CPU result](cpu_preflight.json) verifies paired checkpoint SHA/metadata, input shapes and shared noise, AA/AD action-span differences only after the common C1, physical future-row removal at stops 12/17/22, and the FM/finite scalar update to one FP32 ULP. The [source manifest](source_manifest.json) freezes 28 input/code/checkpoint files. CPU preflight made **0 GPU calls**.

From `/home/qma/work/GWM`, repeat CPU validation with:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python submission/experiments/EXP-009_v3_fm4_af4/run_fm4_af4.py --preflight
```

The approved C1/C2 stages used the following command form from project root. They ran sequentially on GPU0; stdout was preserved for every call, with no retry:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python -u submission/experiments/EXP-009_v3_fm4_af4/run_fm4_af4.py --model fm4 --stage prefill --gpu 0
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python -u submission/experiments/EXP-009_v3_fm4_af4/run_fm4_af4.py --model fm4 --stage second --path AA --gpu 0
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python -u submission/experiments/EXP-009_v3_fm4_af4/run_fm4_af4.py --model fm4 --stage second --path AD --gpu 0
```

The three calls were repeated with `--model af4`. Judge reviewed all four C2 segments before issuing the separate C3 marker; the same runner then used `--stage third --path AA|AD` to extend each branch to 73 RGB frames. The total cap was 38 forwards, 8 VAE decodes and 0.60 GPU-hours, including failed attempts; execution finished before the 09:00 HKT deadline. GPU3/4 carried other users' jobs and were not selected. The single-card sequence preserved one shared budget ledger.

The runner's originals live under `H3-World/outputs/EXP-009_v3_fm4_af4/{fm4,af4}/`; the per-stage JSON, `budget.json`, MP4, endpoint latent and raw KV are retained there. Small original logs, metrics, contact sheets and MP4 copies are in [artifacts](artifacts/raw). [make_comparisons.py](make_comparisons.py) built the real four-panel [AA 56f](artifacts/comparisons/AA_FM4_AF4_with_8NFE_56.mp4), [AD 56f](artifacts/comparisons/AD_FM4_AF4_with_8NFE_56.mp4), [AA 73f](artifacts/comparisons/AA_FM4_AF4_with_8NFE_73.mp4) and [AD 73f](artifacts/comparisons/AD_FM4_AF4_with_8NFE_73.mp4) videos. Both source and output videos passed [full decode/FPS/PTS checks](video_decode_check.json); the first 39 raw RGB frames match exactly. The four-panel videos include earlier FM8/AF8 references and label protocol differences.

The visual gate was joint: full new-frame inspection of AA/AD motion direction, person and parking-lot structure, ghosting and boundary jumps. Optical flow and MAD are supporting measures. Engineering correctness, direction response and video quality have separate conclusions. [Judge's final review](judge/FINAL_REVIEW.md) accepts FM4 limited feasibility with partial quality and rejects AF4's combined 4-NFE result because AD C3 loses person structure. This short-trained student line is stopped without extra 2-NFE or training runs.
