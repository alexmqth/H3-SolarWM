# EXP-010/v1 Worker core result — FM8 AA73 → AA124

2026-10-11 HKT. **The separately released shared-core stage is complete and stopped at 124 RGB frames for Judge review. C7/C8 have not started.** It used the existing EXP-006 FM8 AA73 clean latent history and C1/C2 cache, then committed its existing C3 once and generated C4, C5 and C6 with ordinary FM8. No new training, AnyFlow or DMD weights were involved. The protocol and limits are in [taskbook](taskbook_v1.md), [frozen manifest](source_manifest.json) and [Judge core release](GPU_RELEASE_CORE.md).

GPU0 calls were sequential: `--stage commit_c3`, `--stage C4`, `--stage C5`, `--stage C6` using [run_fm8_sw.py](run_fm8_sw.py). Each exact stdout original is copied in [core logs](artifacts/core/logs); large cache and latent tensors remain in `H3-World/outputs/EXP-010_v3_fm8_sw158/`. There were no failures or retries. The [core budget snapshot](artifacts/core/budget.json) records **28 forwards = 24 sampling + 4 clean commits, 3 VAE decodes, 565.082 GPU seconds = 0.156967 GPU-hours**, within the phase's 28/3 and total task's 62/7/0.75 GPU-hour limits. Peak allocated VRAM across core calls was about **26.455 GiB**. Core wall time includes repeated 33B loading, raw CPU cache I/O and whole-prefix VAE decoding; sampling NFE alone is not an end-to-end timing result.

| Stage | RGB length after stage | Horizontal flow px | Boundary / inside gray MAD | Cache ancestors after commit | CPU raw KV bytes | Worker visual assessment |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| C3 clean commit | 73 (unchanged) | — | — | C1–C3 `[0,1,2]` | 12,465,024,000 | No new video; builds FM8's own history KV |
| C4 | 90 | +0.836 | 9.035 / 3.455 | C1–C4 `[0,1,2,3]` | 15,297,984,000 | Person and garage recognizable; leg trail and view jump |
| C5 | 107 | +0.732 | 5.175 / 4.269 | C1–C5 `[0,1,2,3,4]` | 18,130,944,000 | Structure persists; translucent lower limbs |
| C6 | 124 | +0.660 | 3.711 / 3.818 | **C2–C6 `[1,2,3,4,5]`** | **14,164,800,000** | Person and garage recognizable; mild residue/pose variation |

Flow is a motion proxy, not a formal action-accuracy score. All 51 new RGB frames were retained. The 124-frame [original video](artifacts/core/shared/rollout_124.mp4) and the C4/C5/C6 chunk clips/contact sheets are in [core artifacts](artifacts/core/shared). This remains a single parking scene and seed, with visual quality **PARTIAL** because of transparent leg trails and chunk boundary perspective changes. It is not evidence for 158 frames until C7/C8 run and pass.

The Worker [CPU core check](core_worker_cpu_check.json) independently decoded all 124 H.264 frames at 24 FPS with unique ordered PTS, checked that the first 73 published raw RGB frames are byte-identical to EXP-006, and loaded all 50 layers of `cache_through37.pt` to verify exact retained ancestors `[1,2,3,4,5]`, 300 cumulative layer commits and 14,164,800,000 cache bytes. Every per-stage JSON records source/endpoint/cache SHA, native eight-step sigmas, unchanged source cache and history, step counts, timings, peak VRAM and frame metrics. Small evidence copies and their hashes are listed in [artifact_manifest_core.json](artifacts/core/artifact_manifest_core.json).

**Stop point:** the shared AA124 and its cache may serve as a source for A-continue/D-switch only if Judge reviews the complete C4–C6 frames and issues both the branch GPU marker and a core-review marker bound to C6 JSON/cache37/RGB124 hashes. Neither marker existed at this Worker stop point. No branch GPU work is implied by the core result.
