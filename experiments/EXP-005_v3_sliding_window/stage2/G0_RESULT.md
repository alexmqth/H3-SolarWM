# EXP-005/v2 G0 — before-eviction full-model regression

G0 completed on GPU0 with the frozen 37-latent AA parking input and EXP-003 C1–C5 CPU raw KV. It compared the accepted interval entry and SW-G at two identical C6 states, then replayed C6 at 30 native FM steps. Both fixed-state velocity pairs were exactly equal (`maxabs=0`, relative RMS `0`). The replay endpoint was exactly equal to the saved EXP-003 C6 endpoint. All 124 decoded RGB frames were pixel-identical to the baseline; the first 107 published frames remained unchanged. The new video decodes as 124 frames, 832×480, 24 fps.

Actual budget: 4 diagnostic + 30 sampling = **34 full forwards**, 1 VAE, **309.111 seconds** occupied, peak allocated **25.598 GiB**, zero training. The raw cache remained unchanged throughout reads. This is a full-model before-eviction regression, not new long-window capability.

Evidence: [raw result](../artifacts/stage2/G0/result.json), [attempt ledger](../artifacts/stage2/G0/budget.json), [replay MP4](../artifacts/stage2/G0/C6_replay_124.mp4), [Judge review](g0_judge_review.json), [runner](g0_runner.py), [frozen input manifest](g0_source_manifest.json). These are byte-identical small-evidence copies of `H3-World/outputs/EXP-005_v3_sliding_window/G0/`; the large endpoint tensor remains external.

G1 uses a separately certified 47-latent input and is separately metered. G0's exact match does not predict post-eviction A/D action quality, generated-history stability, or Local RoPE benefit.
