# EXP-005/v2 G1 — SW-G real eviction and A/D continuation

**Worker execution completed; Judge accepted limited feasibility with visual quality PARTIAL.** G1 used the CPU-certified 47-latent extension, the accepted AA124 generated history, Original H3 + released action LoRA, native 30-step FM, current-prefix feedback and strict causal persistent raw KV. The old C6 endpoint was clean-committed exactly once: all 50 layers then retained ancestor indices `[1,2,3,4,5]` for C7, and each path's C7 commit moved its C8 ancestry to `[2,3,4,5,6]`. The shared C6 cache was not mutated by either branch's sampling.

Both paths use the same initial image, C6 history and newly appended noise; C7 changes only the current action from continued A to D. C8 uses each path's own C7 history, so its A/D difference is a closed-loop comparison, not a same-state counterfactual.

| Path / new RGB | Horizontal flow (px/frame) | Boundary gray MAD | Inside gray MAD | Sampling seconds | Visual observation |
| --- | ---: | ---: | ---: | ---: | --- |
| A C7, 124–140 | +0.819 | 7.74 | 4.92 | 177.6 | Single character, scene intact |
| D C7, 124–140 | −1.575 | 14.37 | 4.78 | 208.5 | Opposite motion; noticeable camera/pose jump at switch |
| A C8, 141–157 | +0.768 | 7.57 | 4.64 | 208.6 | Structure recovers after brief limb/color residue |
| D C8, 141–157 | −1.185 | 16.97 | 4.94 | 202.3 | Character/scene remain visible; white trail persists behind character |

C7 same-state action separation by this flow proxy is **+2.394 px/frame** (`A − D`). The proxy alone cannot certify action semantics or visual quality. Existing RGB prefixes were preserved byte-for-byte at both boundaries; A/D 141-frame and 158-frame MP4s independently decoded at 24 fps and 832×480. Visual stability and continuity remain partial, especially for D.

Actual cost: **120 sampling + 3 clean commits = 123 full forwards**, **4 VAE decodes**, **1068.666 s** occupied on GPU0, peak allocated **26.121 GiB**, peak reserved **26.543 GiB**, peak process RSS **91,512 MiB**. No training, AnyFlow, DMD, Local RoPE or C9 was run. CPU KV after eviction is **14,164,800,000 bytes** for five 5-latent ancestors. Only historical video KV is bounded; retained RGB/latent prefix and decode cost are separate.

Evidence: [A/D labeled 158-frame comparison](../artifacts/stage2/G1/G1_A_vs_D_158.mp4), [A full rollout](../artifacts/stage2/G1/A/rollout_158.mp4), [D full rollout](../artifacts/stage2/G1/D/rollout_158.mp4), [machine-readable result](../artifacts/stage2/G1/result.json), [budget](../artifacts/stage2/G1/budget.json), [source manifest](g1_source_manifest.json), [long-fixture certificate](long_fixture_review.json), [Judge review](g1_judge_review.json). Per-chunk JSON, 17-frame MP4 and all-new-frames contact sheets are archived under `artifacts/stage2/G1/A/` and `G1/D/`; large latent and KV tensors remain external.

The long input preserves all old 37-latent conditions and extends the saved action embeddings/time grid. It is **not** equivalent to a default full-length H3 packed rebuild. This is one parking scene and seed, 158 frames total; it does not demonstrate generalized long-horizon stability or overall end-to-end speedup. G1 is a SW-G continuation result, not an AnyFlow or Stage2/DMD result.
