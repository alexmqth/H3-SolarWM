# AnyFlow step16 39-frame visual/action diagnostic

Reviewed snapshot: 2026-10-08T03:47:25.206307+08:00

**This checkpoint is not accepted.** The video below is a diagnostic and does not replace the meeting demo.

[Original / AnyFlow 4-step / AnyFlow 8-step, A and D](anyflow16_AD_original_4step_8step_diagnostic.mp4)

- Both rows contain all 39 source frames at 24 fps (1.625 s), with no failed tail removed. Each source is 832×480; the display scales each tile to 624×360. Timings and forward counts are overlaid.
- 4 steps/chunk: marked ghosting and fog in the latter half; person/garage structure breaks down. A=-1.3153, D=-1.2244, A-D=-0.0909. Visual and action gates fail.
- 8 steps/chunk: substantially more recognizable person and garage, but blur/ghosting remains, especially late D. A=-1.1709, D=-1.4752, A-D=0.3043. Action gate fails; this is not a stable or action-faithful final model.
- Original H3 30 full-sequence steps: A=+1.1813, D=-0.8421, A-D=2.0234. Shared initial image, action-specific prompt, video/audio noise and action spans were checked as exact saved-tensor matches.
- Review used all-frame contact sheets for A4/A8/D8, sampled frames for D4, and the annotated comparison frame30. Raw source hashes and review scope are in `visual_review.json`. Decode verification and visual acceptance are separate.

The weighted validation loss decreases slightly while the high-noise endpoint raw residual worsens. The current evidence does not establish useful 4-step learning after 16 updates. It also does not establish that AnyFlow itself is ineffective: the completed FM, step00/intermediate and clean-history controls instead show that this short training recipe has not established useful AnyFlow learning. Uniform-grid and frozen-time controls remain separate experiments.

The current shifted 4-step grid ends with sigma≈0.425→0, while official Stage1 uses a uniform grid ending 0.25→0. Current shifted 8-step ends ≈0.241→0. A same-checkpoint uniform-grid experiment is the next sampling ablation after the frozen pilot; this is a hypothesis, not a diagnosed cause. Stage2 remains deferred.

Single-run CPU-offload timings on shared hardware do not establish speedup. Archived original H3 offload reserve is unknown, so GPU peak values are not a matched residency comparison. MAD/boundary differences describe activity/continuity rather than perceptual quality.

## Additional FM / step00 observations (04-hour snapshot)

FM16 4/8-step A/D numeric gates also fail (separation0.2338/0.3149). Sampled FM4 A frames preserve much more person/garage geometry than AnyFlow4. The untrained AnyFlow step00 A also shows marked late fog/ghosting, so trainable-time parameter updates cannot be the sole cause of this degradation. See `FM4_AnyFlow00_AnyFlow16_A.jpg`; this is sampled visual evidence, not a quantitative quality score.

Source review independently found that official H3 Stage1 freezes the target-time MLP while the old local pilot trained it. That is an optimizer-policy discrepancy, not an established visual root cause. Controlled uniform-grid and frozen-time runs are queued separately.

## Completed clean-history control

AnyFlow16 native4 clean-history A=-0.405301, D=-0.918850, A-D=0.513549; the direction/separation gate still fails. In `clean_vs_generated_4step.jpg` (frames 0/8/16/17/28/38), clean history restores recognizable person/garage structure at later chunk starts, but A frame28 still ghosts badly and D also blurs. This uses original-teacher history/anchors, not deployable generated-history inference. Visible resets at chunk boundaries raise boundary RGB MAD to A17.752/D12.307 versus generated A7.008/D6.778. Teacher context and current generated output may disagree, so those resets cannot be interpreted as a pure cache bug or a fair rollout quality score. First-chunk failures persist without generated history.
