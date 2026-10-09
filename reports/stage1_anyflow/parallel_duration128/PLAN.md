# Bounded Stage1 duration continuation: same logical batch on four GPUs

Status: source64 full A/D4/8 review completed on2026-10-08 at13:16HKT; selected for bounded continuation, not yet launched.

## Purpose

Check whether additional unchanged-protocol optimization moves actual A/D generated-history videos toward the short gate. A decreasing scaled loss, restored Adam, or faster execution is not success. This is Stage1 AnyFlow work; it does not start Stage2.

## Fixed experiment

39 RGB frames, 12 latent frames, three chunks5+5+2; full differentiable clean history during training, generated history during evaluation; CPU persistent raw KV, RGB dual anchors, causal action rows/feedback, seed13 and original conditioning/noise. Full52-block QKVO/FFN rank8 bank has43,237,376 trainable parameters; original visual/action adapters and cloned target-time MLP remain frozen. Global logical batch4, LR3e-5, training shift12, held-out/inference shift2.22. The original four-forward AnyFlow loss, including diagonal samples, is unchanged.

## Execution and budget

Use GPUs3,4,5,6 only after occupancy checks. GPU0 is exclusively authorized and not limited to25GiB. Reserve6GiB with offload is retained, targeting the already-tested~40GiB allocated training range. Each GPU holds a full frozen H3 replica and one of the SAME four logical samples: this is sample parallel, not SP or model sharding. Gather the two raw diffusion losses for the original adaptive scale; SUM already/global-batch-scaled gradients. More total GPU/host memory is required. Prior real33B read-only single-batch timing is not a full training speedup guarantee.

Start from source step64 with original adapters, Adam moments, update history, teacher identity and logical/CPU/CUDA RNG. First run64→68 and check actual pre-update restoration plus all four final parameter/Adam/RNG states. If any check fails, stop. Then68→96 and A/D8 evaluation. If the numeric gate passes, also generate4-step videos and stop for visual review. Otherwise96→128 and A/D4/8, then stop unconditionally. No automatic training beyond128.

## Acceptance

A>0, D<0, A−D>1.0 AND intact person/parking structure across all39 frames. Then require matched-dose FM, independent inference seed, A→D→A/D→A→D temporal grounding, and only then124-frame W/S/A/D. These further checks are not included in this finite controller. The numeric gate alone never accepts Stage1.

## Reproducibility limits

Source64 weights/Adam/RNG must restore exactly. CUDA serial versus parallel accumulation is not bitwise deterministic; earlier real33B gradient differences were comparable to repeated original backward runs. Preserve the isolated runtime rather than editing main trainer or existing jobs. The sole copied benchmark change corrects recorded training shift; sampler/model math does not change. No new data, LR, rank, anchor, attention pathway, solver, or diagonal shortcut is added.

## Source64 decision

Both full64 queues completed. Shift2.22 at8 steps/chunk: A−1.072072,D−1.287603,separation0.215531. Shift12 at8: A−0.589203,D−1.212299,separation0.623096 (previous32:0.172691). All four shift12 source64 complete39-frame sheets and Original/16/64 matched12/24/30/38 frames reviewed:4-step still severe fog/ghosting after~20–22;8-step retains subject/garage with late blur/transparency and wrong A direction. Review is static full-frame inspection, not real-time playback. No gate passes. Shift12 A endpoint raw worsens32→64 while D improves, so internal losses do not certify the action trend. Select shift12 because its8-step generated-history separation has started moving appreciably under an unchanged protocol, and its sampling matches the official training shift; do not call it accepted or proven superior generally. Only one additional64-update budget, not an open-ended run. The32-update FM comparison did not show an AnyFlow advantage; AF64 vsFM32 is unequal training dose and cannot establish that advantage.

Before launch at13:18, GPU0 acquired an unrelated r2_real_models.py process(PID2253489,~1.8GiB,high utilization). The idle guard refused to launch; no child or run.json was created. Use previously verified idle GPUs3–6 and evalGPU3, preserving the original preflight/manifest before this hardware-only reassignment. User authorization for idle GPUs already exists. GPU0 has no25GiB policy cap; it is simply excluded while occupied.
