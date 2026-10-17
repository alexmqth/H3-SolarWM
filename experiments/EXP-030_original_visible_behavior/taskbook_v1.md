# EXP-030/v1 — Matched Original-visible / V3 action behavior

Judge authorization: 2026-10-17 12:20 HKT. User approved “开始吧” after the four-new-continuation proposal. Branch A / causal mechanism diagnostics; parent frozen Original H3 + released Action LoRA and V3 baseline. Assigned to original Exp Worker thread 01a0ece6-63b6-7f42-8c39-cd3ec39a9c80. CPU gates PASS authorizes automatic finite GPU execution; no further routine approval is needed.

## Question and interpretation

Does a clean-history, bidirectionally recomputed Original-visible reference preserve an A/D behavioral response missing from V3 when both receive exactly the same saved generated history and current noise?

Original-visible is a controlled inference reference, NOT the native end-to-end Original H3 pipeline. Its comparison with V3 changes historical hidden-state recomputation, attention topology and prefix computation as a protocol package; do not attribute a difference to a single causal edge. EXP-020 already ran native s3 mixed actions but supplies no matched A/D positive control. EXP-028/029 are closed; no more prefix-edge or role-gradient sweeps.

## Conditions and immutable inputs

Only four new outputs: s0_43866101 A/D, then s3_b784d995 A/D. Reuse four EXP-029 ON_A/ON_D V3 outputs and their exact input identities. Saved C1 is 12 latents / 39 published RGB frames; C2 is 5 latents / 17 new published frames. Full output is 56 frames at 24fps, not a newly generated 5-second video.

Use EXP-014 fixtures/<scene>.pt, G1/<scene>/FM30/first12.pt and published_39.npy under H3-World/outputs/EXP-014_v3_multiscene_teacher/, resolving exact noise/audio/packed conditions from EXP-029 run_exp029.py and manifests. Hash actual tensors and sources, not guessed path equivalence. Historical action remains A; current action A or D occupies latent indices12:17. A means man strafes left with camera following; D means right with camera following. Keep released weights, Single I0, native video position, action routing, audio inputs, initial noise, scheduler flow_shift2.22, 30 sampling NFE/31 schedule points, output resolution832x480 and V3 numerical policy/attention backend where applicable. Explicitly list inherent reference differences; do not silently change V3 to make the reference work.

Original-visible recomputes all17 visible latents each sampling call, clamps first12 to the saved clean history, uses native clean-history video time and ordinary sigma-dependent prefix time, returns only current5 velocity. No future video/action/padding is visible. Historical published RGB must remain bit-identical. Internal historical representations may change; describe this rather than calling it the same historical KV. No V3 GPU replay or new KV commit is needed.

## CPU preparation and gates

Reuse EXP-028 original_clean.py as audited starting point, not an already validated sampler. Its strict 0<sigma<1 guard excludes the full schedule start. Explicitly inspect actual native schedule endpoints and safely support required endpoint(s); retain old frozen helper and implement in this experiment. Check native call/mask contracts rather than simply removing assertions.

Bounded CPU preparation <=45min, <=4 CPU workers, <=32GiB RSS excluding any full model loading. Verify imports; exact four-branch selector; own-action layout and visible17 crop; history clamping through every scheduler update; noise identity; native timestep/position/prefix/audio protocol; zero added adapter; actual sampler endpoint support; no gradients; correct saved endpoint/RGB before media encoding; codec CPU round trip; counters/timeout/admission/unique paths. Reuse existing proven tests and use lightweight mocked contracts where real weights would require GPU. Report clearly what CPU checks cannot establish. Freeze source/input hashes and CPU_PREFLIGHT.json before launch. If unavailable inputs, unsupported protocol, or substantial architecture changes are required, stop and report.

## Resource authorization

- One GPU only, allowed/preference [1,6]. Current hardware L40 ~45GiB; shared use permitted. Do not require utilization0 or an entirely empty card. Do not touch other users' processes. Recheck free memory >=32768MiB twice >=10s apart and just before launch. Explicit ABOT_VRAM_RESERVE_GIB=18; CPU offloading enabled using existing environment. This admission is conservative feasibility, not a guarantee of peak usage.
- Aggregate model-process wall <=5400 seconds INCLUDING loading, sampling, decode and failed work. At most120 top-level DiT attempts =4x30 sampling; zero separate probe/commit/replay forwards, zero backward/optimizer updates, <=4 VAE decodes; zero VAE/text/image encoding or downloads. CPU media assembly is allowed.
- GPU allocated <40GiB; process RSS <96GiB; new disk <=3GiB. Record measured allocated/reserved peaks and CPU memory. No use of GPUs0/2/3/4/5/7 for this task.
- Latest GPU launch 2026-10-17 14:25 HKT; absolute hard termination16:00 HKT (08:00UTC), graceful stop15:59:45. Use external process-group timeout/cleanup and an aggregate call ledger. This is fresh daytime authorization, not the expired overnight guard. If no admissible GPU by cutoff, report blocked without a lingering watcher.
- One GPU attempt. OOM/nonfinite/model exception or invalid history protocol stops the task; preserve completed outputs and all cost. CPU-only media repair from saved arrays is allowed, but no silent GPU retry. No automatic continuation into GT-history, longer windows, extra episodes/seeds, FM8, dual adapters, AnyFlow, DMD or training.

## Execution and stopping

Run s0 A/D first; then s3 A/D within the same bounded allocation if computational validity holds. First actual sampling call counts in120; it is the runtime gate (finite output, native shapes, immutable historical latent/publication, correct sigma/action graph). Report actual PID/GPU/log/first-forward to Judge, not only scheduler intent. Stop on resource/protocol failure. Behavioral weakness itself is a valid result; do not rerun seeds to improve appearance. If the reference is clearly structurally unusable already on s0, inform Judge and stop expansion rather than spending remaining budget blindly.

## Evaluation and deliverables

Record source/config/input hashes, environment, precision/backend, full sampling schedule, process and branch timings, sampling/decode times, memory and120-call ledger. Save generated endpoints and RGB before ffmpeg. Raw evidence goes under H3-World/outputs/EXP-030_original_visible_behavior; source/config/logs/report under this experiment directory. No overwriting EXP-014/028/029 or formal V3.

Deliver four new videos and two scene-specific labeled2x2 comparisons (V3 A/D from EXP-029 versus Original-visible A/D), all17 new frames plus boundary contact sheets and useful actor crops. Explain OV=Original-visible, A/D, NFE, clean history, and the39old+17new frame provenance in README. Report character relative-to-background motion, camera-follow ambiguity, visible response within0.71s, anatomy/ghosting/boundary. RGB MAD/global flow are auxiliary, not action accuracy; allow ambiguous verdicts. Do not promise robust latency or camera/character disentanglement from this short window. Compare within-condition A/D effects before comparing protocols. Shared GPU timings are descriptive, not an algorithmic speed benchmark.

Submit report.md with completed/failed conditions and summed cost, then stop all runners/watchers and message Judge. Judge audits videos, chooses accept/revise/stop and publishes. Worker does not concurrently commit/push.

## Decision after this task

If reference responds on s0 and s3 while V3 is weak on s3: evidence of a protocol-package control gap; consider one independent episode or targeted factorization, separately authorized. If both weak on s3: lower priority of V3-specific explanation; do not infer native H3 inability. If s0 reference fails: inspect applicability before expanding. GT/Self A/D and longer response windows are conditional future proposals, not authorized here. No formal model upgrade follows from output divergence alone.
