# EXP-028/v1: bounded Action–Video LoRA-gradient pilot

2026-10-17 HKT. Judge authorization following user's instruction to continue. This replaces preparation-only status for this task. Original Exp Worker implements and executes; frozen V3 and prior experiments remain unchanged.

## Question and decision

Does a fixed rank-8 adaptation subspace exhibit materially different token-role gradient geometry under Original-visible and correct V3 on two factual ABot examples? This is a small operational comparison, not a single-edge intervention or proof of semantic objective conflict. Results decide whether a role-specific adaptation intervention merits further behavioral testing. Negative cosine alone never authorizes training.

Main simplification: **measure exact LoRA-space role contributions only**. Do not compute or store dense full-weight role gradients, and do not claim to have measured their geometry. For the existing scale s and projection activation X, use U=X A^T and accumulate g_B,r=s D_r^T U_r. At B=0, g_A,r=0 by construction. Capture gradients at the correct residual/cast boundary so that summing roles reproduces actual autograd B.grad. Stream token tiles and retain rank-space results; full activation/cotangent dumps and CPU dense products are out of scope.

## Frozen inputs

- Original H3 + released Action LoRA only; use existing H3 environment. No new downloaded weights/data or SolarWM environment work.
- Clip 1: `43866101158a538d22c52ac5b2fa606b_src1245_rgb56`.
- Clip 2: `b784d995827f53977a8067e5b9dc4b5e_src1635_rgb56`.
- Existing EXP-023 factual encodings/action labels, GT C1 first12 latents, C2 next5. Verify source/episode/I0/prompt/action spans and hashes before model loading.
- sigma .25 with CPU noise seed113; sigma .75 with seed127. Each graph receives identical current noisy latent/target/action per clip/sigma.
- Fixed layers 0 and49, rank8 diagnostic wrappers; copy each layer's A from EXP-025 checkpoint_000, B remains exactly zero. Both graphs use the same A. Record actual shape (8,5376)/(21504,8), scale and checkpoint SHA.
- QKV is head-interleaved (56,3,128); derive actual offsets from production module. Partition all current/history action/video, non-action prefix and other rows disjointly. Preserve finer role provenance in output.

## Graphs and interpretation

Original-visible recomputes the visible17 latent window under original bidirectional routing with C2-only factual FM loss. V3 uses native Single I0/current-prefix, own-action feedback, Global RoPE, native timestep and sigma0 detached GT C1 raw KV, max_history5. Future text/video beyond C2 is physically cropped before refiner. No route/time/prefix changes to make results look better.

These graphs differ in history computation, prefix and topology. Match all other source conditions and report the differences. Writer gradients under detached V3 are NOT_IN_GRAPH. Some role/Q projections can have exact structural zero gradients, especially in the final layer; report zeros and undefined cosines without replacing layers. Small-sample negative alignment cannot establish broad interference or model quality. No optimizer is created or stepped.

## CPU implementation and gates

Implement only in this experiment directory; reuse frozen helpers or isolated imports. At most60 minutes active CPU preparation, 4 workers, RSS32GiB; no 33B model load during this phase. Validate:

1. Role masks, QKV ordering, LoRA chain rule with production-like casting and checkpoint recomputation deduplication on small CPU modules. Include prefix/other and zero-B cases. Freeze error criteria before GPU (FP32 role sum relative RMS <=1e-3 and max-abs <=1e-3*max(1,max_abs(reference)); exact structural zeros handled separately).
2. Explicit grad-enabled Original-visible path and V3 training attention path. Do not assume old inference-only audit helpers support backward. Preserve their attention semantics; scope any differentiable replacement and test on CPU.
3. Source/checkpoint/input hashes, full command, role definitions, cost estimate for rank-space GEMMs and expected capture memory. Freeze configuration and ledger before GPU.
4. Zero wrapper output identity is checked in GPU calls explicitly budgeted below. Wrapper must be enabled in differentiable passes even when B=0.

Passing these objective gates authorizes automatic continuation to the finite GPU pilot without another user confirmation. If a gate cannot pass within CPU budget, submit a specific failure and stop; no broad redesign.

## GPU budget and execution order

One empty L40. **GPU-process wall time <=2400 seconds**, includes model load, offload/reduction, failure, cleanup; allocated peak <40GiB; process RSS <96GiB; <=4 CPU workers. No concurrent model processes for this task.

| Calls | Count | Backward |
|---|---:|---:|
| V3 clean GT C1 commit, one per clip | 2 | 0 |
| Main current forward: two clips x two sigmas x two graphs | 8 | 8 |
| Unwrapped identity reference: first clip x two sigmas x two graphs | 4 | 0 |
| Total hard cap, including failed attempts | 14 | 8 |

Reuse first-clip main forwards as wrapped identity outputs. Compare before backward to the corresponding unwrapped reference with declared precision/backend and relative RMS <=1e-6 / max-abs <=1e-6*max(1,max_abs(reference)); report exact equality when attained. If numerically inconsistent, stop and preserve evidence. No retries or changing backend within the run. Checkpoint recompute is separately counted/timed, not hidden in sampling NFE.

Order: first clip/sigma .25 identity and both graphs' role-sum checks, then remaining first-clip sigma, then second clip. Reuse each clip's detached cache across sigmas. Only run remaining calls after first-pair shape/memory/gradient audits pass. No new media, text/VAE encode/decode, training updates, dual adapters or dense-gradient survey.

## Idle GPU admission

At issuance all eight GPUs are occupied by other projects. Do not launch on a partially free GPU or terminate any existing process. CPU preparation starts immediately. Worker may start a bounded background admission controller after CPU PASS:

- Preferred candidate devices 0,1,6,7; one GPU only. Recheck UUID, no compute processes, memory used <1024MiB twice separated by at least10 seconds, and host MemAvailable >=100GiB before loading. Use a project experiment lock to avoid duplicate GWM launches, and recheck immediately before launch. If state changes, release lock and wait.
- Poll at most once per60 seconds; record WAITING_FOR_IDLE_GPU and last device snapshot. Child launch exactly once; failed/OOM attempt stops rather than retrying on another GPU.
- Admission expires **2026-10-17 13:15 HKT (05:15 UTC)**. No launch after expiry. A launched job retains its 2400-second cap. After09:00 respect project-wide maximum3 GPUs; this task uses1.
- Persist controller PID/log/status/expiry. If waiting at handoff, explicitly report waiting, not running an experiment. On expiry submit resource-blocked status; no indefinite scheduler.

## Deliverables and stop conditions

CPU_PREFLIGHT.json, source/config/hash manifest, run command, real scheduler status, GPU_LEDGER.json, per-clip/sigma/graph/layer/QKV role counts/norms/dots/cosines with numerical floor, autograd role-sum errors, identity errors, graph/commit/recompute counts, elapsed and peak GPU/CPU memory, and report.md. Keep only bounded summaries; no giant activation dumps. Update root report.md and message Judge when CPU gates pass and when GPU finishes or admission expires. Worker does not operate Git or alter Judge management files.

Stop on CPU/gpu time/call/memory cap, source mismatch, unexplained role-sum or identity failure, ambiguous checkpoint invocation, OOM/nonfinite output, unexpected route edits, or scheduler expiry. Do not silently relax precision thresholds or replace clips/layers. A completed negative result is useful; no automatic training escalation. If geometry differs, state the exact subspace and graph contrast, and propose one intervention tied to behavior. If it does not differ, stop this sampled candidate without claiming all optimization mechanisms are excluded.
