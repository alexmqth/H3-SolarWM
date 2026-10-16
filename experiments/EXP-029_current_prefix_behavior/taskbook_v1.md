# EXP-029/v1: same-state current-prefix behavioral intervention

2026-10-17 HKT. Judge authorizes original Exp Worker to implement, pass CPU checks and execute this finite experiment automatically. User's overnight mandate ends09:00 HKT; all experimental activity must finish by08:59:45. No further user confirmation is required within this scope.

## Question

Does allowing **non-action prefix queries to read current video keys** materially change action response or visual continuity under otherwise identical V3 generation? EXP-028 did not establish a strong role-conflict reason for dual-adapter training. This task tests behavior behind a known lawful conditioning path; it is not a new architecture, training run or claim of novelty.

## Inputs: reproduce the actual strong/weak phenotype

Use preserved EXP-014 generated histories, not new GT histories:

- `H3-World/outputs/EXP-014_v3_multiscene_teacher/fixtures/s0_43866101.pt` and `s3_b784d995.pt`.
- `G1/<scene>/FM30/first12.pt` and `published_39.npy` in the same root.
- Both fixtures already contain matching A/D prompt embeddings, full37 native layout, initial noise, audio noise and Single I0. The encoder asserted equal A/D action spans. Verify actual saved tensors and provenance; do not assume equality merely from code.
- Preserve historical action A rows. For AD, replace only local action embedding spans12:17 using the D donor, following EXP-014 `run_teacher.visible`; crop all future spans/video before refiner. No new text/image/video encoding or downloads.
- Same Original H3/released Action LoRA, no new adapters; fp32 native policy, Global RoPE, own-action routing and action feedback, partition12→5, native scheduler30NFE/chunk with flow_shift2.22, original seed13 fixture noise. All4 branches per scene share exact I0/history/current noise/audio/positions.

Historical cache was pruned previously; rebuild it **once per scene under correct V3** using saved generated C1, clean sigma0 commit. All4 current branches reuse the identical detached raw KV. Do not recompute history under the OFF topology. Record KV hashes before/after, original history/fixture SHA and source episode. This isolates only the current C2 edge and retains the actual generated-history state associated with the prior phenotype.

## Four branches per scene

| Branch | Current action | Non-action prefix reading current video |
|---|---|---|
| ON_A | A | Enabled, frozen V3 |
| ON_D | D | Enabled, frozen V3 |
| OFF_A | A | Disabled |
| OFF_D | D | Disabled |

Only remove `prefix_feedback_mask[ann < 0, current_video_offset:] = True`. Action rows retain their own current-video feedback, current video retains own-action reads and past raw KV. All other mask entries and query/key sequence shapes stay identical. Implement the OFF branch through an isolated context/attention function without changing frozen files. If an old helper enforces current-prefix ON, adapt the experiment wrapper explicitly; do not bypass its other time/index/action/cache checks.

## CPU gates (<=45 minutes,4 workers,RSS32GiB)

1. Actual A/D fixture spans/layout match; only current12:17 action embeddings differ; historical A/non-action embeddings and positions match. Future text/video physically absent.
2. On a small graph and real token-group layout, ON/OFF masks differ exactly in the declared non-action prefix rows/current-video columns. Action/current-video/history visibility and native time contract unchanged. Include future isolation and no writes to frozen cache.
3. Real entrypoint imports, independent per-kind budget counters, CPU media path validation, source/checkpoint/input hashes; no mistaken clean-KV commit index assertion. Native schedule has31 points and30 updates with no off-by-one.
4. Confirm all required preserved inputs exist; no silent replacement, re-encoding, regenerated C1 or change to factual/source labels.

## GPU budget and execution

Single GPU1 preferred; may share when at least40GiB actually free in two checks>=10seconds apart and immediately before launch. No util=0 or empty-process requirement; do not touch unrelated tasks. If GPU1 unavailable, wait with bounded polling; no unapproved GPU fallback. CPU PASS permits automatic launch.

- GPU-process wall<=3600seconds inclusive load/offload/commit/sample/decode/cleanup; allocated peak<40GiB, process RSS<96GiB, disk growth<=5GiB.
- **242 DiT attempts = 2 clean history commits + 8 continuations x30 sampling calls.** No backward/updates, text/VAE encoding, extra forward diagnostics or automatic retry. **8 VAE decodes maximum.** Failed attempts count.
- Run scene s0's ON_A,ON_D,OFF_A,OFF_D, then s3 in the same order. Reuse model and the per-scene cache; release each scene's unused cache before the next. Decode with a documented native/offload policy shared by all branches; do not keep avoidable unused tensors on GPU.
- Preserve source-generated first39 published frames exactly. Each branch shows those39 plus17 newly generated frames at832x480/24fps. Label the provenance clearly: these are short continuations, not newly generated full5-second videos. If full VAE decode changes prior-frame values, save the true decode diagnostics and explicitly apply the same frozen-history publishing contract across all branches; do not conceal boundary effects.
- Register controller/timeout/actual child PID,start_ticks,pgid in `research/overnight_20261017/PROCESS_REGISTRY.json` before model computation. Respect guard's08:59:45 stop and09:00 hard cutoff; timeout is min(3600,remaining time to08:59:45). No launch if insufficient planned time; report resource wait rather than claim running.

## Outputs and evaluation

Save exact configurations/commands/hashes, source layout and token roles, CPU results, per-call ledger, per-branch sampling/decode timing, GPU allocated/reserved/process memory, CPU KV bytes/RSS, latent endpoints and raw RGB locally. Publish8 original continuation MP4s and2 labeled2x2 comparisons (rows ON/OFF, columns A/D); comparison assembly is CPU-only, not extra decode calls. Save frame contact sheets for all17 generated frames plus boundary, and SHA/media metadata. Large tensors remain local/ignored.

Report A/D branch difference within ON and within OFF, and ON/OFF difference for each fixed action. Inspect actor structure, motion/pose, scene/camera movement, ghosting and boundary; distinguish character-relative-to-scene evidence from camera-induced screen motion. Global flow and pixel MAD are auxiliary, not an action correctness classifier. Explicitly mark ambiguous camera/character direction. Check whether OFF alters general visual quality rather than specifically control.

Compare ON output with preserved EXP-014 endpoints/videos where possible; report precision/backend-related discrepancy rather than silently demanding bitwise agreement with another historical runtime. The new ON/OFF comparison must use the same runtime. This task alone does not demonstrate a new method or generalize across unseen scenes/seeds.

Stop on unexpected mask/source/time changes, cache mutation, nonfinite tensors, failed budget counters, OOM or budget/deadline. Preserve partial outputs; no automatic scene/seed/NFE sweeps. Worker messages Judge after first scene and on completion, then waits. Judge reviews all generated frames and decides whether a behavior mechanism exists or this path should be closed. No automatic dual LoRA, AnyFlow, DMD or longer rollout.
