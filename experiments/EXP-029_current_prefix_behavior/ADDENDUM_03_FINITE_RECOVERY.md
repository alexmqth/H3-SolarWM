# EXP-029/v2 — finite recovery of the missing s3 OFF_D branch

Judge, 2026-10-17 11:49 HKT. User now explicitly requests trying GPU1/6. This is fresh daytime execution authorization, superseding the expired overnight admission only for this finite recovery. It does not retroactively change the overnight budget or its timeout outcome.

## Audited original state

The original attempt ran on GPU0 from 2026-10-17 03:10:16.545 to04:10:14.361 HKT, approximately3597.816 process seconds. It stopped on its wall-time cap, not OOM. Ledger:235 DiT (2commit+233sampling),7decode,0training; allocated25.4408GiB/reserved26.5313GiB; RSS83.5365GiB. All four s0 branches and s3 ON_A/ON_D/OFF_A have complete results. The final s3 OFF_D has23 sampling attempts in the ledger and no saved endpoint/video. Its RESULT.json still reports initial0 because progress was not flushed each step; use the ledger to account cost. All registered original processes are gone at Judge audit; the overnight guard reached its09:00 cutoff.

Seven completed outputs and failed-run records remain immutable. The original failed status is retained. No whole-batch retry.

## Recovery scope and budget

- One GPU, preference[1,6]. User availability permits either; do not occupy both for this one missing branch. Keep daytime maximum3 rule satisfied.
- At least32768MiB actual free twice >=10 seconds apart plus immediately before launch. Same `ABOT_VRAM_RESERVE_GIB=18`, precision/backend and original model/released Action LoRA.
- One attempt, process wall<=1200seconds including loading, compute, decode and cleanup. Use an external timeout with termination/kill cleanup. Latest start12:35 HKT, absolute hard stop13:00 HKT. Do not reuse the expired overnight scheduler/guard as new authorization.
- Exactly at most31 DiT attempts: one correct V3 ON clean C1 commit for s3 from the saved EXP-014 first12, then30NFE **OFF_D only** from the original saved C2 initial noise. No extra comparison forwards. At most1 VAE decode;0backward/updates/encoding/download.
- GPUallocated<40GiB, RSS<96GiB, new disk<=1GiB. Original and recovery costs remain separate and are summed in the final report. Original233sampling includes the incomplete branch; total authorized consumption becomes at most266DiT/8decode, including that interrupted work.
- Cannot resume from step23: no checkpoint of that partial state was saved. Re-run only this30NFE branch; never label it as a7-step resume.

## Protocol and CPU gates

Reuse the frozen scientific computation: same fixture and historical A rows, current D12:17, same audio/I0/Global RoPE/native31-point scheduler, own-action feedback, current non-action prefix feedback OFF, CPU raw KV and original39 published RGB frames. Rebuild C1 under ON once and compare its digest against original s3 cache digest `2cd1f8e6d4659e74fa70618c0b0317272c18c575244bd55f1c2df14d2cfd8ca9` before sampling. On mismatch stop and report; no extra GPU diagnosis/retry.

Use a dedicated `recovery01/` runner, authorization, ledger, status/log and output directory. Preserve v1 source/CPU hashes and raw evidence. CPU gates cover real imports, exact one-branch selection and budget counters, future cutoff/timeout, no expired09:00 checks, effective reserve18 environment, source/input SHA and output-path isolation. Reuse prior mask/condition tests; do not broaden CPU diagnostics. Freeze the seven completed endpoint/RGB/video hashes before execution and verify unchanged afterwards.

CPU PASS authorizes automatic launch within this budget; original Worker must message Judge with actual PID/GPU, not only a scheduler intent. Report first forward and final exit. On OOM/nonfinite/cache mismatch/budget failure, stop without another attempt.

## Delivery

Recover one56-frame/17-new-frame video and contact sheet. CPU assembly may produce the missing s3 2x2 comparison from three frozen originals plus recovery OFF_D; do not overwrite original completed media. Publish all eight branch sources in a manifest that distinguishes original and recovery runtimes. Record environment, shared GPU conditions, sampling/decode/wall times and allocated/reserved/process memory. Shared GPU wall time is not a controlled ON/OFF speed benchmark.

Complete the delayed EXP-029 Worker report and preserve the overnight termination audit. Judge reviews behavior across all new frames, then decides acceptance and publishes. No additional scenes, seeds, NFE variants, LoRA, AnyFlow or DMD are authorized.
