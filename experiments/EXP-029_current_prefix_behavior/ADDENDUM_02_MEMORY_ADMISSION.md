# EXP-029 admission addendum 02: explicit CPU offload reserve

2026-10-17 HKT, Judge. Admission revision 3. This supersedes only the free-memory threshold and model memory placement policy in v1/addendum01. The finite scientific comparison, output count, precision, noise, conditions, 242 DiT/8 decode/0 update budget, single GPU candidate order [1,0,6,7], 3600 process-second cap and 09:00 hard cutoff are unchanged.

## Evidence and reason

Worker traced EXP-029 to EXP-014 `run_teacher.load_model`, whose environment inherits `ABOT_VRAM_RESERVE_GIB=5` from `infer.py` when unspecified. EXP-014 G2 s0 reached 38.181 GiB allocated. Therefore **32 GiB free is not sufficient evidence for the existing policy**; the prior 40 GiB admission was justified.

EXP-019 explicitly set `ABOT_VRAM_RESERVE_GIB=18` and ran the same 12-history + 5-current latent shape, H3 FP32 policy, CPU raw KV and tiled 56-frame VAE decode (tile256/overlap64). Its four generation rows in `METRICS_TABLE.csv` report reserved peaks 27.0469–27.1973 GiB and allocated peaks 24.9450–25.2939 GiB. This supports testing that memory placement policy with 32 GiB free, leaving about 4.8 GiB beyond the measured allocator peak for context/other overhead and differences. It is an estimate, not a guarantee of EXP-029 peak consumption or immunity to another process growing.

## Authorized implementation

1. Only if the existing scheduler still has attempt_count=0 and no launched child, freeze its status, authorization and CPU manifest under `admission02_previous/`, verify PID/start_ticks/command identity, and terminate that waiting controller only. Confirm exit before editing executable files. If it has launched, do not interrupt or mutate the running experiment; report to Judge and retain revision2 for that run.
2. Explicitly pass `ABOT_VRAM_RESERVE_GIB=18` in the child environment. Set authorization admission_revision=3 and min_free_gpu_mib=32768; scheduler eligibility must use this value consistently. Record effective reserve and threshold in launch status. Preserve two checks >=10 seconds apart and immediate prelaunch check.
3. Keep h3_fp32/weights, attention backend, native schedule and all paired conditions unchanged. No quantization, resolution reduction, altered token packing, or experiment retry. All eight branches use the same memory policy. The historical EXP-014 comparison remains a compatibility observation, not a strict same-runtime equality requirement.
4. Update CPU admission boundary tests (32767 reject/32768 accept), effective environment validation and frozen source/input SHAs. Verify environment reserve is parsed by the actual loader without initializing CUDA. No need to rerun unrelated diagnostic experiments.
5. Preserve old controller registration as historical; register the replacement controller and actual child identities under the same deadline guard. Preserve attempt_count=0 only because no GPU process was launched. Do not use a controller relaunch to reset consumed experiment budget.
6. CPU PASS permits automatic launch of the one existing authorized GPU attempt. The allocated<40GiB limit remains an outer stop bound, not a reservation. No automatic OOM retry or precision/offload retuning. Report actual allocated/reserved peak and process memory; timing includes any additional CPU transfers within the original wall limit.

Worker implements and reports CPU result and new live controller PID/start identity to Judge. No new experiment or training is authorized by this admission change.
