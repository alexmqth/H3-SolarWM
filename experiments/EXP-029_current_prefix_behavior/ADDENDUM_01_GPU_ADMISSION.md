# EXP-029 shared single-GPU admission candidates

2026-10-17 HKT, Judge. GPU1 was subsequently occupied by a new unrelated ~20GiB process after EXP-028 released it. Its free memory is again below the threshold.

To avoid unnecessarily pinning a single-device experiment to one busy card, the Worker may admit **one GPU from [1,0,6,7], in that preference order**, under the existing >=40960MiB double/immediate pre-launch free-memory checks. This is an alternative-device choice, not parallel execution or extra GPU budget. Do not terminate or alter other jobs. All model/data/protocol/budget/cutoff constraints remain unchanged. No special environment or distributed inference is authorized.

CPU-preflight the scheduler selection and environment mapping before launch; CUDA_VISIBLE_DEVICES must identify the selected physical GPU and model code uses logical cuda:0. Register actual process identities with the overnight guard. Stop at the09:00 deadline even if no eligible card appears; report waiting honestly. If no device has enough memory, continue only authorized CPU work and wait.
