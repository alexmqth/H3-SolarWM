# EXP-028 admission clarification: sufficient free memory, not an empty card

2026-10-17, Judge, following the user's question about requiring a completely empty GPU.

An empty GPU is not a model requirement. It was a conservative scheduling rule. This addendum replaces the empty-card/no-other-process condition in taskbook_v1; all experiment inputs, calls, time budget and expiry remain unchanged.

Measured reference: EXP-027 completed recovery peaked at 23.748 GiB allocated / 28.504 GiB reserved; across its attempts the maxima were 24.449 / 29.137 GiB. These measure the previous cached-C2 workload, not EXP-028's Original-visible full17 backward. EXP-028 peak is still unmeasured. Provisionally require **40 GiB actually free** (40960 MiB from nvidia-smi) for its first profile, including allowance for potentially larger backward and runtime overhead. This threshold is an admission estimate, not a measured requirement or a guarantee against OOM.

The worker may share a candidate GPU with existing processes if free memory is at least40960MiB in two checks separated by >=10seconds and immediately before launch. Log other PIDs/memory for attribution; do not terminate, alter or evict them. Other processes need not be absent. Preserve the single-project lock, host-memory check, candidate list0/1/6/7, one-run limit, <=60-second polling frequency, admission deadline and failure-stop rules. Observe other occupancy when reporting performance; shared-GPU timings are not dedicated-GPU throughput measurements.

Update the admission controller and its CPU checks before starting/restarting it. Re-freeze changed source hashes in the preflight manifest. If a controller is already running, replace only this task's own waiting controller; never interrupt a launched experiment or unrelated processes. No additional GPU calls or retries are authorized.

At this review, the largest free memory was GPU1's22706MiB (~22.17GiB), below even the prior29.14GiB allocator reservation peak. Therefore this clarification does not currently make a GPU eligible. Once EXP-028 provides real peaks, Judge may lower future admission thresholds with measured margin.
