# EXP-028/v1 Worker report: Action–Video LoRA role gradients

2026-10-17 HKT. **Execution complete, mechanism claim not established; pending Judge review.** The recovered pilot completed two factual ABot clips × two sigma values × Original-visible/V3, measuring LoRA-B gradients in the fixed rank-8 subspace at layers 0 and 49. It used Original H3 plus released Action LoRA, EXP-025 `checkpoint_000.pt` A factors and exactly zero B factors. There were no optimizer updates, videos, VAE/text encodes, AnyFlow or DMD calls.

## Protocol and provenance

Both branches use the same clip, I0, action rows, C2 noisy latent, C2 FM target, sigma and CPU noise seed (113 for 0.25; 127 for 0.75). Original-visible jointly recomputes clean GT C1 and noisy C2 under Original Single-Egress bidirectional attention, marks C1 fixed at native clean video time, and applies the loss only to C2. V3 reads detached sigma-0 GT C1 raw KV through `own` action routing, current-prefix feedback and Global RoPE. Original-visible has no persistent KV. The two graphs also differ in prefix state and video topology, so this is an operational graph comparison, **not** a single-edge causal ablation. Both use 17 visible latents and physically exclude future action/video rows beyond C2. The separately audited [Original clean reference](original_clean.py) replaces `local_transition.transition_forward`, whose same-sigma noisy history did not match the approved clean-history question.

[CPU_PREFLIGHT.json](CPU_PREFLIGHT.json) records the SHA of code, released Action LoRA, EXP-025 checkpoint, both encoded clips and model config/index. The 13 base transformer shard sizes and locations are already recorded in [EXP-020 source manifest](../EXP-020_original_mixed_action_reference/SOURCE_MANIFEST.json); this pilot did not rehash all 62 GiB of base shards. The CPU gate verified two real layouts, Single-Egress mask directions, the clean-history fixed-time wrapper, exact zero-B output, BF16 cast boundary, checkpoint cotangent deduplication, and the 14th-DiT/8th-backward reservation boundary. CPU preflight is implementation evidence only.

The first admitted GPU1 process failed on `ModuleNotFoundError: run` before CUDA device setup, model load or any DiT/backward. Its exact [log, controller state, preflight and source snapshot](attempt00/) are preserved. It lasted 3.268 seconds by controller timestamps, conservatively charged 4 seconds. Under [startup recovery authorization](ADDENDUM_02_STARTUP_RECOVERY.md), the corrected process ran once on shared GPU1, UUID `GPU-c4c2b3db-791b-ecdc-978f-d56295593124`, from 17:19:46.799 to 17:29:27.330 UTC. Controller PID 1291827, timeout PID 1294061, Python PID 1294062; start ticks/PGID were registered in the project [process registry](../../research/overnight_20261017/PROCESS_REGISTRY.json). Another process (PID 1964241, about 2.8 GiB) shared the card, so elapsed time is not a dedicated-GPU throughput measure.

## Measurement gates and cost

| Item | Observed |
|---|---:|
| Top-level DiT attempts | 14 = 2 C1 commits + 8 measured + 4 unwrapped identity references |
| Backward / optimizer updates / videos | 8 / 0 / 0 |
| Recovery process wall time | 580.532 s, scheduler launch to exit |
| Recovery in-process ledger time | 572.230 s; excludes 8.302 s of startup/exit outside the ledger |
| Cumulative actual process wall time | 583.800 s, including first failure's 3.268 s |
| Cumulative conservatively charged time | 584.532 s, replacing the first failure with 4 s |
| Peak allocated / reserved VRAM | 23.753 / 26.770 GiB |
| Peak process RSS recorded | 55.654 GiB |
| Max wrapped-vs-unwrapped velocity relative RMS | 0.0 across four references |
| Max role-sum vs autograd `B.grad` relative RMS | 6.701 × 10⁻⁷ across 16 layer measurements |
| LoRA-A gradient at B=0 | exact structural zero in all 16 measurements |

Full per-clip, sigma, graph, layer and Q/K/V role counts/norms and cosines are in [ROLE_RESULTS.json](recovery01/ROLE_RESULTS.json); [SUMMARY.json](recovery01/SUMMARY.json) and [GPU_LEDGER.json](recovery01/GPU_LEDGER.json) give paired differences, resource costs and call sequence. All eight measurements passed the fixed numerical gates. The runner retained rank-space role summaries only; **exact role dot products and cross-graph gradient matrices were not persisted**. They cannot be reconstructed exactly from the saved norms/cosines, and this item of the intended deliverable remains missing. No extra GPU pass is authorized to fill it.

## Observed geometry

Mean action-versus-video LoRA-B cosine over the four clip/sigma states (aggregate action and video roles, within each graph):

| Layer / QKV | Original-visible | V3 |
|---|---:|---:|
| L0 Q | +0.0024 | +0.0048 |
| L0 K | +0.0003 | −0.0095 |
| L0 V | −0.0004 | +0.0074 |
| L49 Q | undefined (action Q exactly zero) | undefined (action Q exactly zero) |
| L49 K | +0.0089 | +0.0044 |
| L49 V | −0.1075 | −0.1119 |

The L0 cosines are near zero in both graphs. L49 V is modestly negative in both, with similar means; V3 is **not consistently much more negative** than Original. Current action role norms are generally smaller than video-current and non-action-prefix norms in this factual FM loss, but the two clips and zero-B rank-8 subspace do not establish a general optimization bottleneck. V3 history action/video writer-role gradients are structurally zero because C1 KV was detached; history can still affect current gradients through reads. In Original-visible, layer 0 historical roles contribute because the full window is recomputed. At layer 49, historical action roles and action Q are zero under the final-layer Single-Egress/current-output geometry. Undefined Q cosine is not a measured conflict.

For each paired clip/sigma, V3 factual C2 FM loss exceeded Original-visible (0.2527 vs 0.2309, 0.1013 vs 0.0762, 0.3215 vs 0.3052, 0.1158 vs 0.1008). Because the graphs differ in topology and history computation, this is neither a single-factor effect nor a video-quality verdict. There was no A/D counterfactual generation or person/camera assessment here. A negative or near-zero LoRA-gradient cosine alone cannot prove action loss conflict or justify training dual adapters; related work already allows dual LoRA roles.

## Decision and next check

This bounded sample **does not support** escalating to more gradient layers, larger LoRA, dual-adapter training, AnyFlow or DMD. The most informative next behavior check is the [proposed one-edge C2 A/D pilot](BEHAVIOR_PILOT_PROPOSAL.md): same C1 raw KV and C2 noise, V3 vs only disabling current non-action-prefix → video feedback, two initial states, short 17-frame C2 clips, character and background motion judged separately. CPU audit shows the one-line attention edge path exists, but the current condition files lack a matched D counterfactual prompt; fixed-layout construction/text encoding must be verified first. This is **only a recommendation**, not an approved or launched GPU experiment.

Worker changed only EXP-028 files and root `report.md`; no Git operation, `progress.md`, `next_plan.md` or frozen model code was modified.
