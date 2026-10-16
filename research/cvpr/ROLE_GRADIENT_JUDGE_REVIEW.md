# Role-gradient CPU preparation: Judge verdict

2026-10-17. **ACCEPTED_CPU_PREPARATION; H3_MECHANISM_NOT_TESTED; GPU_NOT_AUTHORIZED.**

Judge inspected the specification, toy source, saved result and GPU draft. An independent CPU rerun in a temporary directory produced identical case metrics and source SHA, with CUDA uninitialized; the original Worker result was not overwritten.

Verified scope: exact local dense-gradient decomposition for two uses of one linear projection, role sums including prefix/other, LoRA chain rule at zero and nonzero B, absence of detached-writer gradients, and a negative-dot counterexample with favorable first-order shared-step terms. Maximum dense sum error 4.44e-16 and LoRA-B error 5.55e-17. This is algebra validation on a tiny FP64 graph, not validation of H3 capture hooks, mixed precision, recomputation deduplication, or role separation benefits.

The proposed 10 DiT / 8 backward / 1,800 GPU-process seconds budget remains a draft. Before execution:

- Quantify capture and reduction compute as well as storage. Tiling reduces memory but does not eliminate the cost of dense role products over tens of thousands of tokens. A CPU reduction of all full-weight products may dominate this budget. Choose exact selected projections or a validated sketch with clear approximation error if necessary; include all extraction time.
- Enumerate any zero-wrapper identity forwards within the hard call ceiling. The draft's main table has no separate calls reserved for an empirical identity comparison. Algebraic identity and a measured production regression are different evidence.
- Preserve the declared operational comparison: Original-visible versus V3 changes prefix, history and topology together; resulting gradient differences do not isolate causalization or a particular edge. Role LoRA geometry at zero initialization is local to that chosen subspace.
- Link the proposed diagnostic to a concrete method decision and behavior evidence. Negative cosine by itself cannot trigger a dual-adapter training run. Avoid expanding this into an all-layer gradient survey.

CPU preparation is closed. No follow-up implementation or GPU task is implicitly authorized. Literature differentiation and a bounded behavior/diagnostic decision remain the next research planning items.
