# Action–Video role separation: Judge assessment

2026-10-17. Research hypothesis and bounded preparation, not an accepted method or GPU authorization.

## Decision

Adopt action–video interaction under causalization as the leading research question. Add optimization interference as H3 alongside information-path mismatch (H1) and historical-state inertia/interference (H2). Do not require the final method to combine routing and dual adapters. A useful result can support one mechanism, both, or neither.

The strongest potential claim is an interaction: causalization changes learned action–video dependencies and adaptation dynamics in a measurable way; a targeted intervention restores held-out control and continuity at matched resources. Token-specific low-rank adapters alone do not establish novelty. Full related-work comparison remains incomplete.

## Verified evidence and citation limits

- The local October 10 routing audit reports delta-cosine means 0.3354 / 0.1175 / 0.6400 for public feedback removal / video causalization / clean frozen commit. These are numerical action-sensitivity changes, not measured losses of semantic control.
- `code/causal/h3_cached.py` requires no-grad commits and detaches copied tensors. EXP-023's `make_cache` is reused by EXP-025/027. Their future C2 loss does not backpropagate into the historical C1 writer computation.
- EXP-025 uses 50 fused-QKV rank-8 increments, each with shapes (8,6720) and (20160,8): 10,752,000 parameters. Two copies would have 21,504,000. Parameter counts do not predict activation memory or backward cost.
- The primary H3-World abstract at https://arxiv.org/abs/2609.01560 was retrieved on 2026-10-17. It confirms language-based character/camera control and temporal attention routing. Its paper claims do not establish our proposed mechanism.
- SGF+ quoted angles 104.2/106.3 degrees and CAER's exact identity/claims have not been independently verified in this review. API timeout/rate limiting is not evidence that these papers do not exist. Treat them as leads, not established comparisons. CVPR edition/deadline remains unverified.

## Essential conceptual corrections

### 1. Token roles are not independent task losses

For a linear projection Y=X W^T and a single loss L, the local occurrence contribution from role r is G_r=D_r^T X_r, where D=dL/dY. Include action, current video, prefix/other rows and every invocation of shared weights. Only roles executed in the differentiable graph contribute. The role sums must reproduce the ordinary gradient within a declared numerical tolerance.

An action row can already contain visual information after attention, and a video row can contain action information. Consequently G_action and G_video are not automatically gradients of an action-accuracy loss and an appearance loss. A negative dot product diagnoses local cancellation between uses of a shared projection, not a complete semantic mechanism.

### 2. Negative cosine is insufficient for a decoupling decision

Report norms, relative role magnitudes, per-layer/Q/K/V distributions and the fraction of negative alignments across clips/sigmas. Near-zero gradients have undefined or unreliable cosine. Equal role averaging or per-token normalization changes the question and must be reported separately from the true loss gradient.

Compare a matched Original-visible reference and correct V3 where feasible, rather than measuring V3 alone. Negative alignment may already occur in a successful bidirectional model. Relevant evidence is a reproducible change associated with causalization and actual behavioral weakness, followed by intervention benefit.

Even for two contributions with negative dot product, a shared SGD step may still improve both local role contributions: the first-order terms depend on norm-squared plus the cross term. Adam preconditioning changes update geometry further. Do not infer optimization failure from a histogram of negative cosines.

### 3. Diagnose the parameter space that the proposed method can change

For W=W_frozen+s B A, map each role's dense gradient to dL/dB=s G_r A^T and dL/dA=s B^T G_r. Full-weight geometry and low-rank trainable geometry can disagree. At the standard B=0 initialization, dL/dA is zero; it is not evidence of absent action influence. Released Action LoRA stays frozen and is distinct from new role adapters.

Use bounded layerwise accumulation or an explicitly validated sketch; do not retain 50 dense fused-QKV gradients or all token activations. One dense fused-QKV gradient has 135,475,200 entries (about 0.505 GiB in FP32), before role copies and activations. Frozen weight `.grad is None` cannot answer the question; hooks/functional probes must retain the required activation derivatives without optimizer updates.

### 4. Detached history and current-token roles are separate questions

In V3's existing detached-cache objective, historical writer contribution is absent from the graph by design, not measured to be zero in the underlying model. Recomputing history differentiably adds a training path and must be a separately declared comparison. Future supervision of a writer requires a differentiable path or an equivalent gradient estimator/replay; it does not follow from merely adding a writer LoRA, nor does it require one particular replay implementation.

Current action/video role separation can be studied without restoring writer gradients. Do not conflate these two interventions or describe the existing C2 objective as SGF.

### 5. A legitimate causal method cannot preserve every original path

Past output/state must not depend on newly arrived actions. Original bidirectional paths through rewritten past representations cannot simply be restored. A candidate needs a lawful current-block conditioning path and a precise cache contract, including action/prefix, refiner, position and indirect multilayer dependencies. First distinguish accidental omissions from unavoidable changes in the causal factorization.

## Minimal evidence sequence and ROI

1. Use existing strong and weak ABot behavior examples to select a small paired behavior check. Same history/action intervention/noise within each comparison; label character and camera outcomes separately. A 17-frame continuation tests local response only, not long-term control or full recovery from delayed response. No broad scene sweep yet.
2. Prepare one bounded role-gradient diagnostic on the same training objective and true recorded action labels. Synthetic A/D with the same factual target is an off-policy probe, not equally valid supervision for both actions. First validate the decomposition and LoRA mapping on CPU; then propose exact GPU forwards/backwards and memory limits. Use few representative layers initially if collecting all layers is expensive.
3. Only if the evidence is useful, compare correct V3, shared rank-8, parameter-matched shared rank-16 and two rank-8 roles with matched data exposure/update/NFE. Specify prefix role assignment. If routing also changes, include routing-only and combined conditions; do not start this matrix before a useful pilot.
4. Require actual action/continuity improvements to justify role separation. If a correct baseline removes the problem, or conflict does not predict intervention benefit, stop this mechanism claim. No indefinite cosine optimization, added modules or short-training MSE chase.

## Immediate handoff

Original Exp Worker: CPU-only preparation, at most 15 minutes / 4 CPU workers / 8 GiB RSS; no GPU, downloads, training updates or new videos. Deliver `ROLE_GRADIENT_SPEC.md`, a small CPU decomposition/LoRA check with output, and a proposed bounded GPU budget. Do not build/load the 33B model or implement full dual-adapter training. Judge owns interpretation and authorization.
