# Role-gradient 33B pilot resource draft — **not authorized**

2026-10-17，Exp Worker。This document supplies a possible bounded GPU task for Judge review. Current authorization is CPU-only. Do **not** run it from this document.

## One decision, two factual clips

The pilot would ask whether token-occurrence role gradient geometry changes between an Original-visible window and the correct V3 detached-history objective on the **same factual ABot C2 FM loss**, and whether the observed geometry is large enough and consistent enough to justify a later behavioral/optimization intervention. It cannot attribute differences solely to causal attention or establish action/visual-task conflict.

Proposed fixed train examples are the already encoded, separate episodes `43866101158a538d22c52ac5b2fa606b_src1245_rgb56` and `b784d995827f53977a8067e5b9dc4b5e_src1635_rgb56` from [EXP-023 encodings](../../experiments/EXP-023_v3_fm_small_data/encoded/). Their **recorded** C2 action scripts and source C2 video latents supply the condition and FM target. EXP-014's synthetic A/D branches indicate a strong and a weak model-generated response on related source windows, but are **not factual labels for this loss**, nor automatically matched to this GT C1 state. CPU provenance must verify exact source windows, I0, script, packed spans, input SHA and episode split. If any selected encoding or recorded action is unavailable, stop; no opportunistic replacement.

Compare two operational graphs only:

1. **Original-visible reference:** frozen Original H3 + released Action LoRA, visible GT `[C1,C2]` window with bidirectional Original routing, C2 FM loss only. It is not full Original inference with unknown future clips.
2. **Correct V3:** same frozen weights, native Single I0/current-prefix and strict chunk-causal C2, GT C1 `sigma=0` no-grad clean commit into detached CPU raw KV; same C2 factual action, target, noise/sigma, Global RoPE and backend precision. Its C1 writer is absent from the C2 gradient graph by design.

This pair changes topology **and** prefix/history hidden-state context. It can compare operational gradient geometry, not isolate which edge causes it. A subsequent edge-only comparison would require its own authorization. No new FM/visual/action adapter checkpoint is loaded. Add a **zero-residual** rank-8 QKV diagnostic LoRA at frozen blocks 0 and 49 solely to create a measurable `B.grad`; optimizer is not instantiated or stepped. Preflight must prove model output identity at B=0 and correct `56×3×128` QKV head/role mapping. Use the same source latent and independently fixed CPU noise seed113 at sigma .25 and seed127 at sigma .75 for both graphs. No counterfactual A/D receives the real target in this pilot.

## Exact ceiling and order

| Activity | Top-level 33B DiT attempts | Backward | Other |
|---|---:|---:|---|
| V3 GT C1 clean commits, one per clip (reused across two sigmas) | 2 | 0 | 2 CPU raw KV states, no writer grad |
| Original-visible C2 factual FM, 2 clips×2 sigma | 4 | 4 | C2 loss only; 17-latent visible window |
| V3 C2 factual FM, 2 clips×2 sigma | 4 | 4 | same target/noise as reference |
| **Hard total** | **10** | **8** | **0 optimizer updates, 0 VAE/text encode/decode, 0 new video** |

Start with the first clip and one sigma under both graphs, check hook count, role partition, `Σ_role g_B,r` vs autograd B.grad and memory **before** the remaining pairs. Its calls consume the 10-call cap. If a graph cannot capture a unique gradient-bearing invocation under checkpointing, or is OOM, stop and preserve the partial ledger; do not retry with new layers/clip or quietly disable role groups. Activation checkpoint recomputes inside backward count in elapsed time and should be separately instrumented, not presented as additional top-level sampling calls. A full preflight must freeze exact code/data/model SHA, command, GPU UUID and output directory before any GPU start.

Suggested single empty L40 hard caps: **1,800 GPU-process seconds**, **10 top-level DiT attempts**, **8 backward**, allocated VRAM **<40 GiB**, process RSS **<96 GiB**, at most 4 CPU workers, no training step or media generation. The frozen EXP-027 C2-only run peaked around 23.75 GiB allocated, but Original-visible full17 backward plus hook/offload storage may be materially larger; this draft does not assert it fits. Load/cleanup, failed calls and CPU offload are charged. Keep only two selected layers' local X/D, stream full-weight role dot products in output-row tiles (e.g. 256 rows) and drop each graph after its backward. No persistent 50-layer dense gradients or all-token tensors. The dense fused-QKV matrix at runtime is 115,605,504 FP32 elements (~0.431 GiB) **per layer before four role copies**; full replication is disallowed.

## Required outputs and interpretation gate

Per clip/sigma/graph/layer/Q-K-V: row counts for current/history action and video, prefix and other; raw role norm and pairwise cosine/dot with numerical floor; LoRA-B role map norm/cosine; aggregate/autograd checksum; checkpoint recompute and detached-commit occurrence counts; GPU/CPU memory and elapsed time. At B=0 all LoRA-A role gradients are zero by chain rule and their cosine is undefined. Report missing history-writer path as `NOT_IN_GRAPH`, not a measured zero. Store summary statistics and bounded hashes, not large activation dumps.

Before inferring H3 optimization interference, Judge must see a **separate behavior check** on matched current action, I0, history and noise, with character movement and camera movement annotated separately. Existing ABot videos may help select clips, but EXP-014 synthetic A/D is not the real action label and EXP-019/020 protocols differ. A 17-frame C2 continuation supports only local response; delayed switching and long-horizon recovery need later sequences. Any additional 30NFE A/D video budget, checkpoint adaptation, shared-rank16 vs dual-rank8 comparison, or routing-only ablation must be a later explicit task. Negative gradient cosine alone, or one layer/one sigma, cannot trigger dual-adapter training. If role geometry is inconclusive or similar in both graphs, stop this candidate rather than enlarging the cosine survey.
