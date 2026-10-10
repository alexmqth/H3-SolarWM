# EXP-005 protocol: frozen CPU stage and separately authorized GPU stage

This document preserves the v1 CPU contract. The executable v2 GPU authorization is [next_plan.md](../../../next_plan.md), with per-stage SHA-bound authorizations and source manifests in [stage2](stage2/). The original v1 document is preserved byte-for-byte in [previous_stage1](stage2/previous_stage1/PROTOCOL.md).

## Frozen parent contract

Use Original H3 and released action LoRA, native Single I0, own-action routing with action feedback and **current non-action prefix feedback**, video sigma ×1000, audio timestep 1000, `fixed_prefix_timesteps=False`, 30-step native FM/Euler shift 2.22, global RoPE, generated history, clean sigma-0 commit and CPU raw KV. Partition is `[12,5,5,5,5,5,5,5]`; history holds the latest five complete chunks. Future action/video rows must be physically trimmed before the model or its text refiner.

`interval_sw` must run under `torch.no_grad()` and the frozen `current_prefix_feedback()` context manager. The entry checks the active attention method and refuses a forward if that context is absent. This explicit fail-closed requirement prevents silent fallback to the older prefix topology. The check is specific to the frozen context implementation/hash in the manifest; a future refactor requires revalidation.

SW-G preserves all global positions. SW-L is identical through index 5. From index 6 onward, the oldest retained ancestor's latent start is the local temporal origin. The H3 native nonuniform temporal grid is rebuilt for retained video only. Prefix positions remain global. Raw K/V stay frozen; local RoPE is read-time metadata, while commits store canonical global RoPE. This position change can affect prefix/video attention and is not a pure cache-memory ablation.

## Stage-one evidence boundary

Tests use small CPU K/V and actual H3 RoPE and packed-layout functions. The real parking input is read only as a 37-latent CPU fixture. The model function is replaced in numerical regression tests, so passing means attention/protocol regression under tested CPU fp32 backend, **not** full 33B generation equivalence or image/action quality. The structural 42/47-latent layouts carry dummy action text and must never be passed to the real model.

At index 5 `[32,37)`, EXP-003 ended without committing C6. Stage two must commit that clean endpoint exactly once to trigger eviction before index 6 `[37,42)`. Required per-layer ancestry: C6 sampling `0–4`, C7 sampling `1–5`, C8 sampling `2–6`. The cache guard rejects missing layers, mismatched rows, duplicate commit counts and empty-cache false positives.

The frozen H3 DiT uses three-axis MM-RoPE; this path does not implement SolarWM camera PRoPE. SW-L therefore tests a local MM-RoPE geometry candidate only. The original packed builder cannot simply be rerun at 42/47 latents: changing total text length and the mirrored video origin shifts old positions. A valid extension must keep old semantic coordinates and embeddings, audit new action rows/positions and future-action isolation, then separately validate audio and new noise. The earlier 37 noise latents must be copied, not regenerated from the same seed in a larger tensor.

Only history **video KV** is bounded by W5. Known action prefix, all retained latent/RGB outputs, VAE prefix decoding and temporary current KV can still grow or peak separately. No end-to-end memory bound is claimed.

## Handoff gate

Before GPU work, produce a certified native >37 input fixture, frozen source/input/checkpoint hashes, exact hardware/backend and a separate approval with budget. G0 old-vs-SW-G same-state/full-model regression comes first; only then consider SW-G C7/C8. SW-L requires a separate decision after SW-G. [GPU_PLAN.md](GPU_PLAN.md) is the original proposal, not executable authorization.

**Historical v1 gate, now superseded:** the CPU phase had zero GPU authorization. The certified 47-latent fixture and the separately approved G0/G1/L1 v2 stages are recorded in [stage2/README.md](stage2/README.md), their `*_authorization.json` files, and their metered `result.json`/`budget.json` files. Later availability of more GPUs does not increase the frozen EXP-005/v2 limits or authorize C9, training, AnyFlow, or DMD.
