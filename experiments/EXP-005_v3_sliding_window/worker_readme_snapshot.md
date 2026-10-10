# EXP-005 / v1 — V3 sliding-window CPU protocol check

**Execution: stage 1 completed (CPU only). Model capability: NOT_TESTED. Judge acceptance: pending.** This experiment prepares two independent candidates on the frozen V3 Baseline: SW-G keeps global positions and limits raw-video KV to five ancestor chunks; SW-L additionally remaps retained video positions to a native local window at read time. Neither candidate has produced a model video.

The parent is the accepted Original H3 + released action LoRA, Single I0, current-prefix feedback, native FM 30-step strict causal KV path from [EXP-002](../EXP-002_native_cached/README.md) and [EXP-003](../EXP-003_native_cached_124/README.md). Its 124-frame result remains the reference; this CPU task does not alter it.

## What was checked

- Explicit latent spans `[0,12), [12,17), …` and exact five-ancestor indices. At index 6, the first eviction changes history from chunks `0–4` to `1–5`; after index 7 it is `2–6`.
- Frozen `H3ChunkCache` commits, layer completeness, row counts, capacity, sigma-0-only clean commit, cache identity during reads and append-only RGB stitching, using small CPU tensors.
- For old indices 2–5, SW-G gives **exact CPU fp32 attention output** against the frozen EXP-003 interval function under the same toy inputs, monkeypatched lightweight model function, and `current_prefix_feedback()` context. The test also compares the actual model-call arguments: current, audio, prompt, anchor, packed positions/action rows, both timesteps, prefix-time flag, own-action routing, action feedback, frame start and prefix length. This is not a 33B-output regression.
- SW-L uses the frozen H3 `(1,4,4,4,4)×5/3` temporal grid, leaves prefix coordinates untouched, changes only retained/current video coordinates after eviction, and stores canonical global RoPE in raw KV. Actual H3 MM-RoPE shows that changing video positions also changes video-to-prefix logits. It is not equivalent to recomputing past hidden states.
- The real parking packed 37-latent input obeys the native grid and is physically trimmed to visible actions/video. Rebuilding a 42-latent packed sequence with the released builder moves old action/I0/video coordinates; it is **not** a safe long-input extension.
- `interval_sw` now fails closed if the accepted external current-prefix feedback context is absent. Post-37 calls are restricted to CPU structural probes until a native long fixture and GPU authorization exist.

`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python -m pytest -q submission/experiments/EXP-005_v3_sliding_window/test_contract.py` → **14 passed in 4.24 s**. [Raw log](artifacts/cpu_tests.log). No 33B load, VAE, GPU forward, training, or video generation occurred.

## Decision and remaining blocker

CPU implementation correctness is supported only within the tested scope. A certified >37-latent native conditioning fixture is still missing. The new action embeddings/rows and positions must extend the original 37-latent document without moving its old text, I0, audio, video, or action coordinates; the old prompt embeddings and first 37 noise latents must also be identical. A structural toy layout cannot satisfy this model-input requirement. The stage-two runner must implement and audit that extension before any post-37 GPU call. The draft [GPU plan](GPU_PLAN.md) is **not authorized**; the user's 2026-10-10 overnight 8-GPU availability (then at most three from 2026-10-11 09:00 HKT onward) is a scheduling fact, not a change to this task's zero-GPU budget.

See [PROTOCOL.md](PROTOCOL.md) for exact conditions, [MANIFEST.md](MANIFEST.md) for source hashes and provenance, [metrics.json](metrics.json) for machine-readable stage-one status, and [FUTURE_ANYFLOW.md](FUTURE_ANYFLOW.md) for a separate future study. Neither AnyFlow nor DMD is part of EXP-005 stage one.
