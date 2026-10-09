# AnyFlow implementation provenance

Reference: [SolarWM](https://github.com/Junchao-cs/SolarWM), revision
`ce1da4e7705391eda8eeda6016c0fd3f614b975e`, Apache-2.0.

- `anyflow_reference.py` vendors `src/solarwm/training/anyflow.py` with an attribution header; mathematical code is unchanged.
- `anyflow.py::anyflow_sample_loss` ports `src/solarwm/backends/minimax_h3/anyflow_loss.py::h3_anyflow_v15_loss` to the H3-World callback's **noise-clean** convention. SolarWM converts its native **clean-noise** velocity first; H3-World's `model_fn_minimax_h3` already applies that sign conversion.
- `H3AnyFlowConditioner` follows the official cloned target-time MLP and fixed 0.25 mixing gate. It uses DiffSynth's native time-embedding module and copies materialized weights into an independent FP32 module, without copying offload wrappers.
- Packed rows are deduplicated by **(current time, target time)**. Text, image anchor, audio and clean-history target times stay on their own current-time diagonals; only denoised video rows receive the requested target time.
- Physical batch 1 is accumulated into logical batches of 4: 2 diffusion samples (`r=t`), 1 endpoint sample (`r=0`), 1 general map sample (`0<r<t`). Adaptive rescaling uses the two raw diffusion losses of the SAME optimizer batch, before Gaussian weights. No stale EMA loss reference is substituted.
- Three detached model evaluations construct the finite-difference target before one trainable prediction. Clean-history KV is reused across these calls and rebuilt after each optimizer update.

Deliberate scale/protocol differences from the official run: H3-World keyboard action routing, 39-frame teacher-generated pseudo-GT, tail QKV rank-8 rather than all-block rank-384, shift=2.22 rather than 12, existing RGB anchor, detached clean-history KV, physical batch 1 with local accumulation rather than distributed global batch 128. These differences are recorded; this is a TF-AnyFlow objective port, not a claim to reproduce the released checkpoint.

`tests/test_anyflow.py` numerically compares loss and parameter gradients against the official H3 loss on diagonal, endpoint, general-map and time-boundary cases. It also checks actual H3 packed time rows, target-time sensitivity, KV non-mutation, gradient checkpointing, optimizer paths, and exact checkpoint reload.

The SolarWM license is reproduced in `SOLARWM_LICENSE`. Synthetic CPU results validate implementation only; generated-video quality requires pretrained 33B evaluation.

2026-10-08 optimizer-policy correction: the official H3 Stage1 runtime enables the cloned delta embedding, then freezes the transformer and optimizes only the block/refiner QKVO/FFN LoRA parameters. The initial local pilot additionally trained the target-time MLP. Its mathematical loss parity does not establish optimizer-policy parity. A frozen-time controlled run is prepared; existing results remain labeled as the trainable-time variant.

Mixed-precision qualification: the local original time MLP computes in BF16 and the cloned target-time MLP in FP32. Diagonal loss identity is algebraic; initial model conditioning is not bitwise identical to the original BF16 path. A CPU probe using the real step00 weights at10 native times found relative embedding L2 differences0.00109–0.00353. This is not a CUDA output-error bound or an established cause of video failure. Official H3 preserves FP32 time-embedding units. Existing controlled runs keep this implementation fixed.
