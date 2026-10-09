# Diagonal AnyFlow compute probe — isolated, no training change

CPU: 20 actual small-H3 cases passed exact raw loss/weight/all parameter gradients, unchanged CPU RNG and cached K/V/RoPE. FP32/BF16, detached/full history, gradient checkpoint/offload, diagonal and off-diagonal cases covered. See `cpu_equivalence.json`.

GPU4: one trained33B AnyFlow step32, A/chunk2, full history, logical batch4, train shift12. Two identical reference backward passes measure CUDA repeatability, followed by one candidate backward. No optimizer update. Inputs and code frozen in `manifest.json`; trained weights remain in source outputs and are not copied here.

At exact r=t, the derivative coefficient is zero. Candidate skips its three no-grad evaluations; current forwards become 10 instead of16 per four-sample logical batch. Off-diagonal cases remain unchanged. Skipped derivative norm is null/unmeasured, not falsely recorded as zero.

GPU result is complete: see [RESULTS.md](RESULTS.md). All common sample values match; shortcut gradient differences are comparable to repeated-reference differences. This is not a bitwise CUDA proof. Compare candidate gradient differences against repeated-reference differences, and report compute timings as a single shared-host observation. This is not a quality experiment, does not satisfy Stage1 acceptance, and has not modified the main trainer or live frozen runtimes.
