# EXP-026/v3 · four-card Stage2-fix load-only evidence

This directory contains an independent four-card patch, the CPU gates and the **one real GPU attempt**. The attempt failed at FlashAttention backend selection after checkpoint shard loading progress reached 14/14. It did not reach FSDP wrapping or EMA restore. No video was generated.

- [Worker report](WORKER_REPORT.md): exact conclusion, failure point, resource usage and preparation deviation.
- [Candidate config](candidate_one_case.yaml) and [official diff](CONFIG_DIFF.json): SP4/DP1 with one slot, CPU-deferred local-only loading; H3 backend contract passes but official data is absent.
- [Patch](cpu_sharded_load.patch), [source hashes](SOURCE_RECEIPT.json) and [isolated patched package](patched_src/src/solarwm): official checkout remains unchanged.
- [CPU boundary result](CPU_TEST.json) and [four-rank Gloo group result](CPU_GROUP_TEST.json): implementation checks, not real CUDA FSDP.
- [GPU launch](gpu_attempt_001/GPU_LAUNCH.json), [outcome](gpu_attempt_001/GPU_OUTCOME.json), [ledger](gpu_attempt_001/GPU_LEDGER.json), [rank logs](gpu_attempt_001/) and [environment diagnosis](ENV_DIAGNOSIS.json).

The real attempt used physical GPUs 0/1/6/7 for 26.316 seconds and then released them. No repeat is authorized by EXP-026/v3. The original H3-World text-action experiments and official SolarWM camera-data protocols remain separate.
