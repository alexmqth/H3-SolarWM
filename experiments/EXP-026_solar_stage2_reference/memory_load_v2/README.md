# EXP-026/v2 · CPU-first model loading patch

**Status: CPU boundary checked; real FSDP/GPU loading not tested.** The official SolarWM checkout remains untouched. [The patch](cpu_sharded_load.patch) applies to official source revision `ce1da4e7705391eda8eeda6016c0fd3f614b975e`; [source hashes](SOURCE_RECEIPT.json) make that condition explicit. The patch changes only `optional.load_transformer()` and the fixed-length inference preparation in `runtime.py`. It preserves the official SP4/DP2 Stage2 profile and uses the existing physical-node `frozen_base_shard_size=8` HYBRID_SHARD path already specified by the official Stage2 training config.

The v1 SP8/single-case candidate was **not** valid under `validate_h3_config`: fixed Stage2 inference requires SP4. With SP8, the default FSDP logical-DP group would also contain one rank, so CPU loading alone would not fix the GPU memory problem. [The v2 candidate](candidate_two_case.yaml) instead requests two authentic, source-balanced official cases. Both generic loader parsing and H3 backend validation pass. Real cases remain unavailable because the dataset downloads were denied; the config is not a reader success.

With `model.cpu_sharded_load: true`, the patch loads the strictly checked transformer on CPU, verifies that the frozen base remains on CPU, injects the same LoRA, moves only the ignored LoRA parameters to the local device, and then invokes the existing FSDP wrapper with eight physical base shards. EMA loading still follows FSDP and copies only LoRA tensors through the existing `load_initial_weights()` path. No attention, camera, RoPE, KV, solver or output code changes. The opt-in helper rejects a topology other than eight ranks, SP4/DP2 and two cases.

The [CPU boundary test](CPU_TEST.json) applies the patch to a temporary package copy and exercises the actual loader and preparation functions with tiny modules. It checks dtype/device routing, strict fallback behavior, load→inject→wrap order, a tiny EMA checkpoint and a fault injection for whole-model `.to(cuda)`. An eight-process **Gloo-only** check executes the existing physical shard-group function and confirms group sizes 8/1 with CUDA uninitialized; see [CPU_GROUP_TEST.json](CPU_GROUP_TEST.json). These are not FSDP model forwards. The [resource estimate](RESOURCE_ESTIMATE.json) warns that eight simultaneous CPU replicas alone require at least 524.75 GiB for base and BF16 LoRA, before temporary loading overhead. Observed available host memory at audit was 546.53 GiB, leaving only 21.78 GiB headroom; GPU loading must not be scheduled under that condition.

Reproduce without GPU from the project root:

```bash
python submission/experiments/EXP-026_solar_stage2_reference/memory_load_v2/make_patch.py
PYTHONPATH=SolarWM/src .venvs/solarwm-h3/bin/python submission/experiments/EXP-026_solar_stage2_reference/memory_load_v2/prepare_v2_config.py
.venvs/solarwm-h3/bin/python submission/experiments/EXP-026_solar_stage2_reference/memory_load_v2/test_cpu.py
OMP_NUM_THREADS=1 PYTHONPATH=SolarWM/src .venvs/solarwm-h3/bin/torchrun --standalone --nproc-per-node=8 submission/experiments/EXP-026_solar_stage2_reference/memory_load_v2/test_cpu_groups.py
```

[Worker report](WORKER_REPORT.md) states the remaining risks and [the load-only proposal](GPU_LOAD_ONLY_PROPOSAL.md) separates a future load check from actual video inference. This patch has not been installed in the official checkout or used to load the 33B model.
