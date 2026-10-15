# EXP-026 / plan v5 Worker 交付：四卡 Stage2-fix EMA load-only

2026-10-16 HKT。**环境与四卡加载门控通过；没有运行模型 forward、训练、codec 或视频。** 详细证据见 [v5 Worker 报告](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v5/WORKER_REPORT.md)。

我在项目独立用户目录安装 CUDA 12.4 开发工具链，从官方 `flash-attn 2.8.3` 源码编译出与既有 `torch 2.6.0+cu124` 和旧 C++ ABI 相符的新扩展。两次构建累计 2,723.522 秒，峰值聚合 RSS 18.50 GiB，均在计划上限内；第一次因构建 PATH 缺 `ninja` 而停止，第二次成功。新 wheel 和 `.so` 的 SHA-256、来源、完整日志及资源记录分别见 [构建清单](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v5/BUILD_MANIFEST.json)、[工具链清单](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v5/TOOLCHAIN_MANIFEST.json)和[构建日志](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v5/BUILD_02.log)。隔离 overlay 的真实 CPU import 与 Diffusers Flash backend requirement 均通过，CUDA 在 CPU 门控进程中未初始化：[ENV_PREFLIGHT.json](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v5/ENV_PREFLIGHT.json)。共享 venv、官方 SolarWM checkout、torch/backend 和模型权重未改。

随后仅运行**一次** GPU0/1/6/7 的四卡 SP4/DP1 load-only。四个 rank 均通过严格 base 加载、物理 shard4/replica1 FSDP、624 个 LoRA 张量 / 2,075,394,048 参数、Stage2-fix EMA step3900 恢复及稀疏数值抽查，全部记录 `load_only_pass` 并释放。耗时 157.989 秒，收费 631.956 GPU 秒；每 rank 峰值 allocated 22.44 GiB、reserved 23.29 GiB，物理显存采样峰值每卡 24,283 MiB，任务 RSS 峰值 58.25 GiB。结束后四张目标卡均回到 0 MiB。[GPU 总账](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v5/gpu_attempt_001/GPU_LEDGER.json)与[rank 日志目录](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v5/gpu_attempt_001/)保留原始证据。v3 原有 105.265 GPU 秒不清零；v4 是 0；EXP-026 累计 737.221 GPU 秒。

**能力边界：** 这是加载/分片可行性证据，尚未执行 FlashAttention CUDA kernel、官方真实数据读取、模型 forward、采样、KV 或视频生成，不能称为 SolarWM Stage2 推理效果通过。Worker 未操作 Git 或更新 `progress.md`，等待 Judge 验收与同步。
