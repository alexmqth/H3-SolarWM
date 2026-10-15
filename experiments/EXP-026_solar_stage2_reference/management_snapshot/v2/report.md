# EXP-026/v1 Worker 交付：官方 SolarWM H3 Stage2-fix 单样本 CPU 准备

2026-10-16 HKT。**状态：CPU 预检完成，官方输入访问受阻；未运行 GPU 推理，也没有官方输出视频。** 本任务授权 0 GPU。EXP-025 已由 Judge 验收，未在本轮重跑。

官方 Stage2-fix EMA 与 33B base 文件存在。官方配置经必要本地路径、单样本和 SP8/DP1 改动后可由官方 loader 解析；但 HF 测试索引下载返回 gated 401，ModelScope 主数据集和 H3 latent 数据集返回 403，实际 sample ID、index row、payload shard、相机轨迹和 reader 结果均未知。下载 0 字节、GPU forward 0 次、训练更新 0 次。默认拒绝 GPU 的启动入口已通过负向检查。

另有独立的 GPU 前置阻塞：目前官方 H3 runtime 在 FSDP 包裹之前把完整 61.73 GiB transformer 加载到单卡；L40 总显存仅 44.99 GiB。SP8 解决单样本调度的 DP 整除问题，但不会分片这一步。因此不能把候选配置或八张卡空闲解释为官方推理已可执行。需先获得官方数据访问、完成真实 CPU reader，再经另一个获批任务验证内存安全加载方案和 GPU 预算。

[完整 Worker 报告](submission/experiments/EXP-026_solar_stage2_reference/WORKER_REPORT.md) · [输入清单](submission/experiments/EXP-026_solar_stage2_reference/INPUT_MANIFEST.json) · [CPU 预检](submission/experiments/EXP-026_solar_stage2_reference/CPU_PREFLIGHT.json) · [配置差异](submission/experiments/EXP-026_solar_stage2_reference/OFFICIAL_CONFIG_DIFF.json) · [下一阶段条件](submission/experiments/EXP-026_solar_stage2_reference/GPU_PROPOSAL.md)。Worker 未操作 Git，也未改 `progress.md`。
