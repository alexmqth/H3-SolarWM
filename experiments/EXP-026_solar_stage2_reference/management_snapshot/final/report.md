# EXP-026/v2 Worker 交付：官方 SolarWM-H3 Stage2-fix 内存安全加载准备

2026-10-16 HKT。**v2 CPU 补丁和小模型检查已完成；真实 33B/FSDP/GPU 加载、官方样本读取和视频推理仍未通过。** 本轮授权 0 GPU 初始化、0 forward、0 下载、0 训练更新，实际执行均为 0。未修改冻结的 SolarWM checkout 或 `progress.md`，未操作 Git。

v1 的单样本 SP8 候选只有通用 YAML 解析通过，H3 backend 合同实际上拒绝固定 Stage2 的 SP8；而 SP8/DP1 下默认 FSDP 也不会分片 base。v2 改为官方 SP4/DP2、两条真实样本的候选配置，`load_config` 与 `validate_h3_config` 都通过，但数据权限尚未开放，真实 reader 未运行。

独立补丁在临时源码副本中通过 CPU 测试：严格 base 加载保持 CPU；LoRA 注入后只将 ignored LoRA 参数送往目标设备；调用官方已有的八张物理卡冻结 base 分片入口。小模型测试覆盖 BF16/FP32 dtype、旧整模型 `.to(cuda)` 故障注入、load→inject→wrap 顺序及小 EMA 数值拷贝。八进程 Gloo 测试确认现有分片组为 8、复制组为 1，全部未初始化 CUDA。**这些测试不等于真实 FSDP/GPU 通过。**

当前主机内存也不足以安全启动八份 33B：base+BF16 LoRA 的八进程算术下界约 524.75 GiB，审计时 MemAvailable 约 546.53 GiB，仅余 21.78 GiB 容纳加载/FSDP临时开销；GPU2–5仍有其他进程。提出下一阶段先满足八卡空闲和至少650GiB可用内存，再由Judge另行授权一次限时load-only检查，不涉及视频。官方数据集HF 401/ModelScope 403阻塞未变，也未重试。

[完整v2报告](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v2/WORKER_REPORT.md) · [补丁与CPU测试](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v2/README.md) · [load-only提案](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v2/GPU_LOAD_ONLY_PROPOSAL.md) · [v1冻结报告](submission/experiments/EXP-026_solar_stage2_reference/v1_frozen/WORKER_REPORT.md)。
