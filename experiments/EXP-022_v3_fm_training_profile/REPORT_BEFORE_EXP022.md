# EXP-021/v1 Worker 执行报告：CPU 准备完成，待 Judge 验收

2026-10-15 HKT。根据 [当前任务书](../../../submission/experiments/EXP-021_v3_fm_baseline_preparation/taskbook_v1.md)完成普通 FM 因果适配的 CPU 准备；没有进行 GPU 调用、训练更新、33B 推理或新视频生成。EXP-019/020 已由 Judge accepted 并冻结；此前根报告的内容保存为 [历史快照](../../../submission/experiments/EXP-021_v3_fm_baseline_preparation/REPORT_BEFORE_EXP021.md)，仅将相对链接调整为新目录可访问。本轮只修改 EXP-021 目录与本文件，没有 Git 提交/推送。

[完整 Worker 报告](../../../submission/experiments/EXP-021_v3_fm_baseline_preparation/WORKER_REPORT.md) · [目录入口](../../../submission/experiments/EXP-021_v3_fm_baseline_preparation/README.md) · [数据清单](../../../submission/experiments/EXP-021_v3_fm_baseline_preparation/DATA_MANIFEST.json) · [CPU 检查](../../../submission/experiments/EXP-021_v3_fm_baseline_preparation/CPU_CHECKS.json) · [独立结构审计](../../../submission/experiments/EXP-021_v3_fm_baseline_preparation/MANIFEST_AUDIT.json)

- 32 条训练、8 条开发集的 56 帧候选源窗口已按 episode 划分，训练与开发不交叉，源窗口互不重叠，constant/switch 覆盖达到 16/16 与 4/4。**只有 4 条训练视频已物化**；28 条训练、8 条开发窗口仍需抽帧、画质核查和 VAE 编码。开发集只有 2 个曾观察 episode，尚无 6–8 个独立盲测场景。
- 普通 FM 数学/接口 CPU 骨架及 4 个单测通过，覆盖噪声/速度目标、未来动作隔离、clean history 梯度边界、adapter 更新和 checkpoint/optimizer/RNG 恢复。真实 V3 推理入口显式拒绝 grad-enabled forward，33B 可微训练 wrapper 尚未实现；因此不能称训练已就绪或质量已改善。[训练计划](../../../submission/experiments/EXP-021_v3_fm_baseline_preparation/TRAINING_PLAN.md) · [测试日志](../../../submission/experiments/EXP-021_v3_fm_baseline_preparation/CPU_TEST_LOG.txt)。
- 后续 GPU profile 必须另获 Judge 授权，先量 1–2 次完整 forward/backward/update 与编码/保存成本，再决定训练规模。[预算草案](../../../submission/experiments/EXP-021_v3_fm_baseline_preparation/GPU_BUDGET_DRAFT.md) 已记录用户资源窗口：今夜最多 8 张空闲卡，明天 09:00 HKT 后最多 3 张；当前任务批准的 GPU 调用仍为 **0**。
- 官方 SolarWM H3 Stage2-fix 权重在本地，但标准 SP4/8 GPU、158f/4NFE 相机推理未执行；必要的 preencoded test index/support 文件本地未发现。[独立准备页](../../../submission/experiments/EXP-021_v3_fm_baseline_preparation/SOLAR_STAGE2_REFERENCE.md)。EXP-020 Original H3 56f/30NFE 文本动作片与官方协议不可直接比较。

Worker 请求 Judge 审核 CPU 来源、接口草案和预算。下一 GPU 阶段目前只是建议，不视为已获放行。
