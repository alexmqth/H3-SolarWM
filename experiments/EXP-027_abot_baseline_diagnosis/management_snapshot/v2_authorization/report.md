# EXP-027/v1 Worker 交付：ABot 基线诊断与下一实验建议

2026-10-16 HKT。**仅完成 CPU 证据审查；0 GPU、0 下载、0 编码、0 模型 forward、0 训练。** 推荐下一次只做“两条真实 ABot clip 的 C2 普通 FM 可拟合性门控”，暂不直接扩展 GT/self-history × action-switching 视频矩阵。完整判断见 [EXP-027 Worker 报告](submission/experiments/EXP-027_abot_baseline_diagnosis/WORKER_REPORT.md)、[证据复核](submission/experiments/EXP-027_abot_baseline_diagnosis/EVIDENCE_REVIEW.md)和[单一 GPU 提案](submission/experiments/EXP-027_abot_baseline_diagnosis/GPU_PROPOSAL.md)。

现有 32 train / 2 dev 的 34 组 I0、条件和 17-latent 编码均在本地；训练集是 4 个 episode、常量/切换动作各 16 条。两条 dev 与训练 episode 不重叠，但早已被观察，不能作为盲测。[数据清单](submission/experiments/EXP-027_abot_baseline_diagnosis/DATA_SUMMARY.json)记录了两条优先 clip 的完整动作时序及编码文件存在性。

EXP-023/025 的训练链路和梯度是真实有效的；[checkpoint 参数差异](submission/experiments/EXP-027_abot_baseline_diagnosis/CHECKPOINT_DELTA.json)也证明 adapter 确实变化。但每条 clip 在每轮实验中只接受 4 次 C2 更新，固定 FM probe 仅改善约 1%，[EXP-025 Judge 审核](submission/experiments/EXP-025_v3_fm_full_coverage/judge/FINAL_REVIEW.md)未发现清晰的动作或画质新增收益。因此先用两条动作清楚的样本、独立随机训练噪声与未用于更新的多 σ probe，判断当前目标能否产生明显的局部学习信号。若两条都不能拟合，就应停止同类 FM 扩预算；若能拟合但视频仍不响应，再针对目标与 rollout/history 的错配设计下一步。这个提案需要 Judge 单独签发 GPU 预算，当前没有启动许可。正式 V3 仍冻结，官方 SolarWM 参考支线按用户决定暂停。
