# 真实ABot验证片段：已完成结果

History treatment: **generated**. Causal **30 steps/chunk**. Original30为整段30次前向。

仅列出真实完成且输入一致的结果；缺失checkpoint不填充成模型输出。GT-history是oracle历史诊断，拼接的预测chunk不是自由rollout。两条真实动作含联合按键/镜头操作，片段间光流差不作为A/D控制恢复证据。

时间来自单次共享主机、预缓存conditioning，不能宣称加速；MAD表示运动/帧差，不是画质。自然户外场景的绝对MAD不能套用停车场阈值。视频画质还需人工检查全部39帧，当前报告生成器不自动判PASS。

- [118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140](118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140_comparison.mp4) — Real ABot GT / Original 30 full-seq / Causal FM0 30/chunk generated
- [dfec8ed3237860eba14d67c089ecd041_D_1750](dfec8ed3237860eba14d67c089ecd041_D_1750_comparison.mp4) — Real ABot GT / Original 30 full-seq / Causal FM0 30/chunk generated

[逐条指标](metrics.csv) · [完整机器记录](metrics.json)
