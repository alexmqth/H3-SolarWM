# 停车场零更新baseline全部完成

**零更新causal的动作区分在30步和8步均失败；30步画面较完整，8步约20–22帧后明显重影。** 所有视频完整39帧，已静态检查全部帧，不是实时播放。

[完整三列两行对比](original_causal0_30_8_AD.mp4)：列为Original30 / Causal0 30步每块 / Causal0 8步每块，行为A / D。视频标注真实步数计数与单次recorded wall。

| 方法 | A | D | A−D | 完整画面 |
|---|---:|---:|---:|---|
| Original30 | +1.181253 | −0.842124 | 2.023377 | 人物/车库完整 |
| Causal0 30/chunk | −0.195516 | −0.186950 | −0.008565 | 人物在，尺度/细节漂移，A/D运动相似 |
| Causal0 8/chunk | −0.021397 | −0.031908 | 0.010510 | 约20–22帧起重影，后段持续 |

[30步完整指标与评审](../parking_step00_30step/VISUAL_REVIEW.md) · [8步完整指标与评审](../parking_step00_8step/VISUAL_REVIEW.md)

Original旧legacy精度，causal0/48统一h3_fp32；anchor/prefix不同，Original比较为pipeline级。输入noise/audio/prompt/anchor/action rows逐张量一致，causal0/48才是同配置训练前后比较。Shared-host单次时间与不同offload策略不能用来宣称公平加速或显存减少。待FM48完成后补训练后结果；本报告没有训练效果结论。
