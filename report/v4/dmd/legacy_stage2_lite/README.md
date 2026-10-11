# 早期 Stage2-lite：旧协议的 critic / DMD 探索

## 名称与目的

DMD = Distribution Matching Distillation（分布匹配蒸馏），critic / fake-score 是拟合学生生成分布的可训练评分角色；Stage2-lite 表示小预算的Stage2部分链路试验。现归入V4的SGF研究档案，执行时早于冻结V3，不是新的V4成果。

目的：在早期因果/anchor方案上检查学生自身采样、fake-score训练、冻结教师与DMD更新能否连起来。旧 RGB anchor / visual adapter 条件与EXP-008的V3 native Single I0不同，不能直接归因比较。

## 视频与结论

- [早期8步、4次更新AD39](videos/stage2_lite_39_8step_4updates_AD.mp4)。8为推理步数，4为训练更新次数。
- [接入RGB visual adapter后的AD39](videos/stage2_lite_rgb_endpoint_integrated_AD_39.mp4)。结构改善，但没有恢复正确A/D动作；报告的A/D光流均为负。

原实验记录链路可运行、训练finite；动作与视觉联合能力未通过，不等于官方SolarWM Stage2完成。保留负结果，当前没有追加训练任务。

[旧实验完整记录](../../../../experiments/03_stage2_lite_critic_dmd/README.md) · [返回V4](../../README.md) · [缩写表](../../../GLOSSARY.md) · [视频来源](video_manifest.json)
