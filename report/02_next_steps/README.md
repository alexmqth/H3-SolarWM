# 下一步：实际模型验证与真实混合动作条件

2026-10-11 09:00的夜间实验已收口。当前没有待执行GPU任务；本轮目录整理不新增实验授权。正式状态以根目录progress.md、执行授权以next_plan.md为准。

## 已有结论

- 普通FM-8NFE已测新首窗、短程场景迁移和Global滑窗158帧，有限可行，质量PARTIAL。
- AnyFlow AF-8NFE停车场续写有限可行，但迁移没有联合收益；AF-4NFE当前配置失败，停止扩展。
- V4的DMD训练工程通过，当前配置生成失败，归档。
- EXP-015/016真实数据和VAE已验收，EXP-017固定动作布局仅通过CPU检查。

## 下一轮建议（草案）

先验证固定动作布局在实际模型中的canonical回归，再编码真实mixed-action条件；固定首个scene，以真实C1作为给定历史，比较FM-30NFE / FM-8NFE的C2局部续写，再决定是否扩其余三图。这是给定真实历史的续写，不是自由长时生成或真实D反事实评测。

草案总预算：8次canonical forward、最多156次续写/commit forward、14次text encode、8次decode、0训练，上限3660 GPU秒（约1.0167 GPU小时）；需落实独立任务与Judge分阶段放行。普通FM真实数据适配和AnyFlow finite-map训练随后分别设计。

[完整下一阶段草案](../../experiments/EXP-017_v3_fixed_action_layout/judge/NEXT_STAGE_DRAFT.md) · [CPU候选审核](../../experiments/EXP-017_v3_fixed_action_layout/judge/FINAL_REVIEW.md) · [数据准备](../v3/fm2anyflow/data_preparation/README.md)
