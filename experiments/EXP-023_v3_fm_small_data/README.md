# EXP-023：V3普通FM小数据拟合

Exp Worker已完成32train/128updates与固定GT/self-generated-history前后对照，Judge已验收任务完成；新增画质/动作收益未证实，不升级V3。FM为Flow Matching；128updates是训练更新数，FM-8NFE为每chunk8次推理求值。固定probe均值MSE从0.154220降至0.152322（-1.23%），但视频尚未证明明确的动作或画质提升。

[Worker完整报告](WORKER_REPORT.md) · [三列对照视频](paired_gallery/) · [前后数值](POST_COMPARISON.json) · [视频审计](FINAL_VIDEO_AUDIT.json) · [预算账本](CUMULATIVE_BUDGET.json) · [任务书](taskbook_v1.md) · [授权](AUTHORIZATION.json) · [固定选样](SELECTION.json)。执行完成不等于画质或动作验收通过。

## 执行补充（优先于原任务书相应条款）

- [补充01：34clip统一源RGB，70次VAE编码/13800GPU秒](ADDENDUM_01_SOURCE_RGB.md)。
- [补充02：forward与checkpoint backward保持相同context](ADDENDUM_02_CHECKPOINT_CONTEXT.md)。编码继续；optimizer更新需先通过CPU梯度回归及首个真实backward重算门控，之后自动继续。旧EXP-022梯度PASS已撤回，前向/KV证据保留。

- [补充03：ffmpeg媒体修复与一次受限恢复](ADDENDUM_03_VIDEO_IO_RECOVERY.md)：decode总上限13，其他上限不变，失败费用累计保留。

- [补充04：BF16身份写入修复，CPU复核后最后一次首片恢复](ADDENDUM_04_BFLOAT16_RECOVERY.md)：449顶层DiT/14decode，其余总时限与训练预算不变。

- [补充05：生产hook作用域CPU验证通过，仅恢复训练与post](ADDENDUM_05_TRAIN_HOOK_RECOVERY.md)：不增加预算，不重复已完成pre结果。真实首步backward门控继续有效。

[Judge最终审核](judge/FINAL_REVIEW.md) · [独立复核](judge/FINAL_AUDIT.json)。
