> **新版导航：** [A机制诊断](../../branches/A_causal_diagnostics/README.md) / [B因果适配](../../branches/B_causal_adaptation/README.md) / [C AnyFlow与DMD](../../branches/C_anyflow_dmd/README.md)。本目录保留原证据路径。下文冻结/no-go指当时E2；后来的V2b仅在第二块局部通过，不代表训练恢复。

# 实验档案导航

**后续训练与视频总入口：** [数据集视频 / AnyFlow 训练结果](../../meeting/DATASET_AND_ANYFLOW.md)，说明实际 objective，并列出本次补齐的 114 个 MP4 和完整视频索引。

原来平铺的**51个实验目录已归入7个分类目录**，30个散落文件也已归档。先看主要结果，需要追溯时再展开历史诊断。

本目录保留历史名称`stage1_anyflow`，实际内容也包含普通FM、真实视频训练、动作机制诊断和DMD准备。当前研究已冻结：E2两臂各4更新没有一致收益，局部动作＋画质联合验收未过，不自动续训或重启AnyFlow/Stage2。

## 七类档案

| 目录 | 内容 | 优先级 |
|---|---|---|
| [01_real_video](01_real_video/README.md) | 真实ABot FM48、噪声采样分布FM48、最新E2 | **先看：训练后的完整结果** |
| [02_causal_diagnostics](02_causal_diagnostics/README.md) | 局部拓扑、action路由、历史条件及同状态差分 | **再看：为什么有效或失败** |
| [03_anyflow_trials](03_anyflow_trials/README.md) | AnyFlow/FM历史变体、checkpoint评测、早期pilot | 研究过程备查 |
| [04_numerical_checks](04_numerical_checks/README.md) | 时间条件、精度、梯度与数值正确性 | 工程追问备查 |
| [05_runtime](05_runtime/README.md) | 历史GPU迁移、并行和续训收据 | 运行追溯 |
| [06_stage2_preparation](06_stage2_preparation/README.md) | shared roles、FMBS、DMD等工程准备 | 不等于完整Stage2结果 |
| [07_protocols](07_protocols/README.md) | 历史计划、机制总览、目录整理记录 | 历史决策与路径查询 |

## 常用结果直达

- [真实视频FM48完整验收](01_real_video/real_abot_fm/FM48_COMPLETE_REVIEW.md)。
- [噪声采样分布对照](01_real_video/fm_density_control/FINAL_RESULTS.md)。
- [最新E2完整结果与六条对比视频](01_real_video/real_transition_windows/FINAL_RESULTS.md)。
- [历史条件改变后的A/D视频](02_causal_diagnostics/history_conditioning/VIDEO_RESULTS.md)。
- [局部动作拓扑](02_causal_diagnostics/local_topology/README.md)。

## 展示与复现入口

[会议主Demo与5分钟讲稿](../../meeting/README.md) · [昨晚真实视频训练展示](../../meeting/OVERNIGHT_PROGRESS.md) · [最终实验报告](../../docs/EXPERIMENT_REPORT.md) · [独立环境验收](../final_acceptance/README.md)

会议主片仍来自旧RGB checkpoint；最新E2与AnyFlow各有自己的结果，不能混用。各实验内部保留原有`report/`、视频、帧图、指标及源码快照结构。

## 旧路径与原始证据

完整旧→新路径映射见[LAYOUT.json](LAYOUT.json)，整理前版本为`62feab7`。例如：

```text
real_abot_fm/            → 01_real_video/real_abot_fm/
history_conditioning/   → 02_causal_diagnostics/history_conditioning/
pilot_snapshot/         → 03_anyflow_trials/pilot_snapshot/
pretrained_gpu2_*.json   → 03_anyflow_trials/early_pilot/
ABCD_STATUS.md          → 07_protocols/overviews/ABCD_STATUS.md
```

报告导航和当前展示脚本已更新。**原始JSON/CSV/log、视频、图片和冻结源码按原字节移动**；历史收据中的旧路径仍描述当时的实验位置，用映射表查找新归档位置，不把旧收据改写成新的实验记录。原工作区`H3-World/outputs/`没有移动。

旧报告里的“正在运行”仅对应其日期。当前状态以[最终报告](../../docs/EXPERIMENT_REPORT.md)和[E2冻结收据](01_real_video/real_transition_windows/FREEZE.json)为准。[整理验收](07_protocols/reorganization_20261009/README.md)记录文件和链接检查。
