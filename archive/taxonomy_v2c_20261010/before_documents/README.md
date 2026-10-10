# H3-World 因果世界模型研究

目标：从 Original H3-World 出发，建立同时具备动作控制、视觉稳定和高效推理能力的自回归视频世界模型，再研究 AnyFlow 少步生成与 On-policy DMD。

当前正式状态见 [progress.md](progress.md)。EXP-001的V2b持续A/D124帧与Original对比已交付；EXP-002/003进一步完成[V3严格缓存124帧可行性](submission/mainline/V3_efficient_causal/README.md)。EXP-004原权重8步续写AA/AD73帧已验收，有限正信号与画质限制见[展示](submission/report/V3_8step_continuation/README.md)；任务已收口，下一决策见[next_plan.md](next_plan.md)。

## 协作入口

| 文件 | 职责 |
| --- | --- |
| [guideline.md](guideline.md) | Judge / Exp Worker 协作、实验预算、版本与验收规则 |
| [progress.md](progress.md) | 当前研究状态、能力边界与任务索引 |
| [next_plan.md](next_plan.md) | 当前唯一有效任务书、预算与验收条件 |
| [report.md](report.md) | 当前 Worker 执行报告 |
| [archive.md](archive.md) | 历史任务、原文快照和 Judge 决策记录 |

## 目录与复现

- [H3-World](H3-World/README.md)：模型源码、因果实现、测试与原始 outputs。工作树包含本地研究改动，复现需使用对应冻结源码及 hash。
- [SolarWM](SolarWM/README.md)：参考实现与研究方法。
- [submission](submission/README.md)：版本定义、研究分支、实验证据与展示材料；独立 Git 仓库。
- [主线版本](submission/mainline/README.md)：V0、V1、并行 V2a/V2b、V3可行性版本。
- [汇报材料](submission/report/README.md)：版本说明、原片、对比视频与讲稿。
- [依赖与复现说明](submission/REPRODUCE.md)：提交包环境、外部权重、源码 patch 和验证方式。具体实验以任务书及冻结协议为准。
- `models/`：共享权重链接；`.venvs/`：已有环境；不复制基础权重到提交包。

GWM 根目录不是 Git 仓库；H3-World、SolarWM、submission 分别管理版本。`/home/qma/work/GWM` 是此工作区的别名。

根目录只保留六份协作文档。旧 README、WORKSPACE、scheduled-sampling 状态及任务总结的原文与 hash 已保存在 archive.md 的 `root_cleanup` 节；历史运行建议不构成当前任务授权。
