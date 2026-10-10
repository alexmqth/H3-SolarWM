# EXP-013 — 多场景 V3/AnyFlow 数据准备（CPU 审计）

本轮只审计现有 6 个 ABot episode / 24 条39帧 clip，并冻结四张 train 初图；没有新模型推理、编码或训练。当前结论与局限见 [Worker 报告](WORKER_REPORT.md)。

| 文件 | 作用 |
|---|---|
| [audit_data.py](audit_data.py) / [运行日志](audit_run.log) | ≤4 CPU 线程的可复跑哈希、完整 clip 解码、源动作重建和视频生产配方复核 |
| [逐 clip inventory](inventory.json) / [源 manifest](source_manifest.json) | 每条 clip 与六个原始 episode 的位置、SHA、帧/PTS、动作列与来源 |
| [源/切分/动作审计](source_split_action_audit.json) | train/validation 隔离、真实共按键、源帧对齐、旧编码协议和异常探针 |
| [候选 manifest](candidate_manifest.json) / [初图联系表](candidate_contact_sheet.jpg) | 按每 episode 最早 `target=A` 确定四张训练初图；标注真实录屏共按键 |
| [拟议协议](PROPOSED_PROTOCOL.json) / [后续阶段预算](FUTURE_GPU_DESIGN.md) | native Single I0、冻结 V3 教师、独立 AnyFlow student 和固定评估门；**仅草案** |
| [任务书](taskbook_v1.md) | 当前获批范围和停止条件 |

录屏文件名 A/D 不代表纯方向控制：四张选定的 A 初图来自真实 A+S+L、A+S、A+S+J、A 片段。未来教师 A/D 仅是以这些初图为起点的合成反事实，不是录屏的真实后续动作。两个 validation episode 已在 EXP-011/012 中看过，只用于固定回归评估。
