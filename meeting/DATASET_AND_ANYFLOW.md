# 数据集视频与 AnyFlow 训练：结果、视频和来源

[按三类模型查看对比视频](model_types/README.md)：零更新routing、普通FM适配、AnyFlow有限区间训练，与数据来源/history条件分开标注。

**真实 ABot 数据训练和 AnyFlow 历史训练的结果都在提交包中。** 2026-10-09 本次补齐此前未归档的单条视频、测量记录和两个中断训练收据，并提供完整视频索引。主会议片仍是旧 RGB checkpoint，本页专门展示后续训练。

## 先分清数据、history 和训练目标

| 实验 | 训练数据及 history 来源 | 实际训练目标 | 预算与结果入口 |
|---|---|---|---|
| ABot FM48 | 数据集中录制的游戏视频及控制记录；GT clean history；16 train / 8 validation，按 episode 隔离 | 普通 teacher-forcing FM | [48 更新完整结果](../reports/stage1_anyflow/01_real_video/real_abot_fm/FM48_COMPLETE_REVIEW.md) · [实际 training.json](../reports/stage1_anyflow/01_real_video/real_abot_fm/training_snapshot.json) |
| ABot 噪声采样对照 | 同一真实数据集、同一划分；GT clean history | 普通 FM；仅改变训练 sigma 采样分布 | [另 48 更新完整结果](../reports/stage1_anyflow/01_real_video/fm_density_control/FINAL_RESULTS.md) · [实际 training.json](../reports/stage1_anyflow/01_real_video/fm_density_control/training_inputs/candidate_training.json) |
| E2 真实动作后果监督 | 真实 ABot transition windows，6 train / 2 validation；局部同 sigma 加噪历史 N | FM-only 与 FM＋action ranking | [两臂各 4 更新及全部评测](../reports/stage1_anyflow/01_real_video/real_transition_windows/FINAL_RESULTS.md) |
| AnyFlow full-history 与续训系列 | Original H3 生成的停车场 A/D teacher 视频/latent 伪标签；训练保留完整 clean-history 梯度 | TF-AnyFlow 有限区间目标；另有独立普通 FM controls | [full-history16](../reports/stage1_anyflow/03_anyflow_trials/full_history_candidate/FINAL_RESULTS.md) · [128→136 有限预算对照](../reports/stage1_anyflow/03_anyflow_trials/interval_consistency_candidate/FINAL_RESULTS.md) |

两轮 ABot48 调用了 `train_stage1_anyflow.py`，但实际参数是 **`objective=fm`**。该脚本同时支持 FM 和 AnyFlow，不能由文件名判断训练目标。AnyFlow 收据里的 `teacher_dir` 指向 `action_A_teacher_39` / `action_D_teacher_39`，而 ABot48 使用 `real_data_manifest=.../abot_bridge/encoded_manifest.json`。本机两个 outputs 树共检查 172 份 `training.json`；目前未找到“录制的 ABot 数据＋AnyFlow objective”已完成训练的证据，不能把这两组实验合并为那项结果。

`real H3` 指真实 33B 模型，`full-history` 指是否保留历史梯度，`GT/clean history` 指历史条件；这些词本身都不说明数据来自真实录制视频。ABot 的“真实视频”是实际录制的游戏视频，不是现实世界实拍。

## 会议直接播放这些完整对比

| 展示 | 完整 MP4 | 观看说明 |
|---|---|---|
| 真实数据 FM48，GT history / 30 steps per chunk | [GT-history 四列对比](../reports/stage1_anyflow/01_real_video/real_abot_fm/report/trained_complete_gt30/dfec8ed3237860eba14d67c089ecd041_D_1750_comparison.mp4) | GT / Original H3 / FM0 / FM48；人物大体保留，边界重置和模糊仍在 |
| 同一场景、同一 FM48，generated history / 30 steps per chunk | [generated-history 四列对比](../reports/stage1_anyflow/01_real_video/real_abot_fm/report/trained_complete_generated30/dfec8ed3237860eba14d67c089ecd041_D_1750_comparison.mp4) | 约 23 帧后残影、26–35 帧人物分解；完整失败后段保留 |
| 同一场景 FM48，generated history / 8 steps per chunk | [8 步完整对比](../reports/stage1_anyflow/01_real_video/real_abot_fm/report/trained_complete_generated8/dfec8ed3237860eba14d67c089ecd041_D_1750_comparison.mp4) | 没有同协议 FM0 8 步视频，不把它解释为训练前后收益 |
| E2 真实 GT-history，局部 30 步 | [Original local N / FM-only4 / FM+action4](../reports/stage1_anyflow/01_real_video/real_transition_windows/review_step4/118eb5d8b75e1b8ac23a4e9ae77af9a9_D_1015_comparison.mp4) | 39 当前帧的局部诊断，不是 124 帧自由生成；动作项没有一致额外收益 |
| AnyFlow full-history16，4/8 steps per chunk | [Original / 4 步 / 8 步，A/D 两行](../reports/stage1_anyflow/03_anyflow_trials/full_history_candidate/full_history16_AD_diagnostic.mp4) | 训练伪标签来自 Original H3；4 步后段雾化，8 步仍模糊、方向未恢复 |
| AnyFlow128 / control136 / auxiliary136，8 steps per chunk | [8 步完整四列 A/D](../reports/stage1_anyflow/03_anyflow_trials/interval_consistency_candidate/original128_control_auxiliary136_8step_AD.mp4) | 左列 Original，随后三种 AnyFlow checkpoint；保留 18–38 帧重影 |
| 同三种 checkpoint，4 steps per chunk | [4 步完整四列 A/D](../reports/stage1_anyflow/03_anyflow_trials/interval_consistency_candidate/original128_control_auxiliary136_4step_AD.mp4) | 有限区间一致性改善不能替代动作和画质验收 |

这些主要短片均保留完整 39 帧；E2 局部长度和全局位置见原报告。GT-history 每块使用参考历史，不能当作自由 rollout 稳定性。自然 ABot 场景含联合键盘/相机动作，不能把两个不同场景的光流相减当作纯 A/D 控制指标。

39 帧、3 chunks 时，8 steps/chunk 是 **24 noisy forwards＋3 clean commits**，30 steps/chunk 是 **90＋3**。Original 30 steps 是完整序列的 30 次去噪。E2 使用局部重算协议，没有 persistent hidden KV，其计数见 E2 报告。计时来自共享硬件单次运行，不能当 warmup 后多次均值。

## 这次补了什么，如何找到全部文件

- 原来平铺的 51 个实验目录已归入 [7 个分类](../reports/stage1_anyflow/README.md)，原始 outputs 不移动。
- 检查 23 个源实验目录中的 **202 个 MP4 路径、197 份不同视频内容**；重复内容按 SHA256 链接到已有文件。
- **新增 114 个 MP4**：真实数据组 28 个，AnyFlow/FM 历史对照组 68 个，运行/续训归档 12 个，数值候选归档 6 个。还补入 224 份关联测量收据与 2 份中断训练记录，共 340 个源文件、约 88.9 MiB。
- 真实数据组 98 个源 MP4 现全部有归档链接，包括 GT/reference、训练前后单条输出和拼接对照。它们不是 98 次独立实验。
- 28 份所列训练记录中，21 份使用 AnyFlow 目标，包括迁移/中断和续训收据，不能把各文件的累计 update 数相加。完成状态和恢复起点均保留。

| 完整清单 | 内容 |
|---|---|
| [真实数据视频索引](../reports/stage1_anyflow/01_real_video/VIDEO_INDEX.md) | ABot48、采样密度对照和 E2 的全部源视频 |
| [AnyFlow/FM 历史视频索引](../reports/stage1_anyflow/03_anyflow_trials/VIDEO_INDEX.md) | pilot、full-history、训练预算和 4/8 步评测 |
| [数值候选视频索引](../reports/stage1_anyflow/04_numerical_checks/VIDEO_INDEX.md) | native-FP32 候选的完整评测 |
| [续训视频索引](../reports/stage1_anyflow/05_runtime/VIDEO_INDEX.md) | GPU 迁移与 68→128 续训相关结果 |
| [源文件→Git 路径、SHA256、实际 objective](../reports/stage1_anyflow/07_protocols/reorganization_20261009/training_video_inventory.json) | 训练与视频逐文件审计 |
| [完整解码检查](../reports/stage1_anyflow/07_protocols/reorganization_20261009/training_video_decode.json) | CPU 检查可播放性和完整帧数；不代表画质通过 |

本次没有新训练、模型推理或重新编码。新 FM48/E2 的 adapter 张量、optimizer/RNG、encoded latents 和原始数据集仍在本机；本次发布的是训练结果、配置/日志、测量及可播放视频。checkpoint 本机位置与哈希见 [昨晚进展](OVERNIGHT_PROGRESS.md) 和 [checkpoint 审计](overnight_assets/sync_audit.json)。旧会议 adapter 不能冒充这些新训练权重。

目前最准确的展示结论是：**真实数据 FM 和伪标签 AnyFlow 两条路线均完成了有限预算实验，但动作与人物结构联合验收仍未通过。** 训练和数值检查的完成，不能写成完整 SolarWM Stage1 或视频效果已经成功。
