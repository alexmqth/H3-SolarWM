# V4-DMD：SGF路线的分布匹配蒸馏尝试

## 名称与目的

**名称解释：** SGF = Self-Gradient Forcing（自梯度强制）；DMD = Distribution Matching Distillation（分布匹配蒸馏）。DMD8表示8-map学生；cycle8表示累计8个学生训练cycle。

**本次目的：** 让少步学生在自己的生成样本上，通过冻结teacher与可训练fake-score critic获得分布匹配训练信号，检查能否改善少步质量。实际仅覆盖固定C1后的当前块，当前配置生成失败，不能称完整SolarWM Stage2已复现。

[其他缩写与计数](../../GLOSSARY.md) · [汇报总览](../../README.md)

## 实验结果与证据

**EXP-008累计8个student cycle训练工程PASS；最终cycle8的AA C2生成FAIL，当前配置停止归档。** 共同FM-8NFE前39帧保留，新增17帧全部彩色噪声、人物与停车场不可辨。AD完成5步后在第6forward中断，没有完整端点或视频；C3未执行。没有追加训练、改超参或挑选中间checkpoint。

[FM-8NFE / AF-8NFE / DMD8三列56帧视频](videos/AA_FM8_AF8_DMD8_cycle8_collapse_56.mp4) · [DMD AA原片](videos/AA_rollout_56.mp4) · [全部新增17帧](../../../experiments/EXP-008_v3_dmd/dmd_eval/artifacts/raw/AA/chunk_12_17_all_frames.jpg)

Teacher为冻结V3 causal H3，fake为独立普通FM/QKV，student从EXP-007 AF-train-32updates（EXP-007 / AF2）继承。三角色分别使用自身clean KV，保留完整8-map反向，fake更新后重建自身KV。只固定C1/current-block on-policy，不是原生双向teacher或完整SolarWM Stage2复现。

| 阶段 | 实际执行 | GPU小时 |
| --- | --- | ---: |
| v1 pilot | 17forward / 3backward / 3update / 0VAE | 0.212416（三卡保守） |
| v2新增7cycles | 99forward / 14backward / 14update / 0VAE | 1.394701（三卡保守） |
| 最终视频及中断 | 16已启动或预记账forward（1中断）/ 1VAE | 0.056973 |

合计约1.66409GPU小时。训练student峰28.102GiB；评估完成阶段峰25.970GiB，中断AD无终态峰值。训练来源、角色隔离、完整梯度链、配套checkpoint/optimizer/RNG独立审计通过，不能据此宣称生成成功。

CPU诊断显示DMD端点与初噪声cosine 0.949，FM-8NFE/AF-8NFE仅0.100/0.114，支持去噪失败；不能单凭这些数据归因某个超参数。Fake loss尖峰和梯度变小保留为观察。本结果只否定已测配置，不代表所有DMD方法不可行。

[Judge最终审核](../../../experiments/EXP-008_v3_dmd/judge/FINAL_REVIEW.md) · [Worker视频报告](../../../experiments/EXP-008_v3_dmd/worker_report_v2_eval.md) · [训练审核](../../../experiments/EXP-008_v3_dmd/judge/TRAIN_V2_REVIEW.md) · [实验入口](../../../experiments/EXP-008_v3_dmd/README.md)

后续EXP-009使用DMD前的原AF-train-32updates（EXP-007 / AF2），与普通FM在4NFE进行有限比较，零新训练。正式V3 Baseline与FM-8NFE/AF-8NFE冻结结果保持。

## 视频来源

本目录 `videos/` 为已完成实验视频的原字节副本，可直接打开；来源、大小和 SHA-256 见[视频清单](video_manifest.json)。
