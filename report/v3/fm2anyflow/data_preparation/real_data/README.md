# 真实训练数据与VAE/输入协议准备

## 名称与目的

**名称解释：** GT = Ground Truth（真实记录）；VAE = Variational Autoencoder（变分自编码器）；I0是初始图像；CPU布局测试只处理位置/索引等元数据。

**本次目的：** 补充真实录屏及实际动作，检查同源视频编码前缀是否一致，并解决未来动作文本长度影响当前位置的问题。视频展示真实录屏与重建；EXP-017只做CPU测试，没有生成视频。

[其他缩写与计数](../../../../GLOSSARY.md) · [汇报总览](../../../../README.md)

## 实验结果与证据

这是后续训练的数据准备证据，**不是新的模型版本，也没有训练结果**。

- [EXP-015：四个固定训练episode的真实56帧与原始动作](../../../../../experiments/EXP-015_v3_real56_data/README.md)：224帧、11keys、native12+5分块与动作预处理未来隔离通过。三个场景为A+S/J/L组合，第四段A→静止→W，没有D反事实GT。
- [EXP-016：SingleI0及真实视频VAE](../../../../../experiments/EXP-016_v3_real56_vae/README.md)：四组同源56/39编码前12latent逐值相同；重建主体和环境保留，普通细节有损。77.722GPU秒、峰7.504GiB、0训练。

## 原始录屏 / VAE重建

左侧是原始真实录屏，右侧是VAE重建；每条56帧、24fps。不能作为动作生成能力展示。

| 场景 | 并排视频 | 重建PSNR均值dB |
| --- | --- | ---: |
| 山路43866101 | [查看](videos/43866101_source_vs_reconstruction.mp4) | 29.736 |
| 草地7199292c | [查看](videos/7199292c_source_vs_reconstruction.mp4) | 29.469 |
| 工业9dc2e588 | [查看](videos/9dc2e588_source_vs_reconstruction.mp4) | 29.251 |
| 暗草地b784d995 | [查看](videos/b784d995_source_vs_reconstruction.mp4) | 34.894 |

## 尚未解决

原生packed位置依赖总文本长度，未来动作句长变化即使被裁剪仍可能影响历史位置。CPU已复现该风险；完整训练输入需要固定位置合同和实际模型回归，不能直接宣称训练ready。

[V3总览](../../../README.md) · [EXP-015 Judge验收](../../../../../experiments/EXP-015_v3_real56_data/judge/FINAL_REVIEW.md) · [EXP-016 Judge验收](../../../../../experiments/EXP-016_v3_real56_vae/judge/FINAL_REVIEW.md)

## 固定布局CPU候选

[EXP-017](../../../../../experiments/EXP-017_v3_fixed_action_layout/README.md)已通过canonical A/D回归及未来变长文本隔离的CPU检查；0GPU/训练，真实text encoder/DiT/KV数值/视频未测。[Judge最终验收](../../../../../experiments/EXP-017_v3_fixed_action_layout/judge/FINAL_REVIEW.md)与[下一阶段草案](../../../../../experiments/EXP-017_v3_fixed_action_layout/judge/NEXT_STAGE_DRAFT.md)分别记录证据和未授权预算。

## 视频来源

本目录 `videos/` 为已完成实验视频的原字节副本，可直接打开；来源、大小和 SHA-256 见[视频清单](video_manifest.json)。
