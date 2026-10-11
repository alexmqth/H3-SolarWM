# V3真实转移数据准备（EXP-015/016）

这是后续训练的数据准备证据，**不是新的模型版本，也没有训练结果**。

- [EXP-015：四个固定训练episode的真实56帧与原始动作](../../../experiments/EXP-015_v3_real56_data/README.md)：224帧、11keys、native12+5分块与动作预处理未来隔离通过。三个场景为A+S/J/L组合，第四段A→静止→W，没有D反事实GT。
- [EXP-016：SingleI0及真实视频VAE](../../../experiments/EXP-016_v3_real56_vae/README.md)：四组同源56/39编码前12latent逐值相同；重建主体和环境保留，普通细节有损。77.722GPU秒、峰7.504GiB、0训练。

## 原始录屏 / VAE重建

左侧是原始真实录屏，右侧是VAE重建；每条56帧、24fps。不能作为动作生成能力展示。

| 场景 | 并排视频 | 重建PSNR均值dB |
| --- | --- | ---: |
| 山路43866101 | [查看](../../../experiments/EXP-016_v3_real56_vae/artifacts/43866101/source_vs_reconstruction.mp4) | 29.736 |
| 草地7199292c | [查看](../../../experiments/EXP-016_v3_real56_vae/artifacts/7199292c/source_vs_reconstruction.mp4) | 29.469 |
| 工业9dc2e588 | [查看](../../../experiments/EXP-016_v3_real56_vae/artifacts/9dc2e588/source_vs_reconstruction.mp4) | 29.251 |
| 暗草地b784d995 | [查看](../../../experiments/EXP-016_v3_real56_vae/artifacts/b784d995/source_vs_reconstruction.mp4) | 34.894 |

## 尚未解决

原生packed位置依赖总文本长度，未来动作句长变化即使被裁剪仍可能影响历史位置。CPU已复现该风险；完整训练输入需要固定位置合同和实际模型回归，不能直接宣称训练ready。

[V3总览](../README.md) · [EXP-015 Judge验收](../../../experiments/EXP-015_v3_real56_data/judge/FINAL_REVIEW.md) · [EXP-016 Judge验收](../../../experiments/EXP-016_v3_real56_vae/judge/FINAL_REVIEW.md)
