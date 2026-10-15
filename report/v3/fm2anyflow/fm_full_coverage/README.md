# 全50层 C1+C2 普通 Flow Matching / EXP-025

## 目的与缩写

验证扩大adapter覆盖并加入首块训练，能否改善此前tail8/C2-only训练没有明确视频收益的问题。
FM（Flow Matching，流匹配）是普通速度场训练；FM-8NFE表示每chunk采样调用网络8次，clean KV commit另计；FM-train-256updates表示256次optimizer更新。C1为首块12 latent，C2为续块5 latent；GT为真实视频历史。Full50 QKV rank8为全部50层query/key/value低秩adapter，共10,752,000训练参数。本轮没有AnyFlow训练。

32train，每clip每block4遍；256updates，C1/C2各128。Single I0、native time、Global RoPE、strict causal与原始Action LoRA冻结。与EXP-023同时改变容量、训练块覆盖及总更新数，是联合方案，不能作单因素归因。2dev来自已观察episode，不是盲测。

## 结果

执行通过；人物结构/动作与连续性仍为PARTIAL，**没有证实新增视频收益**。C1 probe略差，C2小幅改善；self-history运动不足和视点偏移未明确解决。不升级正式V3，不自动增加rank、epochs、AnyFlow或DMD。有限预算负结果不能说明充分训练后的FM或H3架构不行。

| 固定FM MSE | 训练前均值 | 训练后均值 | 均值比变化 |
| --- | ---: | ---: | ---: |
| all | 0.145759580 | 0.144883364 | -0.6011% |
| C1 | 0.137298802 | 0.137618846 | +0.2331% |
| C2 | 0.154220358 | 0.152147881 | -1.3438% |
| train_C1 | 0.140055349 | 0.140351936 | +0.2118% |
| train_C2 | 0.162248781 | 0.159937400 | -1.4246% |
| dev_C1 | 0.134542255 | 0.134885756 | +0.2553% |
| dev_C2 | 0.146191934 | 0.144358361 | -1.2542% |

C2固定probe较EXP-023的−1.2306%仅多约0.113个百分点，不支持显著优势结论。数值为固定状态速度MSE，不能当视频质量分数。

## 成本

| 项目 | 实测 |
| --- | ---: |
| 硬件 | 1×NVIDIA L40 / GPU0 |
| 全任务GPU进程时间（含前后评测） | 11566.613秒 / 3.213 GPU小时 |
| 训练阶段 | 9666.122秒 / 161.10分钟 |
| 顶层DiT / backward / updates | 570 / 256 / 256 |
| VAE decode / 新VAE或text encode | 12 / 0 |
| 全过程峰allocated / reserved | 24.449 / 28.139 GiB |
| C2 CPU raw KV | 6,799,104,000 bytes |

RSS 66.472GiB是账本终点值，不是已证明的RSS峰。每片采样耗时见下表；采样、commit、decode及端到端口径不能互换。逐调用耗时见原始RUN.log，总费用见GPU_LEDGER.json。

| 片段 | 训练前sampling秒 | 训练后sampling秒 |
| --- | ---: | ---: |
| train_constant_GT39_generated17 | 43.395 | 51.575 |
| train_switch_GT39_generated17 | 42.549 | 54.129 |
| dev_constant_GT39_generated17 | 39.179 | 62.335 |
| dev_switch_GT39_generated17 | 41.777 | 63.234 |
| dev_constant_self56 | 98.121 | 195.085 |
| dev_switch_self56 | 108.490 | 112.287 |

## 六组三列视频

每片56帧、24fps（2.33秒），三列为GT / 零新增adapter / 256updates，生成均FM-8NFE/chunk。GT39_generated17仅后17帧新生成，前39帧显示真实历史；self56为两块自生成，前后模型各自生成C1，因此不是同历史C2消融。

- [dev_constant_GT39_generated17_GT_zero_256](videos/dev_constant_GT39_generated17_GT_zero_256.mp4)
- [dev_constant_self56_GT_zero_256](videos/dev_constant_self56_GT_zero_256.mp4)
- [dev_switch_GT39_generated17_GT_zero_256](videos/dev_switch_GT39_generated17_GT_zero_256.mp4)
- [dev_switch_self56_GT_zero_256](videos/dev_switch_self56_GT_zero_256.mp4)
- [train_constant_GT39_generated17_GT_zero_256](videos/train_constant_GT39_generated17_GT_zero_256.mp4)
- [train_switch_GT39_generated17_GT_zero_256](videos/train_switch_GT39_generated17_GT_zero_256.mp4)

来源与SHA见[VIDEO_MANIFEST.json](VIDEO_MANIFEST.json)。[完整实验](../../../../experiments/EXP-025_v3_fm_full_coverage/README.md) · [Judge审核](../../../../experiments/EXP-025_v3_fm_full_coverage/judge/FINAL_REVIEW.md) · [前一配置EXP-023](../fm_small_data/README.md) · [返回分组](../README.md)。
