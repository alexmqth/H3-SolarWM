# ABot两片C2普通FM拟合 / EXP-027

## 名称、目的与范围

FM（Flow Matching，流匹配）是本轮训练目标；**FM-train-64updates**表示累计64次optimizer更新，**FM-8NFE**表示每个生成块采样调用网络8次，历史clean KV commit另计。C1为首块12latent，C2为后续5latent；GT-history表示使用真实视频的已完成首块作为历史。本轮没有AnyFlow训练。

目的：此前32clip实验每clip仅4次C2更新，本轮集中两片各32次，检查是否出现明确局部拟合与视频收益。全50层QKV rank8、10,752,000新训练参数；Original H3与released Action LoRA冻结。S+D和A+S来自不同I0/episode，不是同状态动作干预。

**Judge结论：执行通过；固定指标小幅改善，新增视频/动作收益未证实。当前候选收口，正式V3不升级。**

## 视频：左0updates，右64updates

每条56帧/24fps，约2.33秒；**前39帧是真实历史展示，仅后17帧（约0.71秒）是新生成**。两侧前39帧原始RGB逐值相等。

| 动作片段 | 训练前后对照 | 同源真实视频 |
| --- | --- | --- |
| S+D：后退并向右侧移 | [播放左右对比](videos/S_D_0_vs_64updates.mp4) | [源视频](videos/S_D_source_rgb56.mp4) |
| A+S：后退并向左侧移 | [播放左右对比](videos/A_S_0_vs_64updates.mp4) | [源视频](videos/A_S_source_rgb56.mp4) |

两片前后人物可辨且有运动，姿态节奏与场景变化接近，未确认新的动作或连续性收益。本轮不包含self-history或动作切换视频，不能外推长时生成。光流幅值不能作为人物方向准确率。

## 指标与成本

| 项目 | S+D | A+S |
| --- | ---: | ---: |
| 固定独立噪声FM MSE变化 | −0.975% | −1.003% |
| 新生成17帧源参考PSNR | 20.80→21.22dB | 22.71→23.05dB |
| boundary MAD | 7.856→7.912 | 8.713→8.686 |
| C2采样耗时，训练前→后 | 49.38→62.14秒 | 57.08→61.39秒 |

采样耗时只含8NFE，不含历史commit、decode、加载/写片，单次耗时差不代表算法速度变化。PSNR更接近一个源轨迹，不等于动作能力通过。

- 1×L40：含首次失败，总3876.365秒（64.61分钟/1.07677GPU小时）。
- 恢复训练阶段2463.095秒，64updates；固定probe为0/32/64三组，各40条。
- 峰allocated24.449GiB，reserved29.137GiB（两次尝试最大）；CPU raw KV6.332GiB。
- 账本终点RSS最大68.147GiB，未记录可验证的RSS峰；逐样本物理显存峰未提供。
- 总303顶层DiT尝试：302完成+1失败；64backward、64updates、4decode、0encode/text。首次commit审计误拒绝已修复，失败费用保留。
- 可选dev self-history未触发，最终4原片+2对比+2源参考，均完整解码并核对SHA。

[视频来源清单](VIDEO_MANIFEST.json) · [原始四视频](../../../../experiments/EXP-027_abot_baseline_diagnosis/recovery_01/videos/) · [完整实验](../../../../experiments/EXP-027_abot_baseline_diagnosis/README.md) · [Judge审核](../../../../experiments/EXP-027_abot_baseline_diagnosis/judge/FINAL_REVIEW.md) · [逐帧审核](../../../../experiments/EXP-027_abot_baseline_diagnosis/judge/VISUAL_REVIEW.md) · [返回分组](../README.md)。
