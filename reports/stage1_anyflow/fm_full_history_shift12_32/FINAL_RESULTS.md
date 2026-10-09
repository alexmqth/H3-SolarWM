# 同条件 FM32 / AnyFlow32：没有验证出 AnyFlow 少步收益

归档时间：2026-10-08T12:08:28.794348+08:00。两种目标的 A/D39f、4/8 steps per chunk 全部完成，均未通过 Stage1 短片 gate。

训练起点、32次更新、全52块rank8 Q/K/V/out/FFN、43,237,376可训练参数、train shift12、完整历史梯度、action/chunk与128个current sigma序列均已匹配。AnyFlow的off-diagonal目标与额外前向是目标本身的差别。实际GPU初始化/采样审计见 [matched_training32_audit.json](matched_training32_audit.json)。

| 方法 | steps/chunk | flow(A) | flow(D) | A−D | Gate |
|---|---:|---:|---:|---:|---|
| Original H3 | 30 full sequence | +1.181253 | −0.842124 | 2.023377 | reference |
| FM32 | 4 | -1.058372 | -1.298657 | 0.240284 | FAIL：A<0 |
| FM32 | 8 | -0.943292 | -1.356802 | 0.413510 | FAIL：A<0 |
| AnyFlow32 | 4 | -1.216895 | -1.197330 | -0.019565 | FAIL：A<0 |
| AnyFlow32 | 8 | -1.305473 | -1.478164 | 0.172691 | FAIL：A<0 |

## 完整帧复核

所有四条FM的0–38帧contact sheet及对应Original/AnyFlow的12/24/30/38帧已检查。4-step：FM仍有模糊/透明，但人物与车库明显比AnyFlow完整；AnyFlow约20帧后严重雾化。8-step：两种方法均保留人物与车库，但都有模糊、后段透明感；未见明确AnyFlow画质优势。A没有呈现原始H3的左移动作，A/D都更像相似的向前运动。此处为静态完整帧与同帧复核，不声称真人实时播放或VBench评价。

[4-step完整三列A/D对比视频](matched_FM32_AnyFlow32_4step_AD_diagnostic.mp4) · [8-step完整三列A/D对比视频](matched_FM32_AnyFlow32_8step_AD_diagnostic.mp4)。左Original、中FM32、右AnyFlow32；均39f/24fps/H264/yuv420p/faststart，已完整解码39帧并检查标签、frame30预览。视频保持原速，时长1.625秒，不循环/补帧。

## 运行记录

| 方法 | 动作 | steps/chunk | E2E s | GPU allocated peak MiB | CPU KV MiB | noisy+commit | 帧间灰度MAD | 边界RGB MAD |
|---|---|---:|---:|---:|---:|---|---:|---:|
| fm32 | A | 4 | 189.96 | 38924.02 | 6484.13 | 12+3 | 3.0265 | 3.6075 |
| fm32 | D | 4 | 227.17 | 38924.02 | 6484.13 | 12+3 | 3.1837 | 3.9276 |
| fm32 | A | 8 | 231.63 | 38924.02 | 6484.13 | 24+3 | 3.2473 | 3.5093 |
| fm32 | D | 8 | 219.54 | 38924.02 | 6484.13 | 24+3 | 3.4369 | 4.3436 |
| anyflow32 | A | 8 | 270.44 | 38984.16 | 6484.13 | 24+3 | 3.7519 | 5.2148 |
| anyflow32 | D | 8 | 261.32 | 38984.16 | 6484.13 | 24+3 | 3.6789 | 4.8099 |
| anyflow32 | A | 4 | 193.98 | 38984.16 | 6484.13 | 12+3 | 4.1702 | 6.2343 |
| anyflow32 | D | 4 | 217.43 | 38984.16 | 6484.13 | 12+3 | 4.0438 | 6.2784 |

所有conditioning与对应Original teacher逐张量一致。39帧=3块，因此4/8步/块分别是12/24次noisy forward，另加3次clean commit。边界为17/34帧；帧间MAD表示运动活跃度，不是质量分数。单次并行运行、共享CPU/offload资源；原始H3的offload预算不同，不能用表中时间/显存计算公平speedup或显存节省。没有完成warmup后多次均值，也没有权重/激活显存独立拆分。

训练32次中：FM/AnyFlow current forwards=128/512；各30次clean commit、120次带梯度历史前向；不含验证和backward checkpoint重算。FM训练含准备/验证3215.65秒、GPU allocated peak41251.94MiB。AnyFlow分两段续训，额外启动/验证不等；同更新数不是等算力比较。

## 结论与后续

在这一个首帧、prompt、seed和两条teacher伪真值上，本次受控实验不支持AnyFlow32的少步收益；FM32本身也未过动作gate。不能外推为AnyFlow方法无效或官方Stage1不工作。GPU0/1已有的AnyFlow64有限训练量对照继续完成；不将AnyFlow64与FM32称为同训练量比较。不扩124f、不开始Stage2，先通过A>0、D<0、A−D>1且人物/场景稳定的短片gate。

## 同一光流指标按时间分段

沿用上述Farneback参数，仅按RGB时间拆分；不新增指标。17/34是现有边界口径，受temporal VAE感受野影响，不能把分段当独立latent chunk因果隔离实验。最后一段只有4个transition，估计不稳定。

| 方法 | 动作 | steps/chunk（Original为full） | 帧0–16 | 帧17–33 | 帧34–38 |
|---|---|---:|---:|---:|---:|
| Original | A | 30 | 0.083398 | 2.220752 | 1.570233 |
| Original | D | 30 | -0.894451 | -0.863608 | -0.630615 |
| fm32 | A | 4 | -0.857184 | -1.198588 | -1.319734 |
| fm32 | D | 4 | -1.291033 | -1.191697 | -1.409212 |
| fm32 | A | 8 | -0.882645 | -0.919771 | -1.306946 |
| fm32 | D | 8 | -1.404145 | -1.170357 | -1.501578 |
| anyflow32 | A | 8 | -1.205984 | -1.345781 | -1.205748 |
| anyflow32 | D | 8 | -1.494039 | -1.429047 | -1.274034 |
| anyflow32 | A | 4 | -1.298811 | -1.376988 | -0.267222 |
| anyflow32 | D | 4 | -1.555982 | -1.084254 | -0.299065 |

两种目标的A在首段均为负，Original首段+0.083398；差异在短片开头已出现。因此本次失败不能全部解释为后段generated-history累积漂移。但光流为场景运动proxy，不是独立动作分类器；不能仅据这张表指定mask、action representation或AnyFlow时间条件中的哪一个是根因。原始逐transition数据在[segment_response.json](segment_response.json)。
