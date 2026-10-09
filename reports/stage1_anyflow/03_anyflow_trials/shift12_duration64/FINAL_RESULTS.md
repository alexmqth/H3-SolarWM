# shift12/full-history：64次后8步动作分离增加，Stage1仍未通过

记录时间：2026-10-08T13:19:28.038242+08:00。所有A/D39f、4/8 steps per chunk已经完成并静态复核完整帧。

8步A−D由32次的0.172691升到64次的0.623096，但A仍为负值；4步后段仍严重雾化。不是Stage1验收通过，也没有证明AnyFlow优于匹配FM。

## 动作学习曲线

| updates | steps/chunk | A | D | A−D | gate |
|---:|---:|---:|---:|---:|---|
| 16 | 4 | -1.227654 | -1.232781 | 0.005126 | FAIL |
| 16 | 8 | -1.312275 | -1.492164 | 0.179888 | FAIL |
| 32 | 4 | -1.216895 | -1.197330 | -0.019565 | FAIL |
| 32 | 8 | -1.305473 | -1.478164 | 0.172691 | FAIL |
| 64 | 4 | -0.903425 | -1.052966 | 0.149541 | FAIL |
| 64 | 8 | -0.589203 | -1.212299 | 0.623096 | FAIL |

Original30：A+1.181253、D−0.842124、分离度2.023377。验收必须A>0、D<0、A−D>1且人物/场景完整。shift2.22同为64次的8步分离度0.215531，完整结果见[另一训练分布](../full_history_duration64/FINAL_RESULTS.md)。只有一个inference seed、两个teacher片段，不能泛化为已验证的普遍收益。

## 完整画面

四条完整0–38帧contact sheet和Original/16/64在12/24/30/38帧对照已查看。4步A/D约20–22帧后强重影、严重雾化，人物和停车场难以辨认。8步A/D保留人物和停车场到最后，仍有模糊、后段透明感；A没有Original的向左移动。分离度增加没有伴随明确的画质修复。这是静态完整帧复核，不等同真人实时播放或VBench。

- [Original / AnyFlow64四步 / 八步，A/D两行](shift12_anyflow64_AD_diagnostic.mp4)
- [Original / AnyFlow16 / AnyFlow64，8步学习曲线](original_anyflow16_anyflow64_8step_AD.mp4)
- [Original / AnyFlow16 / AnyFlow64，4步学习曲线](original_anyflow16_anyflow64_4step_AD.mp4)

视频保留全部39帧，24fps、H264/yuv420p/faststart，1.625秒原速，无裁剪失败后段；原始832×480，展示有缩放。完整解码通过，展示标签与第30帧另外复核。不会替换meeting主Demo。

## 实测推理记录

| action | steps/chunk | E2E s | GPU allocated peak MiB | CPU KV MiB | noisy+commit | 灰度帧间MAD | 边界RGB MAD |
|---|---:|---:|---:|---:|---|---:|---:|
| A | 4 | 182.09 | 38984.16 | 6484.13 | 12+3 | 3.8354 | 6.0062 |
| D | 4 | 179.46 | 38984.16 | 6484.13 | 12+3 | 3.8418 | 6.1022 |
| A | 8 | 225.52 | 38984.16 | 6484.13 | 24+3 | 2.9907 | 4.1526 |
| D | 8 | 225.63 | 38984.16 | 6484.13 | 24+3 | 3.3460 | 4.7583 |

所有conditioning与对应Original逐张量相同；seed13、generated history、RGB dual、causal action rows/feedback、推理shift2.22固定。4/8是每块步数，3块合计12/24 noisy+3 clean commits。CPU KV不是完整主机RSS，MAD是运动活跃度指标而非画质。单次共享主机执行、Original offload预算不一致，不声称公平speedup或峰值节省，也不是warmup后多次均值。

冻结runtime记录错误：cached.json中的anyflow_training_shift误写2.22，真实checkpoint config与training.time_sampling=12。旧文件保持原样，新runtime只修metadata读取。

## 训练和后续决策

新增32次含准备/验证5108.91秒，allocated40667.62MiB。原visual/action/time冻结，43,237,376参数的全52-block rank8 QKVO/FFN bank更新。完整[0/16/32/64固定验证](training_curve_00_16_32_64.json)：A endpoint32→64为59.536766→61.048344（变差），D22.905317→19.990650（改善）。实际sigma/r/类型/权重一致；原训练日志未记录实际GPU noise哈希，不作该哈希相等声明。自适应scaled loss不是动作/画质的替代指标。

匹配32次的FM8分离度0.413510高于AF32的0.172691；AF64现在虽到0.623096，却多了训练更新，不能据此证明AnyFlow优于FM。四卡同batch候选可在不改loss/data/global batch的前提下缩短后续训练等待。基于这条开始变化的8步学习曲线，选择shift12做一次有限64→68验证→96评测→最多128评测，短片gate仍严格保持；不扩长视频、不进入Stage2。
