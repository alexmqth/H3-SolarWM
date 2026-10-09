# AnyFlow96 / 8 steps per chunk：动作分离回落，D后段画质恶化

记录：2026-10-08T14:15:04.837262+08:00。A/D39f评测全部完成；Stage1未通过。

A=−0.483647、D=−0.729654、A−D=0.246007。相较64次0.623096，分离度回落；A仍方向错误。96次D约20帧后重影明显增强、后段雾化，人物/停车场结构显著劣于64次。不能把最新checkpoint默认作为最好版本。

## 同一训练协议的8步学习曲线

| optimizer updates | A | D | A−D |
|---:|---:|---:|---:|
| 16 | -1.312275 | -1.492164 | 0.179888 |
| 32 | -1.305473 | -1.478164 | 0.172691 |
| 64 | -0.589203 | -1.212299 | 0.623096 |
| 96 | -0.483647 | -0.729654 | 0.246007 |

Original30参考：A+1.181253/D−0.842124/分离度2.023377。短片gate仍为A>0、D<0、A−D>1且人物/场景完整；64次也未过gate，只是当前较好的参考点。

## 可播放对照和画面复核

[Original / AnyFlow64 / AnyFlow96，A/D两行](original_anyflow64_anyflow96_8step_AD.mp4)。全部39帧/24fps、H264/yuv420p/faststart，1.625秒原速，完整解码和第30帧标签已检查。视频保留全部失败后段，不替换meeting主Demo。

两条全0–38帧contact sheet，以及Original/64/96在12/24/30/38帧的对应画面已静态查看。A仍保留人物和车库，但约20帧后模糊/透明和重影未消失，相较64次无明确视觉修复。D后段比64次明显恶化，约22帧起人像和环境叠影，末段严重雾化。此处是完整静态帧复核，不等于实时播放或VBench。

## 实测记录

| action | E2E s | allocated peak MiB | CPU KV MiB | noisy+commit forwards | 灰度帧间MAD | 边界RGB MAD |
|---|---:|---:|---:|---|---:|---:|
| A | 262.64 | 38984.16 | 6484.13 | 24+3 | 2.9961 | 4.5575 |
| D | 264.37 | 38984.16 | 6484.13 | 24+3 | 3.1267 | 4.2853 |

8步是每块步数；3块合计24 noisy+3 clean commits。generated history/CPU raw KV/RGB dual/causal action rows及feedback/seed13/推理shift2.22均不变，全部conditioning与对应Original逐张量一致。新runtime正确记录训练shift12。MAD是活动量，低值可能来自模糊；CPU KV不等于主机总RSS。单次共享主机时长且Original offload配置不同，不作公平speedup或显存节约声明，也未取warmup后多次均值。

## 训练误差与实际效果分离

真实GPU4–7完成68→96，新增28次及加载/验证1491.87秒，allocated峰值40504.14–40505.35MiB/卡。四replica参数/Adam/三种RNG一致，旧visual/time冻结，四组bank都更新；实际68恢复全部相等。完整[训练记录](TRAIN96_RESULTS.md)及[固定0/16/32/64/68/96验证](training_curve_00_16_32_64_68_96.json)已归档。A endpoint64→96从61.048降至35.395、D19.991→18.416，但generated-history动作分离与D画质反而退步。不能用clean-history内部拟合改善代替实际rollout效果。

按此前唯一的有限预算，队列已从完整step96继续最多128，再评测A/D4/8。96只评测8步，不把之前64次的4步结果冒充96；没有自动扩124f或开始Stage2。128之后必须重新基于完整结果决策，不自动继续加更新数。
