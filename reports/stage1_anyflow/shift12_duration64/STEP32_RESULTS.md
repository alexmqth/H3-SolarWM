# Full-history / train shift12 / AnyFlow32：完整短片仍失败

记录时间：2026-10-08T11:38:14.383176+08:00。这里只评估总32次checkpoint；64次仍在训练。

[完整39帧：Original H3 / AnyFlow32 4 steps / AnyFlow32 8 steps，A/D两行](shift12_anyflow32_AD_diagnostic.mp4)

视频H.264/yuv420p/24fps，已完整解码验证，并检查标注预览。4步由GPU4补评测，8步由GPU1原队列生成；读取同一step32。完整失败后半段保留，未替换meeting主视频。

| Method | steps/chunk | A flow | D flow | A−D |
|---|---:|---:|---:|---:|
| Original H3 | 30 full sequence | +1.181253 | −0.842124 | 2.023377 |
| shift12 AnyFlow16 | 4 | −1.227654 | −1.232781 | 0.005126 |
| shift12 AnyFlow16 | 8 | −1.312275 | −1.492164 | 0.179888 |
| shift12 AnyFlow32 | 4 | -1.216895 | -1.197330 | -0.019565 |
| shift12 AnyFlow32 | 8 | -1.305473 | -1.478164 | 0.172691 |

A均为负，不满足A>0、D<0、A−D>1与画面完整的联合gate。增加到32次没有显示动作恢复。

| Action | steps/chunk | E2E s | allocated GPU MiB | CPU KV MiB | noisy + commits | Gray MAD | Boundary RGB MAD |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 4 | 193.98 | 38984.16 | 6484.13 | 12+3 | 4.170234 | 6.234297 |
| D | 4 | 217.43 | 38984.16 | 6484.13 | 12+3 | 4.043825 | 6.278421 |
| A | 8 | 270.44 | 38984.16 | 6484.13 | 24+3 | 3.751948 | 5.214816 |
| D | 8 | 261.32 | 38984.16 | 6484.13 | 24+3 | 3.678931 | 4.809912 |

四条全部0–38帧contact sheet与Original/step16/step32在12/24/30/38帧的对照均已审阅。4-step的A/D约20帧后仍严重雾化、重影，人物和车库结构退化；8-step主体保留但仍模糊，D后段透明感存在。相对16次没有明确改善。这里是静态完整帧复核，不冒称真人实时播放。

完整输入对照均与原始H3逐张量一致：image/prompt/action rows/video noise/audio noise。train shift12，validation/inference shift2.22；全覆盖rank8、full-history训练、RGB dual、CPU KV、causal action/feedback均固定。两条GPU上的推理相同checkpoint、相同权重和设置，只改变NFE。

本轮与GPU0/3训练并行，耗时是单次记录，非warmup多次均值，不作speedup主张。MAD不是画质分数，gray帧差和RGB边界统计口径不同。冻结runtime的anyflow_training_shift日志字段误标2.22，实际训练12由checkpoint/setup/training.json证明。

训练raw loss和全部128个实际sigma/r对计划的核验见[GPU_RESULTS.md](GPU_RESULTS.md)。GPU3匹配full-history/shift12/FM32还在运行，不能提前比较方法优劣；相同更新数也不等训练算力。GPU1原控制器已经继续32→64，GPU0另一训练分布分支继续；没有进入Stage2。
