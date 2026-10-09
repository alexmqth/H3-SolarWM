# Train shift12 AnyFlow16：四条评测完成，未通过

完成于 2026-10-08T10:35:56.707520+08:00。仅训练时间采样及其Gaussian权重改为12，公共validation/inference仍2.22；其余完整历史梯度、FP32、全QKVO/FFN rank8、初始化/数据/动作/anchor与16次预算固定。

[完整39帧视频：Original / 4 steps / 8 steps，A/D两行](shift12_anyflow16_AD_diagnostic.mp4)

四条全部帧已解码并通过0–38帧contact sheet复核，另与Original/FM/训练shift2.22按12/24/30/38帧比较。4-step两动作约20帧后仍严重雾化和重影；8-step两动作保留人物与车库主体但模糊，A仍朝错误方向。未见相对shift2.22的明确少步画质收益。

| Training shift | steps/chunk | A flow | D flow | A−D |
|---|---:|---:|---:|---:|
| 2.22 | 4 | -1.248505 | -1.224589 | -0.023916 |
| 2.22 | 8 | -1.304568 | -1.487077 | 0.182509 |
| 12 | 4 | -1.227654 | -1.232781 | 0.005126 |
| 12 | 8 | -1.312275 | -1.492164 | 0.179888 |

Original H3：A=+1.181253、D=−0.842124、分离度2.023377。两种训练shift均未满足A>0、D<0、A−D>1与视觉完整。

| Action | steps/chunk | E2E s | GPU allocated MiB | CPU KV MiB | noisy + commit | Gray MAD | Boundary RGB MAD |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 4 | 183.09 | 38984.16 | 6484.13 | 12+3 | 3.9201 | 6.0099 |
| D | 4 | 234.62 | 38984.16 | 6484.13 | 12+3 | 4.0412 | 6.4186 |
| A | 8 | 268.77 | 38984.16 | 6484.13 | 24+3 | 3.8185 | 5.3323 |
| D | 8 | 264.12 | 38984.16 | 6484.13 | 24+3 | 3.6973 | 4.8124 |

训练含准备/验证2516.05秒，allocated峰值41310.95MiB。完整训练初始化/噪声/64个sample及历史前向审计通过；[公共验证表](GPU_RESULTS.md)显示A endpoint下降、D上升，weighted变化很小。固定noise validation与训练采样覆盖均有独立记录。

以上为多卡并行时的单次offload运行，不是warmup均值或speedup证据。GPU allocated与CPU KV分列，但没有独立测量权重/激活瞬时占用；不得从峰值简单减出可靠的activation指标。MAD是像素变化而不是画质评分。

[同checkpoint的teacher/generated history诊断](../shift12_history_diagnostic/FINAL_RESULTS.md)已完成。两者均未通过；teacher历史能约束后续场景，但有oracle reset跳变，不能作为正式Demo。

下一步只有同协议训练量对照：GPU0 shift2.22、GPU1 shift12各自恢复step16/Adam/RNG到总32→最多64，保持其它条件固定。16次微小更新不足以否定AnyFlow方法，也不构成当前有效的证明。Stage1仍未验收，Stage2暂缓。
