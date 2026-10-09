# Full-history AnyFlow16：四条完整评测均未通过

完成于 2026-10-08T10:04:19.828032+08:00。GPU0/reserve6，训练allocated峰值40.34GiB。保留clean-history梯度后，16次更新仍未恢复A方向，也没有显示少步画质收益。

[完整39帧诊断：Original / 4 steps / 8 steps，A/D两行](full_history16_AD_diagnostic.mp4)

视频为H.264/YUV420P/24fps，保留后半段失败画面，标注NOT PASSED与真实forward计数。会议主视频未替换。

| Training | steps/chunk | A flow | D flow | A−D |
|---|---:|---:|---:|---:|
| detached16 | 4 | -1.198536 | -1.242114 | 0.043578 |
| detached16 | 8 | -1.305347 | -1.496471 | 0.191124 |
| full-history16 | 4 | -1.248505 | -1.224589 | -0.023916 |
| full-history16 | 8 | -1.304568 | -1.487077 | 0.182509 |

Original H3参考：A=+1.181253，D=−0.842124，A−D=2.023377。当前A均为负，未达到A>0、D<0、A−D>1与视觉完整同时成立的gate。

| Action | steps/chunk | E2E s | GPU allocated MiB | CPU KV MiB | noisy + commits | Gray MAD | Boundary RGB MAD |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 4 | 210.80 | 38984.16 | 6484.13 | 12+3 | 4.1677 | 6.2543 |
| D | 4 | 206.97 | 38984.16 | 6484.13 | 12+3 | 4.0305 | 6.2965 |
| A | 8 | 267.35 | 38984.16 | 6484.13 | 24+3 | 3.8129 | 5.2343 |
| D | 8 | 257.68 | 38984.16 | 6484.13 | 24+3 | 3.7220 | 4.9371 |

## 视觉与解释

四条全部0–38帧已通过全帧contact sheet复核，另与Original/FM16/detached16逐项对比12/24/30/38帧。4-step的A/D均在约20帧后严重雾化；8-step两条保留人物与车库主体但仍模糊。相对detached16无明确改善。静态全帧审阅不能替代真人实时播放判断动作自然度。

所有保存的conditioning与原始H3逐张量相同。推理仍为39f、3 chunks、CPU raw KV、RGB dual、causal action rows/feedback、seed13与shift2.22。两组更新数相同但full-history额外56次带梯度历史前向，训练计算量不同。独立GPU存在反传重复性误差，不把微小数值差异全归因于history梯度。

时间为与GPU1训练并行时的单次recorded run，无warmup多次均值，不能声称速度收益；Gray MAD与Boundary RGB MAD口径不同，也不是画质分数。

训练损失和初始化审计见[GPU_RESULTS](GPU_RESULTS.md)。这只否定了本轮16-update配置已产生收益，不说明AnyFlow或完整历史梯度无用。后续固定配置检查训练量；GPU1的训练shift12受控16次对照继续。Stage1未验收，Stage2暂缓。
