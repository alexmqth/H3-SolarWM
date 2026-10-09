# full-history / shift2.22：64次完整评测仍未通过

记录时间：2026-10-08T12:56:34.789330+08:00。A/D39f、4/8 steps per chunk全部完成并检查。A方向仍错误；4步后段严重雾化，8步人物/场景保持但模糊、透明感和动作偏差仍在。Stage1未通过，Stage2暂缓。

## 动作学习曲线

| 更新数 | steps/chunk | A | D | A−D | Gate |
|---:|---:|---:|---:|---:|---|
| 16 | 4 | -1.248505 | -1.224589 | -0.023916 | FAIL |
| 16 | 8 | -1.304568 | -1.487077 | 0.182509 | FAIL |
| 32 | 8 | -1.309516 | -1.497935 | 0.188419 | FAIL |
| 64 | 4 | -1.000683 | -1.153995 | 0.153312 | FAIL |
| 64 | 8 | -1.072072 | -1.287603 | 0.215531 | FAIL |

Original30 reference：A=+1.181253、D=−0.842124、A−D=2.023377。当前8步分离度16→32→64为0.182509→0.188419→0.215531，尚未恢复动作方向。4步flow靠近零不能证明改善，因为后段严重雾化会降低有效运动。

## 画面与可播放证据

四条视频全部0–38帧contact sheet、Original/16/64在12/24/30/38帧的对应画面已检查。4步A/D约22帧后强重影、雾化，人物和车库逐渐难以辨认；8步两条保留主体和车库到38帧，但仍模糊且有透明感，A没有恢复Original中的左移动作。这是静态完整帧与同帧复核，不等同真人实时播放或VBench评价。

- [Original / 4-step / 8-step，A/D两行](full_history64_AD_diagnostic.mp4)
- [Original / AnyFlow16 / AnyFlow64，8-step学习曲线](original_anyflow16_anyflow64_8step_AD.mp4)
- [Original / AnyFlow16 / AnyFlow64，4-step学习曲线](original_anyflow16_anyflow64_4step_AD.mp4)

全部为39f/24fps/H264/yuv420p/faststart，完整解码通过，按原速1.625秒保留所有失败后段；展示缩放，原视频为832×480。诊断视频不替换会议主Demo。

## 实测运行记录

| 动作 | steps/chunk | E2E s | GPU allocated peak MiB | CPU KV MiB | noisy+commit | 帧间灰度MAD | 边界RGB MAD |
|---|---:|---:|---:|---:|---|---:|---:|
| A | 4 | 213.16 | 38984.16 | 6484.13 | 12+3 | 3.7956 | 5.7326 |
| D | 4 | 212.29 | 38984.16 | 6484.13 | 12+3 | 3.8728 | 5.7155 |
| A | 8 | 261.91 | 38984.16 | 6484.13 | 24+3 | 3.6738 | 4.9343 |
| D | 8 | 260.55 | 38984.16 | 6484.13 | 24+3 | 3.4453 | 4.8840 |

全部conditioning与对应Original逐张量一致；CPU raw KV、generated history、RGB dual、causal action rows/feedback、推理shift2.22、seed13保持固定。4/8步是每块步数，3块合计12/24 noisy+3 clean commits。MAD描述活动量，不是画质；边界为17/34帧。单次并行主机时间且Original offload预算不同，不计算公平speedup或显存节省，没有声称warmup后多次均值。

## 训练证据与下一步

完整0/16/32/64固定验证曲线在[training_curve_00_16_32_64.json](training_curve_00_16_32_64.json)。新增32次训练及准备/验证5241.37秒，allocated峰值40668.18MiB；原visual/action/time冻结，全覆盖rank8 bank四组更新。endpoint32→64 A49.928→47.883、D24.234→18.441，diffusion raw也下降；internal拟合改善未带来当前rollout验收。

本轮覆盖两个39f teacher伪真值、64 optimizer updates、256 loss samples。这是有限训练量的结果，不是官方大规模Stage1的充分训练证明，也不能据此断言AnyFlow方法无效。GPU1的shift12总64对照继续；收齐后结合学习曲线决定下一轮预算。四卡同batch训练候选只完成真实33B只读与CPU多步/Adam恢复验证，尚未进行33B多卡optimizer训练。
