# FM sampling density 对照：有限预算收尾

2026-10-09 02:57。48 次训练、54 个 noise 点、GT/generated 各 18 个同状态 action 点、10 条完整 39 帧候选视频及 8 个报告组全部完成。局部动作/画面门槛 FAIL。本轮完成不代表 Stage1 完成；按用户新要求停止扩展旧 A–D，转向三个有决策意义的实验。

| 停车场方法 | A flow | D flow | A−D |
|---|---:|---:|---:|
| Original30 full-seq（旧精度参考） | +1.181253 | −0.842124 | 2.023377 |
| 旧 shift12 FM48，30/chunk | −0.182680 | −0.163995 | −0.018685 |
| 新 shift2.22 FM48，30/chunk | −0.190239 | −0.158556 | −0.031684 |
| 旧 shift12 FM48，8/chunk | −0.025848 | −0.043304 | 0.017456 |
| 新 shift2.22 FM48，8/chunk | −0.042918 | −0.116288 | 0.073371 |

30 步两条新候选的全 39 帧静态图中，人物大体完整、运动变化偏弱；A/D 无正确相反方向。8 步两条在约 23 帧后人物边缘重复、半透明/重影明显，不能用更小 MAD 宣称改善。这里是全部帧 contact sheet 检查，不是实时播放评审。自然两场景的六条视频已另行逐帧检查，第二场景 generated30/8 人物分解；[完整自然视频结论](NATURAL_VIDEO_RESULTS.md)。

后两 chunk 的 12 点：fixed-generated history 新 delta cosine 0.035019，旧为 0.029239；GT 新为 −0.002418，旧为 0.011453。54 点等权 raw MSE 新比旧 +0.195%，低/中略降而高段上升；采样因素未修复 action geometry。全 18 点统计与 12 点不可混算。

[30 步对比](report/parking_30/README.md) · [8 步对比](report/parking_8/README.md) · [几何](GEOMETRY_RESULTS.md) · [noise](report/noise/INTERPRETATION.md)。四个停车场并排 MP4 已完整解码，均 H264/yuv420p/24fps/39f/faststart，归档哈希核对通过。

时延为共享机器单次结果；新30步 A/D 为539.92/576.95秒，新8步为232.40/221.46秒。两种 density 的GPU/offload环境记录不同，不用时延或显存差声称算法收益。新候选峰值allocated26814.75MiB，CPU raw KV6484.13MiB；30/chunk为90 noisy forwards + 3 commits，8/chunk为24 + 3。Original与causal条件/精度不同，只是pipeline级展示；新旧density之间是匹配的训练因素对照。

终态核查见[closeout.json](closeout.json)：10项GPU队列任务、8组CPU报告全部exit0，原训练/控制器/评测进程已退出，306项冻结runtime及queue/report源hash保持。没有续训或新GPU任务。收尾脚本初次因默认shell无ffprobe失败，未更改模型或结果；改用项目已有PyAV完整解码后通过。

下一步不是继续48/136预算，也不启动原来准备的单步paired-velocity GPU试验。新方案见提交包的 `three_experiments/PROTOCOL.md`；会议主视频保持现状。
