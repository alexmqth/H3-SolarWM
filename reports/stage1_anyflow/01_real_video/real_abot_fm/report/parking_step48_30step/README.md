# 停车场已知动作正控：完整 A/D 对照

Causal 为 **30 steps/chunk**，三块共 90 noisy forwards + 3 clean commits；Original 为整段30次。

[完整两行对比视频](original_causal_AD.mp4) · [完整指标CSV](metrics.csv) · [机器记录](metrics.json)

| 方法 | flow(A) | flow(D) | A−D | 仅数值门槛 |
|---|---:|---:|---:|---|
| Original H3 30 full-seq | +1.181253 | -0.842124 | 2.023377 | True |
| real-FM0 parking30 | -0.195516 | -0.186950 | -0.008565 | False |
| real-FM48 parking30 | -0.182680 | -0.163995 | -0.018685 | False |

最终画质/action验收仍须检查完整39帧，自动表格不判视觉PASS。Original旧legacy精度、causal0/48统一h3_fp32，且anchor/prefix协议不同，因此与Original是pipeline对比；训练前后causal才是匹配配置。

时间是单次共享主机记录，Original offload reserve未知，不声称加速或显存优化。GT与generated-history不同场景的总flow不能代替这里的同图A/D。
