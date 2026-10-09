最新完整64次结果：[FINAL_RESULTS.md](FINAL_RESULTS.md)。下文保留当时快照。

# shift12/full-history：总64次训练完成，视频评测尚未齐全

时间：2026-10-08T13:09:03.067937+08:00。Stage1尚未通过。

新增32次含准备与验证 5108.91 秒，allocated峰值 40667.62 MiB。四组bank更新，原visual与target-time冻结。

| updates | A endpoint raw | D endpoint raw |
|---:|---:|---:|
| 0 | 59.624653 | 22.804770 |
| 16 | 54.556305 | 24.329163 |
| 32 | 59.536766 | 22.905317 |
| 64 | 61.048344 | 19.990650 |

4 steps/chunk A/D已完成：A=−0.903425、D=−1.052966、A−D=0.149541。两条全部39帧与Original/16/64同帧已静态复核；约20–22帧后重影并严重雾化，人物与停车场难以辨认。没有明确视觉修复。8步A已完成，flow=−0.589203；全39帧与Original/16/64同帧已复核，人物/停车场保留，但后段透明、模糊，A仍非原始向左运动。D8仍在运行，尚不能报告64次的完整8步分离度。

A endpoint在16→32→64继续变差，D改善；不能用自适应缩放后的weighted总loss下降宣称画质改善。验证sigma/r/类型/权重逐项一致，原训练日志没有actual GPU noise hash，因此不宣称其哈希一致。

原冻结runtime的cached.json将训练shift误标2.22；真实checkpoint配置与training.time_sampling为12，保留旧文件及更正说明。
