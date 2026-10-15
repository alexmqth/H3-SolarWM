# EXP-025/v1 · Full50 C1/C2 ordinary FM

研究问题：在有限真实数据预算内，全50层 QKV rank8 与 C1/C2 等频普通 FM 的联合方案，是否比此前 tail8/C2-only 方案产生清楚的视频或动作收益？这是联合方案，不是容量和训练块覆盖的单变量消融。

**结果：训练与评测链路完成，能力收益未证实；Judge已accepted任务执行，视觉/动作PARTIAL。** 32条 train、2条已观察 dev；C1/C2各128次更新，最终6条56帧固定视频与checkpoint0的6条对照。C2固定probe MSE约下降1.3%，C1微升；视频没有跨角色一致的明确画质或动作优势。没有进行同状态A/D反事实或长视频测试。

从 [完整Worker报告](WORKER_REPORT.md) 开始；[任务书](taskbook_v1.md)、[预算和资源账本](GPU_LEDGER.json)、[固定数值摘要](METRICS_SUMMARY.json)、[12原片独立审计](INDEPENDENT_VIDEO_AUDIT.json)、[6对照媒体审计](comparisons/MEDIA_AUDIT.json) 提供追溯。

| 对照 | MP4 |
| --- | --- |
| train constant, GT39→17 | [GT / zero / 256](comparisons/train_constant_GT39_generated17_GT_zero_256.mp4) |
| train switch, GT39→17 | [GT / zero / 256](comparisons/train_switch_GT39_generated17_GT_zero_256.mp4) |
| dev constant, GT39→17 | [GT / zero / 256](comparisons/dev_constant_GT39_generated17_GT_zero_256.mp4) |
| dev switch, GT39→17 | [GT / zero / 256](comparisons/dev_switch_GT39_generated17_GT_zero_256.mp4) |
| dev constant, self56 | [GT / zero / 256](comparisons/dev_constant_self56_GT_zero_256.mp4) |
| dev switch, self56 | [GT / zero / 256](comparisons/dev_switch_self56_GT_zero_256.mp4) |

模型权重和 optimizer checkpoint 的大型 `.pt` 文件仅保留本机；元数据 `checkpoint_000/064/128/256.json` 给出文件SHA。运行代码为 [run_exp025.py](run_exp025.py)，[finalize_cpu.py](finalize_cpu.py) 只用已保存RGB制作对照。未改冻结的EXP-023/024源码，未操作Git。

[Judge最终审核](judge/FINAL_REVIEW.md) · [汇报视频与成本](../../report/v3/fm2anyflow/fm_full_coverage/README.md)。
