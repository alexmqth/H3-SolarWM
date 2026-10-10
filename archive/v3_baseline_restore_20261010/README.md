# 恢复V3 Baseline及Sliding Window候选分类

2026-10-10，用户最新决定取代短暂V2c分类。正式参考为V3 Original Feasibility Baseline；独立候选V3-SW-G、V3-SW-L均未获得生成能力验收。

- [迁移路径和原始SHA](migration.json)：74个文件，迁移前Git基点`575530a3b5d660f8ac3559502a848d08c3e2a1b1`。
- [逐文件核验](validation.json)：65个文件逐字节相同；其余7份当前说明/分类metadata更新，2段8步展示片恢复为EXP-004冻结V3原片。
- `before_documents/`保存变更前当前说明；`previous_display_videos/`保存此前CPU重标的V2c展示片。展示文件不重新编码。
- 107项冻结证据检查通过，包括正式124帧视频/配置/代码及EXP-002/003原始报告、日志和产物。本次没有GPU推理、训练或新增视频。

规范目录为`report/v3/{v3_baseline,v3_sw_g,v3_sw_l}`与`mainline/v3/`对应分组。`mainline/V3_efficient_causal`保留兼容入口；EXP-004正式目录恢复`experiments/EXP-004_v3_8step`，短暂V2c实验名保留反向链接。

[当前V3导航](../../report/v3/README.md) · [EXP-005 CPU证据](../../experiments/EXP-005_v3_sliding_window/README.md)。目录与命名恢复不改变历史能力结论。
