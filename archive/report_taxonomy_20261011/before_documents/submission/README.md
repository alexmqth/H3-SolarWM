# H3-SolarWM：因果世界模型研究

目标：在H3-World上统一动作控制、基本视觉稳定和高效因果推理，再研究少步生成与On-policy DMD。

**已验收正式参考：[V3 Original Feasibility Baseline](mainline/v3/v3_baseline/README.md)**，严格因果/persistent raw KV/Global RoPE，AA/AD124帧有限可行性。画质与连续性仍PARTIAL。

**当前：[EXP-005 V3 Sliding Window结果](experiments/EXP-005_v3_sliding_window/README.md)**。SW-G（Global）158帧有限可行性通过，画质PARTIAL；SW-L工程通过但无已证实收益，当前无训练路线归档。281forward/9VAE/0.662707GPU小时，Baseline保持冻结。

- [汇报与视频](report/README.md) · [浏览器](report/index.html)
- [V2家族](report/v2/README.md)：RGB Anchor / Same-σ
- [V3家族](report/v3/README.md)：Baseline / SW-G / SW-L
- [主线定义](mainline/README.md) · [研究总览](REPORT.md)
- [复现与依赖](REPRODUCE.md) · [研究分支](branches/README.md)

原“V2c”分类按用户最新决定恢复为V3 Baseline，历史迁移记录保留。EXP-004普通FM8步续写只验证到73帧，首窗仍借用30步结果，不能视为AnyFlow训练完成。


**新增EXP-006：[全程V3-FM8](report/v3/v3_fm8/README.md)** 已验收新首39+AA/AD73有限可行性，quality PARTIAL，0.136905GPU小时、零训练。当前转入独立V3-AF初始化/单步训练准备，尚无AnyFlow能力验收。
