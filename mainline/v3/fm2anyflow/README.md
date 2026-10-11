# V3 FM → AnyFlow

FM = Flow Matching（流匹配）。本分组的目的：降低每块采样成本，对照原权重直接减步与新AnyFlow有限映射训练的收益。

[FM-8NFE版本证据](fm8/README.md)为每块8步普通采样，零新训练；[报告总览](../../../report/v3/fm2anyflow/README.md)收录FM-8NFE续写、全程FM-8NFE、FM-4NFE/AF-4NFE、AF-8NFE、迁移和数据准备。AnyFlow没有证明整体优于FM-8NFE，当前失败配置保持归档。

本目录记录定义，视频统一在报告各尝试的videos/。
