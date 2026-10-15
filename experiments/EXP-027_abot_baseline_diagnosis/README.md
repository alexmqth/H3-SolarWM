# EXP-027：ABot 基线诊断与下一轮实验准备

当前阶段：CPU盘点与单一GPU方案准备，尚未运行训练/推理，因此没有新视频。

目的：在H3-World语言动作与V3 causal/KV协议上，区分小预算训练尚无学习信号与自生成历史/动作切换问题，确定值得投入的下一项实验。ABot为项目现有Gameplay数据来源；GT为真实视频历史；FM为Flow Matching，NFE为每chunk推理网络求值次数，training updates单独计数。

[任务书](taskbook_v1.md) · [授权](AUTHORIZATION.json)。复用EXP-023/025的实际数据与证据，SolarWM官方参考不是前置条件。
