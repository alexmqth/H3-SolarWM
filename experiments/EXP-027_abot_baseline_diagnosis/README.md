# EXP-027：ABot 基线诊断与下一轮实验准备

当前阶段：CPU盘点已接受，v2两片C2普通FM拟合已授权执行，结果待交付。单L40≤6000GPU秒，64updates，固定独立噪声probe及4个必要前后视频；不以10%代理指标作为模型验收硬门槛。

目的：在H3-World语言动作与V3 causal/KV协议上，区分小预算训练尚无学习信号与自生成历史/动作切换问题，确定值得投入的下一项实验。ABot为项目现有Gameplay数据来源；GT为真实视频历史；FM为Flow Matching，NFE为每chunk推理网络求值次数，training updates单独计数。

[当前任务书](taskbook_v2.md) · [授权](AUTHORIZATION_v2.json) · [Judge审核](judge/V1_REVIEW.md) · [CPU证据](EVIDENCE_REVIEW.md)。复用EXP-023/025的实际数据与证据，SolarWM官方参考不是前置条件。
