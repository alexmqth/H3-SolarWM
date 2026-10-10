# Research Branches

这些是主线的机制诊断与先行训练探索，不是一条已经全部成功的checkpoint继承链。

- [A · Causal Mechanism Diagnostics](A_causal_diagnostics/README.md)：分块、条件、路由、KV、VAE与时间位置。
- [B · Causal Adaptation & Action Recovery](B_causal_adaptation/README.md)：FM适配、mix/replay、action监督、RGB视觉修复和真实ABot。
- [C · AnyFlow & DMD Explorations](C_anyflow_dmd/README.md)：finite-map数值、16/64/128/136试验及critic/DMD准备。

状态：`implementation verified`只说明接口/数值；`preliminary trained`只说明做过更新；`quality failed`表示未过对应动作/视觉门槛；`not yet validated`表示缺少能力证据。可同时出现，不能用“训练完成”替代“能力通过”。

[逐子实验机器索引](EXPERIMENT_INDEX.json) · [完整旧档案入口](../reports/stage1_anyflow/README.md) · [主线](../mainline/README.md) · [汇报包](../report/README.md)
