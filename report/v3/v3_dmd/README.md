# V3-DMD：真实8-map训练工程已验证

**EXP-008/v1单cycle工程可行性通过；尚无DMD视频或质量收益结论。** 正式V3 Baseline及AnyFlow/FM8证据冻结保留。

Teacher为冻结V3 causal H3，fake为独立普通FM/QKV，student从EXP-007 AF2 step32的新AnyFlow权重继承。三角色分别在GPU0/2/5使用各自clean C1 KV；student保留全部8-map计算图，fake消费detach生成端点、更新后重建自身KV。不是原生双向teacher或完整SolarWM Stage2复现。

实际17forward、3backward、3update、0VAE，254.899秒，保守三卡合计0.212416GPU小时。Allocated峰值teacher25.070、fake25.879、student27.965GiB。独立审计确认全部8map梯度有效、target/QKV确实更新、配套checkpoint/optimizer/RNG与冻结来源一致。

仅固定FM8 C1上的AA C2 current-block on-policy，独立训练噪声；不能由一次更新或loss判断动作、画质或多块长期能力。下一v2已批准累计8cycles并用匹配AF3/FM8的视频检验，训练和视频分阶段预算，不自动扩展。

[实验入口](../../../experiments/EXP-008_v3_dmd/README.md) · [Worker报告](../../../experiments/EXP-008_v3_dmd/worker_report_v1.md) · [Judge审核](../../../experiments/EXP-008_v3_dmd/judge/PILOT_REVIEW.md) · [独立审计](../../../experiments/EXP-008_v3_dmd/judge/pilot_audit.json) · [有限延续任务书](../../../experiments/EXP-008_v3_dmd/taskbook_v2.md)
