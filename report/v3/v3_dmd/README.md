# V3-DMD：真实8-map训练工程已验证

**EXP-008已完成累计8个student DMD cycle并通过训练工程审核；最终cycle8视频正在推进，尚无质量收益结论。** 正式V3 Baseline及AnyFlow/FM8证据冻结保留。

Teacher为冻结V3 causal H3，fake为独立普通FM/QKV，student从EXP-007 AF2 step32的新AnyFlow权重继承。三角色分别在GPU0/2/5使用各自clean C1 KV；student保留全部8-map计算图，fake消费detach生成端点、更新后重建自身KV。不是原生双向teacher或完整SolarWM Stage2复现。

实际17forward、3backward、3update、0VAE，254.899秒，保守三卡合计0.212416GPU小时。Allocated峰值teacher25.070、fake25.879、student27.965GiB。独立审计确认全部8map梯度有效、target/QKV确实更新、配套checkpoint/optimizer/RNG与冻结来源一致。

仅固定FM8 C1上的AA C2 current-block on-policy，独立训练噪声；不能由一次更新或loss判断动作、画质或多块长期能力。下一v2已批准累计8cycles并用匹配AF3/FM8的视频检验，训练和视频分阶段预算，不自动扩展。

[实验入口](../../../experiments/EXP-008_v3_dmd/README.md) · [Worker报告](../../../experiments/EXP-008_v3_dmd/worker_report_v1.md) · [Judge审核](../../../experiments/EXP-008_v3_dmd/judge/PILOT_REVIEW.md) · [独立审计](../../../experiments/EXP-008_v3_dmd/judge/pilot_audit.json) · [有限延续任务书](../../../experiments/EXP-008_v3_dmd/taskbook_v2.md)

## 有限延续已完成

v2新增7cycles，99forward/14backward/14update/0VAE，27.89分钟wall，三卡1.394701GPU小时，student峰28.102GiB。全部训练噪声从pilot RNG独立重放一致，cycle4/8配套checkpoint通过审核。Fake loss曾在cycle4升至31.53，后约2.5，student梯度变小；surrogate降低不代表画质提升。

[训练Worker报告](../../../experiments/EXP-008_v3_dmd/worker_report_v2.md) · [Judge训练审核](../../../experiments/EXP-008_v3_dmd/judge/TRAIN_V2_REVIEW.md) · [视频放行与停止条件](../../../experiments/EXP-008_v3_dmd/EVAL_RELEASE.md)。最终cycle8最多35forward/4VAE/.35GPUh；C2若持续崩溃则停止C3，不追加训练。
