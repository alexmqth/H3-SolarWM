# V3-AF：AnyFlow新student研究中

**当前只验收训练工程可行性，生成能力尚未测试。** 正式V3 Baseline结果冻结；普通FM8与AnyFlow分别评价。

EXP-007 AF0 CPU和AF1真实33B单步训练通过。初始化对角输出与V3逐值一致，target-time/QKV有有效梯度与更新，基础权重不变，更新后按自身权重重建历史KV。累计22forward/4backward/1update/0VAE、186.58153秒，峰值26.63145GiB。

下一阶段已批准最多累计32updates，随后使用共同FM8首窗、相同动作/噪声、各自权重KV比较8NFE续写。当前没有新AnyFlow代表视频，不将早期旧协议产物作为完成证据。

[实验入口](../../../experiments/EXP-007_v3_anyflow/README.md) · [Judge审核](../../../experiments/EXP-007_v3_anyflow/judge/AF1_REVIEW.md) · [当前冻结任务书](../../../experiments/EXP-007_v3_anyflow/taskbook_v3.md) · [普通FM8参考](../v3_fm8/README.md)
