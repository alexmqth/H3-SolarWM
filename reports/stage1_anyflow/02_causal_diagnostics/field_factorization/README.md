# 同状态 field 因果化对照（实验 A）

[完整结论](RESULTS.md) · [逐点指标](metrics.csv) · [汇总收据](analysis.json)

12 个状态、9 个角色、224 次真实 H3 前向；没有 optimizer 更新。未训练 causal 的整体 velocity cosine 为 0.996253，但 A/D 差分 cosine 仅 0.060153。此前匹配的 FM32/AnyFlow32 都没有恢复动作几何。

所有角色采用同一 SDPA 后端和同长度完整历史重算。**这是完整输入的 field 诊断，不能当作 persistent-KV 推理等价性证明。** Native FlexAttention 对照也完整保留；动作小差分在 BF16 下对后端有敏感性。原始 receipt 记录本机路径；源 runner 依赖原 outputs 布局和此前 probe，提交包不含模型、delta tensors 或 conditioning。

后续是实验 B 的真实 ABot causal FM 桥接；A 结果本身不是视频修复。

![同状态field与动作差分](field_geometry.png)
