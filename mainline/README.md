# Mainline：版本定义与研究路线

- [V0 Original](V0_original_bidirectional/README.md)：原始双向H3参考。
- [V1 Native Chunk-Causal](V1_native_chunk_causal/README.md)：因果/KV工程基础，动作与视觉退化。
- [V2家族](v2/README.md)：V2a RGB Anchor、V2b Same-σ两条并行修复方案，无权重继承。
- [V3家族](v3/README.md)：V3 Original Feasibility Baseline为已验收124帧正式参考；下设Sliding Window与FM → AnyFlow。
- [V4 SGF / DMD](v4/README.md)：分布匹配蒸馏尝试，无已验收生成模型。

V3采用Original H3 + released Action LoRA，并未串接V2a/V2b的训练checkpoint。EXP-005已完成GPU验证：SW-G158帧有限可行性通过；SW-L工程成立、生成PARTIAL且无已证实收益，当前无训练方向归档。FM/AnyFlow已完成有限实验；DMD归入[V4 SGF研究](v4/README.md)，当前配置生成失败。

[汇报导航](../report/README.md) · [版本路线图](../report/roadmap.md)
