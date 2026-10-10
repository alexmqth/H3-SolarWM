# V3-SW-G — 独立研究候选

**状态：CPU准备与验证；GPU生成能力未测试，未正式验收。**

保留Baseline全部条件与Global RoPE，泛化显式chunk计划，实际淘汰第6块commit后的第1块，只保留最近5个祖先。

当前任务为[EXP-005](../../../experiments/EXP-005_v3_sliding_window/README.md)。优先推进SW-G；SW-L不预设优于Global，不通过改prefix、时间或动作mask挽救结果。

[阶段二GPU提案（未授权）](../../../experiments/EXP-005_v3_sliding_window/GPU_PLAN.md) · [未来FM8 / AnyFlow独立计划](../../../experiments/EXP-005_v3_sliding_window/FUTURE_ANYFLOW.md) · [V3总览](../README.md)
