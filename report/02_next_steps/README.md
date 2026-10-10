# 下一步：V3 Sliding Window分阶段验证

V3 Baseline继续作为正式124帧参考。EXP-005先完成CPU实现与正确性验证，再提交有限GPU任务书，等待Judge批准。

优先SW-G：首淘汰前复现冻结V3，随后第7/8块验证真实淘汰、最近5个具体祖先indices、KV容量、动作/切换、完整画面与逐块成本。SW-L保持相同窗口，仅改变video位置协议，独立判断；不能通过悄悄改prefix/time/action mask修复。

[任务入口](../../experiments/EXP-005_v3_sliding_window/README.md) · [GPU提案](../../experiments/EXP-005_v3_sliding_window/GPU_PLAN.md) · [后续V3-FM8 / V3-AF独立设计](../../experiments/EXP-005_v3_sliding_window/FUTURE_ANYFLOW.md)。当前不启动GPU或训练。
