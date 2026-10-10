# V3：正式基线与两个Sliding Window候选

**V3 Original Feasibility Baseline继续作为已验收的正式基础版本。** V3-SW-G和V3-SW-L独立命名、独立配置、独立验收，候选结果不覆盖Baseline。此前短暂使用的V2c分类按用户最新决定撤回，历史迁移记录保留。

| 属性 | V3 Baseline | V3-SW-G | V3-SW-L |
| --- | --- | --- | --- |
| Backbone | Original H3 + released Action LoRA | 相同 | 相同 |
| 新训练 | 无 | 无 | 无 |
| Strict causal / Persistent raw KV | 是 / 是 | 保持 | 保持 |
| 历史窗口 | 已验收范围最多5个祖先 | 固定最近5个祖先 | 固定最近5个祖先 |
| 历史淘汰 | 124帧结束前未触发验证 | 实际启用、待GPU验证 | 同左 |
| Video位置 | Global RoPE | Global RoPE | 显式Sliding Local RoPE |
| 超过6块 | 未验收 | 待测试，优先 | 待测试，独立对照 |
| 定位 | 正式参考、124帧有限可行性 | 检查真实淘汰与KV容量 | 检查位置外推和适配风险 |

- [V3 Baseline：124帧正式证据](v3_baseline/README.md)
- [V3-SW-G：Sliding Window + Global RoPE](v3_sw_g/README.md)
- [V3-SW-L：Sliding Window + Local RoPE](v3_sw_l/README.md)

EXP-005当前只授权CPU实现与验证，GPU阶段等待独立Judge批准。CPU结构/数值检查不等于33B视频能力验收。后续V3-FM8和V3-AF另立任务，普通FM减步与AnyFlow finite-map训练分别评价。
