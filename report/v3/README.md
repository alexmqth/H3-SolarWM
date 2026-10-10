# V3：正式基线与两个Sliding Window候选

**V3 Original Feasibility Baseline继续作为已验收的正式基础版本。** V3-SW-G和V3-SW-L独立命名、独立配置、独立验收，候选结果不覆盖Baseline。此前短暂使用的V2c分类按用户最新决定撤回，历史迁移记录保留。

| 属性 | V3 Baseline | V3-SW-G | V3-SW-L |
| --- | --- | --- | --- |
| Backbone | Original H3 + released Action LoRA | 相同 | 相同 |
| 新训练 | 无 | 无 | 无 |
| Strict causal / Persistent raw KV | 是 / 是 | 保持 | 保持 |
| 历史窗口 | 已验收范围最多5个祖先 | 固定最近5个祖先 | 固定最近5个祖先 |
| 历史淘汰 | 124帧结束前未触发验证 | 已验收C7/C8真实淘汰 | 同左 |
| Video位置 | Global RoPE | Global RoPE | 显式Sliding Local RoPE |
| 超过6块 | 未验收 | 158帧有限可行性通过 | 158帧工程通过、生成PARTIAL |
| 定位 | 正式参考、124帧有限可行性 | 检查真实淘汰与KV容量 | 检查位置外推和适配风险 |

- [V3 Baseline：124帧正式证据](v3_baseline/README.md)
- [V3-SW-G：Sliding Window + Global RoPE](v3_sw_g/README.md)
- [V3-SW-L：Sliding Window + Local RoPE](v3_sw_l/README.md)

EXP-005/v2已完成281forward/9VAE，实际0.662707GPU小时。SW-G优先保留；SW-L有动作响应但重复场景/亮度跳变更多，无已证实收益，当前无训练方向归档。正式Baseline不变。后续V3-FM8和V3-AF另立任务，普通FM减步与AnyFlow finite-map训练分别评价。


## 全程普通FM8

[EXP-006 V3-FM8](v3_fm8/README.md)已验收：新8步首39+AA/AD73，有限可行性通过、quality PARTIAL。43forward/5VAE/0.136905GPU小时，零新训练。下一项为独立V3-AF，普通减步与finite-map训练分别记录。

## AnyFlow研究进展

[V3-AF](v3_anyflow/README.md)：累计32次finite-map训练和8NFE AA/AD73续写有限可行性通过，quality PARTIAL；共同FM8首窗，尚无整体优于FM8的证据。正式V3 Baseline冻结，后续独立DMD pilot。
