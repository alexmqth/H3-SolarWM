# V3：严格因果、真实KV与可用动作/视觉的可行性验证

V2b在EXP-001中完成持续A/D的124帧可行性验收；仍每步重算所有历史，没有persistent KV。V2a有缓存/视觉修复经验但动作失败。V3尚未完成。

当前关注同一配置是否同时具备strict chunk-causal、真实persistent video KV、可辨动作、可用多窗口人物/场景及可信效率记录。模型未经大量针对性训练，可以接受模糊、动作不够自然等原型限制；不要求成熟产品质量。

EXP-002 native Single I0/12→5→5的current-prefix严格缓存候选已验收73帧；同history A/D、真实KV与基本结构有有效证据。下一任务只延伸同一协议到124帧并记录实际成本，具体预算以根next_plan.md为准。旧RGB-dual固定状态诊断负结果保留，不追溯改写。

若有有效能力信号，再做同协议124帧和效率验证；持续无效则停止该方向，依据证据选择有限适配或换路线，不默认追加大规模训练。AnyFlow/DMD在可信因果基线上再推进。

[EXP-001验收](../../experiments/EXP-001_v2b_124/judge/FINAL_REVIEW.md) · [路线图](../roadmap.md) · [汇报导航](../README.md)
