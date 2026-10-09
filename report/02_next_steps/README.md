# V3 Planned：统一两条路线的能力，随后才是AnyFlow与DMD

**目标：Efficient Causal Generation + Visual Stability + Action Fidelity。** V2a提供CPU KV/视觉修复经验，V2b提供局部动作与结构正控；它们不是可直接拼接即合格的两个组件。

真正缺少的是经过适当causal adaptation的模型：不依赖history/current video双向重算，仍有正确动作条件能力。训练目标和拓扑选择须基于受控证据，不能因两支各有优点就直接宣称统一可行。

| 次序 | 需要回答的问题 | Go条件 / 当前边界 |
|---|---|---|
| 局部能力 | strict chunk-causal是否保留动作与结构 | 足够30steps、同history/noise与当前A/D；视频方向/结构共同评审，不单看cosine |
| 缓存 | 实际cache与匹配时间/输入的重算是否等价 | 逐层K/V、RoPE、velocity跨sigma；区别matched rebuild与sigma0 clean commit；已有诊断wrapper证明受控等价，非默认路径自动通过 |
| 多块 | 能否在自己的history继续且不崩 | 第三/第四块、多场景，再124f；目前V2b只到第二块56f |
| 效率 | 历史复用是否带来端到端收益 | 统一硬件/offload、warmup、多次均值、真实首屏与每块计时；V2a也没有整体加速证明 |
| AnyFlow | 可否降低每chunk采样 | 在已可信causal30上比较4/8步，finite/diagonal与teacher endpoint双指标及完整视频 |
| On-policy DMD | 能否改善生成历史分布偏移 | 真self-rollout、frozen teacher、trainable fake score与正确生成梯度；已有lite质量失败/工程准备不能当完整Stage2 |

V2a的124f视觉证据不覆盖20秒；V2b的56f局部动作与视觉证据不覆盖124f。V3无checkpoint和视频。本次只是重整研究路线，没有启动上述实验。

[路线图](../roadmap.md) · [汇报导航](../README.md)
