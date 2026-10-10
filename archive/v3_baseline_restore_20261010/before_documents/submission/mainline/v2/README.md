# V2：并行修复路线

V2a、V2b、V2c是针对V1视觉与动作退化的三条并行研究路线。字母标识方案，不表示V2a → V2b → V2c的权重继承。V2b和V2c使用Original H3 + released action LoRA；V2a有自己的视觉适配。

| 路线 | 研究思想 | 主要优势 / 已验证范围 | 主要限制 |
| --- | --- | --- | --- |
| V2a：RGB Anchor | 通过视觉条件一致性和 adaptation 维持画面 | 124帧基本人物/场景结构，支持严格因果与KV | 动作控制失败；20秒后段退化 |
| V2b：Same-σ | 保留可见历史与当前视频的联合双向去噪 | 持续A/D124帧动作与基本结构可用 | 计算昂贵，无persistent KV；切换和连续性有限 |
| V2c：Strict Causal + Persistent KV | 原生条件、current-prefix与冻结历史KV实现严格因果生成 | AA/AD124帧动作、结构与真实KV复用可行；另有8步续写73帧证据 | 瞬态人体形变、连续性PARTIAL；8步首窗仍借用30步 |

- [V2a RGB Anchor](v2a_rgb_anchor/README.md)
- [V2b Same-σ](v2b_same_sigma_local_bidir/README.md)
- [V2c Strict Causal + Persistent KV](v2c_strict_causal_kv/README.md)

[汇报与视频分组](../../report/v2/README.md) · [主线总览](../README.md)
