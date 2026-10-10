# V2：视觉与动作退化的三条修复路线

V2a、V2b、V2c是针对V1视觉与动作退化的三条并行研究路线。字母标识方案，不表示V2a → V2b → V2c的权重继承。V2b和V2c使用Original H3 + released action LoRA；V2a有自己的视觉适配。

| 路线 | 研究思想 | 主要优势 / 已验证范围 | 主要限制 |
| --- | --- | --- | --- |
| V2a：RGB Anchor | 通过视觉条件一致性和 adaptation 维持画面 | 124帧基本人物/场景结构，支持严格因果与KV | 动作控制失败；20秒后段退化 |
| V2b：Same-σ | 保留可见历史与当前视频的联合双向去噪 | 持续A/D124帧动作与基本结构可用 | 计算昂贵，无persistent KV；切换和连续性有限 |
| V2c：Strict Causal + Persistent KV | 原生条件、current-prefix与冻结历史KV实现严格因果生成 | AA/AD124帧动作、结构与真实KV复用可行；另有8步续写73帧证据 | 瞬态人体形变、连续性PARTIAL；8步首窗仍借用30步 |

## 按路线查找

- [V2a · RGB Anchor](v2a_rgb_anchor/README.md)：版本、原片、比较和代码阅读快照。
- [V2b · Same-σ / Local Bidirectional](v2b_same_sigma_local_bidir/README.md)：持续A/D124帧、Original对照与切换限制。
- [V2c · Strict Causal + Persistent KV](v2c_strict_causal_kv/README.md)：124帧可行性与成本。
  - [EXP-002 · 73帧缓存证据](v2c_strict_causal_kv/73frame_evidence/README.md)
  - [EXP-004 · 原权重8步续写](v2c_strict_causal_kv/8step_continuation/README.md)

## 命名与证据边界

2026-10-10按用户研究分类，将此前称为“V3 Efficient Causal”的严格缓存路线归入V2c。已验收结果、权重、推理协议与实验编号均不因此改变。EXP-001 / v3中的小写v3是任务书修订号，与模型V3无关；更早历史文件也曾用旧V3指代现在的V2b，须结合日期与协议识别。

今后有变体的版本按 `report/vN/vNa_<method>/`、`vNb_<method>/` 分组，各组有README导航；同一路线的步数/长度实验放在该路线内。未来V3的研究定义尚未发布，不为空版本建立已完成声明。

[总导航](../README.md) · [并行路线图](../roadmap.md) · [主线定义](../../mainline/v2/README.md)
