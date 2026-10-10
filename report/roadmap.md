# 研究路线：冻结V3 Baseline，再验证Sliding Window

```mermaid
flowchart TD
  V0["V0 Original H3"] --> V1["V1 因果化：动作/视觉退化"]
  V1 --> V2a["V2a RGB Anchor + adaptation"]
  V1 --> V2b["V2b Same-σ 联合双向去噪"]
  V1 --> B["V3 Original Feasibility Baseline：Global / strict causal / KV，124帧"]
  B --> G["V3-SW-G：最近5祖先 + Global，优先验证"]
  B --> L["V3-SW-L：相同窗口 + Local，独立对照"]
  B -. "后续独立任务" .-> FM["V3-FM8：普通FM减步"]
  B -. "后续独立训练" .-> AF["V3-AF：target-time finite-map student"]
```

箭头表示研究关系，不表示V2a→V2b→V3的权重继承。V3-SW-G/SW-L当前为CPU阶段，尚未进行GPU生成验收；Baseline的124帧证据继续保留。Local raw KV位置重映射不等价重算所有历史；也不预设Local优于Global。

[V3三个协议](v3/README.md) · [V2修复方案](v2/README.md) · [CPU与分阶段任务](../experiments/EXP-005_v3_sliding_window/README.md)
