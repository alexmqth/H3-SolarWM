# 研究路线：V3因果缓存、少步生成与V4 SGF

```mermaid
flowchart TD
  O["V0 Original H3"] --> V1["V1 因果化：视觉/动作退化"]
  V1 --> V2["V2a RGB Anchor / V2b Same-σ"]
  V1 --> B["V3 Baseline：因果缓存，30步124帧"]
  B --> SW["Sliding Window：Global / Local"]
  SW --> G["Global 158帧有限可行；Local当前无收益"]
  B --> FM["普通FM减步：8步 / 4步"]
  B --> AF["AnyFlow：新学生finite-map训练"]
  AF --> E["AF-8NFE有限续写；迁移及AF-4NFE存在失败"]
  AF --> D["V4 SGF / DMD：当前训练工程通过，生成失败"]
  FM --> DATA["后续：真实数据、布局协议、实际模型回归"]
  E --> DATA
```

箭头表示研究关系。V2a/V2b/V3不是连续权重继承；具体EXP-008 DMD学生继承EXP-007 AF AF-train-32updates。正式V3 Baseline保持原权重与协议，V4为SGF研究分类，当前没有已验收生成模型。

[Sliding Window](v3/sliding_windows/README.md) · [FM → AnyFlow](v3/fm2anyflow/README.md) · [V4 SGF / DMD](v4/README.md) · [下一步草案](02_next_steps/README.md)
