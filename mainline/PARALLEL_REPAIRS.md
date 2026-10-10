# 修复路线与V3候选关系

V2a RGB Anchor与V2b Same-σ是针对V1退化的并行修复方案。V3 Original Feasibility Baseline独立使用Original权重，在严格因果与persistent KV下完成124帧有限可行性验收。上述研究经验相互参考，不构成checkpoint依次继承。

在冻结V3之上，SW-G只增加真实历史淘汰和长区间支持，SW-L再单独改变video位置映射。[三个V3协议对照](v3/README.md)。候选不覆盖正式Baseline。
