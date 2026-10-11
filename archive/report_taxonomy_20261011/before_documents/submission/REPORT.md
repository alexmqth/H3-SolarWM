# H3-World因果世界模型：当前研究状态

2026-10-10。正式参考恢复命名为 **V3 Original Feasibility Baseline**。V2a RGB Anchor、V2b Same-σ保留各自研究身份；没有顺次权重继承关系。

## 已有证据与独立候选

| 版本 | 协议与现有证据 | 主要限制 |
| --- | --- | --- |
| V2a RGB Anchor | 视觉条件一致性与adaptation，124帧基本结构 | 动作控制失败，20秒后段崩坏 |
| V2b Same-σ | 历史与当前联合双向去噪，持续A/D124帧 | 无persistent KV，计算昂贵，切换PARTIAL |
| V3 Baseline | Original H3 + released Action LoRA，30步、Global RoPE、strict causal/persistent raw KV，AA/AD124帧已验收 | 瞬态形变、连续性PARTIAL；真实淘汰与超过6块未验收 |
| V3-SW-G | 保留Baseline条件，固定最近5祖先，Global RoPE | EXP-005 CPU准备；GPU待批 |
| V3-SW-L | 相同窗口，仅video位置改为显式Sliding Local RoPE | 独立候选，不等价历史重算；GPU待批 |

## 实验入口

- EXP-001：V2b持续A/D124帧与Original对照，动作切换PARTIAL。
- EXP-002/003：V3 Baseline严格因果、真实KV、同历史动作响应及自身历史124帧可行性。
- EXP-004：原权重普通FM8步续写到73帧；首39帧仍复用30步结果，AA拖影明显，不能称全程8步或AnyFlow。
- [EXP-005](experiments/EXP-005_v3_sliding_window/README.md)：Sliding Window实现、CPU正确性与GPU提案。本轮0 GPU推理、0训练，候选生成能力NOT_TESTED。

[V2汇报](report/v2/README.md) · [V3三个版本](report/v3/README.md) · [主线定义](mainline/README.md) · [浏览器演示](report/index.html) · [五分钟讲稿](report/TALK_5MIN.md)。

## 下一步与判定尺度

先审核首次淘汰前的V3回归，再有限验证SW-G第7/8块，最后独立决定SW-L。真实长输入认证、GPU入口与预算账本是审批前置；现阶段不启动GPU。详见[分阶段GPU提案](experiments/EXP-005_v3_sliding_window/GPU_PLAN.md)。

按可行性判断普通画质缺陷；持续动作失效或严重结构崩坏时停止相应方向，不为微小指标改进反复实验。五祖先只约束历史video KV，不能据此宣称完整系统恒定内存或无限时长质量。

[V3-FM8与V3-AF独立计划](experiments/EXP-005_v3_sliding_window/FUTURE_ANYFLOW.md)区分直接FM减步和新的target-time-conditioned finite-map训练。旧协议AnyFlow产物不作V3-AF完成证据。

[比较口径](report/COMPARISON_PROTOCOL.md) · [历史报告](archive/legacy_reports/REPORT_before_reorganization.md)
