# H3-World因果世界模型：研究状态

## 研究路线

V0建立原始双向动作/视觉参考；V1引入严格分块因果与persistent raw KV，并暴露视觉与动作退化。V2a、V2b、V2c是对此问题的三条并行修复方案，字母不代表权重继承。

| 方案 | 研究思想 | 现有优势 | 主要限制 |
| --- | --- | --- | --- |
| V2a RGB Anchor | 视觉条件一致性 + adaptation | 124帧基本结构 | 动作控制失败，20秒后段崩坏 |
| V2b Same-σ | 保留历史/当前联合双向去噪 | 持续A/D124帧动作与基本结构 | 计算昂贵、无persistent KV，切换有限 |
| V2c Strict Causal + Persistent KV | 原生条件 + 冻结历史KV | AA/AD124帧动作/结构/KV可行 | 瞬态形变，连续性PARTIAL |

原V3 Efficient Causal现重分类为V2c。V2b/V2c均使用Original H3 + released action LoRA，V2c没有继承V2a训练adapter；版本分类不改变实验配置和结论。

## 实验与汇报

- EXP-001：V2b持续A/D124帧与Original对照，动作切换PARTIAL。
- EXP-002：V2c共同历史下的动作响应、严格因果与真实KV，续到73帧。
- EXP-003：同一V2c协议AA/AD124帧可行性，AA中段明显形变后恢复。
- EXP-004：V2c原权重普通FM8步续写到73帧，0新增训练，0.129449 GPU小时。首39帧仍借用30步结果，AA拖影明显。

[V2分组与视频](report/v2/README.md) · [主线定义](mainline/v2/README.md) · [浏览器演示](report/index.html) · [五分钟讲稿](report/TALK_5MIN.md)。

## 系统与能力边界

核心实现见[causal KV](code/causal/h3_cached.py)、[AnyFlow探索](code/causal/anyflow.py)、[DMD-lite探索](code/causal/stage2_lite_dmd.py)。具体版本对应的真实冻结代码、输入和成本以实验目录的manifest为准。历史AnyFlow/Stage2训练与工程运行不代表当前基线已通过这些方法。

目前是单scene/seed、有限训练下的可行性验证。动作proxy只作辅助，画质/严格连续性有明确缺陷；不把不同运行时刻、不同首窗复用范围的增量计时称为公平完整E2E速度比。

## 下一研究决策

优先验证V2c首窗也用8步，检查直接减步是否能独立启动并续自己的历史。证据不足则及时停止；不为微小收益增加消融，也不默认靠更多训练挽救失败。AnyFlow、On-policy DMD与未来V3研究目标另行定义。

[路线图](report/roadmap.md) · [比较口径](report/COMPARISON_PROTOCOL.md) · [历史报告](archive/legacy_reports/REPORT_before_reorganization.md)
