# V2a / V2b：并列修复路线

| 维度 | V2a：RGB-Anchor Causal | V2b：Same-σ History / Local Bidir |
|---|---|---|
| 共同问题 | 因果rollout中生成历史/条件与原生模型行为不匹配 | 同一研究问题的另一条路线 |
| 核心思路 | 修复图像条件协议，配合视觉适配 | 恢复原生条件和尽可能接近原生的局部联合去噪 |
| 图像条件 | RGB-consistent dual anchor | Single I0 |
| 历史 | 自己生成的clean endpoint经clean commit写入KV | 自己生成的history按当前σ临时加噪，每步重算 |
| Attention | Strict chunk-causal video attention | 已可见history与当前video局部双向；未知未来移除 |
| Persistent hidden KV | 支持，CPU raw video K/V | 不支持 |
| 采样 | 8steps/chunk | 30steps/chunk |
| 新增训练 | visual QKV + endpoint/boundary/replay；使用已训action residual | 无；Original + released action LoRA |
| 视觉证据 | 124f人物/场景相对完整；20秒崩坏 | 56f中第二块人物结构基本完整 |
| 动作证据 | A/D方向门槛失败 | AA/AD/DA/DD第二块方向正确 |
| 主要不足 | 动作控制；更长时程也退化 | 历史重算成本、效率与长时可靠性未验证 |

二者没有checkpoint继承，亦非单因素消融。V2a仅124f视觉证据，V2b仅56f第二块局部动作/结构。未来V3需真正统一严格因果、历史复用与能力，不是checkpoint拼接。

[能力比较视频](../report/00_comparison_gallery/V2a_vs_V2b.mp4) · [路线图](../report/roadmap.md)
