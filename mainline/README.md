# Mainline · 原始能力、因果化与两条并列修复路线

V0 → V1 → {V2a, V2b} → V3 Planned → AnyFlow → On-policy DMD。这是研究路线；V2b权重从Original出发，V2a与V2b之间没有checkpoint继承。

| 版本 | Action fidelity | Visual stability | 已验证范围 | Persistent KV | 采样 | 新增训练 | 当前结论 |
|---|---|---|---|---|---|---|---|
| V0 Original | A/D方向正控 | 124f基本完整 | 124f及已有长片参考 | 否 | 30整段；另存50步 | 无，released LoRA | Reference，非GT |
| V1 Native causal | 显著下降 | 本代表停滞、过亮/背景退化；其他早期协议有重影 | 124f工程rollout | 是，CPU raw video KV | 8/chunk | 本代表无 | 工程可行，联合质量失败 |
| V2a RGB-Anchor | A/D方向失败 | **124f人物结构相对稳定**；20s失败 | 124f视觉证据；243/481f负结果 | 是，clean commit的历史hidden KV | 8/chunk | visual adapter +已训action residual | Longer-horizon Visual Stability Demonstrated（仅124f scope） |
| V2b Same-σ local bidir | 持续A/D在124f有可辨响应；切换仍有限制 | 124f人物/场景基本可用；边界与节奏有缺陷 | 单停车场、seed13、六块124f | **否**，每步重算全部可见历史 | 30/chunk | **无**，Original + released LoRA | Sustained A/D feasibility accepted；非V3 |
| V3 Efficient causal | 目标：保留 | 目标：保留并扩大范围 | **尚未实现** | 目标：strict causal + KV | 先可信30步，后AnyFlow | 待定 | Planned Unification，无模型/视频 |

- [V0 Original Bidirectional](V0_original_bidirectional/README.md)
- [V1 Native Chunk-Causal](V1_native_chunk_causal/README.md)
- [V2a RGB-Anchor Causal](V2a_rgb_anchor_causal/README.md)
- [V2b Same-σ History / Local Bidir](V2b_same_sigma_local_bidir/README.md)
- [V3 Efficient Causal — Planned](V3_efficient_causal_planned/README.md)

[汇报包](../report/README.md) · [路线图](../report/roadmap.md) · [研究分支](../branches/README.md)
