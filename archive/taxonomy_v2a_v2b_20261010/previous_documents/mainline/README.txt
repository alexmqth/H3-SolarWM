# Mainline · 模型与协议演进

版本号描述研究路线，不保证checkpoint逐代继承。尤其V3从Original重新出发。

| 版本 | Action fidelity | Visual stability | 长时 rollout | Persistent KV | Sampling steps | 新增训练 | 验收状态 |
|---|---|---|---|---|---|---|---|
| V0 Original | A/D 正控；W/S 以视频为准 | 124f 基本完整 | 有243/481f参考；仍有几何变形 | 否 | 30整段；另存历史50步 | 无，released LoRA | Reference，不是GT |
| V1 Native causal | A/D显著减弱；本代表A符号错 | 停滞、透明/重影等退化 | 124f执行完成 | 是，CPU raw video KV | 8/chunk × 8 + 8 commits | 本代表无 | 工程可行，质量/动作未过 |
| V2 RGB联合修复 | A符号错；未恢复 | 124f相对改善；20s失败 | 124/243/481f均有片 | 是，CPU raw video KV | 8/chunk | visual QKV + action residual | 视觉局部改善，联合验收失败 |
| V3 Same-σ C12→5 | 两份自身history下当前A/D方向正确 | 第二块人物结构保持 | 仅56f、2块 | **否**，T2逐sigma联合重算 | 30/chunk | 无，从Original重新出发 | **Current Best Local Causal Rollout Candidate** |
| V4 Efficient causal | 待验证 | 待验证 | 待验证 | 目标：是 | 先30/chunk可信，再少步 | 待定 | Planned，无checkpoint/视频 |

- [V0 · Original H3-World (Bidirectional Baseline)](V0_original_bidirectional/README.md)
- [V1 · Native Chunk-Causal H3-World](V1_native_chunk_causal/README.md)
- [V2 · RGB-Anchor Causal H3-World](V2_rgb_anchor_causal/README.md)
- [V3 · Same-σ History / C12→5 H3-World](V3_same_sigma_history/README.md)
- [V4 Efficient causal — Planned](V4_efficient_causal_planned/README.md)

[精简汇报](../report/README.md) · [研究分支](../branches/README.md)
