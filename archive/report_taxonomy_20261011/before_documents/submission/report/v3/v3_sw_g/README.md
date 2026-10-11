# V3-SW-G — Sliding Window + Global RoPE

**EXP-005/v2已验收158帧有限可行性，画质/连续性PARTIAL。** 正式V3 Baseline仍冻结为124帧参考，本候选单独记录。

保留Original H3 + released Action LoRA、Single I0、native timestep、own-action/action feedback/current non-action prefix feedback、strict causal、30-step FM、clean raw KV commit与Global RoPE。唯一窗口改动是显式长分块及真实最近5祖先淘汰。

## 证据

- G0两组真实模型old/new forward误差0；30-step C6 endpoint与124RGB逐值复现冻结Baseline。
- G1同AA124历史、同新增噪声下继续A或切换D；两路径各到C8/158帧。C7祖先C2–C6、C8祖先C3–C7，全部50层检查通过。
- 历史video KV保持14,164,800,000 bytes；已发布RGB逐值不变。仅video KV有界，不代表prefix、完整输出、VAE或全系统内存有界。
- A C7/C8水平flow +0.819/+0.768；D −1.575/−1.185。方向与首块切换响应可辨，flow仅为辅助。
- 主体与停车场结构可用；A C8瞬态拖影，D切换边界跳变、C8持续半透明拖影。接受可行性，不要求成熟画质，不为这些缺陷追加C9或扫参。

[G1 Judge审核](../../../experiments/EXP-005_v3_sliding_window/stage2/g1_judge_review.json) · [完整新增帧视觉记录](../../../experiments/EXP-005_v3_sliding_window/judge/visual_notes.json) · [EXP-005](../../../experiments/EXP-005_v3_sliding_window/README.md) · [V3总览](../README.md)

## 范围

单停车场seed13，前124帧复用冻结AA历史，仅新增两块。158帧约6.58秒，尚非长时或无限稳定验证。固定prefix的长输入是经认证的显式扩展，并非默认H3整段长packed重建。原权重、零新增训练。后续少步/AnyFlow必须独立冻结协议与预算。

[SW-G A/D完整对比](../../../experiments/EXP-005_v3_sliding_window/artifacts/stage2/G1/G1_A_vs_D_158.mp4) · [最终Judge审核](../../../experiments/EXP-005_v3_sliding_window/judge/STAGE2_FINAL_REVIEW.md)。Local有限对照已完成，未显示联合收益，优先保留SW-G。
