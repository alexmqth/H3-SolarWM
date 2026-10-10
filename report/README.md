# H3-SolarWM：汇报导航

**正式参考：V3 Original Feasibility Baseline。** Strict Causal + Persistent raw KV + Global RoPE，AA/AD六块124帧有限可行性已验收，零新增训练；瞬态人体形变和连续性PARTIAL如实保留。

## 按版本查找

- [V0 Original](V0_original/README.md) / [V1 Native causal](V1_native_causal/README.md)
- [V2家族](v2/README.md)：[V2a RGB Anchor](v2/v2a_rgb_anchor/README.md)、[V2b Same-σ](v2/v2b_same_sigma_local_bidir/README.md)
- **[V3家族与协议对照](v3/README.md)**：
  - [V3 Baseline：正式124帧参考](v3/v3_baseline/README.md)
  - [V3-SW-G：Sliding Window + Global](v3/v3_sw_g/README.md)，优先候选
  - [V3-SW-L：Sliding Window + Local](v3/v3_sw_l/README.md)，独立位置对照
- [EXP-004：V3普通FM8步续写证据](v3/v3_baseline/8step_continuation/README.md)，首39帧借用30步，不代表全程FM8或AnyFlow。

## 直接展示

[Original / V3 Baseline AA124](v3/v3_baseline/videos/Original_vs_V3_Baseline_AA_124.mp4) · [V2b / V3 Baseline AA124](v3/v3_baseline/videos/V2b_vs_V3_Baseline_AA_124.mp4) · [V3 AD124](v3/v3_baseline/videos/V3_Baseline_AD_124.mp4) · [浏览器演示](index.html)

[路线图](roadmap.md) · [比较边界](COMPARISON_PROTOCOL.md) · [指标](METRICS.md) · [讲稿](TALK_5MIN.md) · [下一步](02_next_steps/README.md)

EXP-005当前仅授权CPU实现与验证；GPU阶段在单独任务书批准前不启动。普通FM减步、AnyFlow finite-map训练、DMD分别记录。当前单scene/seed与增量计时不能支持无限长稳定或完整公平E2E加速。
