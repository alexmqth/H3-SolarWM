# H3-SolarWM：汇报导航

**正式参考：V3 Original Feasibility Baseline。** Strict Causal + Persistent raw KV + Global RoPE，AA/AD六块124帧有限可行性已验收，零新增训练；瞬态人体形变和连续性PARTIAL如实保留。

## 按版本查找

- [V0 Original](V0_original/README.md) / [V1 Native causal](V1_native_causal/README.md)
- [V2家族](v2/README.md)：[V2a RGB Anchor](v2/v2a_rgb_anchor/README.md)、[V2b Same-σ](v2/v2b_same_sigma_local_bidir/README.md)
- **[V3家族与协议对照](v3/README.md)**：
  - [V3 Baseline：正式124帧参考](v3/v3_baseline/README.md)
  - [V3-SW-G：Sliding Window + Global](v3/v3_sw_g/README.md)，158帧有限可行性通过
  - [V3-SW-L：Sliding Window + Local](v3/v3_sw_l/README.md)，独立位置对照已归档，无已证实收益
  - [V3 FM30 四训练初图教师候选](v3/v3_teacher_targets/README.md)：AA/AD56并排与逐场动作适用性；两图D反向未证实
- [EXP-004：V3普通FM8步续写证据](v3/v3_baseline/8step_continuation/README.md)，首39帧借用30步，不代表全程FM8或AnyFlow。

## 直接展示

[Original / V3 Baseline AA124](v3/v3_baseline/videos/Original_vs_V3_Baseline_AA_124.mp4) · [V2b / V3 Baseline AA124](v3/v3_baseline/videos/V2b_vs_V3_Baseline_AA_124.mp4) · [V3 AD124](v3/v3_baseline/videos/V3_Baseline_AD_124.mp4) · [浏览器演示](index.html)

[四训练初图AA/AD 56帧画廊](v3/v3_teacher_targets/README.md)：[s0](v3/v3_teacher_targets/videos/s0_AA_vs_AD_56.mp4) · [s1](v3/v3_teacher_targets/videos/s1_AA_vs_AD_56.mp4) · [s2](v3/v3_teacher_targets/videos/s2_AA_vs_AD_56.mp4) · [s3](v3/v3_teacher_targets/videos/s3_AA_vs_AD_56.mp4)。[Judge已有限验收](../experiments/EXP-014_v3_multiscene_teacher/judge/FINAL_REVIEW.md)协议与来源；`s2/s3`的D反向未证实，不是多场景动作训练成功。

[路线图](roadmap.md) · [比较边界](COMPARISON_PROTOCOL.md) · [指标](METRICS.md) · [讲稿](TALK_5MIN.md) · [下一步](02_next_steps/README.md)

EXP-005/v2已完成有限GPU验证，281forward/9VAE/0.662707GPU小时。SW-G优先保留，Local不升级；后续任务另行授权。普通FM减步、AnyFlow finite-map训练、DMD分别记录。当前单scene/seed与增量计时不能支持无限长稳定或完整公平E2E加速。

## 新增158帧对照

[SW-G A/D](../experiments/EXP-005_v3_sliding_window/artifacts/stage2/G1/G1_A_vs_D_158.mp4) · [Global/Local A](../experiments/EXP-005_v3_sliding_window/artifacts/stage2/L1/G1_vs_L1_A_158.mp4) · [Global/Local D](../experiments/EXP-005_v3_sliding_window/artifacts/stage2/L1/G1_vs_L1_D_158.mp4)。前124帧相同，C7同历史，C8各自历史；人物基本可辨，Local场景/亮度跳变更多。


## 全程普通FM8

[V3-FM8](v3/v3_fm8/README.md)：EXP-006从首窗即8步，AA/AD73有限可行性通过、quality PARTIAL。为后续V3-AF提供有限匹配NFE对照，区别于EXP-004的30步首窗续写。
