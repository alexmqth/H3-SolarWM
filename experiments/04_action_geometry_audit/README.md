# Breakthrough 04：定位 action geometry 失败不是简单 routing 问题

## 问题

RGB anchor 修复视觉后，A/D 仍然产生同向负 horizontal flow。需要区分未来 action rows 不可见、共同场景漂移和 causal score field 本身改变三种假设。

## 诊断

1. 把 action prefix 改为 `all` 作为 visibility upper bound，结果 A=-1.1535、D=-1.4733、A-D=0.3198，不能恢复方向。
2. 同 seed 加入 STILL counterfactual，STILL=-0.7129；从 A/D 中扣除后仍没有正确左右方向，说明共同 scene/camera drift 不是主因。
3. 在同一个 generated state 上比较 frozen causal 与原始 bidirectional teacher：delta norm 为 7.642 与 8.704，ratio=0.878，但 cosine=-0.015。幅度相近，方向几乎正交。
4. hidden action residual、action-prefix residual、released H3 action-LoRA、tail4/tail8 action-QKV 和 Stage2-lite DMD 均未达到 flow(A)>0、flow(D)<0、A-D>1.0。

## 证据

[routing 上界](../../meeting/action_geometry/stage2_routing_all_rgb_AD_39.mp4)、[端点监督（含 state mismatch 的第一版）](../../meeting/action_geometry/stage2_action_qkv_latent_endpoint_AD_39.mp4) 和 [FINAL_ACTION_DIAGNOSTIC.md](../../reports/action_alignment/FINAL_ACTION_DIAGNOSTIC.md) 保存了 routing upper bound、endpoint smoke 和最终短片 gate 的证据。

## 结论

当前主要问题是 causal attention + generated-history 造成的 action-conditioned score geometry mismatch，而不是简单缺一条 action routing edge。继续 single-state gain、anchor、solver sweep 不具有清晰归因价值；需要 multi-state/multi-seed action supervision 或更完整的 rollout distribution matching。
