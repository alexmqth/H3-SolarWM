# V3-FM8 — 从首窗开始的普通FM8

**EXP-006已验收单停车场seed13的AA/AD73帧有限可行性，quality PARTIAL，零新训练。** Original H3 + released Action LoRA，Single I0/native timestep/current-prefix/own-action、Global、strict causal + persistent raw KV；首12及后续5-latent块全部8步。与EXP-004不同，本轮首39帧也是新8步生成。

A/D C2方向+0.846/−0.321，C3 +0.505/−0.826；人物/场景基本可用，边界跳变和肢体透明残影仍存在。实际43forward/5VAE/0.136905GPU小时，allocated peak26.061GiB。73帧之外的全程FM8未测；普通FM减步不等于AnyFlow训练。

[Judge审核](../../../experiments/EXP-006_v3_fm8_full/judge/FINAL_REVIEW.md) · [AA73原片](../../../experiments/EXP-006_v3_fm8_full/artifacts/videos/AA_FM8_73.mp4) · [AD73原片](../../../experiments/EXP-006_v3_fm8_full/artifacts/videos/AD_FM8_73.mp4) · [30步参考/新FM8 AA](../../../experiments/EXP-006_v3_fm8_full/artifacts/videos/AA_FM30_saved_first_vs_FM8_new_first_73.mp4) · [30步参考/新FM8 AD](../../../experiments/EXP-006_v3_fm8_full/artifacts/videos/AD_FM30_saved_first_vs_FM8_new_first_73.mp4)

对照的生成首窗与后续历史不同，不能称同raw KV单变量消融或公平E2E速度比较。正式V3 Baseline及SW-G结果冻结保留；本项提供后续V3-AF匹配NFE的有限对照。
