# V3-FM8 — 从首窗开始的普通FM8

**EXP-006已验收单停车场seed13的AA/AD73帧有限可行性，quality PARTIAL，零新训练。** Original H3 + released Action LoRA，Single I0/native timestep/current-prefix/own-action、Global、strict causal + persistent raw KV；首12及后续5-latent块全部8步。与EXP-004不同，本轮首39帧也是新8步生成。

A/D C2方向+0.846/−0.321，C3 +0.505/−0.826；人物/场景基本可用，边界跳变和肢体透明残影仍存在。实际43forward/5VAE/0.136905GPU小时，allocated peak26.061GiB。EXP-006自身范围为73帧，EXP-010后续已扩展158帧；普通FM减步不等于AnyFlow训练。

[Judge审核](../../../experiments/EXP-006_v3_fm8_full/judge/FINAL_REVIEW.md) · [AA73原片](../../../experiments/EXP-006_v3_fm8_full/artifacts/videos/AA_FM8_73.mp4) · [AD73原片](../../../experiments/EXP-006_v3_fm8_full/artifacts/videos/AD_FM8_73.mp4) · [30步参考/新FM8 AA](../../../experiments/EXP-006_v3_fm8_full/artifacts/videos/AA_FM30_saved_first_vs_FM8_new_first_73.mp4) · [30步参考/新FM8 AD](../../../experiments/EXP-006_v3_fm8_full/artifacts/videos/AD_FM30_saved_first_vs_FM8_new_first_73.mp4)

对照的生成首窗与后续历史不同，不能称同raw KV单变量消融或公平E2E速度比较。正式V3 Baseline及SW-G结果冻结保留；本项提供后续V3-AF匹配NFE的有限对照。

## 全程FM8 + SW-G 158帧（EXP-010）

2026-10-11已验收：接续EXP-006自己的AA73至共同AA124，再C7/C8 A继续/D晚切换各158。有限可行PASS，quality PARTIAL；首个D块响应弱（flow+.034），第二块反向更清楚（−.858），切换响应PARTIAL。A C7/C8为+.439/+.587。人物/车库可辨，透明腿部残影、视角/几何边界跳变保留。真实两次淘汰，50层历史video KV恒定14,164,800,000 bytes，旧RGB不变。

新增62forward/7VAE/0训练/.315863516GPUh，allocated峰26.59566GiB。单停车场seed13、158帧6.58秒，不外推无限长度/泛化；prefix与VAE成本仍可增长，冻结long47 fixture不等价默认长输入重建。30步参考与FM8使用不同生成历史，不能称同KV消融或公平E2E速度排名。

[最终审核](../../../experiments/EXP-010_v3_fm8_sw158/judge/FINAL_REVIEW.md) · [A158](../../../experiments/EXP-010_v3_fm8_sw158/artifacts/branch/A/rollout_158.mp4) · [D158](../../../experiments/EXP-010_v3_fm8_sw158/artifacts/branch/D/rollout_158.mp4) · [30步/FM8 A并排](../../../experiments/EXP-010_v3_fm8_sw158/artifacts/comparisons/A_SWG30_vs_FM8_SWG8_158.mp4) · [D并排](../../../experiments/EXP-010_v3_fm8_sw158/artifacts/comparisons/D_SWG30_vs_FM8_SWG8_158.mp4)。
