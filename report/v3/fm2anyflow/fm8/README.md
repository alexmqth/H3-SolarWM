# V3-FM8：从首窗开始的普通8步采样

## 名称与目的

**名称解释：** FM = Flow Matching（流匹配）；FM8是每个视频块8步普通采样，FM30为30步。数字不表示帧数或训练次数。本尝试零新训练。

**本次目的：** 降低V3每块30步的采样成本，同时摆脱EXP-004对30步首窗的依赖。EXP-006检查全程8步73帧；EXP-010再检查8步与Global滑窗组合到158帧。

[其他缩写与计数](../../../GLOSSARY.md) · [汇报总览](../../../README.md)

## 实验结果与证据

**EXP-006已验收单停车场seed13的AA/AD73帧有限可行性，quality PARTIAL，零新训练。** Original H3 + released Action LoRA，Single I0/native timestep/current-prefix/own-action、Global、strict causal + persistent raw KV；首12及后续5-latent块全部8步。与EXP-004不同，本轮首39帧也是新8步生成。

A/D C2方向+0.846/−0.321，C3 +0.505/−0.826；人物/场景基本可用，边界跳变和肢体透明残影仍存在。实际43forward/5VAE/0.136905GPU小时，allocated peak26.061GiB。EXP-006自身范围为73帧，EXP-010后续已扩展158帧；普通FM减步不等于AnyFlow训练。

[Judge审核](../../../../experiments/EXP-006_v3_fm8_full/judge/FINAL_REVIEW.md) · [AA73原片](videos/AA_FM8_73.mp4) · [AD73原片](videos/AD_FM8_73.mp4) · [30步参考/新FM8 AA](videos/AA_FM30_saved_first_vs_FM8_new_first_73.mp4) · [30步参考/新FM8 AD](videos/AD_FM30_saved_first_vs_FM8_new_first_73.mp4)

对照的生成首窗与后续历史不同，不能称同raw KV单变量消融或公平E2E速度比较。正式V3 Baseline及SW-G结果冻结保留；本项提供后续V3-AF匹配NFE的有限对照。

## 全程FM8 + SW-G 158帧（EXP-010）

2026-10-11已验收：接续EXP-006自己的AA73至共同AA124，再C7/C8 A继续/D晚切换各158。有限可行PASS，quality PARTIAL；首个D块响应弱（flow+.034），第二块反向更清楚（−.858），切换响应PARTIAL。A C7/C8为+.439/+.587。人物/车库可辨，透明腿部残影、视角/几何边界跳变保留。真实两次淘汰，50层历史video KV恒定14,164,800,000 bytes，旧RGB不变。

新增62forward/7VAE/0训练/.315863516GPUh，allocated峰26.59566GiB。单停车场seed13、158帧6.58秒，不外推无限长度/泛化；prefix与VAE成本仍可增长，冻结long47 fixture不等价默认长输入重建。30步参考与FM8使用不同生成历史，不能称同KV消融或公平E2E速度排名。

[最终审核](../../../../experiments/EXP-010_v3_fm8_sw158/judge/FINAL_REVIEW.md) · [A158](videos/A_rollout_158.mp4) · [D158](videos/D_rollout_158.mp4) · [30步/FM8 A并排](videos/A_SWG30_vs_FM8_SWG8_158.mp4) · [D并排](videos/D_SWG30_vs_FM8_SWG8_158.mp4)。

## 两固定其他场景短程验证（EXP-011）

工业/村落从各自FM8首窗续写AA/AD至56帧，有限可行PASS、quality PARTIAL。工业人物/场景与动作响应保留；村落FM8颗粒/边缘残影明显更重于FM30，但短程分叉仍可辨。零新训练、两固定初图/单seed，不证明长时泛化。[Judge验收](../../../../experiments/EXP-011_v3_scene_transfer/judge/FINAL_REVIEW.md)。

## 视频来源

本目录 `videos/` 为已完成实验视频的原字节副本，可直接打开；来源、大小和 SHA-256 见[视频清单](video_manifest.json)。
