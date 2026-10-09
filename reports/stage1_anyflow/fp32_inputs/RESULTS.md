# FP32 输入扰动诊断：真实33B完成，结果混合

完成时间：2026-10-08T06:27:20.156407+08:00。24项测量、8次逐bit回滚；322.8秒，allocated峰值37736.6MiB。

相同frozen16权重、旧clean CPU KV、噪声幅值和有限差分方向；仅改变边界计算与输入扰动的精度。未恢复原生FP32权重。

| Action / chunk / sample | Legacy raw | FP32 boundary raw | FP32 boundary + input raw |
|---|---:|---:|---:|
| A / 0 / endpoint | 23.511124 | 25.384254 | 22.816929 |
| A / 0 / flow_map | 0.184112 | 0.178079 | 0.173387 |
| A / 2 / endpoint | 146.207169 | 59.419834 | 67.016144 |
| A / 2 / flow_map | 0.171378 | 0.169404 | 0.167619 |
| D / 0 / endpoint | 15.565413 | 17.205088 | 17.926905 |
| D / 0 / flow_map | 0.190069 | 0.184026 | 0.180233 |
| D / 2 / endpoint | 17.866949 | 21.598722 | 23.033852 |
| D / 2 / flow_map | 0.229302 | 0.225745 | 0.223078 |

A/chunk2 endpoint显著下降，但另一些endpoint上升；general-map在这些固定状态上小幅下降。没有导数真值，不能把相对legacy的差异称为导数误差，也不能据此宣称视频质量改善。

[control_replay_audit.json](control_replay_audit.json)：与前一独立诊断重叠的16项legacy/boundary测量完全复现。固定旧KV和方向使本实验只解释同状态数值敏感性，不能代表完整重新训练的效果。

完整训练另见[native FP32 candidate](../native_fp32_candidate/README.md)，其原生权重恢复和FP32训练噪声策略与本诊断不同。
