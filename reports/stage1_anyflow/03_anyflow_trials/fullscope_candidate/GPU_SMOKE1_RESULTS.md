# 全Q/K/V/out/FFN rank8：真实GPU单次更新通过

快照：2026-10-08T07:33:44.389173+08:00。GPU0独占、reserve6、CPU raw KV。

真实33B完成1次AnyFlow optimizer update，43,237,376训练参数。单update 102.09秒，含准备及前后验证451.63秒，allocated峰值37629.48MiB（36.75GiB）。无OOM。这些单次数字不是与其他offload状态的公平速度/显存收益比较。

QKV、out、FFN及refiner四组均有有限非零梯度并实际更新；原visual adapter严格不变，时间MLP严格冻结。所有训练参数都属于新bank。初始A/D固定noise的8个样本验证记录与上一轮native初始化完全相同，见gpu_initial_function_audit.json。

| Fixed-noise validation | Initial | After one update |
|---|---:|---:|
| A weighted | 0.118052 | 0.118067 |
| A endpoint raw | 59.624653 | 69.482544 |
| A flow_map raw | 0.168970 | 0.169805 |
| D weighted | 0.170266 | 0.170327 |
| D endpoint raw | 22.804770 | 21.892948 |
| D flow_map raw | 0.228104 | 0.227544 |

A endpoint变差、D endpoint略降，单次更新尚不构成训练收益证据。没有full-scope生成视频，不声称画质或action gate通过。

已经精确恢复自身step01并继续到总16次更新。真实GPU加载审计中，包含完整stage1_lora.pt的四套adapter、Adam、logical/CPU/CUDA RNG、历史记录和teacher身份均逐项相同。见gpu_resume_audit.json。后续按固定协议生成A/D39f的4/8 steps/chunk对照。

这里验证的是硬件与优化器链路；16次仍是初始学习曲线，与官方训练数据/批量/步数的规模相差很大。实际效果应由完整视频、方向/分离度和独立seed等验收，不能把能fit或更新完成称为Stage1完成。

逐投影B因子审计：312/312个逻辑线性投影从零B变为非零，排除只有A因子weight decay变化造成的假更新；见gpu_projection_update_audit.json。
