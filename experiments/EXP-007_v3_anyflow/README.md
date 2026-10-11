# EXP-007 — V3 AnyFlow新student

## 名称与计数

本实验的训练状态为 **AF-train-32updates**，8次采样求值配置为 **AF-8NFE**；完整写作 **AF-train-32updates + AF-8NFE**。AF2是第二个训练阶段，新增31次optimizer updates；加上AF1的1次，累计32次。AF3是视频评估阶段。AF2/AF3不是2/3 NFE。后续EXP-009复用同一checkpoint评估AF-4NFE，未新增训练。

原`step_32/`路径、阶段文件名及冻结证据继续用于复现。[统一命名及旧名映射](../../report/GLOSSARY.md)。

## 已审核范围

- AF0：真实tiny-H3 CPU协议/梯度/有限映射测试通过；[CPU审核](AF0_REVIEW.md)。
- AF1：真实33B初始化与单次finite-map更新工程通过；[Judge审核](judge/AF1_REVIEW.md)、[独立checkpoint审计](judge/af1_audit.json)、[Worker完整报告](worker_report.md)。初始化diagonal逐值一致，target/QKV有有效梯度与更新，更新后自建KV变化，冻结base不变。
- 两attempt累计22forward/4backward/1update/0VAE/186.58153秒，peak allocated26.63145GiB。首次路径类型错误及修复重跑单独保留，未覆盖记录。
- AF2：累计32updates训练工程通过；[Judge审核](judge/AF2_REVIEW.md)、[Worker报告](worker_report_v3_af2.md)。新增31updates/527forward/124backward/1.191072GPUh。
- AF3：**共同FM-8NFE首窗后的AA/AD73帧8NFE续写有限可行性通过，quality PARTIAL；尚无优于FM-8NFE的整体收益证据。** [Judge审核](judge/AF3_REVIEW.md)、[独立审计](judge/af3_completed_audit.json)。35forward/4VAE/0.131389GPUh，含首次失败。不是从首窗全程AF。

## 当前任务与协议

[EXP-007/v3任务书](taskbook_v3.md)已完成AF2训练与AF3匹配8NFE视频验收，不继续扫参。AA/AD冻结V3 C2 generated数据交替，每步按student当前权重重建C1 clean KV；评估首窗来自新FM-8NFE，C2同历史、C3各自历史。后续独立DMD任务以根目录next_plan为准。

Original H3 + released Action LoRA、Single I0/native time/current-prefix/own-action/action feedback、Global、strict causal/persistent raw KV、12后5均保持。新last8 rank8 QKV和target-time MLP gate.25，与普通FM-8NFE减步独立。

## 证据与复现

[AF1冻结配置](config_v2.json) · [源清单](source_manifest_v2.json) · [小证据与外部checkpoint清单](artifact_manifest_v2.json) · [初始化/单步入口](run_af1.py) · [首次失败](AF1_ATTEMPT1_FAILURE.md) · [修复授权](AF1_RETRY_AUTHORIZATION.md)。大权重、optimizer与RNG保持H3-World/outputs/EXP-007_v3_anyflow_af1_attempt2/，由清单SHA追溯，不入Git。

后续[DMD独立设计](FUTURE_DMD.md)要求角色、方向、真实生成链梯度和各权重自建KV，当前尚无DMD GPU任务。
