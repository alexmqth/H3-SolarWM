# 当前任务状态与下一步

2026-10-10 HKT，Judge。**EXP-005/v2已完成并验收，本轮GPU队列关闭。** 用户批准的有限G0/G1/L1均已执行；没有将剩余额度转给新实验。

## EXP-005最终判定

- G0 PASS：真实旧/新forward、C6 endpoint、124RGB逐值一致。
- G1 accept：SW-G A/D各158帧有限可行性，真实淘汰、最近5祖先、历史不可变、动作方向与结构通过；quality PARTIAL。
- L1 accept execution / archive candidate：cache工程正确、动作信号保留，场景/亮度更不稳定，无已证实联合收益；停止当前无训练Local方向，不追加C9或调参。
- V3 Baseline仍为冻结正式124帧参考，SW-G独立保留，不覆盖。
- 实际281forward/9VAE/0训练/0.662707GPU小时/单GPU0，预算1.70GPU小时。全部GPU进程结束。

[冻结v2任务书](submission/experiments/EXP-005_v3_sliding_window/taskbook_v2.md) · [Judge最终验收](submission/experiments/EXP-005_v3_sliding_window/judge/STAGE2_FINAL_REVIEW.md) · [实际报告](report.md)

## 下一优先提案：EXP-006 / v1 — V3-FM8全程减步

**状态：研究提案，尚未授权GPU或训练。** 只在下一轮明确授权后运行，不沿用EXP-005的剩余GPU预算。

| 字段 | 提案 |
| --- | --- |
| Track / Parent | Mainline效率可行性 / 冻结V3 Baseline Global causal/KV |
| 核心问题 | 从首窗就用普通FM8步，是否仍有可用结构和A/D响应？ |
| 假设 | EXP-004续写8步的正信号可能延伸到首窗，但未验证 |
| Baseline | 原V3 30-step，优先复用已保存同场景seed13/噪声/条件 |
| 固定变量 | Original H3 + released Action LoRA、Single I0、native timestep、own-action/action feedback/current-prefix、Global位置、clean KV commit、12后5分块 |
| 唯一改动 | 所有生成块（含首窗）用native FM8步，shift2.22不扫描 |
| 输入/执行 | 同I0/原噪声：首12 latent→39帧，再AA/AD各两个5-latent块→73帧；各接自己历史，无GT重置 |
| 拟议预算 | 40sampling+3commit=43forward，5VAE，0训练，1卡，≤0.75GPU小时；项目≤3卡 |
| 验收 | 首39结构可用、A/D方向/切换、全新增帧/boundary、历史不变、逐块sampling/commit/decode/内存与完整配置 |
| 停止 | 首窗持续严重崩坏、nonfinite/OOM、协议错误或上限即停；不自动重试/扫描4/12/16步/改shift |
| 交付 | runner/input/config hash、日志/预算、原片/代表对照、报告与Judge独立审核 |
| 不包含 | 新LoRA、AnyFlow/DMD、Local修复、长时扩展或额外scene/seed |

这轮要解除“首窗借用30步”的混杂。8步产生的新历史与旧30步历史不同，所以闭环质量对照需要明确历史来源；不得冒称同raw KV单状态消融。若旧30步首窗条件不匹配，应列出缺口并另计参考预算，不擅自新增30步推理。

## V3-AF后续独立研究

[详细设计](submission/experiments/EXP-005_v3_sliding_window/FUTURE_ANYFLOW.md)：target-time-conditioned student、finite-map训练、r=t初始化与时间符号验证、按student权重刷新历史KV、匹配8NFE/输入历史/总成本。普通FM减步与AnyFlow训练分别评价；旧AnyFlowcheckpoint不作V3-AF完成证据。本次不训练。
