# Worker 当前执行报告

2026-10-11 HKT。**EXP-006/v1 已完成并通过 Judge 有限可行性验收；EXP-007/v2 AF1-warmup 是当前正在准备的独立批准任务。** 正式进展及后续授权以 Judge 维护的 `progress.md`、`next_plan.md` 为准。

## EXP-006/v1 — V3 普通 FM8 从首块开始

完整 Worker 报告、原始小证据、对比视频和逐块指标见 [EXP-006](submission/experiments/EXP-006_v3_fm8_full/README.md)。固定 Original H3 + released action LoRA、Single I0、native timestep、strict chunk causal、current-prefix、Global RoPE、persistent CPU raw KV、12+5+5 latent、seed13/相同输入噪声、native shift2.22。首 39 帧也由本轮 8-step 新生成，后续 AA/AD 两条各到 73 帧。没有新训练、AnyFlow 或 DMD。

GPU0 五个独立阶段累计 40 sampling + 3 clean commit = 43 full forwards、5 VAE、492.856 GPU 秒（0.1369 GPUh），最高 allocated 26.061 GiB。第二块同首窗历史的 A/D 水平光流分别 +0.846 / −0.321，第三块各自历史分别 +0.505 / −0.826。人物与停车场保持可辨，AA 第三块存在持续透明肢体残影，画质为 PARTIAL。Judge 判定 `PASS_FINITE_FEASIBILITY_QUALITY_PARTIAL`，范围限停车场单 seed、73 帧，不代表 124 帧或跨场景质量。并排视频与保存的 30-step 首块有不同 generated history，是跨协议展示，不能作为纯步数速度或质量消融。

## EXP-007/v2 — V3 target-time student AF1-warmup

Judge 已批准 [任务书](submission/experiments/EXP-007_v3_anyflow/taskbook_v2.md) 的一次 logical batch4、一个 optimizer update；上限 22 forwards、4 backward、0 VAE、0.75 GPUh。AF0 CPU 核查已通过，但 AF1 GPU 训练和 checkpoint 尚未完成；不能把 AF0 写成真实 33B Few-step 质量结果。Worker 正在准备独立 runner，保持原 V3 history/attention/time 协议和训练预算，不加载旧 AnyFlow 产物。
