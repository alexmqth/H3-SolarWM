# EXP-022/v1 · V3 普通 FM 真实训练入口与有限 profile

[Worker 结果报告](WORKER_REPORT.md)：CPU 数据与来源检查通过；单卡 checkpoint0 前向/KV 精确等价、2 次真实 FM 更新和保存重载完成，**Judge更正为needs_revision：前向/KV通过，训练梯度协议待修复验证**。模型画质和动作能力尚未评测。

[冻结任务书](taskbook_v1.md) · [授权与上限](AUTHORIZATION.json) · [CPU 来源/布局检查](CPU_PREFLIGHT.json) · [来源 SHA](SOURCE_MANIFEST.json)

本任务仅检验工程训练通路与实测资源，不评估视觉质量或动作恢复。模型协议为 Original H3 + released Action LoRA、V3 Single I0/native time/Global RoPE/own-action/current-prefix feedback、clean raw KV 和 `[12,5]` 分块。训练目标是普通 FM `epsilon-clean`；无 AnyFlow target time、DMD、视频采样或新 VAE/text 编码。

数据：EXP-021 固定的 36 条新增候选 RGB 已[物化](RGB_MATERIALIZATION.json)并逐条完整解码；另 4 条复用 EXP-015。每条有 8 帧 contact sheet；[总览](RGB_OVERVIEW.jpg)与[质控摘要](DATA_QA.json)提供快速检查。新增数据**尚未** VAE 编码，不会混入本次 profile。本次真实 FM 使用已有 b784d995 `full17/prefix12` latent 和 EXP-019 E fixture。

代码：[独立可微 V3 wrapper](interval_train.py) · [CPU preflight](preflight.py) · [GPU runner](profile_gpu.py) · [RGB 构建](materialize_rgb.py)。冻结推理入口未修改。[GPU账本](GPU_BUDGET.json) · [原始日志](GPU_RUN.log) · [KV等价](CACHE_EQUIVALENCE.json) · [velocity等价](FORWARD_EQUIVALENCE.json) · [更新数值](UPDATES.json) · [重载](RELOAD_CHECK.json) · [最终CPU审计](CPU_FINAL_AUDIT.json)均已落盘。后续建议与成本分解见[预算草案](NEXT_STAGE_BUDGET_DRAFT.md)。Judge 负责最终验收与 Git 发布，Worker 不提交 Git。

[Judge最终审核](judge/FINAL_REVIEW.md) · [独立核查](judge/INDEPENDENT_AUDIT.json)。后续32条/128updates小数据拟合归EXP-023。

**最新更正：** [checkpoint反向重算context风险与验收撤回](judge/REVIEW_CORRECTION_01.md)。旧step02不可作正确V3梯度证据，修复在EXP-023执行。
