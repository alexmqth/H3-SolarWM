# 后续 32 条真实小数据拟合预算建议（未授权）

EXP-022 仅证明单条 b784d995 的 C2 普通 FM **2 updates 训练通路**。两个 loss 对应不同 σ（0.5 和 0.75），不可写成下降曲线。tail8 QKV rank8 的容量、首块训练、动作保真、视觉质量和跨 episode 泛化均未验证。以下预算须由 Judge 另发任务书；用户夜间至多 8 张空闲卡、次日 09:00 后至多 3 张，不代表应同时占用这些卡。

## 实测锚点

本次 1 条已编码样本、batch1、同 GPU0：完整进程 413.20 秒，含首次 load/冷启动、reference/新入口等价、2 updates、保存/重载。更新 1 的 clean C1 commit+当前 C2 forward+backward+optimizer 为 `17.051+9.403+2.619+0.171≈29.244s`；更新 2 为 `14.738+9.155+2.168+0.001≈26.062s`。这只是**同一预编码样本**的稳态操作成本，不包括新 36 条视频的 VAE/text 编码、数据交换、验证、失败重试。首次 reference commit 为 147.759 秒，冷启动差异不能忽略。CPU KV 每次约 6.799 GB；GPU allocated 峰 20.293 GiB、reserved 峰 22.852 GiB；事件中最大 RSS 64.704 GiB，连续 RAM 真峰未测。activation checkpoint+CPU saved tensor offload 已开启，backward 时间含其重算；精确重算层数未埋点。

## 分阶段提议

| 独立阶段 | 固定工作量与建议硬上限 | 放行下一阶段的证据 |
| --- | --- | --- |
| 编码 pilot | 固定 2 条新增 clip（1 constant、1 switch；选定后不换），单空闲 GPU；最多 2 video VAE encode、2 I0 encode、8 text encode、0 DiT；建议 ≤900 GPU 秒、峰显存≤44 GiB | RGB→17 latent 正确、前12 latent 与同源39帧精确匹配、动作17 span与fixture来源一致；完整计时和 SHA |
| 其余数据编码 | pilot 通过后固定余 34 条新增 clip；最多 34 video VAE +34 I0 encode，文本调用按 pilot 去重实测设上限；建议暂留 ≤3000 GPU 秒但须用 pilot 实测修订，不提前批量启动 | 32 train/8 dev 各自 episode 来源、完整编码、无缺失或越界；已有4条只复用，训练/开发不串集 |
| 训练第一停点 | checkpoint0→8 updates，batch1、普通 FM、σ抽样策略冻结、同一 adapter/optimizer；同类 commit 每步重建。建议单GPU ≤600 秒、≤16 顶层 DiT（每步1 commit+1 train）、8 backward/8 update | loss/梯度finite、base/Action LoRA不动、checkpoint恢复，预留实际失败预算；不依据训练 loss 宣称画质 |
| 固定验证第一停点 | 2 条开发 clip：1 constant、1 switch，checkpoint0/step8 在**同 seed/NFE、GT history**下各生成一次 C2 8NFE；约 36 顶层 DiT（4×[1 commit+8 sample]）、4 decode，建议 ≤1200 GPU 秒。另固定同状态 A/D velocity 反事实，最多 4 顶层 forward（共≤40）；所有内容先固定，不按结果换场景 | 逐帧人物/背景、边界、动作方向联合审查；score/action 指标不能单独判定通过。开发仅2个曾观察 episode，仍非盲测 |
| 条件扩展 | **只有第一停点联合不恶化且有明确收益**，再增加至16 updates，重新匹配验证；32 updates 和 self-generated-history 续写各需单独停点与额度 | 长时漂移与 GT-history 能力分开判断；不要把局部训练直接升级正式 V3 |

按 26–30 秒/次，仅训练核心 32 updates 约 14–16 分钟；考虑加载、不同样本 I/O、检查点与冷启动，建议正式 32-update 训练任务先留 **≤1800 GPU 秒**，并在 8/16 updates 阶段硬停复核。这个上限是保守草案，不能从本次单样本 profile 推断会有足够模型容量或画质收益。完整编码和阶段验证分别记账，不混入训练吞吐。建议后续验证仍以 8NFE为首轮固定口径，若要30NFE或从 I0 自由生成，必须单列预算；self-generated history、同状态 A/D 换动作和首块训练也各需明确实验。
