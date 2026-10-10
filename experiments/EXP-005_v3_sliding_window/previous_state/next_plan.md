# EXP-004：V2c原权重8步续写的有限可行性

发布：2026-10-10，Judge。用户要求继续下一阶段；V2c正式基线已冻结于submission commit `d039352941708d4c7c757d696009b67c3e0e2468`。本任务已完成并关闭；以下保留原执行范围，不再授权追加GPU运行。

## 1. Task / Version / Track / Parent

- Task ID / Plan Version：EXP-004 / v1。
- Worker Status：completed；Judge Acceptance：accepted。
- Research Track：Mainline efficiency / 少步基线；不是AnyFlow训练。
- Parent：已验收V2c（Original H3 + released action LoRA，native Single I0/current-prefix/严格chunk causal/CPU persistent raw video KV）。

## 2. 唯一问题 / 假设 / 决策价值

**仅把新生成chunk的native FM采样从30步减为8步，不训练、不改其它协议，能否保留同历史A/D响应，并在自己的8步历史下续到73帧的基本人物/场景结构？**

假设是已有权重可能有足够的减步余量，未预设通过。本轮不是AnyFlow，也不把普通Euler减步结果用于证明AnyFlow成败。

已有8步V1/V2a与旧AnyFlow试验使用不同图/anchor/权重，不能回答本轮V2c同协议问题；EXP-002/003只验证30步。先做本轮比新开蒸馏训练更便宜。

- 正结果：后续优先验证8步自己的历史能否延伸；再决定是否需要训练，而非直接开大AnyFlow任务。
- 负结果：停止原权重直接8步路线，不扫4/12/16步、shift或gain；保存30步V2c与8步负对照。再依据可修复信号决定独立的有限少步适配任务，绝不自动扩训。

## 3. Baseline / Controlled / Changed Variables

唯一主动变化：new chunk的sampling steps从30改8，使用原生scheduler按8步生成sigma序列，flow shift仍2.22，普通FM/Euler更新不变。不从30步列表随意截断前8步。

固定Original权重和released action LoRA、停车场832×480、seed13、full37 noise、Single I0、native timestep、global RoPE、固定audio、h3_fp32边界精度/backend/offload、current-prefix own-action连接、sigma0原图clean commit和CPU raw KV；首12已有history与KV相同，首39已发RGB相同。

30步对照直接复用EXP-002的AA/AD56与73帧、端点/原始指标。首12由既有30步流程生成，本轮复用；因此结论只能称**8步续写**，不能说整片从零均为8步、已测8步首屏或完整8步E2E。

## 4. Inputs / Provenance

- 已冻结首12：`submission/experiments/11_causal_12_then5_selfhistory/states/first12_A.pt`，SHA `242a1db06bc3423fe21207af5d2ccf59c9f27f07c9d88f22ce9f77921f6b0eea`。
- 直接加载`H3-World/outputs/EXP-002_native_cached/first12_A_cache.pt`；无需重新prefill或identity诊断，保持祖先KV表示。
- 已发首39 RGB从EXP-002的published_56.npy截取并核对EXP-002前缀摘要，AA/AD共用相同像素，不从新解码覆盖。
- conditioning/noise/LoRA沿用EXP-002/003冻结runtime和source manifest；使用同一全局位置与动作裁剪。
- EXP-002第二/第三块的30步latent仅作对照，不能作为8步路径第二块后的历史。

## 5. Execution Plan

### S0：最小入口与预检

在`submission/experiments/EXP-004_v2c_8step/`建独立config/runner；优先复用EXP-002 interval及既有路由，不重写注意力实现。新改动仅8步schedule、复用首窗与账本/输出路径。CPU检查覆盖实际8步schedule到终点、预算、输入/首RGB/缓存摘要；继承已通过因果和cache测试，不为未改部分重复GPU诊断。

冻结实际源码/config/命令。每次forward和VAE尝试调用前记账，包括失败；禁止自动重试。项目其它用户GPU不得干预。

### S1：同A历史下AA/AD分别8步续第二块到56帧

两个分支读取同一首12及raw KV，噪声/anchor/audio/全局位置/RGB前缀相同，仅当前动作A或D不同。interval[12,17)、index1；8步sampling只读历史，记录内存cache identity/storage/version/commit计数不变，冻结39RGB并追加17。

先看两条新增帧。如果严重结构崩坏或动作条件明显失效，则停受影响路径并保留真实负结果，不再自动续该路径；普通细节/边界/局部proxy异常按可行性尺度判断。观察足够改变决策即可收口。

### S2：值得继续的路径续自身第三块到73帧

只将自己的8步第二块clean endpoint按原sigma0/native/原stop17动作条件提交一次，index1；不得换回30步第二块。然后[17,22)、index2再8步sampling，只读已提交cache；冻结56RGB，追加17。最后第三块不做无用commit。

每条最多两新chunk，最多四条8步采样。Worker可在授权内逐块快速视觉检查后继续，不必等待Judge逐块许可；明显失败按停止条件报告。

### S3：交付并停止

给出8步AA/AD完整56/73原片、左V2c 30步右8步的共有长度并排片。清楚标明首窗仍是复用30步历史，新块是8步；若路径提前停则按真实长度对比，不补片或循环。第二块可同history比较；第三块两种steps/动作路径的历史已不同，要区分。

提交完整report、代码/配置、来源/产物manifest、实际命令/日志、逐块成本与账本。提交后停GPU，由Judge验收、归档和Git发布，不自动开展下一实验。

## 6. Budget / GPU

- 0 optimizer update、0额外真实模型diagnostic、0首窗prefill、0新增基线生成。
- 新sampling≤32（最多4×8），第二块clean commit≤2；完整denoiser总≤34。最后块不commit，失败/重试计入上限。
- VAE decode≤4；RGB re-encode0；单scene/seed，仅AA/AD。
- 全任务≤0.5 GPU-hours，首个GPU进程起≤90分钟elapsed；每卡allocated≤44GiB。加载/缓存搬运/commit/VAE/占卡检查等待/失败都计GPU-hours。
- 现在已过2026-10-10 09:00 HKT，**项目总GPU≤3，本任务最多1卡**。启动前重新确认空闲卡；当前优先空闲GPU0，但不能仅凭编号猜测可用。不得终止或迁移他人进程。

## 7. Evaluation / Acceptance

分别报告执行有效性、动作、基本视觉、缓存与效率。静态全部新增帧和必要native细节必须查看；不能仅看末帧。普通质量缺陷允许，但明显持续的人物分解/控制消失须明确失败，不能为了减步利益隐藏。

主要对照为同V2c协议30 vs8。第二块匹配历史/初始noise/条件，主动变量只有schedule步数；第三块接各自生成历史，属于闭环协议结果。记录实际sigma列表、8次sampling、commit数、cache未变、RGB冻结与checkpoint/hash。

记录每块sampling、commit、VAE、wall、cache tokens/bytes和GPU峰值。sampling含缓存搬运；若未另测搬运子项就标未拆分，不为补计时复制路由或重复运行。峰值应覆盖本进程真实sampling/commit/VAE；缺项披露。比较8步与既有30步的单次观察成本，注明不同时刻共享硬件、首窗复用；不要直接宣称端到端3.75倍加速。

结果是负也可以接受任务执行；8步能力按PASS/PARTIAL/FAIL区分。只因8步推理完成或flow符号符合不宣布能力通过。8步成功也不改名为AnyFlow或宣布训练完成。

## 8. Stop / Out of Scope / Deliverables

协议错误/未来泄漏/历史回改/NaN/OOM/预算超限立即停相关进程；严重持续质量或动作失败停相关路径，充分证据后停止扩展。无4步、其它步数、noise/shift/gain/mask扫描、长至124帧、新scene/seed、训练、AnyFlow/DMD或新Original基线授权。

独立目录：README、config、源码与hash、metrics、MANIFEST、原片/并排片、日志和budget；根report.md真实状态。Judge审阅后更新progress/next_plan/archive/submission，正常Git合并与推送。本任务只授权上述≤34前向。

## 9. Judge 验收与下一决策（2026-10-10）

**accepted / 本轮关闭。** AA/AD8步续写都到73帧；同history第二块flow +0.780621/−1.509824，自己的8步历史第三块+0.978349/−0.732276。基本结构与动作响应有限通过；AA后段透明肢体/腿拖影明显，画质/严格连续性PARTIAL。首39帧复用30步，不能称全程8步或AnyFlow。

实际32sampling+2commit=34forward、4VAE、0训练/额外诊断/首窗prefill，466.01794 GPU秒（0.12944943 GPU小时），峰值25,682.07MiB；GPU0顺序执行，已释放。代码与来源、own-history/cache/RGB冻结、原片与4个并排片核验完成。详见[正式审核](submission/experiments/EXP-004_v2c_8step/judge/FINAL_REVIEW.md)与[展示](submission/report/v2/v2c_strict_causal_kv/8step_continuation/README.md)。

**下一项建议：首窗也使用8步，再检验自己的8步历史，解除对30步前缀依赖。** 先短窗判断结构是否可用，再有限延伸；不为小幅proxy变化追加调参，不自动开AnyFlow/DMD或训练。当前没有EXP-005任务授权；启动前另立预算和停止条件。本轮结果保留为V2c增量证据，正式30步124帧版本不变。

## 10. 分类更新（2026-10-10）

本任务现属于V2c Strict Causal + Persistent KV，原称V3；任务编号EXP-004/v1、预算、结果和关闭状态不变。实验入口为`submission/experiments/EXP-004_v2c_8step/`，汇报为`submission/report/v2/v2c_strict_causal_kv/8step_continuation/`。V2a/V2b/V2c是并行方案，未来V3定义尚未发布。
