# EXP-027/v2：两片 ABot C2 普通 FM 可拟合性验证

2026-10-16 HKT，Judge授权原Exp Worker立即实现并执行。用户已确定ABot主线；CPU检查与真实首步梯度门控通过后自动连续运行，无需额外marker。v2覆盖GPU_PROPOSAL的预算/视频门槛，原提案保留历史。

## 单一研究问题

原32clip实验每clip只有4次C2更新。本轮集中在两条清楚常量动作片，各32次新噪声更新，检查固定独立噪声下局部拟合和匹配视频是否出现清楚学习信号。不是泛化、同状态动作因果对照或充分训练结论。

## 冻结条件

两片：43866101158a538d22c52ac5b2fa606b_src0621_rgb56（S+D）与7199292ca430b478bfd0d83cf2e519d8_src1670_rgb56（A+S）。两片不同I0/episode。复用EXP-023已有source-RGB编码，不下载/重编码。

Original H3 + released Action LoRA冻结，EXP-025 checkpoint_000.pt全50层QKV rank8零增量初始化（SHA65fc21a9afa45db9efa7de4b89c6323df694cbd601142c9a86669afe9276c659），10,752,000可训练参数，重新初始化optimizer。保持原Single I0/native time/Global RoPE/own-action/action feedback/current-prefix feedback/12+5 strict causal/clean raw KV/h3_fp32/SDPA。新代码只放EXP-027，不修改023/025或正式V3。

## 训练与固定probe

仅训C2、GT C1历史，交替两clip，64updates、每clip32次，batch1、AdamW lr1e-4/betas(.9,.95)/weight_decay .01，独立训练RNG seed13，noise和sigma~Uniform(0,1)每update新采样，排除端点。每update先用当前adapter clean commit GT C1，不跨更新复用KV；历史不反传，禁止未来目标进入历史。沿用修复后的完整forward/backward重算current-prefix作用域与冻结参数检查。

保存0/32/64的adapter、optimizer和RNG。各checkpoint用2fit+既有dev constant/switch共4片，sigma=.10/.25/.50/.75/.90 ×独立probe seeds113/127，共40probe/阶段、120总。probe状态与训练RNG隔离；同checkpoint/clip只需一次GT clean commit，合计12额外commit。记录逐clip/σ/seed MSE、变化幅度和finite，分别报告train/dev；dev已观察，非盲测。

## 视频与验收

正常完成且数值健康时，**必须**生成两条fit片GT39+生成17的checkpoint0/64匹配对照，共4视频；预算预留，不能因MSE不足10%全部跳过。相同输入、采样seed、初噪声、FM-8NFE/chunk、flow shift2.22，输出完整56帧24fps并清楚标注仅后17帧生成。若在训练后统一评测前后权重，CPU预先验证保存重载，准确记录额外加载时间。

只有两fit片独立噪声平均MSE各改善≥10%，且低σ≤.25和高σ≥.75两段均改善、所有数值finite，才自动追加既有dev_switch的self-history checkpoint0/64两片。其C1状态随模型不同，不能声称同历史对照。最多6视频/6VAE decode；无新VAE/text encode。

10%仅为可选视频费用开关，不是模型验收的硬阈值。Judge综合学习曲线、实际运动、结构和边界决定是否继续；形变不单独否决，MAD下降不能掩盖静止。64updates后停止，不自动加rank/LR/epochs/AnyFlow/DMD。无信号只说明本预算候选不支持扩展，不能宣称模型永远不可拟合。

## 顺序、计数与预算

CPU准备与复用来源/四条encoded输入/零adapter checkpoint SHA检查、production采样加载器及媒体保存验证后，启动一次受控单卡任务。最先最多6DiT做disabled/enabled零adapter C1/current、C1/commit、C2/current入口回归，核对velocity/KV及协议。首个真实C2更新确认50层梯度、重算路由、历史未变、冻结参数无梯度后自动继续；其调用已计入64updates，不另外跑profile。

总上限：**6000单卡GPU进程秒，336顶层DiT尝试，64backward/64updates，6decode，0encode/0text encode**。构成：训练128、probe120+历史commit12、四fit视频36、可选dev视频34、零入口回归6。未用计数不能用于额外实验/重试。checkpoint内部activation重算不另计顶层DiT，费用计入时间。

单张空闲L40，优先GPU0；不抢其他用户进程。最多1卡，遵守全项目日间≤3/夜间≤8。外部watchdog覆盖加载/保存/评测/释放，5980秒停止并6000秒内清理本任务组。为末尾4个必要视频预留800秒，setup/回归与初probe后记录实际吞吐；预计无法在余量内完成训练/probe/必要视频则在完整update边界停止并报告，不自动延长。阶段参考train3200/probe1400/video1000/setup400秒，但以总6000硬上限与必要视频预留为准，不新增阶段重试。

allocated≥40GiB、RSS≥96GiB、OOM、非finite、路由/冻结梯度/来源/未来隔离错误或超时立即停止；不自动重跑。实现CPU自检≤30分钟/4workers/RSS32GiB；媒体仅CPU从已保存RGB恢复，不重新采样。运行前冻结精确config/代码hash/账本，记录物理GPU与PID；启动/32updates/结束消息Judge。

## 交付

CPU_PREFLIGHT、冻结来源、配置/代码、零回归与首步梯度记录、训练顺序/独立噪声/固定probe结果、checkpoint receipt、逐阶段耗时/显存/KV/RSS/调用账本、4必要+最多2可选原片及对比/视频SHA、WORKER_REPORT与根report.md。所有失败费用计入，准确报告实际完成数。Worker不操作Git/progress；Judge审核并push。
