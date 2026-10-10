# EXP-011/v1 — 两个固定初图的 V3 FM30 / FM8 短程迁移

Judge，2026-10-11 05:40 HKT。**当前只授权 CPU 实现和输入来源核查；预处理、首窗生成和续写分别需要 Judge GPU marker。** EXP-010 已验收并结束。正式 V3 Baseline 与全部已保存结果冻结。

## 问题、依据与决策价值

Research Track：Mainline evaluation；Parent：V3 Baseline 原生 causal/KV 协议与 EXP-006 普通 FM8。问题：停车场验证过的无新训练 V3 FM30 / FM8 能否在两个固定、视觉差异较大的既有 ABot 初图上保持基本人物/场景，并产生短程 A/D 响应？变量为输入场景与采样步数；按场景分别比较两个完整闭环方案。

当前验收主要覆盖一个停车场，继续减少 NFE 或修补已失败配置的 ROI 较低。若两种步数均可行，保留 FM8 并优先整理协议/数据；若 FM30 可行而 FM8 明显失败，则缩小减步能力边界、后续考虑多场景适配；若两者均失败，则先确认原生协议迁移与动作条件的限制，不立即扩大训练或筛场景。这是两例迁移检查，不构成统计泛化结论。

## 固定输入

仅使用下列现有 PNG 与 `annotation_pilot.json` 中对应 `caption.scene_static`：

1. `H3-World/data/abot_bridge/clips/validation/118eb5d8b75e1b8ac23a4e9ae77af9a9/A_1140.png`：户外工业场景。
2. `H3-World/data/abot_bridge/clips/validation/dfec8ed3237860eba14d67c089ecd041/D_1750.png`：中世纪村落场景。

核对 `encoded_manifest.json` 的 episode/split 和 `preparation.json` 原始视频来源。二者为已有固定 validation episodes，没有根据本轮输出选择。它们曾用于旧协议评估，不声称对全部历史实验或 H3 预训练绝对未见。ABot 为游戏录屏，不称现实世界摄像数据。

**旧 encoded 文件协议为 `global_retimed_rgb_prefix_last_image_dual_v2`，禁止复用其 clean latents、prompt、packed conditions 或 KV。** 新输入只以原始 PNG 和静态描述为源，重新生成 native Single I0 条件；不使用 narrative、未来视频帧、旧 teacher endpoints 或 GT actions。图像原尺寸 832×480，记录实际预处理与哈希。

## 不变协议与比较设计

Original H3 + released Action LoRA，零新增权重/训练；Single I0、native timestep/audio1000、own-action routing、action feedback、current non-action prefix feedback、clean KV commit、Global RoPE、strict causal + detached persistent CPU raw KV、max_history5、h3_fp32 边界与已验收 attention backend保持一致。不得把当前随动代码代替冻结 runtime 而不审查。

参考 EXP-006 的原生 37-latent 输入构建与裁剪方式，生成仅到 C1=12 latent/39 RGB，再 AA/AD 各一个 5-latent C2 到 56 RGB。使用相同完整 packed horizon，避免步数比较同时改变 prefix/video/camera 位置。若必须不同构建，CPU 报告先明确并由 Judge 决定，不能暗改协议。

每场景两个方法共享同 I0、静态文本、A/D action script、seed13 原始初噪声/audio噪声、空间尺寸和 native shift2.22；各方法 C1 从同噪声独立生成，FM30=30-step，FM8=8-step。随后各自 sigma0 提交自身 C1，AA/AD C2 共享该方法同 C1 cache 与初噪声；AD 仅在 C2 切 D。C2 不必提交终端 KV。A/D 文本由同一冻结动作脚本产生，记录确切句子与当前动作 rows。

FM30 指 **V3 causal 30-step**，不能标作 Original H3 双向参考。两个方法 C1 与后续生成历史不同，因此比较不是同 KV 消融。不到滑窗淘汰长度，不借本任务宣称滑窗新结论。

## 阶段与预算

### P0：CPU 准备（已授权）

冻结两图、静态描述、split/source 与配置；审查 native 输入构建调用链、当前/未来 action 隔离、chunk/crop 索引与位置、精度/后端；提供完整运行 import 预检、预算闸门和 source manifest。明确新 fixture 的 GPU 编码函数、各类调用数量和复用方式；编码不得触发 denoiser sampling。设计成功/失败均保存的账本，无自动重试。旧停车场证据与模型权重只引用，不复制。

### P1：输入 GPU 编码（另放行）

最多两个 scene fixtures，**0 denoiser forward、0 training、0 视频 decode**，总上限 .25 GPUh；Worker 在 P0 报告中列出文本/图像/VAE encode 的精确调用预算后，Judge 写明 counts 再放行。编码后做 CPU 检查：A/D 的 I0/noise/camera/video位置一致；非动作文本相同；差异仅是 action rows；Single I0 位置与原生结构成立；保存新 fixture SHA。不得直接读取旧 Dual Anchor 编码。

### G1：新首窗（另放行）

两场景 × (FM30 30 + FM8 8) = **76 sampling forward、4 VAE decode**；0 commit。保存四个新 39 帧片及全部帧供审查。Judge 按各 scene/method 判定是否继续；出现持续全噪声、主体/场景消失时该配置停止续写，保留另一个对照的有限机会。一般画质缺陷为 PARTIAL。

### G2：自身历史 AA/AD C2（按首窗分别放行）

四个 scene/method 各 1 clean commit，共4；两场景 × 2分支 × (30+8) =152 sampling；合计 **156 forward、8 VAE decode**。新生成 8 段17帧，最终各56帧。若 G1 某配置停止，则减少实际预算，不挪用其余额加别的场景/种子。

推理总上限 **232 forward =228 sampling+4commit，12 decode，0 backward/update，≤1.25 GPUh**；加 P1 总 ≤1.5 GPUh。每卡 allocated ≤44 GiB；空闲磁盘保持至少60 GiB，新增 tensor/fixture/缓存预估≤50 GiB，禁止删除历史证据腾空间。09:00 HKT 绝对截止，之前仅使用实时真正空闲的卡并计所有卡 GPU 小时；不碰其他用户进程。无需为了用满卡而并行，可按两场景独立执行。09:00后项目最多3卡，当前任务不自动续跑。

## 评价与停止条件

按 scene/method 检查首39全部帧及全部新17帧、必要原分辨率；主体/场景、动作方向/切换、ghosting和边界变化分开报告。停车场光流符号不直接迁移为新场景控制准确率；结合两个反事实分支和原动作句子，无法判定则 PARTIAL/不确定。缓存/历史 RGB 不变、sigma grid、真实 steps、内存和每段 sampling/commit/decode/加载总耗时纳入审计。不只凭 finite forward 或 MP4 可解码宣布能力通过。

OOM/nonfinite、协议/来源/position错误、预算触顶、磁盘风险或持续结构失败则停止对应后续启动，保留失败日志，无自动重试/改种子/换初图/调 prompt。无 C3、无长视频、无 FM4/AF/DMD/训练、无场景筛选。必要实现修复先 CPU 复核并重新冻结，GPU 重跑需 Judge 单独决策且计入预算。

## 交付与验收

独立目录保存代码/config/source/input manifest、输入来源/编码预算和结果、CPU 测试、分阶段授权、命令/原始日志/账本、所有原片/全帧图、各场景 FM30 vs FM8 AA/AD 并排片、每块指标、Worker 报告和 Judge结论。视频清楚标注场景、NFE、自身生成历史、Single I0。大权重/KV仅保留原路径与摘要，不纳入 Git。

执行成功、模型能力与迁移结论分开，只有 Judge 正式接受后才更新主线与报告导航。不同结果按上述决策收口，不自动扩展。

# EXP-011 P0审核与P1输入编码放行

2026-10-11 05:53 HKT。Judge独立CPU入口预检与test_p0均PASS，最终代码清单26项已逐项核对；PNG/静态描述来源、native full37 Single I0、A/D独立句子嵌入、GPU调用记账与09:00截止已审查。准备期修正独立入口导入顺序、BF16/FP32噪声摘要和内存KV签名；没有失败GPU尝试。真实新fixture正确性仍待编码后审核。

**批准GPU0仅执行P1两初图编码：最多6次text encoder forward、2次image VAE encode、0视频encode、0denoiser forward、0decode、≤900 GPU秒。** 启动前确认实际空闲、单卡allocated≤44GiB、磁盘≥60GiB。不改冻结代码/config，不自动重试。完成后跑CPU audit_fixtures.py并提交真实新输入哈希和账本。

G1四个首窗、G2续写尚无GPU授权；P1成功不等于生成能力通过。总任务232推理forward/12decode、编码.25+推理1.25GPUh与09:00截止不变。

## P1首发失败与一次修复重跑授权（05:56 HKT）

首发在模型加载前因sha字符串路径类型失败，0模型调用；保守计费4.386365982秒，原始日志/账本/首发源码与marker已归档并核对哈希。修复统一Path(path)，初始化纳入异常处理，独立test_p0通过。Judge批准P1 attempt2一次重跑，仍6text/2image encode、900秒累计预算、0去噪/0decode；失败成本不重置。代码manifest现为11363412837dfdca85a611761fc1a59b730cb2c541117942fbbcf14f24866976，无协议变动。G1/G2仍未授权。

# EXP-011 P1模型目录修正与attempt3授权

2026-10-11 06:00 HKT。attempt2因独立P1没有import infer所设置的模型根目录而解析出空权重列表，0模型调用，已终止；其日志/result/预算和marker原样存attempt2_archive。累计P1账本8.667172311秒，所有模型调用计数仍0。账本包括Worker明确标注的保守失败启动开销，原始账本保留。

Judge CPU使用相同ModelConfig、禁止下载、显式DIFFSYNTH_MODEL_BASE_PATH核查通过：14个已存在文本权重分片、1个VAE文件、processor目录正确指向既有H3模型。证据P1_MODEL_PATH_CPU_CHECK.json。

批准attempt3一次编码启动，命令环境必须按P1_APPROVED.json显式指定冻结模型根。代码manifest不变（11363412837dfdca85a611761fc1a59b730cb2c541117942fbbcf14f24866976），同6text/2image encode、0denoiser/0decode、累计900秒；不重置失败账本、不改图/文本/协议。G1/G2仍未授权。此修复只处理资源解析，不能作为模型能力证据。

# EXP-011 P1验收与G1首窗放行

2026-10-11 06:02 HKT。P1输入编码接受，生成能力未评估。Judge独立CPU审计确认新两fixture来源、原生CPU噪声逐值重建、Single I0、A/D非动作条件和所有保留Global坐标；Worker检查own-action mask及未来action删除通过。真实6text/2image encode、0denoiser/0decode，含两次失败启动累计70.564697678秒，allocated峰40.576426GiB。失败与修复证据保留，输入/模型协议没有改变。

现在批准GPU0顺序执行industrial/village × FM30/FM8四个G1首窗，合计最多76 sampling forward、0commit、4VAE decode，沿用推理累计4500秒与09:00截止。启动环境按G1_APPROVED.json，尤其ABOT_VRAM_RESERVE_GIB=18保持已验收推理offload设置，显式模型根禁止下载。每个首窗从各场景相同seed13噪声独立生成，全部39帧需审查，严重结构崩溃的配置不继续G2。无自动重试、无新训练。

G2仍未放行；P1工程成功不代表画质/动作迁移通过。正式V3结果保持冻结。

# EXP-011 G1首窗验收与G2逐配置放行

2026-10-11 06:15 HKT。四个G1首窗已完成，Judge看过全部156帧和必要原分辨率细节。工业FM30/FM8人物/场景与运动可辨；村落两方法中段树遮挡后人物重现，FM8的颗粒感、树叶与人物交叠残影更明显。四配置首窗有限可行接受，quality PARTIAL；本阶段不宣称A/D切换或persistent KV续写通过。

独立CPU核查四条实际native sigma、源fixture/初噪声/prompt/Global坐标、FP32边界、端点/RGB/video SHA、39帧24fps全解码及PTS全部PASS。实际76sampling、0commit、4decode，推理账本577.338099311秒，allocated峰25.090662GiB；P1另外70.564697678秒。详细逐配置审计和visual_notes见judge目录。

**现在分别批准industrial/village × FM30/FM8的G2。** 每配置提交自身C1 clean KV一次，再共享同C1 KV/尾噪声生成AA与AD各17新帧至56。两个FM30各61forward/2decode，两个FM8各17forward/2decode，G2合计156forward=152sampling+4commit、8decode。全任务推理232forward/12decode、4500GPU秒及09:00截止不变；GPU0/ABOT_VRAM_RESERVE_GIB=18等环境与G1保持一致，真实空闲检查、44GiB显存和60GiB磁盘底线。

每个配置有独立G2 marker，绑定已验收首窗row/endpoint/RGB SHA。不覆盖首39帧，检查实际内存KV签名和全50层indices。看完整新增画面与A/D差异，光流仅辅助。一般缺陷PARTIAL；协议/数值/资源问题或持续结构崩溃停止后续启动，无C3/新训练/调参/自动重试。
