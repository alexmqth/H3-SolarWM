# Worker 当前执行报告

2026-10-11 HKT。**EXP-015/v1四段真实56帧数据准备已完成，等待Judge验收；0 GPU调用、0模型编码/训练。** 四个固定train episode的56帧MP4/I0、56×17真实动作和17latent原生池化已生成；224帧完整解码/PTS、全部11原始key、首39动作前缀、C1动作预处理未来隔离及实际FFmpeg源PTS逐帧索引均通过。s3的C2真实动作为A→静止→W，四段没有D反事实GT。近邻RGB像素探针在低运动s1有歧义，但原FFmpeg select后的56个源PTS都精确匹配。构建阶段FFmpeg实际OS线程峰值超出任务书4线程上限，已在[完整Worker报告](submission/experiments/EXP-015_v3_real56_data/WORKER_REPORT.md)披露。后续VAE前缀因果性和packed文本长度影响global位置仍是独立闸门，当前没有任何模型质量结论。[数据入口与四视频](submission/experiments/EXP-015_v3_real56_data/README.md) · [输出清单](submission/experiments/EXP-015_v3_real56_data/OUTPUT_MANIFEST.json) · [源PTS审计](submission/experiments/EXP-015_v3_real56_data/SOURCE_PTS_AUDIT.json)。

2026-10-11 HKT。**EXP-014/v2 两条缺失AD均按Judge marker串行恢复完成，Worker CPU审计PASS；本轮所有GPU进程已退出，训练更新0。** GPU0先跑`s1`后跑`s3`，每图30 sampling/1decode/0commit，分别159.631/159.061 GPU秒，峰值约38.178GiB。原C1、AA、50层KV、半途AD行与旧39RGB不变，两条56帧恢复视频可完整解码。[s1并排视频](submission/experiments/EXP-014_v3_multiscene_teacher/artifacts/recovery_v2_comparisons/s1_7199292c_AA_vs_recovered_AD_56.mp4)显示D反向较清楚，但A持续方向弱；[s3并排视频](submission/experiments/EXP-014_v3_multiscene_teacher/artifacts/recovery_v2_comparisons/s3_b784d995_AA_vs_recovered_AD_56.mp4)两路几乎同向，D反向未证实。加上此前`s2`同向，四训练图不能不加筛选当作全都动作正确的teacher。[逐场Worker用途标签](submission/experiments/EXP-014_v3_multiscene_teacher/WORKER_TARGET_ASSESSMENT.json)与[累计预算上下界](submission/experiments/EXP-014_v3_multiscene_teacher/WORKER_CUMULATIVE_BUDGET.json)已单独冻结；完整协议、光流、显存与质量边界见[恢复GPU报告](submission/experiments/EXP-014_v3_multiscene_teacher/RECOVERY_V2_GPU_REPORT.md)。等待Judge正式验收及下一任务；不自行启动AnyFlow/DMD或扩场景。

2026-10-11 HKT。**EXP-014/v2 仅CPU恢复准备已完成，GPU补跑仍待Judge新marker。** 针对T2中断的`s1/s3`两条AD，新建独立[恢复入口](submission/experiments/EXP-014_v3_multiscene_teacher/recover_ad.py)和冻结的config/source/code manifests；实际C1、AA、50层原cache、D action span、初始噪声、Global位置和31点sigma均经CPU复核。两图预检均PASS；缺marker的GPU入口预期拒绝且未创建输出，write-ahead账本的30F/1decode单图上限测试PASS。恢复只会从已保存C1/cache重新计算缺失AD，不重做C1/AA/commit，不覆盖原半途账本；本阶段新增GPU模型调用与训练更新均为0。完整来源SHA、命令和后续GPU门槛见[恢复CPU报告](submission/experiments/EXP-014_v3_multiscene_teacher/RECOVERY_V2_CPU_REPORT.md)。

2026-10-11 HKT。**EXP-014 T2 已按本轮 marker 停止并完整归档实际结果；新训练更新仍为0。** 三张固定 train 初图均完成 native V3 FM30 C1 39帧及 AA 56帧。工业道路 `s2_9dc2e588` 正常完成同历史 AA/AD，90sampling+1commit/3decode、516.294 GPU秒，CPU协议审计PASS，人物/场景可辨；但中央光流 AA+68.385、AD+50.074 都为正，D反向未证实，动作目标只可标 PARTIAL。`s1_7199292c` 与 `s3_b784d995` 的 AD 中途收到SIGTERM，原会话exit143，分别在账本预留第78/80次sampling后停止；只有C1+AA通过独立CPU审计，缺失的AD不能作为动作或质量FAIL，更不能算合格配对teacher。中断耗时未写入`gpu_seconds`，结合Judge观察时钟，T2三图合计用时有保守界限[1387.090,1547.401] GPU秒，未自动重试。逐图视频、退出账本、审计及恢复边界见 [T2 Worker报告](submission/experiments/EXP-014_v3_multiscene_teacher/T2_WORKER_REPORT.md)。等待Judge决定是否接受s2作为有限目标、是否值得仅补做两条缺失AD；本轮没有启动AnyFlow、DMD或新训练。

2026-10-11 HKT。**EXP-014 T1首张FM30教师目标已完成并经Judge逐帧与实际KV验收：有限可用、quality PARTIAL。** 固定`s0_43866101`在同一自生成39帧C1后分叉AA/AD到56帧，90sampling+1commit/3decode、532.261GPU秒、峰38.181GiB；50层C1 cache每层index0、6,799,104,000 bytes，旧39RGB逐值不变。新17帧中央Farneback水平光流AA+42.147/AD−27.459，人物/山路可辨，仍有局部软化。并排视频、原片、逐调用账本与CPU审计见 [T1 Worker报告](submission/experiments/EXP-014_v3_multiscene_teacher/T1_WORKER_REPORT.md)。Judge已签T2 marker，剩余三张固定train初图分别在GPU0/1/2用独立账本并行运行；不增加训练、seed或C3。

2026-10-11 HKT。**EXP-014 P1四张 native Single I0 fixture 已完成并经双重CPU审计PASS。** P1实际12 text/4 image encode、0video/denoiser/decode/update，110.214 GPU秒、allocated峰40.578GiB，四fixture哈希与来源详见 [P1 Worker报告](submission/experiments/EXP-014_v3_multiscene_teacher/P1_WORKER_REPORT.md) 和 [Judge独立审计](submission/experiments/EXP-014_v3_multiscene_teacher/judge/P1_INDEPENDENT_AUDIT.json)。P1仅证明原生输入已正确编码，后续视频能力分别在T1/T2评价。

2026-10-11 HKT。**EXP-014/v1 P0 CPU准备已完成并经 Judge 独立验收。** 四张EXP-013确定性train初图的来源/config/代码SHA已冻结；新入口复用EXP-011 native Single I0/full37、strict causal/current-prefix/Global、FM30 C1→AA/AD C2和每图91forward/3decode协议。P0检查 train/validation 排除、PNG/静态文本来源、stop12/17未来row裁剪、50层index0缓存、阶段marker缺失拒绝和每次GPU调用前预算闸门均PASS。`run_g1`与`run_g2`改在同一scene进程顺序执行，C1后会显式释放旧模型并检查残余显存再加载C2。**P0阶段实际0 GPU调用、0编码、0模型推理、0训练**；各后续阶段均需自己的Judge marker，不因夜间8卡空闲提前启动。[P0报告与可执行命令](submission/experiments/EXP-014_v3_multiscene_teacher/P0_REPORT.md) · [实验入口](submission/experiments/EXP-014_v3_multiscene_teacher/README.md)。

2026-10-11 HKT。**EXP-013/v1 CPU 数据审计和下一阶段设计已完成并经 Judge 验收。** 本轮 GPU 调用/编码/denoiser/训练更新均为0。现有 ABot 为24条39帧 clip、6个 episode；train 4 episode/16 clip、validation 2 episode/8 clip，episode 与源视频/annotations SHA 无跨 split 重复。24条视频、PNG、17列动作、源索引、静态 prompt、源哈希均完成逐项核对；24/24 动作可由原 annotations 精确重建。3条邻帧像素探针不明确的 clip 又按原 FFmpeg 配方从源片重建，三者 MP4 SHA 与现存文件完全一致。四训练候选初图按每 episode 最早 `target=A` 冻结，真实录屏标签分别是 A+S+L、A+S、A+S+J、纯A；未来纯 A/D 是合成反事实控制，不能冒充录屏 GT。旧 Dual Anchor 编码不符合 V3 native Single I0，当前39帧 clip 不含 C2 GT。[完整 Worker 报告、manifest、联系表和拟议 GPU 预算](submission/experiments/EXP-013_v3_multiscene_data_plan/WORKER_REPORT.md)。Judge已据此发布EXP-014 P0；09:00 HKT后最多3卡的资源变化已写入实验任务书。

2026-10-11 HKT。**EXP-012/v1 已完成并经 Judge 最终验收：工程/协议 PASS，AF2 step32 未通过两个固定其他场景的联合迁移验证。** 真实34 denoiser forward、4 decode、0训练/编码、354.403 GPU秒；AF自身50层KV、共同FM8首窗/噪声和旧39RGB不变均通过协议审计。但未见稳定联合收益：工业 AD 反向响应明显弱于普通FM8，村落 AA 后半块人物/前景形成持续透明条纹，Judge判结构FAIL。四条56帧匹配对照、原片、审计和账本见 [EXP-012 Worker 报告](submission/experiments/EXP-012_v3_af8_scene_transfer/WORKER_REPORT.md)、[Judge 最终验收](submission/experiments/EXP-012_v3_af8_scene_transfer/judge/FINAL_REVIEW.md)，CPU准备和冻结哈希见 [P0 报告](submission/experiments/EXP-012_v3_af8_scene_transfer/P0_REPORT.md)。这一结果只限制 AF2 step32 在两固定初图的单块迁移，不推广为整个 AnyFlow 方法失败；本轮无后续GPU扩展。

**此前 EXP-011/v1 已完成并经 Judge 最终验收：**P0/P1、四个39帧 G1 首窗以及四配置 AA/AD 56帧 G2 续写，均为两块有限可行、画质 PARTIAL。两张固定 ABot validation 初图上，四配置均通过来源/缓存/旧RGB不变/完整解码的 CPU 审计。推理共228 sampling + 4 commit = 232 forward、12 decode、1781.710 GPU秒；P1另6 text/2 image encode、70.565 GPU秒，无训练。[G2 Worker 完整报告与视频](submission/experiments/EXP-011_v3_scene_transfer/G2_WORKER_REPORT.md) · [Judge 最终验收](submission/experiments/EXP-011_v3_scene_transfer/judge/FINAL_REVIEW.md) · [G1 报告](submission/experiments/EXP-011_v3_scene_transfer/G1_WORKER_REPORT.md) · [P1 报告及失败记录](submission/experiments/EXP-011_v3_scene_transfer/P1_WORKER_REPORT.md)。这只证明两个预定场景、一个seed、两块的短程迁移，村落FM8有更明显的树叶/人体重影，不代表124帧长期泛化或AnyFlow/DMD成功。更早的EXP-010普通FM8+SW-G 158帧、EXP-009 FM4/AF4与EXP-008 DMD结果见下文。正式进展和后续授权以 Judge 维护的 `progress.md`、`next_plan.md` 为准。

## EXP-006/v1 — V3 普通 FM8 从首块开始

完整 Worker 报告、原始小证据、对比视频和逐块指标见 [EXP-006](submission/experiments/EXP-006_v3_fm8_full/README.md)。固定 Original H3 + released action LoRA、Single I0、native timestep、strict chunk causal、current-prefix、Global RoPE、persistent CPU raw KV、12+5+5 latent、seed13/相同输入噪声、native shift2.22。首 39 帧也由本轮 8-step 新生成，后续 AA/AD 两条各到 73 帧。没有新训练、AnyFlow 或 DMD。

GPU0 五个独立阶段累计 40 sampling + 3 clean commit = 43 full forwards、5 VAE、492.856 GPU 秒（0.1369 GPUh），最高 allocated 26.061 GiB。第二块同首窗历史的 A/D 水平光流分别 +0.846 / −0.321，第三块各自历史分别 +0.505 / −0.826。人物与停车场保持可辨，AA 第三块存在持续透明肢体残影，画质为 PARTIAL。Judge 判定 `PASS_FINITE_FEASIBILITY_QUALITY_PARTIAL`，范围限停车场单 seed、73 帧，不代表 124 帧或跨场景质量。并排视频与保存的 30-step 首块有不同 generated history，是跨协议展示，不能作为纯步数速度或质量消融。

## EXP-007/v2 — V3 target-time student AF1-warmup（已完成）

按 [任务书](submission/experiments/EXP-007_v3_anyflow/taskbook_v2.md) 和 Judge [修复重跑授权](submission/experiments/EXP-007_v3_anyflow/AF1_RETRY_AUTHORIZATION.md)，AF1 已在 GPU1 完成一次真实 logical batch4 更新。[完整 Worker 报告](submission/experiments/EXP-007_v3_anyflow/worker_report.md) 记录来源、命令、两个 attempts、模型检查和 checkpoint。Attempt1 在加载模型前因 runner 类型错误退出，0 forward、3.539 秒；attempt2 使用独立目录，22 forward、4 backward、1 update、0 VAE，183.043 秒。两次合计 186.582 秒，allocated 峰值 26.631 GiB。初始化同状态对角输出逐值一致；target-time 与 QKV 均有有限非零梯度和参数变化；更新后 student C1 KV 重新生成且变化。step0/step1 checkpoint、optimizer 和 RNG 已保存并经 CPU 重载/hash 核对。**这是工程/训练可行性核查，不包含视频或 Few-step 质量验收**；后续预算待 Judge 发布。

## EXP-007/v3 — AF2 有限训练与 AF3 视频评估（均已执行完毕）

Judge 已发布 [v3任务书](submission/experiments/EXP-007_v3_anyflow/taskbook_v3.md)，授权从 AF1 step1 配套权重/optimizer/RNG 恢复，新增最多31次更新、527 forward、124 backward、0 VAE、2 GPU小时。独立 [AF2 runner](submission/experiments/EXP-007_v3_anyflow/run_af2.py) 与 [配置](submission/experiments/EXP-007_v3_anyflow/config_v3_af2.json) 的 CPU preflight PASS，源 hash 冻结。GPU1 上的 AF2 已结束，原日志 `H3-World/outputs/EXP-007_af2_run.log` 和逐调用账本 `H3-World/outputs/EXP-007_v3_anyflow_af2/budget.json` 均保留。每次更新前重新构建当前 student 的 C1 clean KV，AA/AD C2 交替；此训练执行结果本身不代表视频能力通过。

历史中间节点：截至 02:33 HKT，AF2 完成累计 step11；step8 稀疏 checkpoint 已由 Judge 独立验收 SHA、配对元数据、optimizer/RNG、AA/AD 顺序及账本。[DMD pilot CPU 准备](submission/experiments/EXP-007_v3_anyflow/dmd_cpu/PILOT_PREP.md) 只建立 3 角色/17 forward 的预算入口和授权门，不构成 DMD GPU 授权。

**AF2 最终结果（03:20 HKT）：**累计 step32 完成，新增31update/527forward/124backward/0VAE，GPU1 4287.861秒、allocated峰值26.767GiB；完整证据见 [AF2 Worker 报告](submission/experiments/EXP-007_v3_anyflow/worker_report_v3_af2.md)。Judge 最终工程审计通过，AF3 已获正式放行。AF3 step32 CPU preflight 核对 checkpoint、来源与 native 8步网格并通过。首次 GPU0 prefill 在 33B 加载前因缺 `benchmark` 导入路径退出，原账本 0forward/0VAE/3.560679秒；[事故记录](submission/experiments/EXP-007_v3_anyflow/AF3_ATTEMPT1_FAILURE.md) 保留。Judge 批准一次修复重跑，沿用原35forward/4VAE/0.35GPUh总账本。

**AF3最终执行结果：**共用C1 prefill及AA/AD第二、第三块全部完成，35forward/4VAE/473.000 GPU秒，allocated峰值约26.34GiB。原片及并排视频均73帧、24 FPS、完整可解码。C2水平光流AA +0.650、AD −0.697；C3各自历史AA +1.006、AD −0.149。主体/停车场基本可辨，但边界姿态跳变、残影保留，AD C3边界灰度MAD为11.275（普通FM8为4.568）。Judge已接受有限可行、质量PARTIAL；没有稳定优于FM8的证据。详见 [AF3 Worker报告与对比视频](submission/experiments/EXP-007_v3_anyflow/worker_report_v3_af3.md)和[Judge审核](submission/experiments/EXP-007_v3_anyflow/judge/AF3_REVIEW.md)。未据此升级正式版本。

## EXP-008/v1 — 真实 H3 DMD 单 cycle 工程可行性

Judge独立放行后，GPU0/2/5各承载teacher/fake/student，一次cycle正常完成：17forward、3backward、3update、0VAE。端到端wall 254.899秒，保守三卡GPU时间0.212416小时；三卡allocated峰值25.070/25.879/27.965GiB。student连续8-map链、fake两次FM warmup及KV重建、teacher/fake同状态评分、student DMD更新全部执行，8个map梯度非零，target-time与QKV参数改变。配套checkpoint、optimizer/RNG及23条逐调用账本已经Worker CPU复核；Judge独立工程审计亦通过。详见[Worker报告](submission/experiments/EXP-008_v3_dmd/worker_report_v1.md)与[Judge审核](submission/experiments/EXP-008_v3_dmd/judge/PILOT_REVIEW.md)。没有新视频或画质收益结论，不自动扩到下一cycle。

v2在独立TRAIN marker放行后已完成从pilot cycle1恢复、再训练7轮到累计cycle8：99forward/14backward/14update/0VAE，端到端1673.641秒、保守三卡1.394701 GPU小时，allocated峰值25.076/25.885/28.102GiB。cycle4/8配套checkpoint已保存，Worker及Judge的恢复、噪声重放、预算和梯度审计通过。[完整Worker报告](submission/experiments/EXP-008_v3_dmd/worker_report_v2.md)记录fake loss在cycle4跃升及后几轮student梯度变小。

随后Judge独立放行最终cycle8的匹配8NFE视频。AA C2新增17帧**持续全画面彩色噪声**，人物与场景消失，故按停止条件取消全部C3。AD C2在停止指令交叉时被SIGINT中断，仅完成5/8步、第6步已计费，未出片；不作AD质量结论，也不自动重跑。累计16forward/1VAE/205.104GPU秒，AA56帧原片和[FM8/AF8/DMD8三列对比](submission/experiments/EXP-008_v3_dmd/dmd_eval/artifacts/failure_comparison/AA_FM8_AF8_DMD8_cycle8_collapse_56.mp4)已归档。DMD8 AA latent终点与初噪声cosine0.949，FM8/AF8约0.100/0.114，支持未有效去噪。详见[Worker负结果报告](submission/experiments/EXP-008_v3_dmd/worker_report_v2_eval.md)与[Judge最终审核](submission/experiments/EXP-008_v3_dmd/judge/FINAL_REVIEW.md)。v2训练工程成立，但cycle8视频能力明确失败，不能作为Stage2成功结果。

## EXP-009/v1 — FM4 与 AF4 的 4-NFE 有限续写（Worker 已执行，待 Judge 最终验收）

按 [EXP-009任务书](submission/experiments/EXP-009_v3_fm4_af4/taskbook_v1.md) 与 C2/C3 两次独立 GPU 放行，在共同 EXP-006 FM8 首 39 帧后，Original H3 普通 FM4 和冻结 AF2 step32 AF4 分别自建 C1 KV，生成 AA/AD 的 C2、C3 到73帧。两模型均使用同一 seed13、初始噪声、Single I0、native shift2.22四步网格；C2同 clean 历史，C3各用自己的生成历史。GPU0顺序运行，零新增训练/重试；账本 **38forward = 32sampling+6clean commit、8VAE、782.290GPU秒（0.217303GPUh）**，allocated峰26.336GiB，C2后CPU raw KV约8.971GiB。所有12条归档原片/四宫格MP4已按帧数、24FPS、分辨率、单调唯一PTS完整解码。

FM4 第二/第三块 AA 水平光流 +0.706/+0.336，AD −0.190/−0.618，人物及停车场仍可辨但残影/边界变化明显，Worker 评估为有限动作可用、画质 PARTIAL。AF4 对应 AA +0.489/+0.268，AD +0.162/−0.007；AD 第三块人物持续分解为大块透明碎片，方向接近消失，AA 第三块也有严重透明重影。AF4 这条冻结 step32 的4NFE路线没有联合收益，不能把训练执行成功写成生成能力成功。这里是共同 FM8 首窗后的少步**续写**，不是整片从零4NFE，也不能用采样步数推断端到端加速。完整原始指标、视频、四宫格与限制见 [Worker最终报告](submission/experiments/EXP-009_v3_fm4_af4/worker_report_final.md)；[AA四宫格73帧](submission/experiments/EXP-009_v3_fm4_af4/artifacts/comparisons/AA_FM4_AF4_with_8NFE_73.mp4) · [AD四宫格73帧](submission/experiments/EXP-009_v3_fm4_af4/artifacts/comparisons/AD_FM4_AF4_with_8NFE_73.mp4)。EXP-009 到此停止，不自动扩2NFE、新训练或DMD。

## EXP-010/v1 — 全程普通FM8 + SW-G 158帧（仅CPU准备）

按最新 [任务书](submission/experiments/EXP-010_v3_fm8_sw158/taskbook_v1.md) 实现了分阶段 runner：复用 EXP-006 AA73 的真实端点、前39/73 RGB 和自身 C1/C2 缓存；先提交已有 C3，再生成 C4–C6 到124帧，经 Judge 看全帧后才允许 A继续/D切换的 C7/C8 到158帧。使用 EXP-005 冻结 long47 fixture、原始 H3 + action LoRA、native 8步调度、Global RoPE、严格因果与最近5祖先 CPU raw KV，零新增训练。C6 commit 预期首次淘汰 C1；C7/C8 逐层核对精确祖先索引。

[CPU预检](submission/experiments/EXP-010_v3_fm8_sw158/cpu_preflight.json)通过36项来源、FM8 AA73 缓存/端点/RGB链、旧前37 latent 条件逐值一致、延长噪声、8步sigma与计划索引检查，**0 GPU调用**。无 marker 的 `commit_c3` GPU入口按预期拒绝。代码、冻结清单和准确限制见 [Worker CPU报告](submission/experiments/EXP-010_v3_fm8_sw158/worker_report_cpu.md)。当前只完成工程准备；没有 EXP-010 视频、显存或动作质量结论，也没有 GPU 放行。

**共享核心阶段更新（到124帧）：** Judge 单独放行后，GPU0顺序完成已有C3 clean commit与C4/C5/C6三个8步块，累计新增28forward（24sampling+4commit）、3VAE、565.082GPU秒（0.156967GPUh），峰值allocated约26.455GiB。C4/C5/C6水平flow依次+0.836/+0.732/+0.660，人物和停车场持续可辨，但腿部透明残影与边界视角变化仍在，Worker评估画质PARTIAL。C6提交第一次真实淘汰C1，50层cache均只留C2–C6，14,164,800,000 bytes；124帧视频完整解码且前73 RGB与EXP-006逐值相同。[Worker核心报告](submission/experiments/EXP-010_v3_fm8_sw158/worker_report_core.md) · [124帧原片](submission/experiments/EXP-010_v3_fm8_sw158/artifacts/core/shared/rollout_124.mp4)。此核心阶段按协议曾暂停，待Judge独立审核后才获分支放行；单独的124帧结果本身不证明晚期动作切换。

**分支最终结果（到158帧）：** Judge 看完共享核心51帧并独立审核真实缓存后单独放行C7/C8。A继续路径C7/C8水平flow +0.439/+0.587；D切换路径 +0.034/−0.858。D首个切换块响应弱且方向含混，第二块才明显反向，不能称立即准确控制。两个158帧视频中的人物与停车场仍可辨，腿部残影与边界视角尖峰持续，Worker评估画质PARTIAL。C7后A/D各自50层raw KV均为精确最近5祖先C3–C7，14,164,800,000 bytes；两分支前124 RGB逐值共用，C8保留各自前141 RGB。全任务新增**62forward（56sampling+6commit）、7VAE、1137.109GPU秒（0.315864GPUh）**，allocated峰约26.596GiB，无失败/重试/训练。完整可追溯证据及视频见 [Worker最终报告](submission/experiments/EXP-010_v3_fm8_sw158/worker_report_final.md) · [A与30步SW-G并排](submission/experiments/EXP-010_v3_fm8_sw158/artifacts/comparisons/A_SWG30_vs_FM8_SWG8_158.mp4) · [D与30步SW-G并排](submission/experiments/EXP-010_v3_fm8_sw158/artifacts/comparisons/D_SWG30_vs_FM8_SWG8_158.mp4)。两边generated history不同，属于能力展示，不能作为单变量步数消融或公平端到端加速比。EXP-010到此停止，不增加C9或调参。

## EXP-011/v1 — 阶段执行历史（P0/P1 当时状态）

下列“尚未启动/待放行”等措辞记录各阶段完成时的历史状态；EXP-011 最新完成状态与全部 G1/G2 证据见本文顶部和 [G2 Worker 报告](submission/experiments/EXP-011_v3_scene_transfer/G2_WORKER_REPORT.md)。

当前唯一执行任务的 [任务书](next_plan.md) 只批准 P0。已核对 industrial `A_1140.png` 与 village `D_1750.png` 两张832×480 validation首帧的 SHA、episode/split、原视频来源以及 `annotation_pilot.json` 的静态场景描述；旧 `encoded/*.pt` 为 RGB dual-anchor 协议，不作为新输入。已实现从原始 PNG 生成 native Single I0/full37 fixture 的 P1 编码入口，以及按 Judge marker 分开的 G1 四个39帧首窗和 G2 各自生成历史的 AA/AD 56帧续写入口。P1 预计合计6次文本编码和2次图像VAE编码，0 denoiser/视频VAE/decode；调用前账本和09:00 HKT/磁盘/显存闸门已编码。

[P0 完整审计、来源及代码清单](submission/experiments/EXP-011_v3_scene_transfer/P0_REPORT.md) 可追溯。CPU import preflight 与合成预算/缓存版本检查通过，0 GPU调用；无 Judge P1 marker 的独立编码入口按预期拒绝。**尚无本任务新 fixture、视频、动作或跨场景质量结论。** P1 新 fixture 必须再经CPU审计，G1视频须完整目视后才能决定是否放行G2；Worker不自行扩大场景、重试或训练。

**P1首次启动失败：**Judge首个P1 marker下GPU0入口在加载模型前因 `sha(__file__)` 字符串路径类型错误退出。没有文本/VAE/DiT调用、没有fixture；原始账本和日志保留，保守计入整次命令4.386366秒。已做最小代码修复、CPU回归测试及重新冻结代码清单，但**未自动重试**。详见[P1失败记录](submission/experiments/EXP-011_v3_scene_transfer/P1_ATTEMPT1_FAILURE.md)；等待Judge对新清单重新发放P1 marker。

**P1第二次启动也在模型编码前失败：**Judge发放attempt2 marker后，轻量loader因独立进程未设置 `DIFFSYNTH_MODEL_BASE_PATH` 而找到空模型列表；0编码、0denoiser、0decode、0 fixture。原日志/结果/账本已归档，保守再计4.280806秒，两次共8.667172/900秒。冻结代码恢复至attempt2的SHA，不自行改协议或再重跑。详见[第二次失败记录](submission/experiments/EXP-011_v3_scene_transfer/P1_ATTEMPT2_FAILURE.md)；待Judge验证命令环境并重新放行。

**P1第三次获批尝试完成：**显式设置冻结 `DIFFSYNTH_MODEL_BASE_PATH` 后，GPU0 对两张原始 PNG 分别编码新 native Single I0/full37 fixture；实际6 text/2 image VAE、0 denoiser/0 decode/0训练。industrial SHA `9f404c91…57107d`、village SHA `eb759021…78775`；两场景 A/D 输入及 own-action/future-row 协议的 CPU 审计 PASS。第三次耗时61.897秒、allocated峰40.576GiB，P1总账（含两次保守失败计时）70.565/900秒。[P1 Worker 报告与证据](submission/experiments/EXP-011_v3_scene_transfer/P1_WORKER_REPORT.md)。**G1/G2仍未启动，也没有新视频质量结论。**
