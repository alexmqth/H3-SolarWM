# Worker 当前执行报告

2026-10-11 HKT。**EXP-010/v1 已按两次独立 GPU 放行完成普通 FM8 + SW-G 的 A/D 158帧实验，Judge 独立审核接受有限可行性；画质 PARTIAL。** 单停车场/seed13下人物与场景仍可辨，晚期 D 切换首块弱、第二块方向才清楚。EXP-009 普通 FM4 有限可用、AF4 AD第三块失败；EXP-008 DMD cycle8 视频失败。正式进展及后续授权以 Judge 维护的 `progress.md`、`next_plan.md` 为准。

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
