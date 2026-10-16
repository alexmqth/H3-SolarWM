# 当前项目进展

## 2026-10-17 EXP-028已下发：从CPU准备进入有限实测

Judge批准原Worker实现并自动在空闲GPU执行角色梯度pilot。只测rank-8空间，2片×2sigma×Original-visible/V3、层0/49；14DiT/8backward/2400秒、0updates/视频。发任务时8卡均占用，先CPU实现；限时等空卡，不抢占。首次GPU1进程启动因import run错误退出，0DiT/0backward；已授权一次CPU修复后GPU1恢复，尚无机制结果。准入已改为允许共享但首次实际free≥40GiB；此前C2实测reserved峰29.14GiB，本轮完整可见窗口反向峰未知。[任务书](submission/experiments/EXP-028_action_video_role_probe/taskbook_v1.md)。

## 2026-10-17 CVPR研究假设更新

Judge接受Action–Video交互为首选研究问题，新增优化干扰H3为待验证假设，不宣称双Adapter必要或梯度冲突已成立。既有路由cosine与detached cache源码已核对；原Worker的CPU分解/LoRA映射交付已完成，Judge独立复跑与原结果一致；仅验证小型FP64计算，不代表H3存在角色冲突。10DiT/8backward预算草案尚未授权。无新GPU、训练或视频。[CPU验收](submission/research/cvpr/ROLE_GRADIENT_JUDGE_REVIEW.md)。前轮CVPR证据图谱与诊断提案已收口并push。[分析与约束](submission/research/cvpr/ACTION_VIDEO_ROLE_REVIEW.md) · [当前任务](next_plan.md)。

## 研究优先级：CVPR motivation / novelty（2026-10-16）

用户明确CVPR投稿目标。当前causal/KV与训练入口是研究基线，尚无已验证的新方法/机制贡献。Judge将下一步改为动作条件通路与历史状态冲突的可证伪机制诊断准备，原Worker仅CPU盘点已有ABot证据与一个最小方案；0新GPU。停止围绕约1%MSE反复加训。相关工作系统查重未完成，novelty不作已成立声明。[研究计划](submission/research/cvpr/RESEARCH_PLAN.md) · [当前任务](next_plan.md)。

## 最新验收：2026-10-16 EXP-027两片ABot拟合已完成

64updates、每fit片32次，固定独立噪声probe均值变化：S+D−0.975%、A+S−1.003%；dev constant−0.905%、switch−0.671%。四条GT39+生成17原片与两组对比完整交付，Judge已查看全部生成帧：前后总体接近，没有清楚新增动作/连续性收益。执行accepted，能力PARTIAL，不升级V3，不自动继续当前候选。[视频与成本](submission/report/v3/fm2anyflow/fm_two_clip_fit/README.md) · [Judge审核](submission/experiments/EXP-027_abot_baseline_diagnosis/judge/FINAL_REVIEW.md)。

含首次审计误拒绝，GPU总3876.365秒（64.61分钟），303DiT尝试（302完成+1失败）、64updates、4decode。峰allocated24.449GiB/reserved29.137GiB；CPU KV6.332GiB，账本终点RSS最大68.147GiB非峰值。GPU0释放，本轮无后续GPU任务。仍聚焦ABot，SolarWM参考停止。小预算局部负结果不否定充分训练/架构。[下一步状态](next_plan.md)。

## 以下EXP-027阶段记录为历史状态，以本次最终验收为准

## 当前主线：2026-10-16 EXP-027/v2 ABot两片拟合实验已授权

用户决定当前就在ABot数据上研究。SolarWM官方参考支线停止，不继续申请/探测下载，不作为V3研究前置；保留既有环境与四卡加载证据。EXP-027 CPU盘点已接受：此前每clip仅4次C2更新，选两条常量S+D/A+S片检验局部拟合有新增信息。v2已授权单L40≤6000GPU秒/64updates/336DiT/最多6decode；正常结束必须输出fit片前后4视频，10%只控制可选dev对照。原Worker按CPU/零回归/首步梯度门控后直接执行。[当前任务](next_plan.md)。正式V3冻结，训练结果尚未产生。

**EXP-027恢复补充01：** 首次运行在GT C1 commit被新审计断言误拒绝，0backward/0update；零回归与40初始probe已通过，369.255GPU秒/51DiT尝试保留。Judge已核对冻结协议并授权一次CPU实际helper修复后恢复；新增≤5500秒、累计≤6000秒，累计337DiT含失败1。模型/数据/超参不变，不重复初始评测。

**EXP-027 recovery_01阶段：** 已完成32updates（每fit片16次），Judge独立核对checkpoint SHA、50层形状/finite参数、100项optimizer状态、RNG和前32训练顺序，首步50层commit/forward/backward及冻结梯度检查通过。中期固定probe正在执行，尚不作学习/视频收益结论。继续原64updates预算。

## 最新状态：2026-10-16 EXP-026 plan v6 输入访问阻塞收口

独立 CUDA 12.4 / FlashAttention 2.8.3 构建与 CPU 导入通过；四张 L40 真实完成 33B 基础模型、LoRA、FSDP 和 Stage 2-fix EMA 加载及抽样数值核对。Judge 已独立复核日志、代码、文件 SHA、EMA 抽样值与卡释放。耗时 157.989 秒 / 631.956 GPU 秒，含之前失败累计 737.221 GPU 秒。每卡峰 allocated 22.443 GiB、reserved 23.287 GiB；CPU RSS 采样峰 58.249 GiB。GPU0/1/6/7 已释放。

**这是环境与加载可行性验收：0 forward、0 training updates、0 视频。** 完整推理/训练峰值与官方生成质量未测，真实相机输入访问仍是独立缺口。plan v6已完成每提供方一次实际输入访问：HF401、ModelScope403，0下载/0GPU；本地无真实输入，reader和视频未执行。已收口，不在权限未变化时重复请求。官方参考不是V3主线前置；没有新增GPU任务。[v6审核](submission/experiments/EXP-026_solar_stage2_reference/judge/V6_REVIEW.md)。[Judge审核](submission/experiments/EXP-026_solar_stage2_reference/judge/V5_REVIEW.md) · [下一步](next_plan.md)。

EXP-025 已 accepted（执行与证据）；全50层 QKV rank8、32train、C1/C2 各128updates 完成。总11566.613GPU秒，训练9666.122秒；allocated峰24.449GiB。固定MSE：C1 +0.2331%，C2 −1.3438%。完整视频未见明确新增视觉/动作收益，能力PARTIAL，不升级V3，不自动加rank/epochs/AnyFlow/DMD。[审核](submission/experiments/EXP-025_v3_fm_full_coverage/judge/FINAL_REVIEW.md) · [六组视频与成本](submission/report/v3/fm2anyflow/fm_full_coverage/README.md)。

EXP-023 tail8/C2-only视频收益未证实；EXP-024工程profile通过；EXP-022旧梯度PASS撤回仍有效。正式V3冻结。小预算训练负结果不代表充分训练或架构失败。SolarWM 官方相机验证与 H3-World/V3 语言动作训练保持独立环境及结论。

## 以下为历史阶段记录，以各记录日期和后续更正为准


**EXP-023补充02/EXP-022验收更正（优先于早先工程PASS）：** 已确认022的current_prefix_feedback只覆盖forward，checkpoint backward重算可能切回另一attention拓扑。022梯度协议PASS撤回，Judge needs_revision；数据/前向KV/执行事实保留。023编码可继续，训练必须先完成CPU梯度回归与第1次真实backward重算hook检查，optimizer前通过则自动继续128updates，失败硬停。外层context覆盖forward+backward；无新增GPU预算，补充01继续有效。[修复与门控任务](submission/experiments/EXP-023_v3_fm_small_data/ADDENDUM_02_CHECKPOINT_CONTEXT.md)。

## 最新执行：EXP-023（2026-10-15）

EXP-022工程通路已accepted：真实2updates/413.200GPU秒，KV/velocity及保存重载通过，质量尚未评测。EXP-023已授权32train/128updates及GT/self-generated-history固定前后视频，由原Worker连续执行。[当前任务](next_plan.md)。下方较早的当前执行段均为历史状态。

## 当前执行：EXP-022（2026-10-15）

EXP-021 CPU准备已accepted；已授权原Exp Worker实现真实可微V3普通FM入口、补齐候选RGB，并直接运行最多2updates GPU profile。单GPU≤3600秒，无额外常规marker等待。训练效果尚未测，[当前任务](next_plan.md)。下文EXP-019/020收口时的“无新GPU授权”为历史状态，已由本任务取代。

## 最新状态：2026-10-15 Judge最终验收

EXP-019和EXP-020均已accepted收口，项目本轮GPU进程全部结束，零新增训练。EXP-019固定动作布局与真实模型协议通过，四scene真实GT历史下17帧续写有限可行；画质/动作PARTIAL，工业道路背景变化明显。Original EXP-020全56帧有天空放射伪影，切换控制未确证。正式V3保持冻结。

本轮累计196DiT、2335.217GPU秒（0.6487GPU小时，独立进程wall相加）。EXP-019四scene生成阶段峰allocated24.945–25.294GiB、CPU KV6.332GiB；文本准备峰26.483GiB。Original峰allocated26.049GiB。GT39+生成17与Original全56条件不同，不能据此比较端到端加速。

[本轮视频/说明](submission/report/v3/fm2anyflow/data_preparation/mixed_action_validation/README.md) · [EXP-019最终审核](submission/experiments/EXP-019_v3_mixed_action_validation/judge/FINAL_REVIEW.md) · [EXP-020最终审核](submission/experiments/EXP-020_original_mixed_action_reference/judge/FINAL_REVIEW.md)。

下一项EXP-021仅授权CPU数据/评测集/普通FM训练入口与预算准备，由Exp Worker执行；没有新GPU推理/训练授权。公开SolarWM Stage2-fix标准相机推理尚未执行，独立准备，不能标为已复现。[当前任务](next_plan.md)。


**发布记录（2026-10-15）：EXP-019/020最终证据、视频、Judge审核及EXP-021 CPU任务书已push GitHub main `04a3aeaa47763862bbe996c2e7ef658531590674`；独立ls-remote与本地HEAD一致，发布核对时工作区干净。原Exp Worker已确认并开始EXP-021 CPU准备，无新GPU授权。**

## 既有正式基线与历史阶段记录（下文日期对应当时状态）


**2026-10-13 最新：EXP-018统一V0–V4基准已完成并accepted，7个配置均新生成124帧并导出严格5秒展示。** [完整成本/质量表和视频](submission/report/unified_5s/README.md) · [Judge审核](submission/experiments/EXP-018_unified_5s_benchmark/judge/FINAL_REVIEW.md)。完整wall：V0≈802.6、V1 417.9、V2a 658.4、V2b 2172.0、V3 Baseline1358.3、V3 FM-8NFE505.1、V4 485.5秒；allocated峰26.26–26.45GiB，V3 CPU KV峰16.89GiB。V3 FM-8NFE比同路线30NFE减少约62.8%完整耗时，但ghosting/漂移仍PARTIAL；V3-30有瞬态人体折叠，V4全彩噪FAIL。单次缓存/版本协议差异和V0近似计时均已明示，不宣称纯算法speedup或显存大幅节省。645forward/0训练，含一次条件编码失败共1.782533单卡墙钟GPUh；本轮GPU任务已结束，没有新GPU授权。

EXP-018完整证据、7条独立5秒视频、七版本总览和两条Original/V3双栏已发布GitHub main：`cf2b3a9eb0dd529e4dbe779c5d062ed3df756eb6`。正常push及独立ls-remote核对一致，submission工作区干净。

**命名规范（2026-10-12）：** `AF-train-32updates`是训练累计更新状态；`AF-8NFE` / `FM-8NFE`是每块推理求值配置；EXP-007 / AF2是训练阶段编号。[完整映射与计数说明](submission/report/GLOSSARY.md)。

更新：2026-10-11 09:00 HKT，Judge。夜间监督已完成，所有获批阶段已收口；当前项目0GPU，日间最多3卡，GPU3/4他人任务未操作。

**最新EXP-017固定action布局候选已通过CPU验收，实际模型/视频未测。** EXP-015真实56帧与EXP-016 VAE前缀/重建已验收，四scene前12latent逐值相等。下一阶段建议先实际canonical回归，再真实混合动作的FM-30NFE/FM-8NFE局部对照；仅有预算草案，尚未启动GPU/训练。正式V3/FM-8NFE保持冻结，弱教师AD和当前AF-4NFE/DMD负结果已归档。[当前任务及交接](next_plan.md) · [下一阶段草案](submission/experiments/EXP-017_v3_fixed_action_layout/judge/NEXT_STAGE_DRAFT.md)。

## 报告分类更新（2026-10-11）

V3下按[Sliding Window：Global/Local](submission/report/v3/sliding_windows/README.md)与[FM → AnyFlow](submission/report/v3/fm2anyflow/README.md)分组；[DMD归V4的SGF研究](submission/report/v4/README.md)，当前没有已验收V4生成模型。教师候选及真实数据放入fm2anyflow/data_preparation，非新模型版本。各报告补充名称、目的与本地视频；原始实验编号/证据保持。

## 正式参考和研究结论

**V3 Original Feasibility Baseline继续作为已验收正式参考。** Original H3 + released Action LoRA，Single I0/native timestep/own-action/action feedback/current-prefix feedback，strict causal + persistent raw KV，30步Global RoPE，AA/AD六块124帧，零新训练。瞬态人体形变和连续性PARTIAL，停车场单seed13。候选不得覆盖冻结正式结果。

| 版本/任务 | 已验证结果 | 当前定位与边界 |
| --- | --- | --- |
| V0 Original H3 | 原始双向动作/视觉参考 | 非strict causal |
| V1 Native causal | 分块/KV工程可行 | 动作与视觉未同时通过 |
| V2a RGB Anchor | 124帧基本结构 | 动作失败，20秒后段崩坏，旧路线归档 |
| V2b Same-σ / EXP-001 | 持续AA/DD124帧动作/基本结构可行 | 历史双向重算，无persistent KV，切换PARTIAL |
| V3 Baseline / EXP-002/003 | AA/AD124帧strict causal/KV/动作/结构可行 | 正式参考冻结，quality PARTIAL |
| V3-SW-G / EXP-005 | 共享AA124后A继续/D切换各158帧，真实最近5祖先淘汰 | 优先Global滑窗候选，quality PARTIAL |
| V3-SW-L / EXP-005 | 同窗口158帧，cache正确、动作方向存在 | 场景几何/亮度跳变更明显，无联合收益，当前无训练Local配置归档 |
| V3-FM-8NFE / EXP-006、EXP-010 | 从首窗8步AA/AD73；共享AA124后A继续/D晚切换各158，真实淘汰 | 有限可行PASS，quality PARTIAL；首晚切换块弱，第二块反向响应可辨 |
| V3-AF-8NFE / EXP-007、EXP-012 | 停车场8NFE AA/AD73有限可行；两场景匹配历史迁移无联合收益 | 工业D切换PARTIAL，村落AA结构FAIL；当前AF-train-32updates停止扩展，普通FM-8NFE优先 |
| V4-DMD8（旧名V3-DMD8）/ EXP-008 | 累计8 student cycle训练工程PASS；AA C2全部17新帧彩噪 | 生成FAIL，当前配置归档；AD中断未评估、C3未执行 |
| V3-FM-4NFE / EXP-009 | 共同FM-8NFE首窗后AA/AD73续写可用 | 有限可行PASS，quality PARTIAL；4步首窗和长时未测 |
| V3-AF-4NFE / EXP-009 | AA透明化加重，AD C3持续人物分解、响应近消失 | AD C3生成FAIL，无联合收益，当前4NFE配置归档 |

V2a/V2b为并行视觉/动作修复路线，没有权重继承关系。用户最终保留V3 Baseline正式命名；短暂V2c命名只存历史记录。普通FM减步、AnyFlow finite-map训练、DMD分别评价，旧协议AnyFlow产物不是新V3-AF证据。

[V3家族](submission/report/v3/README.md) · [V2家族](submission/report/v2/README.md) · [浏览器视频入口](submission/report/index.html)

## 关键已验收证据

### SW-G / SW-L

EXP-005：14项CPU正确性、实际模型G0两状态零差异、C6 endpoint与124RGB逐值复现冻结V3通过。C6 commit淘汰C1，C7读取C2–C6，C8读取C3–C7；所有50层核查，历史video KV恒定14,164,800,000 bytes（13.19GiB），旧RGB不变。

Global A C7/C8 flow +.819/+.768，D −1.575/−1.185，动作与切换可辨；保留边界跳变和D C8拖影。Local动作仍存在，但重复场景/亮度变化更多。同Global历史/C8 noisy state下Local velocity相对RMS差异18.25%，不是无损重定位或历史重算等价。

实际281forward/9VAE/0训练/.662707GPUh，allocated峰26.621GiB。只证明158帧约6.58秒和两次淘汰；仅历史video KV有界，prefix、latent/RGB与全前缀VAE成本仍可增长。冻结long47 fixture不等价默认长packed重建。[最终审核](submission/experiments/EXP-005_v3_sliding_window/judge/STAGE2_FINAL_REVIEW.md)。

### 普通FM-8NFE

EXP-004曾仅测30步首窗后的8步续写；EXP-006补上首12 latent也新8步生成，AA/AD自己的历史续写至73帧。C2 flow +.846/−.321，C3 +.505/−.826；人物/场景可用，边界与肢体残影保留。43forward/5VAE/0训练/.136905GPUh，峰26.061GiB。[正式审核](submission/experiments/EXP-006_v3_fm8_full/judge/FINAL_REVIEW.md) · [视频报告](submission/report/v3/fm2anyflow/fm8/README.md)。

### AnyFlow8与DMD

EXP-007 AF1单更新真实模型工程通过；AF2恢复到累计AF-train-32updates，新增31updates、527forward/124backward/1.191072GPUh。AF3固定AF-train-32updates、匹配8NFE继续AA/AD73，有限可行PASS、quality PARTIAL，但没有优于FM-8NFE整体证据；35forward/4VAE/.131389GPUh（含首次导入失败），峰26.336GiB。[AF2审核](submission/experiments/EXP-007_v3_anyflow/judge/AF2_REVIEW.md) · [AF3审核](submission/experiments/EXP-007_v3_anyflow/judge/AF3_REVIEW.md)。

EXP-008从AF-train-32updates（EXP-007 / AF2）启动真实33B三角色DMD，完整8-map反向、角色隔离、自身KV与checkpoint/optimizer/RNG工程通过。最终cycle8 AA C2新17帧全彩噪，生成FAIL；AD完成5 sampling、第6forward中断，无端点/视频，为NOT_EVALUATED；C3未执行。总约1.66409GPUh，其中最终评估16已启动/预记账forward含1中断、1VAE/.056973GPUh。只否定当前有限配置，不推广为所有DMD方法失败，不继续加训练或筛checkpoint。[最终审核](submission/experiments/EXP-008_v3_dmd/judge/FINAL_REVIEW.md) · [负结果视频](submission/report/v4/dmd/README.md)。

### 4NFE对比

EXP-009固定普通FM-4NFE vs AF-train-32updates（EXP-007 / AF2） AF-4NFE，共同FM-8NFE首39，各自KV和C3历史。FM-4NFE C2 AA/AD flow +.706/−.190、C3 +.336/−.618，有限可用/quality PARTIAL。AF-4NFE C2 +.489/+.162、C3 +.268/−.0075；AD C3持续人物分解成大块透明碎片，生成FAIL。Judge看过8段全部136张新增帧和必要原分辨率细节，不仅依赖proxy。当前AF-4NFE配置停止，无2NFE或新增训练；AF-8NFE有限可行结论保持。

28项来源、各自KV、匹配输入、旧RGB不变与完整解码审计通过。38forward/8VAE/0update/782.290028秒=.217302786GPUh，峰26.33646GiB。[最终审核](submission/experiments/EXP-009_v3_fm4_af4/judge/FINAL_REVIEW.md) · [完整视频报告](submission/report/v3/fm2anyflow/fm4_vs_af4/README.md)。

## 最新FM-8NFE滑窗与当前任务

EXP-010完整验收：自有全FM-8NFE AA73续到AA124，再C7/C8 A继续/D晚切换到158；C7 flow A+.439/D+.034，C8 A+.587/D−.858。人物/车库可用，透明腿残影与场景边界变化；D首切换块弱，次块响应更清楚。真实50层KV indices1–5再2–6，恒定14,164,800,000 bytes，旧RGB不改写。新增62forward/7VAE/0训练/1137.108658秒=.315863516GPUh，峰26.59566GiB。仅停车场seed13、158帧6.58秒、冻结long47位置，不宣称无限长度或公平E2E速度。[正式审核](submission/experiments/EXP-010_v3_fm8_sw158/judge/FINAL_REVIEW.md)。

EXP-011最终：新native Single I0的工业/村落FM-30NFE与FM-8NFE四配置首39及各自AA/AD56均有限可行通过、quality PARTIAL。Judge检查156首窗帧+136新增帧及原分辨率细节；四套实际50层C1 KV、原39RGB不变、动作/噪声/native网格与完整视频审计PASS。村落FM-8NFE颗粒/植被和人物边缘重影明显更多；动作分叉仍可辨。232forward/12decode，加6text/2image encode，推理1781.710秒、P1含失败70.565秒，总0.514521GPUh，零训练。两固定初图/seed13/56帧，G2各方法自身历史，非广泛泛化或同KV步数消融。无C3。[最终验收](submission/experiments/EXP-011_v3_scene_transfer/judge/FINAL_REVIEW.md) · [视频](submission/report/v3/fm2anyflow/scene_transfer/README.md)。

## 资源与协作规则

用户授权Judge自主审核/布置下一阶段并持续监督到2026-10-11 09:00 HKT；期间使用实际空闲卡，不人为限制总卡数、不抢占他人进程。GPU3/4目前有他人VLLM进程，不操作。09:00后项目最多3卡；新GPU任务必须有编号、预算、冻结输入与Judge分阶段授权，无需用户重复批准已授权范围。

Judge管理next_plan/progress/archive与Git发布，现有Exp Worker实现/执行/提交报告；不创建新agent。普通画质缺陷不机械卡死，协议真实性必须成立。负结果归档，不为占满GPU追加实验。正式结果均冻结，根目录保持六个管理Markdown文件。

## 最新发布与历史

EXP-012完整负结果已发布`efff5c535912b52b53a507da9d0b9ba4ab2ceb53`，63文件，normal push及独立远端main核对一致。39项归档、28个原始副本、33链接、4比较片解码及内容对位通过。P0代码/工业放行发布`f9bf33406de252e15dbf88a76134ff12c7495b40`。

EXP-011完整迁移验收与8比较片发布`43c56e85bb4e42b0ca38a9ecfb547be09e5fcf43`；EXP-010完整FM-8NFE+SW-G158证据`3db745dec1c1d0a4ebd136ed9237012b45a15614`。完整历史见archive与各实验Judge文件。

## 最新AF-8NFE迁移结论

EXP-012两固定其他场景、共同FM-8NFE C1、native8NFE：工业AA/AD flow+28.14/+11.07，AD反向响应未证实；村落+70.33/−68.63，AD可用但AA后半人物/前景大片分叉透明残影、结构FAIL。两套实际50层cache与匹配条件/旧RGB/视频审计PASS。34forward/4decode/354.403秒=.098445GPUh，0训练；不把工程PASS当能力PASS，不追加C3/训练/调参。[最终审核](submission/experiments/EXP-012_v3_af8_scene_transfer/judge/FINAL_REVIEW.md)。

## EXP-013数据准备验收

24个39帧clip、936帧完整解码/原源二值动作独立核查PASS；4train episodes/16clip，2validation episodes/8clip，源和episode隔离。四初图按最早A固定，不按质量挑选。真实录屏有A+S/L/J组合，未来纯A/D teacher是反事实伪标签；旧Dual Anchor编码不能复用，现有clip没有C2 GT。独立AnyFlow训练仍为草案，新评估优先复用FM-8NFE参考，将新增预算控制为126forward/15decode。0GPU/0训练。[正式验收](submission/experiments/EXP-013_v3_multiscene_data_plan/judge/FINAL_REVIEW.md)。

EXP-013数据审计与EXP-014任务书已发布`a216ca863bdb4de62ab7a9dc3bf467f46fe63449`，24文件；正常push及独立ls-remote确认一致。26个报告链接检查无缺失，归档数据与已独立审计摘要相同。EXP-014仅P0 CPU准备获批，尚无GPU marker。

## EXP-014 P1编码运行（07:34 HKT）

P0独立预检与真实runtime摘要PASS；GPU0 P1编码四train图已启动，12text/4image预算，600GPU秒，0denoiser/0训练。T1/T2未授权。代码清单SHA `5afe075f4db7e5b6b107ffabbfcd1feb408dbccd173649b6548c74f876173331`。[P0验收](submission/experiments/EXP-014_v3_multiscene_teacher/judge/P0_REVIEW_AND_P1_RELEASE.md)。

## EXP-014 P1验收/T1已启动（07:37 HKT）

四native fixture独立CPU验收PASS，12text/4image encode实际110.213939GPU秒，峰40.577766GiB，无重试。首图s0_43866101的FM-30NFE C1+AA/AD C2已在GPU0运行，91forward/3decode/1350秒上限；余三图T2未放行，新训练0。[P1审核](submission/experiments/EXP-014_v3_multiscene_teacher/judge/P1_REVIEW_AND_T1_RELEASE.md)。P0代码与编码授权已发布`2244dbcfb84feba79dd8c18846febf0e6c4f8272`，远端核对一致。

EXP-014 P1归档与首图授权已发布`928286da635119ef61e7411af1ddb4c82be70321`，12文件、远端main摘要核对一致；5份P1原始副本逐字节相等，6个报告链接有效。

## EXP-014首图T1验收/余三图运行（07:49 HKT）

首图39+17+17全帧及原分辨率检查通过，AA/AD flow辅助+42.147/−27.459，人物/场景可辨，teacher有限可用/quality PARTIAL。实际50层index0、同C1/noise/Global位置、旧39RGB不变与视频核查PASS。91forward/3decode/532.260704GPU秒，峰38.181200GiB。余三固定图T2已获marker并在GPU0/1/2独立并行，逐图验收；0训练。[T1审核](submission/experiments/EXP-014_v3_multiscene_teacher/judge/T1_REVIEW_AND_T2_RELEASE.md)。

## EXP-014 T2中断与v2准备

s2工业道路AA/AD中央ROI flow+68.385/+50.074，背景主要同向，D反转未证实；结构可用，不能把AD当正确动作监督。s1/s3各在AD完成17/19步后进程消失，预记账采样78/80、各1commit/2decode，无AD端点/视频，不作生成FAIL。可见cgroup OOM计数均0，具体退出原因待原会话证据，不假定OOM。已冻结原账本并记录各515.555GPU秒的保守上界。v2仅CPU恢复准备：复用已保存C1/KV/noise，最多新增两AD共60F/2decode，单scene一次串行恢复，GPU另发marker；全任务402F/12decode、teacher5400秒上限。

EXP-014/v2恢复独立CPU核查PASS，代码末次变化仅预检幂等/增加验证，逆补丁精确还原原SHA，生成逻辑不变。RECOVERY_APPROVED绑定b294c0d1…新清单，仅s1再s3串行；新增60F/2decode/1200GPU秒，0commit/0训练。原T2中断GPU耗时按上下界报告，不以未关闭账本的0作为免费。

## EXP-014最终收口（08:25 HKT）

两条缺失AD已顺序恢复且独立协议审计PASS，292张唯一帧已完整观察。四图结构有限可用，动作标签逐scene保留上述限制。402denoiser预记账、至少400确认完成（含4commit），两个中断末调用不确定；12decode、12text/4image encode，总[0.652293,0.696824]GPU小时，0训练。恢复20原始副本及2比较片完整解码/对位通过。GPU任务均已结束。新训练草案不构成执行授权。

EXP-014最终证据与EXP-015 CPU任务书已发布`2991fec4649de08a33a5523058e6a6a0b6a27863`，58文件；normal push和独立ls-remote一致。4份展示视频与源SHA相同、56个报告链接有效；原始日志和历史快照保留原空白，不为格式修改证据。

## EXP-015真实56帧数据验收（08:39 HKT）

固定四train源224帧/原11keys/native17span与C1动作未来隔离PASS，8,412,751bytes，0GPU/训练。s3尾部实际A→静止→W，无D覆盖；FFmpeg线程超原约定如实披露，后续4CPU affinity。变长未来text会影响原生packed Global位置的CPU复现已成立，不将真实数据直接写成训练ready。下一EXP-016只做VAE前缀和roundtrip验证。[审核](submission/experiments/EXP-015_v3_real56_data/judge/FINAL_REVIEW.md)。

EXP-015全部CPU证据与EXP-016任务书已发布`e9a703db64817deb7e7360dd58fbec9981b4c08e`，40文件、32报告链接，独立远端main核对一致。

## EXP-016 P1 GPU放行（08:44 HKT）

P0真实runtime/source/入口检查PASS，P1_APPROVED已签发，GPU0单卡一次，4image/8video encode/4decode、600GPU秒，0text/DiT/训练。08:50前启动、08:57停新调用。数值前缀与重建质量待实际结果。

## EXP-016 VAE最终验收（08:48 HKT）

四段真实56/同源39编码前12latent逐值相等，12个实物tensor/来源/finite独立核对PASS；224重建帧及关键全分辨率对比可用，普通有损细节。4image/8video encode/4decode、77.721815GPU秒=.021589393GPUh、峰7.503959GiB、0训练，进程退出。完整action packed未来隔离仍待独立解决。下一EXP-017仅CPU固定布局候选，不修改冻结V3。[审核](submission/experiments/EXP-016_v3_real56_vae/judge/FINAL_REVIEW.md)。

EXP-016实物编码/视频证据及EXP-017 CPU任务书已发布`866325c26c0591f47499796f75e7dbff861ace0d`，56文件、14原始副本/35报告链接检查PASS，独立远端main一致。

## EXP-017 CPU最终验收（08:57 HKT）

四scene固定布局候选CPU通过：canonical A/D逐字段复现、未来短/长内容与长度24次独立干预可见布局不变、历史三轴语义位置不变。初版极短future提前拒绝已修复并保留旧attempt证据。0GPU/forward/训练；真实text encoder/DiT/KV数值和视频仍NOT_TESTED，不升级正式模型。下一阶段为未执行草案，canonical双入口若均重算应计8forward而非4，后续14text/156forward/8decode另定预算。[最终审核](submission/experiments/EXP-017_v3_fixed_action_layout/judge/FINAL_REVIEW.md)。

最终EXP-017与09:00交接已发布`0823a8d25f38cc6ac763991aa199a264cd7ae919`；normal push和独立ls-remote一致，submission工作区干净。Worker会话idle并确认9点结束，本项目0GPU。


2026-10-11 报告分类整理已发布GitHub main：`1119f165d6aae5cd62d4533b270875b65d9afd96`。150文件，V3 Sliding Window / FM → AnyFlow、V4 SGF/DMD、术语目的与视频入口已同步；normal push成功，独立ls-remote核对与本地HEAD一致，submission工作区干净。


## 2026-10-12 qma迁移与空间清理

用户明确授权迁移项目并清理旧实验。当前实体目录`/home/qma/work/GWM`，模型在`models/`，qma拥有全部保留文件。已清理可重建KV、早期optimizer状态及重复环境，保留正式证据/权重和后续输入；已删除文件不再声称实物仍可检查。详见[迁移记录](submission/archive/qma_migration_20261012/README.md)。本次无GPU/训练，新研究任务仍需独立预算与Judge放行。

迁移最终检查通过：项目+模型从约808.50 GiB降至275.52 GiB，净减少532.98 GiB；旧work模型目录已删，旧home项目仅余4KiB空目录（父目录权限限制）。qma实体目录/全部文件归属、最新四scene CPU布局、冻结来源SHA、报告链接及视频检查通过；无新GPU/训练。

迁移与清理记录已发布GitHub main：`fa479630851497a878b8b178cd2aedc5d76dc669`，远端独立核对一致，submission工作区干净。项目后续工作目录统一为`/home/qma/work/GWM`。


## 2026-10-12 汇报补全：Original与causal直接对比

[集中报告](submission/report/original_vs_causal/README.md)已汇总推理耗时、峰值显存、CPU KV、质量与视觉连续性，并在报告首页直接展示Original/V3 Baseline及Original/SW-G + FM-8NFE两个对比入口。新增片仅CPU拼接已有持续A视频的共同124帧，右侧完整158帧另列，淘汰后C7/C8不在124帧节选内。新片124帧/24fps完整解码、PTS、SHA和抽样面板/帧对齐检查通过；没有新推理、训练或模型能力验收。Original完整E2E与causal增量/多分支统计不能混算加速；缺少匹配A→D参考及同口径从零E2E/统一质量评测，均已写明。

2026-10-15交接确认：已向固定Worker会话发送EXP-022立即执行指令；Worker已回复“EXP-022/v1已开工”，CPU准备后按授权直接启动GPU profile。EXP-021证据与EXP-022授权已push main `ea5a0c86bc0f6558f024886b4a66bdf95e80824c`，独立远端SHA一致。

发布与交接：EXP-022最终证据及EXP-023任务/授权已push main `7f9f327e17357c222523687cb15b3fc5eb014e02`，独立ls-remote与HEAD一致；固定Worker已确认EXP-023开工，按授权连续执行。

**EXP-023补充01（优先于上文旧预算）：** 已确认旧4train latent来自重编码MP4。本轮34clip统一source RGB重新编码I0/full17，原4条不再复用旧latent。VAE encode上限70、编码4200GPU秒、总13800GPU秒，其余128updates/评测/资源不变；已批准Worker直接执行。[补充任务书](submission/experiments/EXP-023_v3_fm_small_data/ADDENDUM_01_SOURCE_RGB.md)。

EXP-023补充01已通过会话消息发给原Worker，要求统一34clip source RGB输入；已push main `ba47b938cfbc1c09c690c3a43fdc9659a8035921`，独立ls-remote与HEAD一致。

EXP-022梯度验收更正与EXP-023补充02已push `53d21688a646ae20972ae091dd75f0edd3ba78c9`，远端SHA独立核对一致；修复执行消息已发给固定Worker，编码继续，optimizer前按新门控验证。

**EXP-023补充03：** 34clip编码完成，checkpoint0回归PASS；首次pre-eval在ffmpeg写片时失败（13DiT/1decode/301.064GPU秒），尚0训练。已授权显式ffmpeg路径+CPU codec检查及一次最小恢复，失败成本累计，decode上限13，其余440DiT/13800秒/128updates等不变。之后直接继续原任务。[恢复任务书](submission/experiments/EXP-023_v3_fm_small_data/ADDENDUM_03_VIDEO_IO_RECOVERY.md)。

EXP-023媒体恢复补充03已push `61be4c5d82200c7e43baa860f2b0111de484efd4`，独立远端SHA一致；已消息通知原Worker立即按授权恢复，pre/post各6片，失败费用累计，模型总预算不变。

**EXP-023补充04：** 第二次媒体故障（BF16.numpy identity）丢失首片，新增失败9DiT/1decode/81.937秒。Worker与Judge完整CPU保存/解码/故障恢复测试已通过，已授权attempt02最后一次首片恢复；总DiT449/decode14，13800GPU秒和128updates等不变。两次失败全部计入，之后媒体错误只从已保存RGB CPU恢复。[补充04](submission/experiments/EXP-023_v3_fm_small_data/ADDENDUM_04_BFLOAT16_RECOVERY.md)。

补充04/CPU媒体独立复核已push `d74d39d250ee102955d08e7fd1894429835ab9b1`，远端SHA一致；已消息通知固定Worker按最终首片恢复授权立即执行。

**EXP-023补充05：** 6条pre-eval已完成。训练首步hook安装早于context校验导致停止，0backward/0update，失败26.962秒/2DiT reserve保留。生产helper已由Worker和Judge独立CPU验证，现授权只恢复train→post，不重做数据/pre。总预算保持449DiT/14decode/128updates/13800秒，真实首步梯度门控通过后自动继续。[补充05](submission/experiments/EXP-023_v3_fm_small_data/ADDENDUM_05_TRAIN_HOOK_RECOVERY.md)。

EXP-023补充05/生产hook CPU复核已push `fd664dec23401711529c508364a10c1a0437fba4`，远端SHA一致；已向固定Worker发送立即恢复train→post指令，不重复已完成pre。

**EXP-023真实首步梯度门控已由Judge读证据复核PASS：** forward50层、backward实际重算49→42均使用current_prefix，历史cache签名不变，base/released Action LoRA无梯度，新adapter梯度finite且非零。修复后训练已开始，核查时完成3updates，继续既定128。此为EXP-023修复后证据，不追认旧EXP-022 step02梯度正确，也不代表生成质量已通过。

EXP-023 64updates阶段已独立核对checkpoint SHA、32clip各两遍、记录顺序/有限参数/optimizer与RNG字段；训练继续到128，质量待post。阶段证据已push `a8c6a7e28a839410b5be3ec60e12068968624bc4`，独立远端SHA一致，权重仅本地。

发布确认：EXP-023最终证据、六组三列展示视频、Judge审核及EXP-024任务书已push GitHub main `a1ec25cb29fb606b9ee3f594a0d881c399b52e06`，独立ls-remote与本地HEAD一致。EXP-024 Worker仍在执行，新增工作文件待完成后单独审核发布。

EXP-024/v2 GPU授权与Judge门控审核已push `ec3618b9b6dcac833da55673d5b1e67757477f75`，远端SHA核对一致；已向固定Worker发送立即运行指令。

运行确认：EXP-024/v2已在GPU0实际启动，Python PID3061434，日志已加载H3及released Action LoRA，nvidia-smi可见本项目进程。CPU代码与profile入口已push b5e0c1e，实际GPU结果仍待Worker完成/交回Judge。

交接与发布确认：固定Worker回复EXP-025/v1现在开工，按CPU/回归门控通过后自动连续执行。EXP-024最终证据与EXP-025授权已push `48f2bc93af04af6fc285367cc79efad632dbe33e`，独立远端main与HEAD一致。

EXP-025启动实物核对：GPU0/PID3409453；102个输入tensor SHA及实现/依赖SHA一致；零adapter 6forward回归velocity逐值相等、完整KV摘要相同，进入前评测。训练与效果尚未验收。

EXP-025首个真实C1/C2梯度门控已Judge复核：两块50层梯度finite非零、路由完整、冻结参数无梯度；前评测6片实物SHA与16probes齐全。256updates训练正在GPU0继续，质量待后评测。

EXP-025 64updates checkpoint已Judge独立复核SHA、50层有限参数/100项optimizer状态、RNG与训练顺序；32clip每block各一遍。训练继续256，阶段证据已push aa681bb，远端SHA核对一致。

EXP-025 128updates checkpoint已Judge独立复核SHA、50层有限参数/100项optimizer状态、RNG与顺序；32clip每block各两遍。核查时131updates，按已测均速剩余训练约4211秒，仅估算，预算内继续。阶段证据已push 3d0966c，独立远端核对一致，效果仍待post。

EXP-025全部256updates已完成并Judge独立核对最终checkpoint SHA/50层有限参数/optimizer与RNG/256条训练顺序；32clip每block各4遍。训练9666.122秒（161.10分钟），低于11000秒阶段上限；post评测正在进行，质量待审。阶段证据已push 8768a3f，独立远端核对一致。

EXP-026交接：已通过消息工具通知固定Worker，他已确认开工（CPU/输入准备）；当前0GPU，尚未声称官方视频推理完成。

GitHub发布：EXP-025最终证据/六组视频/Judge审核及EXP-026任务授权已push f4238eb8717bd90388181ef5fff9c77721c921b4，远端main独立核对一致。Worker已实际读取官方配置并检索loader代码，正在执行EXP-026。

EXP-026最新阻塞：Worker报告官方HF实际下载401/GatedRepoError，ModelScope主库与H3 latent库403，本地无对应token。样本下载阻塞；仍执行不依赖数据的CPU/config/loader与显存布局审计，0GPU。完整输入可读性未通过，等待具体访问需求报告。

EXP-026/v2执行中更正：Worker发现v1只通过通用load_config，H3 backend要求SP4，SP8候选未通过专用合同；默认FSDP按逻辑DP组分片，SP8/DP1仍不能分片模型。优先保留官方SP4并检查既有跨物理卡分片支持；必要的2样本仅可提案，不新增GPU授权。超出最小加载修复则停止，不重写分布式路线。

2026-10-16 Judge更正：固定158帧官方backend允许4卡SP4/DP1单样本，独立config验证PASS。八卡仅是当前v2补丁/提案限制，不是模型最低要求；后续优先考虑显式物理四卡分片。真实加载/生成尚未测，无新GPU授权。详见submission/experiments/EXP-026_solar_stage2_reference/judge/FOUR_GPU_CORRECTION.md。

EXP-026/v3交接确认：原Worker已消息回复开工，实际读取任务书/授权并查询GPU0/1/6/7空闲及内存，正在memory_load_v3独立实现。CPU门控通过后自动执行一次load-only；当前尚未收到GPU启动PID。授权已push 0cda53c，远端核对一致。

EXP-026 plan v5交接确认：原Worker已消息回复开工，实际读取任务书/授权并检查资源，正在盘点NVIDIA runtime headers、用户目录CUDA12.4安装路线及官方源码。CPU门控通过后自动执行一次四卡load-only，当前尚未启动GPU。任务授权已push 7b202a8c8db704b1880f6c5d47356bb9ca2fc172，远端核对一致。
