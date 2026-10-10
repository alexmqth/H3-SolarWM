# 项目历史归档

创建时间：2026-10-10T02:14:35+08:00。维护者：Judge。

本文件保存历史证据入口、接管记录及原文快照。当前正式研究状态仍以 [progress.md](progress.md) 为准，可执行任务以 [next_plan.md](next_plan.md) 为准。历史文字中的“当前”“下一步”和旧版编号仅描述当时状态，不构成今天的执行授权。

## 2026-10-10：根目录清理与 EXP-001 发布

用户明确要求清理冗余根目录Markdown，并将V2b长视频与Original对比列为下一优先级。Judge发布 [EXP-001 / plan_version 1](next_plan.md)：先验证四路径第三／第四块至90帧，门槛通过后续至124帧；任务执行not_started，验收pending。本次仅整理文档和发布任务，零GPU前向/训练。

### 旧任务关闭与验收边界

旧`Next Plan v5`行政关闭：Judge decision=`stop`，acceptance=`cancelled`，reason=`superseded by user-directed EXP-001`。这是授权入口替换，不是把旧E2或AnyFlow的能力失败改为通过。旧计划、完整历史进展已在本文件SNAPSHOT区原文保存；原来根目录没有对应Worker完成报告，后补空模板也已在ROOT_CLEANUP区保存，不伪造验收凭据。

### 文件处置

根目录保留README、guideline、progress、next_plan、report、archive共六份Markdown。README改为导航与复现入口；progress由4,197行收敛为八项当前摘要，旧历史及疑似其他项目段落仅留档；report绑定EXP-001。下列五份过期/重复文档经逐字节归档核验后从根目录删除，其完整原文在本文件`root_cleanup`节。

| 原根目录文件 | 字节数 | 原文 SHA-256 |
| --- | ---: | --- |
| `SCHEDULED_SAMPLING_EXPERIMENTS.md` | 6838 | `b428ea00cc0f193cba2343faf67448b3ff7c1eac819d8cf88f84be5c869a1ae1` |
| `STATUS_REPORT.md` | 6418 | `0d22314d8bdb3167a6164db7c15dbfd548ab2e4cb05138a3a8d0ceb6afe75448` |
| `TASK_COMPLETION.md` | 6632 | `49d9d032db6c6b2c342722beab1be260469e8e8b42dc6678368d41a13f996f4a` |
| `TASK_SUMMARY.md` | 4110 | `2325b26b09fa54c0b85647994d7c2f8cc367c0573beb02393f085b9ad44fba91` |
| `WORKSPACE.md` | 2024 | `167bbd457e0043fa004c40f61c99ffed281e62354c253949b9e8b4c126bea561` |

检查了这些文件在现行Markdown及非Markdown代码/脚本中的引用；发现的引用位于被一同归档的历史材料。历史快照保留当时路径，不将旧链接冒充现行入口。旧README全文和有用的环境/复现说明仍可追溯；当前README指向有效复现文档。

### 当前研究判断

优先把V2b局部正控扩成可比较的多窗口证据，以确定是否值得推进更长时程或严格因果/KV迁移。124帧约5.17秒是本轮默认目标，不宣称已经解决10/20秒长期稳定。任务保持完整历史重算、原生条件和固定音频协议；与Original的联合音视频差异明确列为混杂。预算上限为612次denoiser、最多并发2GPU、8 GPU-hours，零训练。

## 2026-10-10：Judge 接管与文件补齐

### 范围与验收边界

用户要求读取 guideline、承担 Judge、补齐缺失 Markdown 并了解项目。已读取协作规则、当前任务书、进展最新章节、主线版本定义、V2b 局部实验报告和固定状态 Action Routing/KV 审计报告，并抽查原始 JSON/CSV。

本次是文档接管与证据盘点：新增 report.md 模板和本归档。没有进行模型前向、训练或新视频生成，也没有独立逐帧复审历史视频。因此下表是历史证据的范围梳理，不是本轮对模型能力的新验收。历史任务未具备完整 EXP 编号/plan_version/Judge 验收链路，不追溯虚构编号或 accepted 状态。

### 项目研究状态（接管时快照）

目标是从 Original H3-World 走向动作、视觉和高效推理同时成立的因果世界模型，再推进 AnyFlow 与 On-policy DMD。

| 路线 | 现有证据 | 关键限制 |
| --- | --- | --- |
| V0 Original | 双向 H3 + released action LoRA，已有动作/视觉参考 | 作为参考模型；生成结果不是 GT |
| V1 Native Chunk-Causal | 分块因果、persistent video KV、生成历史 rollout 工程可行 | 动作与视觉未同时通过 |
| V2a RGB-Anchor Causal | 124 帧人物结构相对稳定；8 steps/chunk、RGB 条件、visual adapter、clean-commit KV | A/D flow −0.7841/−1.0075，动作失败；20 秒长片后段严重退化 |
| V2b Same-σ Local Bidir | 单停车场、seed13、12+5 latent；第二块两 history 的 A/D flow 为 +2.1464/−1.7519、+2.6076/−1.4143，历史报告记录局部结构与动作正结果 | 仅两块 56 帧；30 steps/chunk、逐 sigma 双向重算；无 persistent hidden KV；第三/四块及124帧未验证 |
| V3 Efficient Causal | 计划统一严格因果、persistent KV、动作与视觉 | 尚无已验收模型、checkpoint 或视频 |
| Branch C AnyFlow / DMD | 存在实现、数值检查和短预算探索 | preliminary exploration；不能表述为少步生成能力或完整 DMD 已通过 |

V2a/V2b 为并行修复路线，V2b 从 Original 权重出发；两个方案的能力不能相加为同一已成功模型。

最新机制证据：[固定状态审计](submission/reports/stage1_anyflow/02_causal_diagnostics/action_routing_kv_audit/execution_20261010/README.md)。P0 原始 CSV 的 run_04 有六个状态；受控 BF16 数值执行、canonical SDPA/GEMM 下历史 K/V、RoPE 与 velocity 误差均为零，cache hash 不变。该结论只覆盖匹配图/时间条件下的缓存操作，不代表默认部署已等价。P1 中无 persistent KV 的 R2 对 Original 动作差分 cosine 均值约 0.0570，说明失配在缓存前已出现；这不直接判断视频动作或画质。

### 历史证据索引（非新增正式验收）

| 历史事项 | 证据入口 | 本次处理 |
| --- | --- | --- |
| V0/V1/V2a/V2b/V3 定义 | [主线目录](submission/mainline/README.md) | 读取与 guideline 核对 |
| C12→5 自身历史局部正控 | [实验11](submission/experiments/11_causal_12_then5_selfhistory/README.md)、[latent manifest](submission/experiments/11_causal_12_then5_selfhistory/manifest.json) | 读取；保留56帧、单场景/seed范围 |
| Action Routing/KV P0/P1/P2 | [完整报告](submission/reports/stage1_anyflow/02_causal_diagnostics/action_routing_kv_audit/execution_20261010/README.md)、[P0 CSV](submission/reports/stage1_anyflow/02_causal_diagnostics/action_routing_kv_audit/execution_20261010/P0_metrics.csv)、[P1 CSV](submission/reports/stage1_anyflow/02_causal_diagnostics/action_routing_kv_audit/execution_20261010/P1_metrics.csv) | 抽查原始指标；未复跑模型 |
| E2 冻结与提交复现 | [冻结收据](submission/reports/stage1_anyflow/01_real_video/real_transition_windows/FREEZE.json)、[复现报告](submission/reports/final_acceptance/README.md) | 保留旧计划指向；未重新验收 |
| V2a/V2b 汇报整理 | [汇报入口](submission/report/README.md)、[整理收据](submission/archive/taxonomy_v2a_v2b_20261010/validation.json) | 收据记载 PASS、22条画廊视频、79份MP4、1938处链接检查；本次仅读取收据 |

### 工作区与版本控制

- `/home/qma/work/GWM` 是 `/home/lpeng/code/mq_PubDataset/GWM` 的符号链接。
- GWM 根目录不是 Git 仓库；根目录管理文档不会自动被 submission 的 Git 覆盖。
- submission 当前 HEAD 为 `b3a8bcfde0d1aba73e1def34bc6760d583279c35`，本次检查工作区干净；本地 origin/main 引用一致。本次没有联网核验远端实时状态。
- H3-World 当前 HEAD 为 `f0c7be2`，README.md、code/abot/infer.py 已有修改，causal 代码、测试、outputs 等有未跟踪内容。复现须引用冻结源码/hash，不能只引用该上游 commit。
- SolarWM 当前 HEAD 为 `ce1da4e`，本次检查工作区干净。

### 交接问题与待决策事项

1. 根目录 report.md、archive.md 原先不存在，现补齐；guideline.md、progress.md、next_plan.md、README.md 已存在。
2. progress.md 为 4,197 行历史流水账，不符合 guideline 的精简摘要要求。原文在下方完整保留，后续正式整编应从中提取八项当前状态，历史与负结果留档。
3. progress.md 原第3937–3992行附近含“冗余感知探针、解码器、8页论文、r68”等疑似其他项目内容；本次保留原字节，不纳入 H3-World 研究结论，来源待核实。
4. next_plan.md 是2026-10-09的“v5 冻结研究、面试交付”，晚于该计划的 C/B 局部实验、KV审计、目录整理已有历史记录，但未并入任务书。旧计划缺 Task ID、完整资源预算及 Judge 验收栏；当前不能据此启动新研究任务。
5. 旧计划中 `submission/EXPERIMENT_REPORT.md` 和 `submission/PROJECT_PROGRESS.md` 已失效。现有对应文件为 [实验报告](submission/docs/EXPERIMENT_REPORT.md) 与 [历史进展](submission/archive/historical_docs/PROJECT_PROGRESS.md)。
6. 下次发布任务前，先处理旧任务的验收归档和有效任务书。研究决策应明确：先扩展 V2b 自生成历史的局部能力范围，还是在已确认的 prefix/历史时间契约下开展严格因果适配；本次未授权其中任何 GPU 实验。

### 原文快照信息

下方按原字节嵌入接管时的两份文件，便于之后精简正式进展而保全历史。快照中的旧版本命名、失效链接和未验收表述均保留；其存在不表示 Judge 同意该结论。后续正式 EXP 归档须追加完整任务书、Worker 报告、验收结论和证据路径。

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| progress.md | 375823 | `16716a40b1f0952d1eca11ae2e01a8a9ba5a6fd92842f83d01491b0bb17d4b3d` |
| next_plan.md | 2204 | `4658a775766336859c65e645e33b0ceaae28d7619f22a2ae158ddbfcab1dedf6` |

<!-- BEGIN SNAPSHOT next_plan.md -->
# Next Plan v5：冻结研究实验，完成面试交付

2026-10-09按用户指定的四项优先级收尾。E2局部动作＋结构验收仍为No-Go；不因工程复现通过而改变该结论。

| 优先级 | 任务 | 当前结果 |
|---|---|---|
| 1 | 收齐E2并冻结 | FM-only/FM+action各4更新，A-history、D-history和两条真实GT-history全部收齐；无一致改善，记录负结果。16份关键证据哈希及14个历史进程身份已核查，见[冻结收据](submission/reports/stage1_anyflow/01_real_video/real_transition_windows/FREEZE.json) |
| 2 | 更新提交包和会议材料 | INTERVIEW_ANSWER、EXPERIMENT_REPORT、REPRODUCE及meeting已统一；明确旧主片checkpoint与AnyFlow/E2/DMD准备的界限 |
| 3 | 最终可复现性验收 | 新venv＋GitHub新DiffSynth源码；15项KV/局部因果测试通过；随机小H3训练smoke通过；真实33B旧adapter生成39f并完整解码，测量和哈希齐全 |
| 4 | 5分钟技术答辩 | 六页提纲和含播放时间讲稿完成；Original vs causal开场，完整20秒失败片和E2负结果作诊断证据 |

本轮没有把E2从4扩到16，没有重启AnyFlow或Stage2。验收训练仅为随机小H3固定合成batch的8次smoke更新，不是研究模型训练。当前GPU共享负载，不新增不公平的124f性能排名，保留历史单次表并明确限制。

入口：[面试回答](submission/INTERVIEW_ANSWER.md) · [实验报告](submission/EXPERIMENT_REPORT.md) · [复现](submission/REPRODUCE.md) · [验收证据](submission/reports/final_acceptance/README.md) · [会议讲稿](submission/meeting/MEETING_SCRIPT.md)。

## 后续研究，仅保留建议，不自动执行

可信局部causal30step → AnyFlow4/8step → on-policy Stage2。先验证局部动作信息流、action-dependent prefix重算、监督覆盖及动作后果时序；不要求先解决全部长时漂移，但也不以Stage2解释全部GT/reference历史下的局部结构问题。

当前交付状态：工程实现与复现可行；动作保真、稳定长视频和端到端加速没有同时成立。完整历史计划和实验过程保留在[PROJECT_PROGRESS](submission/PROJECT_PROGRESS.md)与各原始报告中。

<!-- END SNAPSHOT next_plan.md -->

<!-- BEGIN SNAPSHOT progress.md -->
## 2026-10-10：V2a / V2b整理包已推送GitHub

已将mainline/branches/report/archive整理、V2a/V2b并列编号、22条当前视频画廊、16条新编号编码及验收材料提交并推送至alexmqth/H3-SolarWM的main分支。提交：b3a8bcfde0d1aba73e1def34bc6760d583279c35（Reorganize research evidence into parallel V2a/V2b reports and video gallery）。

推送前fetch确认远端仍为1a93713，无新增提交或合并冲突；推送后git ls-remote确认远端main与本地HEAD一致，ahead/behind=0/0，submission工作区干净。原审计中的未提交/未推送字段是发布前快照。未运行任何模型训练或推理。

## 2026-10-10：正式改为V2a / V2b并列路线，未来V3统一；视频与文件验收完成

按最新纠正，将RGB联合视觉修复改为V2a、Same-σ局部双向重算改为V2b，未来高效严格因果改为V3 Planned。V2a与V2b针对同一生成历史/条件问题，没有顺承或checkpoint继承关系；V2b从Original H3恢复Single I0/native time，不使用V2a visual adapter。

当前目录为submission/mainline/{V0_original_bidirectional,V1_native_chunk_causal,V2a_rgb_anchor_causal,V2b_same_sigma_local_bidir,V3_efficient_causal_planned}；report同步采用V2a/V2b，保留Research Branches A/B/C。路线图、总README、逐版十项说明、5分钟讲稿、浏览器演示页和现行来源清单均更新。下方首次整理记录保留当时旧编号，当前应以本条和migration映射为准。

V2a：8steps/chunk、RGB dual+visual adapter、clean-commit CPU KV，124f相对结构稳定，但A/D方向失败，20秒后段严重退化。V2b：30steps/chunk、Single I0、Same-σ/T2局部联合重算、无新增训练/无persistent hidden KV，仅56f第二块AA/AD/DA/DD局部动作与结构正结果，未验证第三/第四块或124f。未来V3需不依赖history/current双向重算而保留动作与结构，再引入AnyFlow和on-policy DMD；不是简单拼接两方案。

实际CPU制作16条新编号对比视频，另存7份版本目录实体副本，当前画廊共22条。包括V2a/V2b分别与V0/V1对比、新增V1 vs V2b，以及V2a vs V2b的A、D、合集和V2b四路径RGB39边界展示。V2a vs V2b明确为能力比较，标出8/30steps、clean KV/Same-σ重算等混杂；共同前56f不替代完整原片。18份带旧V2/V3屏幕标签MP4归档并保持原SHA，无覆盖原视频。

最终验收PASS：修订前3141个文件无丢失、无意外改写，3094个原字节保留，47个现行叙述/导航更新；1059项重点证据保护检查通过。79份打包视频（62唯一内容，含归档）全部H.264/yuv420p/24fps全片解码与PTS通过；338份文档的1938处本地链接无断链。V2b冻结源码hash、71项核心复制、15项代码快照验证通过；22份Python语法通过。report实际隔离复制通过：94文件、117581957bytes（112.1MiB），无软链接依赖。

本次只有文件整理、CPU视频编码与验证，无新训练或模型推理。入口：[最新汇报](submission/report/README.md)、[并列路线图](submission/report/roadmap.md)、[整理总结](submission/REORGANIZATION_SUMMARY.md)。收据在submission/archive/taxonomy_v2a_v2b_20261010/validation.json。本次变更仍在本地，未新增Git提交或推送。

## 2026-10-10：Mainline + Research Branches + Report 整理完成（无新模型实验）

按用户新要求完成文件审计、版本映射、仓库导航重构、真实MP4的CPU并排制作和汇报材料。基准submission commit为1a93713e218b8fd48d3fa5c47f25638068c012e0；开始时工作区干净，fetch后远端没有新增提交。

新增submission/mainline/{V0_original_bidirectional,V1_native_chunk_causal,V2_rgb_anchor_causal,V3_same_sigma_history,V4_efficient_causal_planned}、branches/{A_causal_diagnostics,B_causal_adaptation,C_anyflow_dmd}和report/。report约106.7MiB，可独立复制用于组会，含本地index.html、统一十项版本说明、核心代码快照、roadmap、5分钟讲稿、原片/失败片及19条横向画廊（18条新编码+1个同义命名副本）。所有新片24fps/H.264/yuv420p、保持真实帧，不插值/慢放/循环。

版本复核：V1主片选2026-10-02零新增adapter、latent dual、seed13、8steps/chunk的A/D124f，flow约−0.01796/−0.02575；最早seed2/4-step独立保留，fixed-mix不再冒充零训练V1。V2是RGB anchor+visual QKV+endpoint/boundary等联合修复，124f视觉改善但动作失败且20s崩坏。V3从Original重新出发，无V2 visual adapter、无persistent hidden KV，T2逐sigma联合重算、C12→5仅第二块局部动作与结构正控；V4只有计划。AnyFlow/DMD归研究分支，不声称V3已完成Stage1/2。

画廊含V1/V2/V3 vs Original、V1 vs V0、V2 vs V1、V3 vs V2，另有逐A/D和V3四路径56f。V3在RGB39切块处高亮；跨版仅比较共同[0,56)，完整124f源片和20秒失败片仍保留。没有匹配Original/V2 A→D、D→A在RGB39切换的视频，明确MISSING_MATCHED_VIDEO。未进行GPU推理补齐。

原始outputs、checkpoint、模型code、MP4和测量JSON/CSV/log未更改；用git mv归档4份历史文档并保留旧路径入口。代码快照有源路径/commit/SHA，V3冻结runner/runtime哈希核对通过；V0/V1旧运行缺少完整冻结runtime，未用当前源码冒充。旧文档102处断链及补入报告1处链接共103处处理，恢复27个被引用的已有资源；唯一找不到的上游ABot海报改成缺失说明，原Markdown字节另存。

验收PASS：2844个原跟踪文件中2823个字节不变，21个文档仅导航/内容整理；无意外修改或丢失。核心71项复制一致、V3冻结源码一致；417个主要导航链接及全仓库链接无断链；56个新增打包MP4（46个唯一内容）全部解码/帧数/FPS/PTS通过；18个新/摘取Python语法通过；report实际隔离复制、无软链接依赖通过。git diff检查完成，原始计时/显存/光流没有重算改写，也没有新增训练或AnyFlow/DMD。

入口：submission/report/README.md、submission/report/index.html、submission/REORGANIZATION_SUMMARY.md。详细收据在submission/archive/reorganization_20261010/validation.json。此次本地整理尚未提交/推送GitHub；远端检查结果不等于已同步新文件。


## 2026-10-10：固定状态Action Routing / KV审计全部完成，P0严格等价通过

按用户P0→P1/P2顺序完成，不训练、不生成新视频、不启动AnyFlow/DMD。Original H3＋released action LoRA、Single I0/native time、同状态A/D、两份自身first12 history＋当前5latent、三sigma。GPU2主实验和GPU4的N桥接均已退出，最多并发两卡。9项CPU测试及真实832×480布局检查通过。

P0先失败再定位数值混杂：默认BF16/SDPA首状态最大KV相对RMS0.033158、velocity0.009138；禁止BF16 reduced reduction后第0层等价，后续仍差；canonical SDPA后历史50层KV全部相同，但裁剪当前分支还有投影差异。再统一GEMM行数，6状态×A/D的全部逐层K/V、RoPE及最终velocity误差均为0，cache hash不变。三轮失败和定位收据完整保留。该受控数值后端未改权重/边/条件，不是生产默认或速度优化，不能说默认部署已bitwise等价。

P1六状态动作差分cos均值：关公共prefix反馈0.3354；仅video三角化0.1175；当前native条件重算→sigma0 clean commit冻结0.6400；增加within-chunk直读0.4113；增加全部past直读0.0645。对应幅度/相对误差逐点保存。R1-P与R2只有历史video query→当前video key边不同，action feedback/输入/时间全部同协议。

统一与Original-visible比较：R2(无persistent KV)平均delta cos已只有0.0570，R3冻结为0.0669，R4-full为0.0717。失配在缓存之前已出现；不能因R4偏离own就断言它每点都更偏离Original。中sigma纯cross直读相对own cos为A-history−0.0035、D-history0.0487，within为0.2241/0.4421，证明直读范围本身明显改变动作函数，但没有新的视频质量结论。

P2 N-history桥接：关公共反馈cos=0.1469/0.0861，随后仅video三角化=0.0195/0.0649；N三路历史值/时间/当前状态完全相同，两个跨进程R1/N重放误差0。C+N局部正控不能被严格causal图等价替换，仍保留旧正控，不扩多窗口。

主体72次A/D对照加P0/prefill/重放/P2/排错，总计133完整33B调用＋8次第0/1层截断诊断，零optimizer/新视频。成功主进程1120.5秒，GPU allocated峰值25.677GiB，CPU raw KV6.332GiB；均为诊断成本，不宣称视频加速。提交包归档所有源码、协议、原始JSON/CSV/log与失败轮次；临时velocity张量只保存本机路径和hash，不复制模型/KV。

[完整结果与复现边界](submission/reports/stage1_anyflow/02_causal_diagnostics/action_routing_kv_audit/execution_20261010/README.md)。结论限于此固定状态协议：同图同时间的缓存基础操作可等价；动作差分对公共prefix、video图和历史时间条件敏感。尚未通过严格causal＋persistent KV的动作/画质联合验收，未修改生产默认或会议视频。

GitHub同步完成：1a93713e218b8fd48d3fa5c47f25638068c012e0，main本地/远端一致，提交包工作区干净。推送前fetch确认远端无新增提交，不需合并；原始CSV/log的行尾与空格按字节保留。

## 2026-10-09：Action Routing / KV审计完成，四路机制对照未执行

按用户新要求先审计、不启动实验。本轮零模型前向/生成/训练，未重跑神经网络测试；AnyFlow/DMD继续暂停。h3_cached/local_topology与submission副本SHA相同，旧源码、结果和视频不变。

确认causal prefix允许V_i直接读A_j(j≤i)，包括同块更早latent及历史chunk，改变Original Single-Egress；直接入口在fresh action prefix，持久KV只存video rows。own仍可通过历史video hidden/KV间接接收旧action，语义正确性待测。恢复own不等于完整Original：基础公共prefix不能读video，feedback默认False，开启后只恢复当前action读取自身video。

现有recompute_forward接收却未使用action_prefix_mode/action_cond/action_adapter，旧causal mask还屏蔽所有prefix-query→video-key，包括action自身反馈；不能直接与causal+feedback cached比较并称仅差KV。两入口均fixed_prefix_timesteps=True，与原生text/action时间不同。latest C12→5是T2窗口内历史/当前双向重算，不是严格chunk-causal重算；N历史逐sigma加噪也不能和clean persistent KV作单因素对照。

旧首块2×2和无缓存field实验已证明路由/公共prefix/causalization在没有persistent KV时也能改变动作差分；后续own＋current-prefix候选未恢复delta方向。尚不能排序action routing、video topology、KV冻结对实际视频的贡献，replay=0不补足该缺口。

已写出Original-visible、严格own causal重算、own persistent、causal-prefix persistent四路；另有公共prefix桥接及同cache即时/分别重建路由控制。主体72次当前A/D forward（prefill/校验另计），全部未执行。固定Original权重、single I0/native time、同历史/状态/噪声/全局位置，先clean history及同commit重建，必要时才4条30step局部视频。当前下一步以此机制审计为准，暂不自动延长C第三/四块。

[完整代码审计、已有证据及最小计划](submission/reports/stage1_anyflow/02_causal_diagnostics/action_routing_kv_audit/README.md)。

## 2026-10-09 23:19：C12→5自身history第二块取得局部正控；B7首窗A未过

按用户授权的有限分块实验全部完成，Original H3＋released action LoRA、30steps、单I0/native时间、full37固定noise/layout、seed13；零训练、无AnyFlow/DMD。C使用此前局部生成器自己的首12 endpoint，接第二5latent，不用GT或Original整段参考history重置；T2每sigma重算，CPU hiddenKV=0。C各分支完整56RGB/2.33s，B首窗22RGB/0.917s；都不是完整124f。

C＋same-sigma历史N：自身A-history当前A/D flow=+2.146440/−1.751855；自身D-history=+2.607594/−1.414301。完整17当前帧、39首段、人物原尺寸细节及RGB36–41边界静态检查：人物保持单体和可辨肢体，响应不同，未见此前严重躯干分解/多重肢体或scene switch；普通模糊/细节形变仍有。结论仅为第二块、单场景/seed的局部动作＋结构PASS，不冒称实时播放评审、精准3D控制或长期泛化。

clean历史对照：A-history +1.506964/−1.026712；D-history −0.176696/−1.312233（当前A失败）。B首7无history：A−0.152628、D−1.178848，人物完整，但A更像向前、方向未过。不能把收益全部归因于分块相位，N历史处理仍重要；不自动延长B/A。

真实VAE前置审计70.64s、峰值6.465GiB；两参考7/12/17prefix的前17/34/51RGB与full decode逐元素相同，末5RGB存在未来依赖。C冻结已显示39RGB，仅追加39:56，旧末5重解码差0.61–0.68/255只记录、不回改。N边界灰度MAD3.85–6.49，没有输出平滑。

全部新预算300noisy＋8diagnostic forward、0 optimizer，GPU2/5/7均结束；四项CPU区间测试通过，三次真实33B旧首12velocity重放误差0，参数version/history/noise hash不变。C第二块30步采样约194.6–197.3s，峰值allocated25.424GiB；B约117–118s、24.763GiB。时间不含模型/VAE/封装，单次共享机器，不宣称加速。CPU hiddenKV0与CPU权重offload分开。

32个MP4完整解码H264/yuv420p/24fps，完整报告与全部正负分支归档到submission/reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_cb；关键里程碑submission/experiments/11_causal_12_then5_selfhistory另保存四分支主片、协议和6份latent状态（不是新adapter）。更新主README、因果基线和会议入口；124f旧主片不替换。若继续，固定C＋N验证第三/第四块自身history，再讨论全长或KV迁移，不恢复训练。

[56帧四分支视频](submission/experiments/11_causal_12_then5_selfhistory/C_selfhistory_N_56.mp4) · [完整结果](submission/reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_cb/README.md)。

GitHub同步：提交ae8e53c归档C/B实验与chunk审计，1a0ad4d另存6份生成latent状态；推送前检查远端无新增提交。提交包原始log/CSV保持字节，未为格式检查改写测量证据。

## 2026-10-09 22:57：按用户新指示启动有限C/B分块验证

用户明确恢复本项只读实验：优先C=[12,5,5,5,5,5]的第二个5latent，B只验证首7latent，A不启动。没有恢复训练、AnyFlow或Stage2，也不自动跑完整124帧。

新目录H3-World/outputs/2026-10-09-22/chunk_partition_cb，已预登记protocol和源码hash。Original H3＋released action LoRA、单I0/native时间、full37固定布局/noise、seed13、30steps/shift2.22不变。C接自己已有首12生成endpoint（A和D分别来自coarse_A/window0_A.pt、window0_D.pt），每份同历史上fork当前A/D，分别评clean及已有same-sigma noised历史条件；不使用Original全视频或GT历史重置。T2逐sigma重算，无persistent hiddenKV。本次范围是C续到56RGB、B生成22RGB。

显式(start,stop)推理接口在小H3的4项FP32/BF16×两历史条件测试通过，覆盖5种区间、旧路径逐元素一致、未来动作隔离、当前动作响应与历史只读。真实VAE两参考的5/7/10/12/17prefix及future12+干预已完成：7/12/17prefix中前17/34/51RGB与full decode逐元素一致，末5RGB会变化。C采用保持已显示39RGB、只追加39:56的协议，并记录末5RGB潜在回改与边界误差。

GPU2 C_A PID1766826、GPU5 C_D PID1766828、GPU7 B_first PID1766827，最多3GPU。三进程均完整加载104个released LoRA补丁，重放旧首12实际solver状态的velocity误差全部0，进入采样。预算300次noisy采样forward加有限诊断；运行中，尚无动作/画质结论。

## 2026-10-09 21:52：审计Action–Video分块与VAE时间相位

对用户提出的2/7/12起始窗口建议完成CPU源码/调度审计，没有加载模型权重、新增训练或生成，研究冻结保持。旧版按完整latent空间行和action span切片，832×480下每latent390 DiT tokens；非均匀(1,4,4,4,4)动作映射一直存在。旧5-latent边界17/34/…/119固定处于模5=0，对齐encoder17RGB主体；候选2/7/12起始处于模5=2，对齐整视频长度约定。后者来自末尾token_drop=3，不代表encoder有特殊首2启动块。

实际VAE调度为5latent主体+2overlap。CPU运行真实padding/frame-plan/streaming函数（神经decoder用shape stub）：首2latent num_chunks=0、返回None，不能直接实现“先输出5RGB”；prefix5补2后可输出17，prefix7/12可输出22/39但末5RGB仍涉及后续融合。此前真实VAE已证实未来latent影响早期RGB；此轮未重跑像素实测，不将phase候选称为已证实的ghosting修复。

CPU核查旧5、旧12、候选A/B/C所有显式切片的完整action/span、390空间行和global RoPE；生产rollout的index*chunk_frames仍需显式区间重构才能支持非均匀首块，未改默认。复跑KV、局部因果及未来action长度位置契约17项测试全过。指出own/causal prefix模式不同，causal模式直接读过去action已改变Original single-egress，pair索引正确不等于原始信息流保留。后续建议先只做VAE prefix/commit对照，再考虑有限Original30步局部C/B正控；没有启动AnyFlow/Stage2。

[完整审计与CPU收据](submission/reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_audit/README.md)。原始视频/checkpoint/冻结收据不改。

## 2026-10-09 21:22：按训练目标划分三类模型并集中展示

按用户定义分成I Original＋causal routing（零更新）、II ordinary FM/visual/action causal adaptation、III TF-AnyFlow finite-map训练。单窗口、固定GT/reference history、自由rollout作为独立评测维度，未再混称checkpoint类别。

新增submission/meeting/model_types：01零训练Original/Causal0 30/8步；02基于既有单条结果CPU并排Original/FM0/FM48（causal均30步）；03同初始化/容量/两条伪标签/16新增更新的FM与AnyFlow，各有4/8步对比。四条主片均完整39f/24fps，另保存两条Original源视频，模型不重跑。28项旧基线证据不变，新manifest记录分类、实际objective、来源hash及解码；FM0容器trained_for_causal标记不当作训练证据，初始化审计证明零输出/零更新。

明确AnyFlow16起点是旧RGB causal初始化，不是ABot FM48；两组并非单一路径顺序训练。更新因果基线、主README、会议讲稿入口、面试回答、报告与Stage1说明。研究冻结和质量No-Go保持。

[三类模型与视频](submission/meeting/model_types/README.md)。

## 2026-10-09 20:12：单独固定因果化基线证据，未宣称质量通过

先fast-forward合并GitHub 91e64c0的docs/experiments整理。按“已发布H3底座 → 可信causal普通步数 → AnyFlow → on-policy DMD”拆分研究目的，保留官方Stage1因果训练＋AnyFlow的配方区别。

新增submission/docs/CAUSAL_BASELINE.md与07_protocols/causal_baseline.json，固定单窗口39f原始正控、Original local N30步参考history候选及两套124f缓存adapter/视频。局部N两history A/D符号正确，D较完整，A仍有RGB59–68多重手臂/躯干重影；每sigma重算，CPU hiddenKV0，不是自由rollout。旧cached124在动作与画质间仍有取舍，无联合合格checkpoint。资产按SHA固定，不复制33B权重、不改原视频。没有新训练或模型推理，研究冻结保持。

[因果化结果与保存版本](submission/docs/CAUSAL_BASELINE.md)。

## 2026-10-09 18:44：后续数据集训练与AnyFlow视频补档

核对本机两棵outputs的172份training.json及23个源实验目录：真实ABot视频训练为FM48、噪声密度FM48和E2两臂各4；已找到的AnyFlow使用Original H3停车场A/D伪标签。目前没有找到录制ABot数据＋AnyFlow目标已完成训练的证据，不按trainer文件名混称。

补齐114个此前未归档的MP4、224份关联测量收据和2份中断训练记录，共340个原始文件约88.9MiB。202个源MP4路径对应197份不同内容，重复内容链接已有归档；真实数据组98个源视频全部可定位。新增会议数据/目标/视频总入口及四类全视频索引，原51个实验目录已归入7类。未新训练、推理、重新编码或改变质量结论，新adapter/optimizer和原始数据仍在本机。

[结果与视频入口](submission/meeting/DATASET_AND_ANYFLOW.md) · [逐文件来源审计](submission/reports/stage1_anyflow/07_protocols/reorganization_20261009/training_video_inventory.json)。

## 2026-10-09 18:34：实验归档分层整理

将reports/stage1_anyflow原51个一级实验目录归入7类，30个散落文件归入early_pilot和overviews；新增分类README与旧→新路径映射。更新报告/会议/进度链接及当前帧图生成器。原工作区outputs未移动，原始JSON/CSV/log、视频/帧图和冻结Python源保持原字节；不新增训练或改变实验结论。

[新目录导航](submission/reports/stage1_anyflow/README.md) · [整理记录](submission/reports/stage1_anyflow/07_protocols/reorganization_20261009/README.md)。

## 2026-10-09 17:31：补充昨晚真实视频训练的展示入口

已核查远端4460166中真实ABot FM48、sigma density FM48及E2的报告/视频均存在；各目录19/12/34个MP4包含参考与拼接，不能当65次独立实验。4份最终训练adapter仍在本机outputs，存在性与哈希已记录，未上传新权重或optimizer。新增[昨晚进展展示](submission/meeting/OVERNIGHT_PROGRESS.md)：真实GT/Original/FM0/FM48四列、GT/generated history差距、零训练N历史条件的A/D方向改善及E2负结果，彼此不混淆。选中4条完整视频再次解码，未新增GPU任务。

## 2026-10-09 17:14：实验冻结与最终提交复现验收完成

按用户四项优先级完成收尾：E2所有预选评测已齐，FM-only/FM+action各4更新，动作项无一致收益，局部A仍重影。冻结16份关键证据并核对14个历史进程身份已结束；不扩16，不重启AnyFlow/Stage2。

重写INTERVIEW_ANSWER、EXPERIMENT_REPORT、REPRODUCE和会议讲稿/六页提纲；主视频保持2026-10-06 `visual_online_rgb_tail16_endpoint_ad2`，包内visual_rgb_tail16两份adapter与源文件SHA256一致。明确主片没有AnyFlow/E2，不把后续DMD工程原语当完整33B Stage2。保留同checkpoint的20秒完整失败片与E2局部A重影三列视频。

独立Python3.10 venv（无system/user site packages）＋GitHub新克隆DiffSynth固定300e3e4及5份patch完成验收。KV/局部因果15测试通过；实际MiniMaxH3DiT随机小配置普通FM训练smoke8更新，loss2.334223→2.331131，参数改变且梯度有限。真实33B推理完整加载released H3 104对LoRA、正确visual/action adapter，39帧24fps H264完整解码，24noisy＋3clean commits。

物理GPU2共享负载、reserve20，单次端到端374.586s，GPU allocated峰值15176.43MiB，CPU raw KV6484.13MiB；A水平flow−1.14536，说明工程验收通过不代表A动作方向通过。没有追加统一124f性能实验，不把单次共享机器结果当速度比较。主片/grid/失败片及6条E2三列共9视频再次完整解码。

补入遗漏cache测试及pytest/OpenCV依赖；升级pip解决新venv安装问题；benchmark增加显式checkpoint/首帧/prompt及哈希、LoRA加载数量检查。此次只改复现入口与记录，不改架构或训练目标。

[验收证据](submission/reports/final_acceptance/README.md) · [会议材料](submission/meeting/README.md) · [冻结计划](submission/NEXT_PLAN.md)。

## 2026-10-09 16:16：E2受控评测收尾；动作监督短试没有额外收益

2026-10-09 16:16：E2两臂各4更新、六组局部视频评测及48次held-out诊断全部完成。停车场两份历史的A/D符号均保留，但A分支重影仍在，FM+action没有一致优于FM-only；局部动作＋结构联合gate仍为No-Go。本轮不自动扩训，不进入AnyFlow/Stage2。

两臂均从Original H3＋released action LoRA出发，只更新tail8 QKV/out共10,092,544参数，LR2e-5、logical batch2；仅objective不同。初始化bank完全相同，梯度replay误差0，训练wall284.76/416.62秒，峰值allocated均27.91GiB。六组评测共360采样＋12identity forward，另48次held-out同状态诊断；全部原作业及两队列已结束，无重启。

停车场同状态对照（A flow / D flow）：

| 固定历史 | Original权重局部N | FM-only4 | FM+action4 |
|---|---:|---:|---:|
| A-history | +1.645495 / −1.546733 | +1.657549 / −1.395918 | +1.678612 / −1.376481 |
| D-history | +0.614104 / −0.673822 | +0.699057 / −0.667578 | +0.657532 / −0.616613 |

四组当前A/D均完整检查42RGB；A分支中段/末段仍有多重手臂、躯干或腿部残影，D分支相对完整。FM-only局部减轻D-history/currentA的重影，动作项未进一步消除。两份真实GT-history24各生成39RGB：一组首帧背景拖影及边界MAD改善，另一组几乎不变，初始姿态/视角突跳仍存在。三bank六case共738当前RGB静态逐帧及原尺寸细节/边界评审完成，不声称实时播放评审。

Held-out正确动作FM均值0.25976668→0.25969061/0.25969900，改善均不足0.03%；正确排序5/8→4/8、4/8。两状态×四sigma不是八个独立视频试验。没有用验证集改lambda、margin或noise权重，错误动作无反事实视频GT。

训练覆盖审计发现8个microbatch无纯A/D，其中3个还含相机按键。仅CPU扫描原训练episode后，A与D各找到一段更纯的源视频候选；源画面人物完整，但A存在按键激活到明显运动的待核查时序。没有移标签、编码或新训练；它们也不是同状态反事实配对。这是下一轮监督可靠性需要处理的缺口，不是已证实的ghosting成因。

当前T2/N每sigma重算可见hidden，CPU hiddenKV=0，不宣称persistent-KV加速。停车场历史来自Original生成，真实GT例子含联合相机和观测派生F，二者均为局部条件实验，不是124f自由rollout。会议主demo保持原验收状态。

[完整结果与六条三列视频](submission/reports/stage1_anyflow/01_real_video/real_transition_windows/FINAL_RESULTS.md) · [监督覆盖和源视频检查](submission/reports/stage1_anyflow/01_real_video/real_transition_windows/SUPERVISION_COVERAGE.md)。完整Stage1及研究目标仍未完成；路线继续保持“可信causal30 → AnyFlow → on-policy Stage2”。

## 2026-10-09 07:38：E2两臂4更新完成，固定状态诊断与局部视频评审中

2026-10-09 07:38：E2同初始化FM-only与FM+action均已完成4更新；48次held-out同状态诊断完成。第一组D-history的A/D方向保留，但A分支仍重影，动作损失尚无优于FM-only的证据。两队列继续评停车场A-history及真实GT-history24；不扩训、不启动AnyFlow/Stage2。

两臂均只训练released action LoRA tail8 QKV/out，初始化bank逐元素一致，4更新参数相对L2变化约0.00677/0.00672，梯度重放误差0；训练wall284.76/416.62s，torch峰值28580.55MiB。运行冻结源码与307文件runtime未改。

Held-out两状态×四sigma的正确动作FM均值：Original局部N=0.25976668、FM-only=0.25969061、FM+action=0.25969900，改善均不足0.03%；正确动作误差低于交换动作的状态计数5/8→4/8、4/8。两状态不能算8个独立样本；没有验证集调参，也不把交换动作误差解释成反事实视频真值。

停车场D-history的A/D flow：zero +0.614104/−0.673822，FM-only +0.699057/−0.667578，FM+action +0.657532/−0.616613。全42帧静态评审与原尺寸细节显示当前A仍有重影，D基本完整；不以符号正确通过局部联合gate。后续A-history、真实GT两场景按原队列收齐，最终结论尚未作出。

## 2026-10-09 07:14：真实GT局部基线已评审，两臂动作后果监督开始

2026-10-09 07:14：E2真实后果数据与前置审计已完成（6train+2validation；24个真实前缀检查差0；17项CPU检查）。33B梯度预检峰值27.81GiB。两条真实GT-history24的30step局部基线人物大体完整，仍有边界跳变和运动偏离；GPU1/4现运行同初始化FM-only与FM+action，各固定4更新。动作+结构完整gate未过，不进入AnyFlow/Stage2。

本轮两个124RGB真实条件下，仅生成history24之后的12latent/39RGB；展示视频附8个共同GT上下文，总47帧。W+A+J场景采样445.15s、MAD19.250、边界MAD23.376；D+L+F场景451.77s、MAD12.425、边界23.757。两条合计60noisy forward，torch峰值25.62GiB，T2每sigma重算、CPU hiddenKV0。这些是单次局部采样时间、不是整视频端到端加速或纯A/D定量保真。全部78当前RGB静态逐帧检查，并看原尺寸人物细节及GT80→生成81边界；4个H264/yuv420p MP4完整解码通过。

训练集8状态校准中，5/8交换动作的FM误差更低；不把这个当反事实视频真值。预登记规则给margin9.30689275e-5、lambda2.48589083；lr2e-5、logical batch2、released action LoRA tail8 QKV/out，双方同数据/noise/初始化，仅loss不同。序贯hinge梯度与联合图2项新CPU检查通过。两臂各停在4更新，再做冻结的真实GT与停车场same-state动作/视觉评测；不能凭训练loss直接扩到16或进入AnyFlow。

进程：FM-only GPU1 PID174575/start_ticks258003891；FM+action GPU4 PID174707/start_ticks258004007；启动参数hash相同，后续等待原进程、不重启。未完成Stage1或完整研究目标。[报告和局部视频](submission/reports/stage1_anyflow/01_real_video/real_transition_windows/BASELINE_RESULTS.md)。

## 2026-10-09 06:56：真实动作后果数据准备与33B反向传播可行性

2026-10-09 06:56：E2真实后果输入已建立6train+2validation，完整124RGB同episode不重叠；新窗口标签/位置合同通过15项CPU检查。真实33B仅released action LoRA tail8 QKV/out的3次backward已通过，峰值27.81GiB、零optimizer；验证编码通过，训练编码和两条GT-history24/30step局部基线仍在运行。E1动作+结构尚未过，不进入AnyFlow/Stage2。

新数据保留完整联合按键、源帧/pose对齐；原39f起点扩长后的重叠片段不再作为独立124f样本。A/D交换负例只改当前横向动作，未知反事实没有视频GT；train后续窗口7个有资格，history24仅1个A，泛化结论须谨慎。相机F限制为窗口内邻格派生并提交，translation不进入条件；F仍是观测速度代理。第一次混合句长坐标检查因约1e-13的FP64平移舍入停下，已留存attempt1；修订直接按固定origin重建时间坐标，实际tiny-H3及混合句长检查通过。

真实33B预检10092544个released LoRA参数，安装前后输出差0；history12正例/负例FM=0.16211884/0.16219865，history24正例0.15563034，梯度有限非零且权重未变；峰值28481.92MiB、wall89.66s。该值是预检成本，不是生成速度。当前GT-history基线只生成后续39RGB、附8GT上下文，不称124f自由rollout。训练需先登记LR/noise/margin/lambda，仍按先4/最多16更新/臂，不扩模型。

[完整新协议与收据](submission/reports/stage1_anyflow/01_real_video/real_transition_windows/README.md)。没有新模型训练完成或质量改善结论，完整目标继续。

## 2026-10-09 06:30：历史条件改变能恢复当前窗口A/D方向，但视觉仍未通过；停止该因素扩展

C/N同状态探针已完成56次真实H3前向，首窗identity、旧C重放及repeat均0。12个保存的C轨迹noisy states上仅fork当前A/D，指标由保存field在CPU独立重算一致。N/C动作差分方向变化很大，但该cosine不是对可信teacher的保真度。两probe wall312.69/313.03秒、torch峰值allocated27709.41MiB；原PID均已退出。

按预注册门槛完成窗口1的两history×A/D、30步共120采样前向。A历史的A/D从C的+1.275624/+1.176016变为N的+1.645495/−1.546733；D历史从+0.309645/−1.482602变为+0.614104/−0.673822。两份历史当前方向都分开：A历史的D符号由错误正值变为负值，D历史原本符号正确、新协议保持；D分支的原多重手臂明显减少；但D历史当前A在RGB59–68存在明显多重手臂和躯干重影，A历史当前A末段有腿部残影。因此不能仅凭符号改善通过。A历史/A的首边界MAD5.250→9.677，其余3分支降低；未见整体场景/I0姿态重置，但不证明长时history adherence。

4条当前片段全部42RGB静态逐帧检查、原尺寸细节及历史38→当前39边界图复核，非实时播放评审。6个MP4完整解码H264/yuv420p/24fps并归档：两张2×2网格左C右N、上A下D，前8帧相同历史＋42帧当前，总50RGB；不是自由rollout。两视频进程wall575.91/577.08秒（完整两动作作业），全程最多2GPU，CPU hiddenKV0、T2逐sigma重算。参数、history/noise及305文件runtime/启动源码hash保持。所有本轮GPU任务已核实退出；窗口2未运行，无新训练，会议未换、未推送。

下一步按用户E2允许的“真实动作后果”分支准备监督，避免拿失真的R-prefix当teacher。只读核查现有原片哈希：12个train＋5个validation旧起点可延伸124RGB而不越界；多数为联合W/S＋A/D，旧39帧计数不能当新124帧标注，也不是同状态反事实配对。先重建多窗口真实RGB/完整action及无未来泄漏条件，保留其他按键/相机；再做训练梯度/显存预检和FM vs FM+action对照。当前仅数据可用性完成，尚未新编码/训练，不宣称E2效果或Stage1通过。

[视频与完整结果](submission/reports/stage1_anyflow/02_causal_diagnostics/history_conditioning/VIDEO_RESULTS.md) · [E2实际准备清单](submission/reports/stage1_anyflow/02_causal_diagnostics/history_conditioning/E2_REAL_TRANSITION_PREPARATION.md)。保留本轮“动作改善但画质不够”的正负证据，完整目标继续。

## 2026-10-09 06:09：E1历史条件实现与CPU检查通过，真实H3同状态探针已启动

上一轮完成coarse12三窗No-Go，本轮执行已登记的C/N历史条件对照，不扩窗口/网络或训练预算。N只在临时模型输入将H与固定历史noise按当前sigma插值，并匹配历史video时间；history与已发出的RGB只读，动作/prefix/单I0/Original权重/12latent不变。4项实际tiny-H3 FP32/BF16测试通过，涵盖无历史与sigma0一致、逐行数据/时间、future隔离、current action和history真实使用。

GPU1的A probe PID3805243/start_ticks257601298、GPU5的D probe PID3805579/start_ticks257601415已核实存活，保持原进程。两路真实33B首窗C/N误差0、旧C重放0；第二窗口三个sigma的C重放均0。新历史协议使动作差分改变，不能据此称方向更正确。总预算56诊断，当前未有新视频/optimizer，E2/E3不启动。

[本轮协议与实现](submission/reports/stage1_anyflow/02_causal_diagnostics/history_conditioning/README.md)。运行源码/协议和305文件runtime冻结，旧视频不改，meeting未换，未推送。完整任务仍在E1。

## 2026-10-09 05:59：E1粗窗口全部收尾，多窗口动作/结构仍No-Go；历史编码未发现未来泄漏

Original＋released action LoRA、native时间、单I0、30步的12latent实验完整结束。两份历史每份三个当前窗口：A历史A/D flow依次为+0.863118/−1.174217、+1.275624/+1.176016、+1.342357/−0.227267；D历史依次为+0.863568/−1.174460、+0.309645/−1.482602、−2.343244/−2.324911。首窗无历史，重复结果不算两份泛化证据。所有12个局部分支全部帧静态检查、原尺寸关键帧复核：第二窗有多重手臂/拖影，A历史第二窗与D历史第三窗动作失败。不能仅凭部分符号正确或窗口扩大宣布通过。

VAE-only边界审计完成：两验证片段×RGB17/34/39/81→latent5/10/12/24，反转未来RGB后past latent最大差全部0；仅过去RGB＋重复末帧得到相同历史prefix。原D_1750尝试因源片只剩40目标帧，在模型加载前退出，记录保留；改用同验证episode的A_1030完成纯RGB因果审计，不冒充A/D配对。因此本轮不支持VAE历史编码泄漏解释失败，也不能把该检查当动作能力证明。

CPU实际tiny-H3新增9状态历史条件审计：历史video time=1，当前video及所有text/action time=1−sigma，历史action与own video时间相差sigma。这符合现有denoise-mask语义，原生retake也支持clean rows，不能直接称bug或唯一根因。下一项有限对照只比较clean历史与同sigma加噪的历史条件协议，保持Original/12latent/单I0/30步/路由不变；计划已登记，尚未实现或启动。局部后果参照不可靠，E2不开始；AnyFlow/Stage2继续受门槛约束，不继续窗口扫描。

本轮两coarse wall1878.81/1881.43秒，torch峰值allocated26531.75MiB；CPU hidden KV=0（每sigma重算），不是persistent-KV加速。共420 noisy＋9 diagnostic forwards，零optimizer。23个MP4完整H264/yuv420p/24fps解码通过、哈希匹配归档；120f拼接明确为39+42+39 oracle local forks，历史来自Original生成，不是GT或自由rollout。305冻结runtime与启动源/协议hash保持。所有原PID＋start_ticks已核实结束，最多3GPU未超额；当前无本轮GPU作业。会议视频不替换，未推送。

[完整报告与可播放视频](submission/reports/stage1_anyflow/02_causal_diagnostics/coarse_window12/FINAL_RESULTS.md) · [后续历史条件对照](submission/reports/stage1_anyflow/02_causal_diagnostics/coarse_window12/NEXT_HISTORY_CONTROL.md)。E1整体/Stage1尚未完成，不把本轮收尾写成完整研究目标达成。

## 2026-10-09 05:32：同输入首窗对照有明确差别，12latent方向恢复；后续窗口继续

新full37 fixture的匹配5latent A/D已完整结束并经全部17帧静态检查、5个MP4解码核验：A=-1.168911、D=-1.174091、分离0.005179，人物可辨但动作近同。coarse12两个reference首窗已完成；A参考首窗A=+0.863118、D=-1.174217、分离2.037335，D参考数值接近。首窗本来无历史，两份不是独立历史泛化证据。

补齐相同前17RGB区间的原Farneback评估，直接截断原解码帧iterator、不二次压缩：coarse12 A=+0.285428、D=-1.198252，相同区间也保留符号差异；匹配5latent指标逐值重放一致。因此差别不只来自39帧比17帧长的统计区间。12latent的生成/解码仍能用整个当前已知窗口，控制与显示粒度不同，不能据此说交互延迟更好。

coarse_A首窗A/D全部39帧静态检查中人物/停车场完整，A转向/运动与D不同；coarse_D首窗完成输入/指标/编解码核验，未单独作第二次人工视觉结论。首窗口T1/T2与repeat仍0。这个结果支持继续窗口间causal路线，但没有后续窗口/same-state geometry/GT局部画面的完整验收，Stage1仍未过。

GPU4的matched_first5原进程已正常退出。GPU1 PID3445415/start_ticks257337890与GPU5 PID3445945/start_ticks257338007仍存活，正在窗口1/2；保持同一进程，不重启/不改source。最多3GPU约束保持，目前本项目2张。E2/AnyFlow/Stage2不启动，训练0次。

[同区间首窗报告和视频](submission/reports/stage1_anyflow/02_causal_diagnostics/coarse_window12/FIRST_WINDOW_RESULTS.md)。首窗视频快照明确不代表完整120f oracle局部串接已完成。submission归档/索引同步；meeting未换、未推送，完整目标继续。

## 2026-10-09 05:25：E1 12latent跨窗口试验已启动，三路同输入对照运行

上一轮完成条件校准与5latent局部No-Go，是实质进展；本轮按登记协议执行更粗窗口。现有Original124f A/D参考、37latent的首图/prompt/layout/video+audio noise一致性及源hash通过。真实VAE prefix12/24/36实测39/81/120RGB，窗口输出39/42/39帧；不会误标成124f free rollout。CPU实际tiny-H3新2项(FP32/BF16)覆盖3窗口、未来隔离/当前动作干预/历史只读/单I0位置与首块identity。

公平性补充：旧39f与本组37latent的noise/layout不同，新增一次同fixture首5latent A/D控制，仅60次forward；coarse两history保持360，诊断9，训练0。GPU0被他人占用时free-memory guard在加载前拒绝启动，已改GPU4；原失败log保留，未修改运行中source/协议。实际GPU1/4/5三个job均按PID/start_ticks核实，首窗真实33B T1/T2及repeat均0。当前还没有完整新动作/视频结果，不宣称E1通过。

[新实验与运行快照](submission/reports/stage1_anyflow/02_causal_diagnostics/coarse_window12/README.md)。保持原进程等待；不重启已运行作业，不扩E2/AnyFlow/Stage2。原305runtime不变，source/协议/CPU/VAE收据已归档；meeting未换、未推送，完整目标仍在执行。

## 2026-10-09 05:09：E1 条件校准全部收尾；整窗口正控恢复，5-latent 局部动作仍 No-Go

本轮全部GPU作业正常退出。原生text/action时间＋单I0的5+5+2实验，两history各6分支/180次noisy forward＋3诊断，已收齐并检查四条完整39帧静态图、解码核验18个MP4。人物和停车场可辨，但局部动作失败：首块A/D约−1.036/−1.002；A历史第二块+1.663/+0.534（同正）；D历史第二块−0.969241/−0.969736；末块两历史A/D也近同向。oracle串接在17/34重置，不能说是free-rollout稳定。

对照的完整12latent已知动作窗口，同Original权重/seed13/noise/30steps：clean text＋双首帧分离0.021764；恢复原生text time后0.798524；再恢复原始单首帧后2.227735（A=+1.250421、D=−0.977313），人物保持完整。这个正控恢复表明条件协议本身是重要因素；分块负结果说明不能仅修条件就宣布causal action保留，更不能把全部失败归因于缺Stage2。

本轮17项CPU路径检查通过（9拓扑＋5时间＋3单I0），真实VAE审计/首块33B identity证据保持；没有新T1后续geometry、E2或E3更新。single局部组wall853.35/861.05s、peak26409.54MiB、CPU KV0（T2重算），这些不是单条39f推理成本。完整9组81个片段/拼接MP4与1个三列诊断视频已完整解码验证并归档。305冻结runtime未改，训练0次；只使用允许范围内最多3GPU，当前全部本轮GPU任务结束。

下一步仍在E1：使用已恢复正控的12latent当前窗口，跨窗口causal、逐sigma重算，在多个固定历史位置检验当前动作；现有Original124f A/D参考latent已经定位，无需新权重/数据。[有限协议](submission/reports/stage1_anyflow/02_causal_diagnostics/local_topology/coarse_window_protocol.json)已登记、尚未启动。若后续窗口仍失败，不继续无限扩窗口，不转AnyFlow/DMD。当前首个验收门槛仍未过，完整任务继续。

[完整结果及局部视频](submission/reports/stage1_anyflow/02_causal_diagnostics/local_topology/CONDITIONING_RESULTS.md) · [三列条件视频](submission/reports/stage1_anyflow/02_causal_diagnostics/local_topology/conditioning_window12_AD.mp4) · [突破归档：仅参照校准](submission/breakthrough/09_original_conditioning_calibration/README.md)。root/submission的计划、进度与验收同步；meeting未替换，未推送。

## 2026-10-09 04:57：E1 恢复完整短窗口动作正控；原生时间＋单首帧分块检验继续

没有增加训练。此前fixed-text-time T2的两份history×三chunk局部正控全部失败；只恢复H3原始text/action time后，A历史第二块A−D=1.412259，但D历史对应块仅0.008370，首块/末块仍弱，未通过跨chunk验收。

独立12latent/39RGB窗口的受控校准已完成：clean text＋双首帧A=-0.745426、D=-0.767189；原生text time＋双首帧A=-0.000594、D=-0.799118；原生text time＋原始单首帧A=+1.250421、D=-0.977313、A−D=2.227735。最后一组全部39帧静态图中人物结构完整、动作明显不同。三个版本同Original权重/初图/prompt/seed13/noise/30steps，逐次只改一个条件。这是完整已知动作窗口正控恢复，不是5latent causal或Stage1通过；也不把单seed与旧teacher2.023跨协议当学习曲线。

源码确认H3原生text/action time=1−sigma，原型固定为1；SolarWM Stage0.5/Stage1也有video-time→clean-time策略，需训练适配，不能说SolarWM有bug。旧field/真实FM几何的“Original”同样用了原型时间，已补归因限定，保留全部旧数据。当前校准仍固定audio noise，原H3 joint audio/video还存在差别；已有可靠窗口正控，不启动额外audio扫描。

新的native-single T2局部检验已在GPU1/5分别启动两种reference，PID2998665/start_ticks257163067与PID2999208/start_ticks257163183已核实，最多3GPU约束保持。恢复单I0时禁止把它retime到历史末帧；3项新CPU检查通过，真实33B首块T1/T2/repeat均0。保持5+5+2、30steps、相同history/noise/动作干预，收齐后再决定。39f串接是oracle local forks，history是Original-generated，不是GT或自由rollout。

完整校准与视频见[条件对照](submission/reports/stage1_anyflow/02_causal_diagnostics/local_topology/CONDITIONING_RESULTS.md)；[三列A/D视频](submission/reports/stage1_anyflow/02_causal_diagnostics/local_topology/conditioning_window12_AD.mp4)。已完成四组native视频共36个MP4完整解码核验，源runtime305文件不改；失败dual参照的geometry仍未启动，E2/AnyFlow/Stage2保持门槛。会议视频未更换、未推送，完整任务继续。

## 2026-10-09 04:16：E1已执行，T1/T2机械检查通过，真实VAE时域审计与局部正控启动

按新三实验路线开始E1，没有恢复旧A–D扩训。隔离实现T1历史KV冻结＋prefix grounding和T2逐sigma窗口重算；305文件runtime冻结。实际tiny-H3 CPU9项通过，包含Original directed mask、首块identity、future动作删除/扰动、KV只读与祖先replay、滑窗eviction、T2动作/sigma变化与原始历史不改。新代码只支持no_grad，不提前声称可用于训练。

真实VAE-only GPU1准备55.22秒、峰值7518.86MiB，已结束。两个ABot场景分别扰动RGB17/34以后，过去5/10latent差均0；GT encoder本协议通过。反过来，decoder未来latent扰动会改变前17RGB，平均绝对变化0.00727/0.00705，因此新演示采用prefix-only解码并冻结已显示RGB。这个发现不解释解码前action velocity失配，不能当作已修复模型。

GPU0/1已经启动两份停车场Original-generated reference下的同状态A/D局部正控，30steps/chunk。PID2087344/start_ticks256911499与PID2087792/start_ticks256911616已实际核实存活；真实33B首块T1/T2 max_abs=0、repeat=0。完整后续chunk/方向/画面仍待结果。保存实际solver states供同状态geometry，不相减不同轨迹。停车场history不是GT；39f拼接是oracle local forks，不能当自由rollout。

新geometry runner已准备，但要求局部正控完整评审后再启动；没有新optimizer或AnyFlow/Stage2训练。当前2GPU、总上限3，会议不换、未推送。完整任务尚未完成。

[实现、测试与VAE证据](submission/reports/stage1_anyflow/02_causal_diagnostics/local_topology/README.md) · [当前计划](next_plan.md)。

## 2026-10-09 02:57：按用户新要求收敛为三个实验；旧密度队列完整结束

当前研究目标已调整：不继续扩展旧A–D工作流，先从Original H3 + released action LoRA恢复chunk-local action information flow。新E1比较严格过去KV冻结/局部双向路由，以及局部Original窗口每个sigma重算；后者允许作为窗口间causal方案，不能冒充标准persistent-KV。先证明无未来信息泄漏与正确的prefix/cache语义，再做首块和两个后续块的同状态A/D及30步局部视频。停车场teacher-generated history与真实ABot GT history分开，原全长teacher不是合法局部监督。

E2只在拓扑/局部正控可信但动作仍弱时开展，优先可靠同起始状态fork出的action consequences，普通FM与action-swapped transition ranking受控比较；不直接相减不同历史teacher velocity。E3为可信causal→AnyFlow少步→on-policy Stage2，不要求进入Stage2前先解决全部长时漂移。最多3GPU约束保持。新协议是设计，不是已实现或已通过结果。

旧FM48 density全部收尾：54noise、GT/generated各18geometry、10完整39f候选视频、8报告组已齐。停车场新30步A=-0.190239、D=-0.158556、A−D=-0.031684；新8步A=-0.042918、D=-0.116288、A−D=0.073371。新30步人物大体完整但控制失败，新8步约23帧后人物重影；全部39帧静态图检查，非实时播放评审。自然场景六视频已记录FAIL。四个停车场并排MP4完整解码为H264/yuv420p/24fps/39f/faststart，归档哈希匹配。原训练/评测/报告进程按/proc及原start_ticks核实退出，306冻结runtime和队列/report源hash保持。

旧local-step准备的tiny-H3 CPU sequential-VJP对照4例通过，全部参数梯度最大差约2.33e−10、repeat输出0、错误重放能拒绝；仅证明梯度实现，无GPU更新。按新要求搁置该方案，不把它当新E2或效果进展。旧计划全文和此CPU准备均保留。未启动E1新GPU任务，未续训48/136，会议视频未更换、未推送。

[当前计划](next_plan.md) · [三实验完整协议](submission/reports/stage1_anyflow/07_protocols/three_experiments/PROTOCOL.md) · [旧密度终态](submission/reports/stage1_anyflow/01_real_video/fm_density_control/FINAL_RESULTS.md)。下方均为历史进展；当时的“继续完整A–D”已由本条取代。

## 2026-10-09 02:32：六条自然视频评审完成；停车场继续；AnyFlow输出梯度核查补齐

上一目标轮完成同状态机制和30步评审，本轮取得两条generated8新证据并检查全部39帧/18–38放大帧：第一场景人物仍可辨但背景持续横向拖影，第二场景前段模糊、23帧后红色残影与人体混合，末段结构不可靠。新旧FM48未见明确质变；本密度分支六条自然场景视频全部完成并已归档完整四列MP4/指标/逐帧图，H264/yuv420p/24fps/faststart解码通过。视觉FAIL，MAD下降不作画质通过。

停车场回归沿原队列继续：A30已完整39帧，flow−0.190239（仍非预期正方向）、frame MAD3.623436、boundary MAD4.003298；D30尚在生成，未提前计算A−D或称全部评测结束。原GPU/CPU控制器与parking父/子PID均按/proc/start ticks核实存活；当前仅GPU1。冻结源码保持，无重启、无新增训练。

C准备新增旧AnyFlow128的512样本自适应权重与输出梯度CPU核查：adaptive scale逐值重放误差0、weighted loss一致；真实latent维度含末chunk2，解析输出梯度范数与3组autograd检查相符。不detach的负对照另有极强梯度抑制，实际冻结源码正确detach。高noise endpoint108样本raw residual均值64.381、weighted仅0.06978；同保存状态的输出梯度范数之和约为数学上去掉adaptive的0.935%。这些是预测velocity输出端导数，不是H3参数梯度贡献；raw含finite derivative项，也不是endpoint质量。不能直接删adaptive、据此归咎普通FM动作失败或宣称C完成。初版CPU负对照复用释放图的fixture失败及修正保留，生产源未改。

全512记录已在submission仅用归档输入重算，audit/CSV/README逐byte一致。C/D仍须可信局部field；完整A–D未完成，48/136预算不扩大，meeting未换、未推送。

[六条自然视频总览](submission/reports/stage1_anyflow/01_real_video/fm_density_control/NATURAL_VIDEO_RESULTS.md) · [8步逐帧评审](submission/reports/stage1_anyflow/01_real_video/fm_density_control/report/natural_generated_8/MANUAL_REVIEW.md) · [C权重/输出梯度核查](submission/reports/stage1_anyflow/04_numerical_checks/adaptive_weight_audit/README.md)。

## 2026-10-09 02:15：同状态动作差分仍近正交；密度对照generated30继续分解

B密度对照48更新后的54点noise及GT/generated各18点A/D已经完整完成，原GPU0探针正常退出。后两chunk各12点单列：generated的新FM48整体cos0.996377、A/D delta cos0.035019、范数比0.672755；旧shift12为0.996389/0.029239/0.645296。GT后两chunkdelta cos0.011453→−0.002418。含首块的18点generated均值为0.013624，不与12点口径混用。动作非零，方向仍未恢复。

输入state/history/action-pair/endpoint匹配、A/D共用只读KV、D独立历史重建等于A、repeat RMSE0、参数version/source核查通过；teacher双向重算历史与student固定KV的结构差异、插值状态及旧收据缺完整teacher/anchor tensor hash仍披露。归档原始记录在CPU重算，逐点CSV及全部分组报告一致。

54点noise相对旧FM48：低段MSE−0.985%、中段−0.804%、高段+2.548%，全54点等权均值+0.195%；sigma1六点全变差。固定Gaussian权重函数、仅改sampling density呈现取舍，没有证据修复动作geometry。

两条GT30和两条generated30均完整39帧并经静态逐帧检查。GT新旧未见明确质变；generated第一场景人物保留但运动偏弱，第二场景23帧起红色残影、24–36帧身体分解、37–38帧近消失，与旧control相似。因此本轮已观察到视觉FAIL，不能由低/中noise MSE下降判通过。四个完整对比视频已归档并验证H264/yuv420p/24fps/faststart完整解码；不是实时播放评审。

原GPU/CPU队列继续generated8和停车场A/D30/8，当前只用GPU1，项目最多3GPU限制保持；不重启原训练/扫描，不改冻结源码、不延长48/136预算。完整A–D仍未完成，C/D可信局部field前置有效，meeting未换、未推送。

[同状态结果与图](submission/reports/stage1_anyflow/01_real_video/fm_density_control/GEOMETRY_RESULTS.md) · [generated30完整评审及视频](submission/reports/stage1_anyflow/01_real_video/fm_density_control/report/natural_generated_30/MANUAL_REVIEW.md) · [noise解释](submission/reports/stage1_anyflow/01_real_video/fm_density_control/report/noise/INTERPRETATION.md)。

## 2026-10-09 01:45：B密度分支48更新正常完成；两路正式评测已启动

上一目标轮完成step32与C时间条件证据；本轮持续核实等待同一训练进程，现已取得实际终态。GPU1原训练PID1563900已退出，完整48-update收据通过最终checkpoint/参数/数据课程/306项冻结源码核验；step48附加审计确认208/208 bank更新、冻结visual零输出、全部Adam finite/step48、完整noise RNG重放符合预注册计划。预算封顶，无追加更新。

训练wall6656.35s（1.85h），update合计6371.65s，allocated peak34683.62MiB（33.87GiB）。192样本低/中/高为30/76/86；旧control为4/35/153。两次GPU/共享负载不同，allocated峰值也不同，不将wall差说成采样策略提速。

固定4验证片段×4sigma共16项，全部最后chunk2：初始化/旧shift12 FM48/新shift2.22 FM48的mid raw MSE为0.181478/0.178895/0.177215，新比旧低0.94%；high为0.086299/0.081888/0.082432，新比旧高0.67%。新16项均优于初始化，13/16优于旧control，但高段有取舍；本组无low样本，不推断低噪声或视频效果。历史完整validation noise tensor哈希缺失的限制保留。[训练比较](submission/reports/stage1_anyflow/01_real_video/fm_density_control/training_report/README.md)已在提交包内无GPU/权重重算，逐行/分组/CSV一致。

原GPU评测控制器1615283/start_ticks255366821继续运行，已真实启动GPU1自然GT30评测3377015/start_ticks255999773和GPU0噪声扫描3377028/start_ticks255999783；CPU报告1898518/start_ticks255496357存活等完整组。当前项目仅GPU0/1；只读观察session7232已因评测启动正常结束，不是GPU任务退出。噪声扫描本次快照39/54点，完整action/视频验收仍待结果。保留原队列，不重启已启动工作。

接下来仍收齐54点noise、36点同状态A/D、10条完整39帧视频并逐帧检查，才决定B局部能力是否可信；不由训练loss宣布action/画质PASS。完整A–D未完成，C/D前置有效，meeting不换、未推送。

## 2026-10-09 01:08：B step32执行审计通过；C真实时间条件与原生实现数值核查完成

上一目标轮完成视频报告链路CPU集成检查并核实原训练存活；本轮继续同一进程，完成两项新证据。B密度分支目前32/48，GPU1训练PID1563900/start_ticks255332670，两个后评测队列1615283/start_ticks255366821、1898518/start_ticks255496357均存活。step32于01:06保存，208/208 bank模块更新、冻结visual仍零输出、Adam finite且step=32、数据/chunk/sigma课程与完整noise RNG重放匹配、306项runtime哈希保持。只读观察session5968已正常完成，不是训练退出；不要重启原任务。

C准备新增真实FP32时间MLP与生产逐token时间检查：4/8步native/uniform、3chunk、diagonal/finite及显式clean-history共148个CPU用例通过。current/target均采用native1−sigma坐标，text/action、RGB anchor、audio、history、padding分别核验；初始化diagonal embedding误差0，SolarWM实际mix函数对照误差0。另执行SolarWM环境原生Diffusers三个未改AST节点，在全部17个实际native时间上与项目time embedder比较，真实FP32权重输出逐元素相同。权重hash匹配B初始化；h3_cached仅注释/docstring差异，计算AST相同，没有忽略源码差异。

初次检查器单anchor packed与双anchor张量不匹配的fixture失败保留；改用生产实际causal_packed后通过，未改生产代码或运行中训练。上述是时间条件核查，没有transformer/attention/KV/视频/optimizer执行，不能替代训练后velocity diagonal、finite-map composition或C阶段效果。[完整C时间证据](submission/reports/stage1_anyflow/04_numerical_checks/time_contract/README.md)。

本轮没有增加GPU实验或训练预算。继续现存队列等待完整48后的54噪声点、36反事实点和10条39帧视频，再决定B局部能力是否可信；完整A–D仍未完成，meeting不换、未推送。

## 2026-10-09 00:49：原训练继续22/48；完整视频报告链路核验通过

训练PID1563900/start_ticks255332670和两个原评测/报告队列再次核实存活；未重启、未扩大48预算、未增加GPU实验。step16参数/Adam/RNG/source检查已通过。固定generated-state的A/D机制结论保持：对齐/直接路由/只读KV正确不等于动作geometry保真，AnyFlow128的delta cos0.03822；Original同权重causal化对照0.060153。

新增CPU报告端到端检查，用已完成旧control作为两列identity fixture，跑自然GT30和停车场8/chunk的真实输入核验、指标、全部39帧contact sheet及渲染。4个临时MP4全部39f/24fps/H264/yuv420p/faststart完整解码，14张全帧图生成，identity统计一致、自动视觉PASS保持false。临时视频已删除，仅保留源码与收据；不是新候选效果。冻结report_density.py及运行中controller源码未改。

新48-update密度分支仍待完整54点噪声/36点反事实/10视频。C/D局部能力前置有效，完整A–D未完成，meeting不换、未推送。

## 2026-10-09 00:40：B密度分支step16审计通过，训练继续17/48

上一目标轮已实现并接入严格结果报告；本轮核验了实际checkpoint并等待同一训练进程，属于具体进展与已核实等待，没有重启或扩大预算。step16于00:37保存，208/208可训练bank模块改变，冻结visual保持零输出，Adam所有已初始化状态step=16且finite，完整逻辑noise RNG和动作/chunk/sigma序列与预注册计划匹配，306项冻结runtime源hash保持。[step16收据](submission/reports/stage1_anyflow/01_real_video/fm_density_control/checkpoint16_audit.json)；step3相同检查也通过。

当前训练17/48，GPU1/PID1563900/start_ticks255332670仍存活；GPU评测1615283/start_ticks255366821、CPU报告1898518/start_ticks255496357均存活等待。只读step16 watcher2105423已完整核验后退出，不是训练退出。只读观察session74602也已完成。

本轮没有新增GPU实验、没有更改训练源/architecture，也没有训练后动作或画质结果。训练48预算、54噪声点/36反事实点/10完整视频及8报告组保持。C/D前置和完整A–D仍未完成，meeting未替换，未推送。

## 2026-10-09 00:20：B密度训练10/48；完整配对报告已接入等待队列

原训练PID1563900/start_ticks255332670、GPU评测队列1615283/start_ticks255366821均已重新核实存活，未重启任务。当前仅GPU1训练，预算48不变。新增CPU报告控制器1898518/start_ticks255496357存活，等待8组完成后生成完整39帧对照和CSV/contact sheet，不占GPU。

报告器新增严格配对核验：54噪声点输入和Original全输出hash一致；18点geometry固定state/action pair、KV只读、teacher范数和可逆源码路径适配均核对。8项CPU identity/负对照通过，会拒绝错state、teacher hash、action pair、KV写入和未审计source。旧geometry缺完整anchor/teacher tensor hash仍披露。比较器fixture不是新模型效果，不输出未完成模型占位列。

自然场景各GT30/generated30/generated8、停车场A/D30/8报告将保留全部39帧；时延/显存口径和MAD非画质的限制明示，自动脚本不判视觉PASS。新候选尚无训练后视频结论，meeting不变。C/D前置与完整A–D目标未完成，未推送。[本轮协议及报告入口](submission/reports/stage1_anyflow/01_real_video/fm_density_control/README.md)。

## 2026-10-09 00:03：真实FM单变量密度对照已运行4/48；DMD完整FMBS梯度准备通过

上一目标轮完成noise证据与归档，本轮继续B，不重复诊断或延长旧训练。训练入口新增独立`training_weight_shift` / `validation_weight_shift`；默认与旧耦合行为完全相同。45项测试通过，另与冻结旧trainer逐元素对照FM/AnyFlow完整历史loss、梯度、RNG；旧resume兼容，更改loss policy被拒绝为精确resume。

新分支只改变训练sigma采样shift12→2.22，Gaussian权重grid固定12；模型/初始化/rank8全block-refiner bank/真实ABot16train8val/动作和chunk课程/noise序列/RGB dual/CPU KV/完整历史梯度/LR/batch/seed均保持。新预算48封顶，从原始零输出初始化开始，与既有完成的FM48对照，不重新训练control。低/中/高noise样本重放为30/76/86，对照4/35/153；固定的是w(sigma)函数，不是每个样本权重、总梯度量或importance corrected objective。

GPU1于23:47启动，PID1563900/start_ticks255332670；00:02核验仍存活，目前4/48，已覆盖chunk0/1/2。首次前向前实际step00 bank/visual张量、optimizer、数据、logical RNG与对照相同；初始4片段16样本完整validation记录也逐项相同。306项runtime仅3项入口/兼容/测试变化，其余303项保持。不能从这些机械检查声称训练后动作/画质改善。

后续评测队列PID1615283/start_ticks255366821已确认存活，等待原训练PID退出且完整48收据通过审计。最多GPU1/0两路，显存不足等待、失败不自动重试。将跑54点噪声扫描、GT和固定step00-generated各18点动作geometry、2自然场景GT30/generated30/generated8，以及parking A/D30/8完整39帧；不换新分支自己的状态冒充同状态比较。见[单变量协议与实时收据](submission/reports/stage1_anyflow/01_real_video/fm_density_control/README.md)。submission保存的是时间点快照，实时进程以outputs和/proc为准。

D准备进一步实现`causal/dmd.py`：noise-minus-clean参数化下fake_x0−real_x0=sigma*(v_real−v_fake)，teacher/critic及normalizer detach，critic目标为noise−student_endpoint.detach()，student必须保留实际生成图。7项CPU测试通过：官方SolarWM masked/unmasked方向与surrogate对照、Gaussian后验方向及反号负对照、tiny-H3 FP32/BF16两块self-generated history＋CPU KV＋完整FMBS＋独立critic AdamW更新，student全梯度等于直接J^Tg、无base/critic旁路梯度、KV不变。[DMD接通证据](submission/reports/stage1_anyflow/06_stage2_preparation/dmd_gradient/README.md)。未启动33B Stage2，不声称critic收敛或视频改善。

完整A–D仍未完成：A诊断完成，B旧FM48失败且本次受控分支在运行，C/D局部能力前置仍有效。最多3GPU限制继续遵守；当前仅训练GPU1，meeting不替换，未推送。

## 2026-10-08 23:35：同状态动作机制汇总；54点分噪声扫描完成

用户要求的固定generated history、当前noisy latent、只改当前chunk A/D验证已完成，统一解释见[动作机制总览](submission/reports/stage1_anyflow/07_protocols/overviews/ACTION_MECHANISM_SUMMARY.md)。部署AnyFlow128的12点delta cosine均值0.03822，差分范数比0.7935–3.5563；对齐/直接路由/只读KV/重复与future内容负对照通过。完整输入范围受控的同Original权重causal化，delta cos已降至0.060153，说明失配先于AnyFlow。真实FM48也未恢复。Teacher双向重算历史与student固定KV的依赖仍不同，不能将低cosine唯一归因于某层权重或泛称漏一次action alignment。

两组FM0/48各54状态噪声扫描于23:24前后完整写出，当前原PID1199123/1199134均已退出；新增optimizer=0，最多2张项目GPU。2个真实ABot held-out自然场景×3chunks×实际8步全部8个sigma＋30步末端0.071108；逐输入hash和两进程Original完整输出hash相同。整体raw MSE 0.213904→0.210760，下降1.47%，52/54点下降；sigma=1均值反而0.276204→0.277907，Original0.163484。低sigma0.071108时Original也为0.514912，FM48为0.569519。因此额外误差横跨噪声段，不能直接认定大幅低噪声upweight是解法。

192个实际训练样本低/中/高段为4/35/153；低段样本2.08%、Gaussian权重总量2.06%、weighted loss6.51%，都不是gradient mass。Endpoint估计为单次z_t−sigma*v，其MSE等于sigma²×velocity MSE，不是完整采样终点或独立画质指标。本扫描为GT history自然联合动作，不是新A/D或generated-history实验。[结果与图](submission/reports/stage1_anyflow/02_causal_diagnostics/fm_noise_audit/README.md)及[解释](submission/reports/stage1_anyflow/02_causal_diagnostics/fm_noise_audit/INTERPRETATION.md)已归档；提交包在无模型/GPU下重算统计，全部分组/CSV相同。

D准备新增共享H3角色管理器：student / Original teacher / independent critic LoRA共用冻结base，禁用/恢复AnyFlow与旧visual residual，不覆盖保留图中的student参数。真实tiny-H3 FP32/BF16共6项检查通过；含critic AdamW更新插在活跃FMBS图中，恢复后student全梯度不变、teacher逐元素等于Original reference、错误角色backward拒绝。主代码与submission代码/测试/证据已同步。[角色与梯度检查](submission/reports/stage1_anyflow/06_stage2_preparation/shared_h3_roles/README.md)。这不是新的DMD训练或画质通过。

下一受控B因素应先解耦sigma sampling density与Gaussian loss weight，固定模型、数据、初始化及有限预算，同时看分sigma误差、当前A/D几何和完整39帧。尚未启动新训练，不继续盲延48/136预算。完整A–D仍未完成，C/D前置未通过；会议视频不替换，本轮未推送。

## 2026-10-08 22:50：同状态动作机制收尾；真实FM48完整验收失败

真实FM48的全部6条自然场景视频、4条停车场A/D和GT/generated各18状态几何完成，所有评测/报告进程正常退出。训练预算48封顶；未追加optimizer。GT delta cos0.017516→0.017641，固定generated delta cos0.012540→0.009336；整体cos改善不等于action保留。generated差分幅度仍约teacher的67%，主要是方向失配而非动作完全无效。

停车场FM48 30/chunk A−D=−0.018685，8/chunk=0.017456，均未通过。全帧静态评审：30步人物/车库大体保留但左右近同，8步约22帧后重影。第二自然场景generated30约23帧后红色残影、26–35帧人物分解；8步更早模糊且末段仍分解。第一场景人物大体保留，但运动/场景仍失真。MAD/boundary下降不能判画质PASS。8个完整对比MP4解码/H264/yuv420p/24fps/faststart验证通过；没有替换meeting。

隔离own/current-prefix候选的真实33B18状态也完成：首块6点Original身份逐输出相同；后续12点delta cos反而0.028777→0.008548，norm ratio0.6716→1.0330，relative error1.1952→1.4432。不能用含首块identity的全均值0.339宣称改善。120current/reference+16clean前向、零训练；候选没有生成视频或进入生产默认。10项CPU机械检查通过不等于实际action/质量通过。

结论支持当前action已正确到达、却未保持Original动作条件velocity geometry；teacher双向历史重算与student固定raw KV依赖仍不同，不能锁定单层权重或简单宣称再对齐一次就好。当前有限FM48未建立C/D所需可信基线，A–D总体未完成。下一步先使用分noise/clean-vs-generated和局部action证据来确定受控训练因素，不继续blind LoRA/anchor/prefix sweep或Stage2扩训。

INTERVIEW_ANSWER已更新：真实AnyFlow训练/136对照做过但效果失败，会议旧checkpoint没有AnyFlow；旧Stage2-lite causal-teacher/单endpoint Jacobian的限制也明示。当前本轮GPU任务全部结束、未推送。

[完整FM48验收](submission/reports/stage1_anyflow/01_real_video/real_abot_fm/FM48_COMPLETE_REVIEW.md) · [同状态机制](submission/reports/stage1_anyflow/01_real_video/real_abot_fm/FM48_GEOMETRY_RESULTS.md) · [跨chunk候选结论](submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate/INTERPRETATION.md)。

## 2026-10-08 22:29：FM48匹配几何完成；隔离路由候选进入真实跨chunk验证

GT与固定step00-generated各18个同状态A/D对照全部完成。FM0→48：GT整体cos0.988193→0.989033、delta cos0.017516→0.017641；generated整体cos0.994733→0.995210、delta cos0.012540→0.009336。普通FM48未恢复动作geometry。输入/source/pair hashes匹配、A/D内KV只读、重复误差0；teacher双向历史重算与student缓存仍是结构差异，不能将低cosine只归因于权重。

停车场30/chunk完整A/D为A−0.182680、D−0.163995、分离度−0.018685（零更新−0.008565，Original2.023377）。全39帧静态评审：人物/车库大体保留，但A/D近同向，未恢复左右控制。两个自然GT-history30也完整，FM48未见明确视觉质变；边界GT重置不能混同自由rollout崩溃。generated30/8与parking8继续等待完整组。

依首块2×2证据建立隔离候选：own-action直接绑定＋公共prefix读当前video，prefix不读历史video KV、历史仍persistent。10项tiny-H3 FP32/BF16 CPU检查通过，首块Original身份误差0、future-action不污染过去输出/KV、当前action与历史KV真实生效、全祖先按原条件replay误差0。它不是已验收修复，未改生产默认、FM48冻结runtime或meeting。

原GPU1两套几何已正常退出（PID4112044不再存在、receipt complete）；22:27在该空出的lane启动只读33B候选探针PID421741/start_ticks254849925。当前项目GPU0/4/1，最多3张，未增训练预算。A–D整体仍未完成，C/D须有可信局部action/生成基础。

[匹配几何与视频证据](submission/reports/stage1_anyflow/01_real_video/real_abot_fm/FM48_GEOMETRY_RESULTS.md) · [隔离候选](submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate/README.md)。

## 2026-10-08 22:06：FM48首条GT-history视频完整检查，尚无明显画质提升

第一个验证场景的FM48 GT30/chunk已完整生成39帧，输入记录与Original/causal0相同。全部39帧静态接触表及0/8/16/24/30/38四行对应图复核：人物可辨，但这条零更新GT历史本来就保留人物；树木/建筑细节漂移、17/34帧边界重置仍在，未见明确质变。非实时播放。GT oracle拼接不等于自主rollout失败，尚未收齐generated-history与纯A/D结果。

Original/causal0GT/causal48GT的水平flow为2.2888/0.8856/0.5612；frame MAD19.0521/17.1435/15.9786，boundary RGB MAD18.7538/47.2777/46.4746。联合按键/镜头运动变弱不直接等于A/D准确率，MAD降低不判质量提升。FM48载入后推理522.29s、allocated GPU26002.10MiB、CPU KV6484.13MiB、90 noisy＋3commits；单次共享主机不声称加速。

完整首条视频、对应帧、39帧图及原始评测/解码收据：submission/reports/stage1_anyflow/01_real_video/real_abot_fm/review/first_trained_gt30/README.md。H264/yuv420p/24fps/39f/faststart完整解码通过。三路评测继续，完整B/C/D未完成，meeting不变。

## 2026-10-08 22:01：真实causal FM48完成；三路训练后评测运行；H3 FMBS生成端CPU接入

真实ABot普通FM全部48更新正常完成，GPU0训练PID799534已退出。总wall10380.96s（2.88h），update合计10029.77s，allocated GPU peak30125.81MiB（29.42GiB）。208个bank模块更新、visual冻结零输出、16train片段×3chunk各一次、权重/optimizer-RNG协议/manifest/306项冻结runtime核查通过；没有AnyFlow/旧action residual。训练完成不等于视频验收。

固定4个held-out片段×4sigma的16项raw FM loss均下降：中段0.181478→0.178895（−1.42%），高段0.086299→0.081888（−5.11%）。训练192样本分布为高153/中35/低4；validation低sigma≤0.240781为0项，不能推断低噪声改善，也不据此盲目增权。实际validation noise tensor hash未保存，source/seed协议与sigma/weight检查的证据边界保留。完整训练报告：submission/reports/stage1_anyflow/01_real_video/real_abot_fm/FM48_TRAINING_RESULTS.md。

后续评测已真实启动：GPU0自然场景队列3560455（子进程3560459）、GPU4停车场队列3560462（子进程3560658）、GPU1 GT几何3638654。GPU1曾因free<34000等待，计划迁移到GPU2前检查发现原队列已自行启动，因此取消迁移，controller2136435恢复继续，未重启任何GPU任务。项目实际0/4/1三卡；自然场景、停车场A/D和完整36点训练后geometry均尚未收齐，不能以部分值宣布FM48效果。

CPU报告controller3449611已排队：各完整视频组/动作对和geometry达到完整状态后生成对比视频、逐项指标及0→48固定state核查；不启动GPU、训练或自动PASS。四个report脚本也已指纹冻结，运行期间不要修改。

D准备新增主代码causal/fmbs.py：最多三段1→t→r→0保留当前chunk完整梯度，接到实际H3 chunk_forward，历史KV列表快照允许调用者推进/淘汰窗口后仍正确反算。随机小H3用自己8-step生成两块历史，最后2-latent执行FMBS；FP32/BF16、checkpoint/offload、有限差分、detach负对照、缓存生命周期及协议检查共6项通过（5.25s）。完整梯度范数0.00521061，detach首段差0.00124804；有限差分0.00515580。通用primitive与此前45组官方对照版本AST相同，未重复旧数学测试。初次合成网格起点1.0000000000000002被拒，修正fixture端点后通过。报告：submission/reports/stage1_anyflow/06_stage2_preparation/h3_fmbs_integration/README.md。

这只是H3生成端接入，尚无新的33B teacher/critic/DMD训练、显存或画质证明，C/D仍未完成；继续先完成B真实视频/动作验收。会议视频不变，不新增训练预算。

## 2026-10-08 21:28：首块两类路由消融完成；真实FM 40/48，训练预算不变

固定两个场景的step00 generated endpoint/noise、三sigma、首块无历史，仅替换当前A/D；在同权重/同token范围/同后端下做2×2路由消融。当前causal（直接读过去action＋禁止通用prefix读取video）delta cosine均值−0.019933；只恢复prefix的视频反馈为0.240543，六状态均提高但未恢复；只恢复own-action读取为−0.061621。Original规则身份正控为1，不能当作模型修复。整体velocity分别仍约0.9918/0.9991/0.9918，进一步说明整体相似不等于动作几何保真。

真实33B共56前向、404.54s、GPU allocated20222.03MiB；GPU1进程3226677已正常退出。两个中sigma部署cached与full-prefix causal动作差分逐元素相同；四次独立Original身份重放RMSE0。CPU真实layout证明只改声明的两类边且首块attention图无未来action路径；不是多chunk泄漏/视频验收。完整解释见submission/reports/stage1_anyflow/02_causal_diagnostics/first_chunk_routes/INTERPRETATION.md。没有optimizer更新，没有新39/124帧视频，不改meeting。

这支持信息流改变足以造成局部动作velocity失配，不能全部归因generated-history漂移或AnyFlow。SolarWM官方condition prefix确实静态；但H3-World动作LoRA的依赖图不同，照搬不保证保真，也不能由本实验推断官方SolarWM失败。主代码h3_cached仅更正own/causal可见性注释，计算AST不变，冻结训练runtime未动。

FM48训练GPU0/PID799534当前40/48，队列2136435继续等待正常退出后进行既定训练后评测，最多0/4/1三张卡。新增CPU比较器对0→48逐状态current/history/action-pair/source哈希和teacher范数核查；两组真实step00 identity及10种坏输入拒绝/不同checkpoint KV许可共13项通过。实际anchor/teacher输出全张量哈希旧probe未记录，限制明确；不把新模型own-generated states冒充固定状态。先完成普通FM局部视频和动作验收，再决定C/D，完整A–D目标仍未完成。

## 2026-10-08 21:00：停车场零更新30/8步全部失败；真实FM 32/48；FMBS数学路径准备完成

停车场A/D四条39帧baseline全部完成，控制器1506669与子进程已退出。Original30为A+1.181253/D−0.842124、分离度2.023377；causal0 30/chunk为−0.195516/−0.186950、分离度−0.008565，8/chunk为−0.021397/−0.031908、分离度0.010510。所有初始conditioning/noise/action rows与对应Original逐张量一致；Original为旧legacy、causal为h3_fp32，anchor/prefix差异披露，不能单独归因一个mask。causal0/48将作为匹配训练前后对照。

全部39帧contact sheet与对应帧静态复核：30步人物/车库较完整但尺度/细节漂移，两动作运动近同；8步约20–22帧起人物透明/多轮廓、地面柱子叠影持续到38帧。30步改善画面但没恢复动作，8步视觉也失败。三列两行完整视频与记录见submission/reports/stage1_anyflow/01_real_video/real_abot_fm/report/parking_baseline_complete/README.md。三个归档MP4完整解码H264/yuv420p/24fps/39f/faststart通过；不替换meeting。

GPU0训练PID799534仍活跃，真实FM已到32/48，step32完整checkpoint保存，CPU权重hash/metadata/optimizer-RNG协议与前32样本curriculum核查通过；这不是画质结果。后续队列2136435继续等待正常训练退出，届时最多0/4/1，当前只有训练占用一张项目GPU。不新增训练预算。

C/D独立方法核查已读取AnyFlow原论文2605.13724与NVlabs官方commit bf9195a，以及本地干净SolarWM ce1da4e。区分SolarWM四步re-noise SGF/replay与AnyFlow最多三段1→t→r→0的FMBS。隔离CPU原型对固定官方方法45组输出/梯度，最大差均1.1921e−7；精确线性ODE组合/解析梯度通过；detach首段负对照梯度差0.282294，说明单次endpoint replay不等于完整FMBS。初次harness缺copy的失败保留并修复，无GPU训练。方法来源/源码/许可/检查收据见submission/reports/stage1_anyflow/06_stage2_preparation/cd_method_audit/README.md。

旧stage2_lite_dmd.py主入口的teacher仍走causal cached graph，不是关掉adapter就恢复双向Original。已仅更正docstring、注释与三个metadata字段；去除这些非计算变化后AST逐项相同，FM48的306项冻结runtime hash不变。未启动新的Stage2、未把CPU准备当C/D完成；后续仍需B可信局部能力及C的对角/嵌入/权重/网格/composition验证。

## 2026-10-08 20:29：固定generated-state 18点完成，动作差分cos0.012540；FM 23/48

两个真实场景×三chunk×三sigma的step00 generated-state探针完整结束：整体velocity平均cos0.994733，当前A/D差分cos0.012540（范围−0.258759～0.142183），范数比0.346～1.012。GT18点为0.988193/0.017516。首块无历史仍失配。每套六组KV内容/commit/参数不变、两场景chunk1重建A/D历史KV一致、重复student/teacher误差0。generated探针84预测+8commit、887.61s、GPU peak26597.34MiB，PID1695577已退出，释放GPU1。

两套state跨组改变endpoint/history/anchor，不是只改history的消融；每个A/D pair内部固定相同状态，后续FM48对照将复用这些固定states。统一SDPA，teacher仍双向重算历史而student使用CPU KV；状态为显式加噪endpoint而非实际solver中间态。路由入口存在≠原始信息流等价，低cosine支持动作条件函数迁移问题，不能唯一定位权重或自动判视频方向。完整36点解释：submission/reports/stage1_anyflow/01_real_video/real_abot_fm/GEOMETRY_BASELINE_RESULTS.md。

GPU0真实FM当前23/48；GPU4停车场零更新回归继续，第一条A30完成flow−0.195516，D及8步尚未收齐，不发布分离度。项目当前实际用2张卡；训练后队列2136435仅CPU等待，完成/审计后最多复用0/4/1。预算不增加、C/D仍有前置门槛，Stage1未通过，不更新meeting。

20:24补充：六个真实packed layout的CPU SDPA mask拦截确认当前action↔对应视频的双向入口均存在；但causal额外允许video读取更早action，关闭通用prefix→当前video，以及当前prefix中的过去action→历史video反馈。该结构差异不是index错位，首块也适用；尚未做逐边模型效应消融，不归因到唯一边。见real_abot_fm/ATTENTION_ROUTE_INTERPRETATION.md。FM48后的固定评测队列2136435已启动CPU等待（替换仅等待的1915121，修正预审脚本对LoRA列表的遍历，step16的208个bank模块CPU审计通过，无GPU工作重启），正常训练退出/审计通过后复用0/4/1，失败不自动重试、不加训练预算。

## 2026-10-08 20:15：真实场景零更新基线完成；GT动作差分近正交，第二场景30步仍崩溃

真实FM训练GPU0/PID799534仍在运行，已完成20/48次更新；保持固定预算，不把零更新视频当训练结果。两个验证episode的Original30、causal0 GT30/chunk、causal0 generated30/chunk共6条39帧视频完成，初始conditioning/noise一致性检查通过。完整contact sheet与对应帧静态检查：户外场景人物保留但运动变弱、场景漂移；中世纪村落generated-history约23–24帧出现红影，26–33帧人物分解，38帧大部消失。Original完整、GT-history人物保留但拼接跳变。不能根据第一个场景声称30步causal稳定；GT拼接边界不等于自主rollout失败。完整视频与解释见submission/reports/stage1_anyflow/01_real_video/real_abot_fm/BASELINE_COMPLETE_REVIEW.md。

GT同状态当前chunk A/D共18点完成：整体velocity平均cos0.988193，动作差分cos0.017516；无历史chunk0为0.034156。A/D历史KV重建哈希相同、读取期间内容/commit/参数version不变、重复前向RMSE0。差分失配在真实GT和无历史条件已存在，不能全部归因于generated-history drift。低cosine仍不是视频方向判决；teacher双向历史重算/student缓存语义差异明示。

第一个真实场景的Original纯A/D也是弱正控：A+0.744212、D+0.711019、分离度0.033192。保留原视频，不套用旧停车场A>0/D<0/separation>1门槛归罪student。新增停车场step00的30/8步回归GPU4串行运行，对照已知Original约2.023分离度；causal0/48统一h3_fp32，旧Original legacy精度差异单独披露。

固定step00 generated-state探针原GPU5因free27204MiB<34000而启动前退出，没有模型结果。失败记录保留；确认退出后于20:11改GPU1/PID1695577运行。项目当前卡0/4/1，不超过3张，不操作他人进程。

真实数据loader与trainer接口已同步主代码和submission/code，新增6项provenance/split/time-bin测试通过，主入口CPU三chunk训练和旧checkpoint恢复兼容已验证；306项冻结训练runtime hash不变。提交包收录6条基线的4个三列诊断视频、GT18几何与纯Original弱正控，不收数据/PT/optimizer，不更新meeting。

后续仍为完整A–D：先收齐FM48训练后GT/generated视频、停车场A/D及相同固定状态geometry；可信局部能力成立后才继续AnyFlow/Stage2，不盲增136预算。当前Stage1效果未通过，C/D未完成。

## 2026-10-08 19:16：真实FM已完成3/48更新；首条独立episode的Original参考完整解码

训练GPU0/PID799534持续运行。真实数据/零输出初始化/首步bank更新/冻结visual保持/无AnyFlow与旧action residual/运行源hash检查通过。GPU5串行基准已有首条Original30完整39帧，采样207.00s，allocated peak25928.91MiB；全39静态检查人物和场景保持，尚无trained-causal效果结论。该真实动作含forward+strafe left+camera pan left，flow+2.2888不能当纯A正确率。

评测入口前两次dtype错误均发生在首个denoiser输出前，失败日志保留；修正为FP32 video/audio/anchor、BF16 text，并在tiny H3原生/cached路径验证后真实运行成功。未改训练runtime或随机输入。新增来源、执行快照、原始参考视频/接触表见submission/reports/stage1_anyflow/01_real_video/real_abot_fm；受控field图与归因结果见field_factorization。完整A–D目标保持未完成，当前最多2张项目GPU使用，不更新meeting。

## 2026-10-08 19:04：实验A完成；真实ABot causal FM桥接启动

输入范围受控的A对照已完成：12个固定generated-state、9角色、224真实H3前向。统一SDPA/完整prefix历史及anchor/action/timestep下，Original权重仅改causal路由（无新增适配）的整体velocity cosine=0.996253，A/D动作差分cosine=0.060153；匹配FM32为0.054469、AnyFlow32 r=t为0.058434。动作几何失配在AnyFlow训练前已经出现，不支持继续盲增AnyFlow updates。共同初始化AnyFlow r=t逐元素保持输出。原生Flex与SDPA的动作差分cos约0.843–0.868，小动作差分对BF16后端敏感；主表均采用同一后端。

A采用完整历史/历史action-prefix重算，不能冒充部署persistent-KV等价性；此前固定KV机制诊断单独保留。两条诊断进程已退出，无optimizer。报告：submission/reports/stage1_anyflow/02_causal_diagnostics/field_factorization/RESULTS.md。

按完整A–D目标进入B：仅下载6个公开ABot episode（视频502,961,486字节），取16train＋8validation个39帧片段，按episode隔离；真实联合按键/镜头动作保留。RGB与动作共同按源帧索引30→24fps，真实VAE latents为[1,24,12,30,52]；dual anchor采用GT prefix decode→末RGB→H3 image encode，与生成条件一致。24份编码/源哈希/时间bin校验完成；两张VAE静态检查train PSNR35.97dB、validation23.28dB，不是生成质量。初次编码的模型路径失败已保留并修复，成功编码800.78s、GPU peak9683MiB。

B在18:59已于GPU0启动真实33B ordinary FM，PID799534。Original H3+发布action LoRA初始化，不继承旧visual/action residual；全block/refiner rank8 QKVO/FFN bank，GT history、全历史梯度、CPU KV、RGB dual。固定48updates、batch4、LR3e-5、training shift12；48更新覆盖16cases×3chunks。保存0/1/3/16/32/48。训练尚未完成，不把loader/CPU smoke当训练或视频通过。与旧FM32相比数据和初始化均改变，不作单变量归因。

下一步评估独立episode的Original30与causal30（GT/generated history分开），8步补充诊断，再做同状态当前A/D反事实。真实A/D-dominant样本含联合按键/镜头动作，不能直接比较不同片段光流来宣称A/D恢复。C需可靠causal基线并补时间嵌入/自适应权重/scheduler覆盖/composition，D需on-policy teacher/critic及Flow Map Backward Simulation对照；这两个阶段未完成。完整门槛：submission/reports/stage1_anyflow/07_protocols/overviews/ABCD_STATUS.md。项目最多同时3张GPU，未更新会议主视频。

## 2026-10-08 18:09：同generated-state当前chunk A/D机制诊断完成，动作速度方向失配

固定AnyFlow128的A/D generated history，在chunk1/2及高/中/低sigma共12点，只替换当前chunk动作。主对照student r=t与Original H3瞬时velocity。匹配dual RGB anchors/prefix时间条件下delta cosine平均0.03822，范围−0.10234到0.20505；student/teacher范数比0.79349–3.55631。原始单anchor/native prefix参照平均0.02968，范围−0.35006到0.27738，两套结果完整保留。动作有响应，但差分方向与teacher弱相关，支持优先调查因果化后的动作条件函数/velocity geometry，不再泛称缺action alignment。

时间索引/action spans/global RoPE/tail anchor检查通过；实际SDPA masks在0/25/49层符合当前causal+feedback设计；4组完整KV哈希/commit及参数version不变。4次student、4次teacher重复前向RMSE均0；两种history的future-only干预RMSE均0（固定layout内容对照）。中sigma文本路径单独干预也缺少teacher几何一致性，不能归因为正确文本路径简单被residual抵消。

结论边界：状态是显式加噪generated endpoint，并非保存的solver中间状态；teacher双向重算历史，student固定KV，内部表征与依赖图本来不同。低余弦不能唯一归因某组权重，也不能排除generated-history分布影响。下一项建议只读三路：Original双向 / Original权重+causal-KV / trained AnyFlow r=t；各自重建自己的history KV，避免混用。三路尚未运行，无新训练、无Stage2。

单GPU6串行，A/D各57 noisy+2 clean，473.87/421.92秒；allocated peak28763.39/27626.28MiB、CPU KV5403.44MiB。所有进程正常完成。完整中文解释、12点表/路径干预/图/源代码/原始收据归档submission/reports/stage1_anyflow/02_causal_diagnostics/generated_action_geometry128/INTERPRETATION.md。没有新视频或画质修复，不替换meeting；Stage1仍未通过。

## 2026-10-08 17:59：同generated-state当前chunk A/D机制检查已启动（只读）

新探针固定AnyFlow128保存的A/D generated rollout，只替换当前chunk动作文本span和residual输入，保留过去/未来、当前明确加噪状态与KV。主比较r=t瞬时速度；有限区间另列。Original H3权重保留released action LoRA，关闭我们新增visual/bank/AnyFlow/residual；分别测匹配dual RGB anchors/固定prefix时间与native单anchor条件。Teacher重算双向历史、student读固定KV差异明确保留。

CPU真实tiny H3验证通过：非零适配后teacher逐元素恢复、student切回逐元素恢复、未来动作不影响当前输出、KV/参数不变。真实layout检查latent5–9→RGB[17,34)、10–11→[34,39)，action rows/global RoPE/tail anchor均匹配。实际33B已在GPU6运行，A之后串行D，项目本轮只用1张卡；因其他任务进入GPU6，权重offload reserve14GiB且启动free≥36000MiB，不改变算术dtype/模型条件。controller3961954、A进程3944211。

首批A/chunk1高/中/低sigma的matched delta cosine约0.03247/0.00356/0.01905；仍仅局部结果，等待chunk2及D历史。中sigma重复student/teacher误差0、未来动作负对照0。实际SDPA masks在0/25/49层核查；不可在全组结束前把这些数值扩成最终结论。源outputs/2026-10-08-17/stage1_generated_action_geometry，归档submission/reports/stage1_anyflow/02_causal_diagnostics/generated_action_geometry128。没有新训练、没有新视频验收、没有Stage2。

## 2026-10-08 17:38：136对照与54个双指标case全部完成；按最新指示优先做动作差分机制诊断

两组均完成从128相同权重/Adam/RNG新增8更新到136，没有追加训练。原loss8步A+0.039588/D−0.541371、分离度0.580959；4步−0.675601/−0.871925、0.196324。辅助8步+0.095252/−0.539922、0.635174；4步−0.665904/−0.865820、0.199916。全39帧和对应12/24/30/38帧静态检查均未解决后段重影/人物透明，4步22–38帧严重雾化。非实时播放。完整四列Original/128/control136/auxiliary136视频及time/GPU/CPU-KV/forwards/MAD/boundary表在submission/reports/stage1_anyflow/03_anyflow_trials/interval_consistency_candidate/FINAL_RESULTS.md；H264/yuv420p/24fps/faststart全帧解码通过，不替换meeting。

双局部指标54个case均完成：低噪声self-consistency相对速度误差128/control/aux为18.12%/18.13%/17.90%；finite到teacher clean平均RMSE0.058802/0.058682/0.058319。平均未见牺牲finite伪目标距离，只是改善不足1%且未转化成视频效果。当前diagonal伪GT误差0.064893→0.067320变大，不能把参考当GT；冻结128参考fit改善幅度大于当前自一致性改善。所有参数version/KV内容不变，128重复输入/finite/diagonal哈希一致。见dual_metric136/RESULTS.md。

低sigma≤0.240781的endpoint仅2/128：D/chunk1与A/chunk0；A/chunk2三种原AnyFlow样本均无这个噪声段。训练shift12同时改变采样density和Gaussian weight，未来需解耦；已有shift2.22负结果保留，不能直接宣称降shift解决。最新用户优先要求动作机制诊断，因此先不启动噪声训练。

下一项：同generated history、同明确加噪z_t，只替换当前chunk A/D，保留过去/未来action；核查实际时间索引、直接attention可见性与KV读写不变，再比较student与Original teacher动作速度差分。AnyFlow finite output是区间平均速度，主对照使用r=t的瞬时速度，finite另列；teacher额外anchor/固定prefix时间等条件差异须明确。旧FM探针曾用clean latent配非零sigma且替换整段action，不能直接复用其几何结论到当前AnyFlow。最多3张GPU持续有效。Stage2未启动，Stage1尚未通过效果验收。

## 2026-10-08 16:51：最多3张GPU；控制组136完整4/8步未通过，辅助组已真实训练

按最新指示，项目同时最多使用3张GPU，不增加136以上optimizer预算。控制组已完成全部4/8步A/D39帧评测：4步A−0.675601/D−0.871925、分离度0.196324；8步A+0.039588/D−0.541371、分离度0.580959。4步全部0–38帧与Original/128/136的12/24/30/38对应帧静态复核：约18帧起重影加重，22–38帧人物/柱子严重叠影和雾化，未修复；非实时播放。完整4/8对照和指标在submission/reports/stage1_anyflow/03_anyflow_trials/interval_consistency_candidate/CONTROL_RESULTS.md，未替换meeting。

实际恢复任务时，原队列已于16:42自动在GPU6启动辅助组，controller1961979/trainer1963046。因此没有终止或重复启动；保留这次运行。真实预更新128的四套adapter、Adam、逻辑/CPU/CUDA RNG及其余固定配置逐项匹配源128。辅助训练未完成，其结论尚待实际视频。

为落实双局部指标，准备隔离只读诊断：128/control136/auxiliary136在同clean history、Original teacher伪GT和保存噪声下，按native8高/中/低噪声区间分别记录当前finite/当前diagonal自一致性、到相同r的teacher插值目标误差，并单独测各t→0的finite endpoint到clean伪GT距离。低噪声另记对冻结128训练参考的fit，避免混同于当前模型自一致性。r>0不能直接与clean比较。每动作每checkpoint159 noisy+3 clean forwards，无optimizer；源outputs/2026-10-08-16/stage1_dual_metric136。分析不替代视频验收。

若辅助136也失败，优先检查低噪声采样覆盖/监督，按噪声段而非一个total loss判断；不扩大LoRA或改architecture。Stage2只做准备，前置条件为可信clean-history局部画面、同状态A/D反事实及有限区间行为；不要求Stage1预先解决全部长时generated-history漂移。Original同状态反事实参照仍未补，不能夸大现有oracle-history动作结论。

## 2026-10-08 16:39：原loss136训练完成，8步A/D仍失败，完整对比视频已归档

原loss控制组128→136的8次更新全部正常结束，新增更新共1512.02秒，含准备/前后验证总1888.10秒，allocated peak40666.14MiB。128 current+8 physical clean+32 differentiable-history forwards，不含validation与backward checkpoint重计算；visual/action/time冻结、四组bank更新通过。实际129–136动作/chunk/sigma/r与重建序列一致，128/132/136保存的logical RNG逐bit相同于重建；实际GPU noise未直接记录。固定验证A endpoint34.422421→22.487106，D15.185422→14.106135，详见TRAIN_CONTROL_RESULTS.md。

正常finite-map A/D39f、8步/chunk完整评测已完成：A+0.039588、D−0.541371、A−D0.580959，低于128的0.759660。全部39帧和12/24/30/38对应画面静态复核：A约22帧后透明/重影、末段更虚；D人物/柱子叠影仍严重，未见修复。静态全帧观察非实时播放。两条输入conditioning与Original逐张量一致，每条24 noisy+3 commits、CPU KV6484.13MiB、allocated38984.16MiB；A/D E2E242.10/252.37秒，单次共享主机不作speedup结论。训练loss下降没有转化为画质或动作改善。

Original/128/control136三列两行完整对比已归档submission/reports/stage1_anyflow/03_anyflow_trials/interval_consistency_candidate/original128_control136_8step_AD.mp4；H264/yuv420p/24fps/faststart、39帧完整解码和第30帧标签检查通过。CONTROL8_RESULTS.md含原指标表、视觉限制与视频链接。未替换meeting。

控制器684856继续A4/D4；辅助队列732565等待本组评测正常结束后用GPU6，从原128相同权重/Adam/RNG重新开始新增8步。辅助组尚未训练，不把控制组负结果当作辅助项结论。不增加既定预算，Stage1未通过、Stage2未启动。

## 2026-10-08 16:17：控制组继续到134/136，step132随机流与冻结策略验证通过

GPU6控制器684856、训练685759与辅助排队732565均已检查真实/proc存活，未重复启动。控制组已完成134次更新，继续原定136上限；辅助组尚未训练，仍等待控制及其评测结束后使用同一GPU6。

新增逻辑采样核查从源128的logical RNG重建后续8步的时间对/样本类别/FP32 CPU噪声序列，已完成129–132的动作、chunk、sigma/r逐项匹配，实际step132保存的logical RNG与重建值逐bit一致。保存了重建CPU noise hashes；实际GPU noise未在训练时直接记录，不能冒充GPU逐张量审计。真实step132的visual/action/time adapter保持不变，QKV/out/FFN/refiner bank均更新。收据与脚本归档reports/stage1_anyflow/03_anyflow_trials/interval_consistency_candidate。

已准备正常finite-map完整39帧contact/matched-frame/并排视频整理脚本，保持Original/128/同8更新两组的标签与计步/单次耗时口径；当前尚无已完成的新视频，未把准备脚本当作效果验收。Stage1效果未通过、Stage2未启动，继续等待这次受控实验。

## 2026-10-08 16:04：有限区间诊断完成，启动原loss/一致性辅助的有限预算对照

同128权重、clean history及固定teacher/noise插值状态的A/D诊断全部完成：各3chunk×3区间，4/8细分参考；随后各3chunk末区间8/16细分也完成。所有原探针进程均退出。重复输入、正常finite端点与8细分端点hash逐项相同，真实CPU KV内容hash/commit次数和模型parameter version不变。4/8参考下高噪声相对速度差约2.1–8.2%、中间约2.9–3.0%，末区间15.0–18.6%；16参考下末区间约16.5–20.4%，finite/16端点差约为8→16参考变化的10–12倍，细分变化约减半，差异仍存在。

实际128训练更新的512逻辑样本中，低sigma≤0.240781的diffusion/endpoint/general map仅6/2/3个（各类型总数256/128/128），说明低噪声覆盖稀少，是可检验因素而非唯一原因。必须保留反证：六个局部case中正常finite端点到Original teacher伪标签的latent误差都小于diagonal16；细分参考不是更准确GT。内部自洽性不等于视频质量。完整数据/图已归档submission/reports/stage1_anyflow/03_anyflow_trials/finite_interval_probe128/RESULTS.md及finite_interval_refinement128。

基于r=t实生成重影更少、末区间差异持续，准备一次明确独立的有限预算训练对照：同128四套adapter/Adam/逻辑-CPU-CUDA RNG，各新增8更新到136，原loss/数据/架构/anchor/采样保持；候选只增加0.25*MSE(v_finite, frozen diagonal16 average velocity)的末区间辅助。官方AnyFlow以外的实验，不是Stage2，不是默认继续增加训练时长；两组等更新数不等算力。只用正常finite-map A/D39f 8/4完整视频决定是否有效，不用参考fit代替验收。

CPU零权重两步与原trainer模型/Adam/RNG逐项相同；辅助六步小H3训练/full-history反传通过，冻结策略通过，错误动作reference拒绝。GPU6控制器684856/训练685759已真实运行，预更新128的四套adapter、Adam、三类RNG与其余配置逐项相同，固定验证A endpoint34.422421/D15.185422与源128一致；正在新增更新。辅助组启动前GPU5被其它任务占1983MiB，未放行、未打断。CPU队列732565确认控制器存活，等待控制组及评测正常结束后复查GPU6空闲，再运行辅助组；辅助组尚未训练。详见reports/stage1_anyflow/03_anyflow_trials/interval_consistency_candidate/README.md。

没有新画质/动作结论，未替换会议视频，未开始Stage2。Stage1先验证局部少步能力，不要求其独自消除全部长时历史漂移；目标仍active。

## 2026-10-08 15:32：澄清Stage2职责，启动有限区间只读探针

用户关于后段崩溃/Stage2的判断有同权重证据支持：AnyFlow128 teacher history比generated history明显减轻人物/车库退化，A/D分离度0.760→1.334；不能要求Stage1先消除全部长时generated-history漂移才能进入Stage2。Oracle状态重置与动作历史泄漏仍限制结论；同权重r=t改善重影也说明有限步条件值得独立诊断，可能与历史分布相互作用。当前AnyFlow尚无匹配Stage2对照，旧FM Stage2-lite不能回答其疗效。详见submission/reports/stage1_anyflow/07_protocols/overviews/STAGE_BOUNDARY.md。

已完成隔离probe代码和CPU验证：解析常速度场、时变场Euler收敛、故意finite-map偏置，以及随机小H3三chunk/三sigma区间的KV/模型不变性。不是33B画质验收。固定128权重、clean teacher历史、teacher/noise插值状态，比较一次finite map与4/8细分r=t积分；覆盖每动作3chunk×3区间，每动作117 noisy+3 clean forwards。参考仅是同模型数值自洽性，不是GT，记录细分参考的不确定性。

最初所有GPU有其它任务，未抢占；15:31重新检查GPU5/6各3MiB后启动A/D，PID173171/173176，真实状态running，1001项冻结输入复核通过。source为outputs/2026-10-08-15/stage1_finite_interval_probe128；报告/CPU收据/代码/启动快照归档submission/reports/stage1_anyflow/03_anyflow_trials/finite_interval_probe128。当前没有真实区间误差结论，不追加训练次数、不变架构/anchor、不开始Stage2。GPU0资源授权保留，但其上仍有其它任务。

## 2026-10-08 15:18：同权重对角条件消融完成，定位有限步预测的视觉退化

GPU0/3两条AnyFlow128同权重r=t诊断均正常完成，控制器3641070/3641075及子进程已结束；当前无本轮在跑的GPU任务。A+0.074321、D−0.587728、分离度0.662048；正常finite-map为+0.040414/−0.719245/0.759660。对角条件动作仍不足，但完整39帧与Original/finite/diagonal同帧静态复核显示人物/车库明显更完整，尤其D后段严重人像/柱子叠影减轻；仍有模糊/轻微透明感及人物退远，不是整体PASS。静态复核非实时播放。

实际审计确认两个分支checkpoint路径/哈希和conditioning/packed tensor相同；积分sigma网格逐元素相同、3块24 noisy+3 commits不变；全部27条model time trace中noisy target_sigma确为current sigma，实际积分target_sigma仍为原next sigma，commit0→0。只通过隔离wrapper改变模型的目标时间条件，未改原runtime或权重。完整Original/finite/diagonal三列A/D视频已H264/yuv420p/24fps全39帧解码及第30帧标签检查，报告submission/reports/stage1_anyflow/03_anyflow_trials/diagonal_inference128/FINAL_RESULTS.md，保留实际tensor/time audit与全部指标。未替换meeting。

这支持有限步预测的使用在本次rollout中加重视觉退化，不能把当前失败全部解释为缺Stage2；单seed结果不证明实现数学错误，也不排除有限步预测与generated-history分布的交互。对角版本不是重训FM，也不能冒充AnyFlow finite-map成功。没有额外optimizer更新、无Stage2。下一步固定同一clean history/noisy state测有限区间预测相对冻结r=t细分轨迹的偏差；有系统误差后再决定小预算一致性校准，正常finite-map 4/8完整视频仍是验收目标。不自动增加128以上训练预算，不做architecture/anchor扫参。Stage1目标保持active、尚未完成。

## 2026-10-08 15:05：128完整评测、历史对照与同历史动作干预完成

128训练及A/D39f4/8评测全部正常结束，原控制器2446816退出。4步A−0.751431/D−0.885575、分离度0.134144；8步A+0.040414/D−0.719245、分离度0.759660。4条全0–38帧及Original/64/128对应画面静态检查完成：4步18–20帧起重影、22帧后严重雾化；8步A人物保留但透明/退远，D后段显著人像/柱子叠影，视觉未过。8步分离高于64的0.623，但不能以此称整体更好。Original/64/128的4/8完整39f grid均H264/yuv420p/24fps完整解码与标签检查通过；静态复核非实时播放。STEP128_RESULTS.md含完整时间/allocated/CPU KV/forwards/MAD/boundary表，不作单次共享主机speedup声明。

为检验用户关于Stage2/历史漂移的解释，在GPU5/6只改history来源：同128 checkpoint teacher-history A+0.504958/D−0.828582、分离度1.333539，对应generated 0.759660。实际首块5 latent均逐元素相同，conditioning/packed tensor一致；后续才不同。Teacher历史下人物/车库明显更完整，A在17/34帧位置/朝向重置、边界MAD13.50（generated4.10），D也有状态跳变。完整39f Original/generated/teacher三列视频及核查在reports/stage1_anyflow/03_anyflow_trials/history128/FINAL_RESULTS.md；不能当free-running PASS。

分时段已有Farneback：128 generated A在RGB[0,17)/[17,34)/[34,39)为−0.074203/−0.114319/+1.039100；clean为−0.051449/+0.692521/+1.542388。D generated−1.206855/−0.527455/+0.186407，clean−1.213613/−0.854157/−0.526932。全片均值与原记录1e−12内一致，跨边界transition另记；末区间4个transition、temporal VAE影响使RGB区间并非独立latent隔离。

进一步新增同历史动作干预A→D→D、D→A→A（只对chunk1归因），分别对照已有held A/D。实际noise/anchor/基础prompt和前5行action embedding不变、后续action embedding确实改变、packed tensor相同；首块latent max_abs=0。A历史当前A/D flow为+0.692521/+0.378091，A−D=0.314431；D历史当前A/D为−0.427330/−0.854157，差0.426827。相对变化方向一致，动作有独立作用，但未出现同history下相反光流符号；无Original同状态参照，不能把动作惯性与控制不足严格区分。Teacher-history的1.334包含不同历史状态，不能据此宣称动作保真。全部39帧、同帧图与可播放2×2对照在counterfactual128/FINAL_RESULTS.md；chunk2 KV会受到改变后的action commit影响，不扩大chunk1归因。

没有追加训练预算、未开始Stage2。下一项同权重推理消融已实际启动：GPU0重新空闲（启动前3MiB），与GPU3分别运行A/D；控制器3641070/3641075，子进程3642246/3642253于15:05仍存活。仅将AnyFlow模型target_sigma设为当前sigma（r=t），实际native8步积分网格、权重、generated history、anchor等不变，clean commit仍0→0；新sampler名称/27条时间trace显式记录。首块9条trace已落盘，未声称视频完成。997项输入冻结，源目录outputs/2026-10-08-15/stage1_diagonal_inference128/。这是定位有限步映射不足的消融，不可将对角结果冒充AnyFlow finite-map成功。GPU0使用reserve6，不限25GiB。

## 2026-10-08 14:34：128训练完成；区分Stage1能力与Stage2历史适应

14:37补记：128首条A4已完成，flow−0.751431。全39帧及Original/64/128同帧静态检查显示约18–20帧起重影，22帧后严重雾化/人物轮廓分解，未见相较64的明确画质修复；并非实时播放。D4/A8/D8仍待完成，不计算128分离度。证据已加入TRAIN128_RESULTS.md。

GPU4–7完成96→128新增32次optimizer更新，含加载/验证1659.72秒、逐更新wall合计1507.65秒，allocated峰值40504.14–40505.49MiB/卡。四replica参数/Adam/logical-CPU-CUDA RNG一致，四组bank更新、原visual/time冻结。实际预更新96与来源权重/Adam/历史/RNG全部一致；前验证8个样本记录与96后验证相同。新增512 current、120 physical clean commits、120带梯度history forwards，不含验证与反向重算。固定0/16/32/64/68/96/128学习曲线已归档，A endpoint35.395→34.422、D18.416→15.185；内部误差不能代替画质。控制器2446816继续既定A/D39f的4/8步评测（本快照A4，3236763存活），完成后停，无128以上自动训练。报告submission/reports/stage1_anyflow/05_runtime/parallel_resume68_to128/TRAIN128_RESULTS.md。

回应用户关于“后段崩溃是否缺Stage2”：官方源码明确Stage1 clean-history TF-AnyFlow，Stage2在自身rollout上用SGF+DMD与trainable critic适应生成分布，因此这是合理原因之一。已有step16同权重teacher/generated-history对照分离度0.664/0.180，支持历史来源影响；但teacher history下A仍错，第0块尚无历史累积时已偏离，不能全归因于Stage2缺失。这是step16证据，不能当作128诊断。oracle历史含动作后的世界状态，还需要同一历史上的当前action反事实对照。

明确更正阶段门槛：Stage1先证明首块/受控历史下合理画面、动作响应与少步能力；若这些成立而free-running后段退化，才有针对性进入Stage2。不要求Stage1预先消除全部长时漂移。39f自生成A>0/D<0/A−D>1、独立seed、switching及124f保持最终交付验收，不全部作为Stage2启动前提。目前仍未证明局部能力已满足，先收齐128视频，不立即启动Stage2。说明及官方定位见submission/reports/stage1_anyflow/07_protocols/overviews/STAGE_BOUNDARY.md；next_plan和验收文档已同步。

只读精度源码核查完成：官方原生H3的residual/scale/gate仍用block dtype，FP32用于六组边界与时间SiLU；当前h3_fp32已实现对应策略。未发现“官方FP32 residual而本地BF16”的证据，未开新精度实验、未改运行runtime。该结论不是两套模型数值等价证明，源哈希与限定见parallel_resume68_to128/PRECISION_SOURCE_AUDIT.md。GPU0不限25GiB授权保留，当前因其它任务占用使用4–7。

## 2026-10-08 14:19：AnyFlow96完整评测退步，既定128续训运行

GPU4–7完成68→96新增28次更新，含加载/前后验证1491.87秒、逐更新wall合计1327.22秒；allocated峰值40504.14–40505.35MiB/卡。四replica bank/Adam/logical-CPU-CUDA RNG哈希一致，原visual/time冻结、四组bank都更新，真实step68恢复全部通过。固定0/16/32/64/68/96验证sigma/r/类型/权重一致：A endpoint64→96为61.048→35.395、D19.991→18.416，diffusion/general-map raw也略降；actual GPU noise hash未记录，不能以scaled total下降作效果验收。

同一96 checkpoint的A/D39f、8 steps/chunk都完成：A−0.483647、D−0.729654、分离度0.246007，低于64的0.623096，A仍方向错误。两条全部0–38帧及Original/64/96在12/24/30/38帧对应画面已静态查看。A保留人物/车库但约20帧后模糊、透明和重影仍在，无明确视觉修复；D后段比64明显恶化，约22帧起叠影、末段严重雾化。静态全帧复核非实时播放。说明clean-history内部拟合改善不等于generated-history rollout改善；最新checkpoint不能默认当最好。

完整Original/64/96三列、A/D两行视频已归档submission/reports/stage1_anyflow/05_runtime/parallel_resume68_to128/original_anyflow64_anyflow96_8step_AD.mp4；39f/24fps/H264/yuv420p/faststart完整解码与第30帧标签检查通过，保留所有失败后段，不替换meeting。STEP96_RESULTS.md包含time/GPU/CPU KV/forwards/MAD/boundary/动作表与限制。两边conditioning逐张量相同；每条24 noisy+3 commits、CPU KV6484.13MiB，单次共享主机时间不称公平speedup。96只评测8步，未冒用64次4步结果。新runtime正确记录trainshift12。

原控制器2446816按既定预算已从step96继续128，torchrun2798817；本快照113/128，真实/proc均存活。实际train_128/step_96与源96的四套adapter、Adam、teacher/update history、固定配置、logical/CPU/CUDA RNG逐项完全一致（gpu_resume96_preupdate.json）。只执行这次已定的128上限及4/8评测，不自动增加预算。64次保留为当前较好的参考，任何新方案须等这条受控曲线完整后再决策。Stage1效果未过gate，Stage2暂缓。

## 2026-10-08 13:45：继续四卡96训练；64次分时段诊断确认动作尚未恢复

已重新核查控制器2446816与torchrun2447766的真实/proc状态，均存活且非zombie。当前GPU4–7训练已完成86次更新，step80的完整trainer_state已落盘；仍在向96推进，没有96视频或验收结果，不重启任务。原有到96评测/最多128的固定预算保持。

复用原Farneback实现，对Original、shift12 AnyFlow16/32/64、shift2.22 AnyFlow64及匹配FM32的全部12条已完成8-step A/D视频按RGB[0,17)、[17,34)、[34,39)分段，重算全片mean与原值在1e−12内一致。shift12 A在32→64：首段−1.205984→−0.710155，中段−1.345781→−0.673391，末段−1.205748→+0.015203；对应A−D分段0.288056→0.446936、0.083266→0.532387、0.068286→1.181313。提升并非只有最后几帧，但前两段A仍负，末段仅4个transition且接近零，不能称动作方向恢复。RGB分段有temporal-VAE时域影响，不是独立latent chunk隔离实验。跨边界transition单列；累计光流图则保留全部transition。

结果/逐transition/视频哈希和已查看的独立累计光流图归档submission/reports/stage1_anyflow/03_anyflow_trials/duration_response/RESULTS.md。使用已有base Python中的Matplotlib绘图，没有修改运行中的h3world环境；没有新增metric、改变gate或训练协议。已经为接下来的实际96/128完整视频准备只读contact-sheet整理入口，未声称预备脚本就是评测结果。Stage1仍未通过，Stage2暂缓。

## 2026-10-08 13:30：真实四卡64→68通过；从step68转到GPU4–7继续96

GPU3–6完成真实33B的4次新增optimizer更新，含加载/前后验证327.30秒，逐更新为62.97/64.11/25.13/23.80秒（chunk2/2/0/0，不能视为各chunk充分吞吐benchmark）。四卡allocated峰值40503.25–40504.67MiB，约39.55GiB/卡，无OOM。实际保存的预更新checkpoint与源64的四套adapter、Adam、teacher/update history、logical/CPU/CUDA RNG和固定配置全部相等；固定验证8个raw loss也相等。最后四replica的参数、Adam、三种RNG哈希完全一致；原visual/time冻结、四组bank均更新。源training.json和恢复证据归档submission/reports/stage1_anyflow/05_runtime/parallel_duration128/TRAIN68_RESULTS.md。这是训练执行验证，非视频验收，未评测step68。

原控制器2337864成功保存step68后，在启动96前发现GPU3新增其它进程占1811MiB，idle guard拒绝并停止；保留其failed状态（原因为硬件占用），不混淆已成功的68训练。没有启动96或重复更新，也未打断其它进程。

13:29:21已建立独立接续outputs/2026-10-08-13/stage1_parallel_resume68_to128/，启动前GPU4/5/6/7均3MiB。控制器2446816、torchrun2447766，当前running、resume68，直接继续96；988项冻结输入复核通过，runtime通过symlink重用原测试版本，仅换physical GPU和已验证起点。96 A/D8通过数字gate则补4并停待视觉等检查，否则最多128评测4/8，无超128自动训练。新报告submission/reports/stage1_anyflow/05_runtime/parallel_resume68_to128/README.md。GPU0不限25GiB授权保留，但现在确有其它任务；使用空闲卡避让。当前Stage1实际画质/action gate仍未通过，Stage2暂缓。

13:32补记：GPU4–7接续已实际完成第69次更新，控制器2446816/torchrun2447766存活。新train_96/step_68与来源的四套adapter、Adam、teacher/update history、固定配置、logical/CPU/CUDA RNG逐项一致；证据gpu_resume68_preupdate.json和current_snapshot.json已归档parallel_resume68_to128。当前在向96推进，不表示96已完成或Stage1效果通过。

## 2026-10-08 13:20：两组64完整评测归档；shift12动作分离开始增加，四卡有限续训启动

GPU0原shift2.22总64队列已全部完成：4步A−1.000683、D−1.153995、分离度0.153312；8步A−1.072072、D−1.287603、分离度0.215531。四条全39帧及Original/16/64同帧复核完成，4步约22帧后严重雾化，8步人物/车库保留但模糊/透明感，动作未恢复。完整诊断和指标在submission/reports/stage1_anyflow/03_anyflow_trials/full_history_duration64/FINAL_RESULTS.md。

GPU1 train shift12总64及全部评测也正常结束：4步A−0.903425、D−1.052966、分离度0.149541；8步A−0.589203、D−1.212299、分离度0.623096。8步16→32→64为0.179888→0.172691→0.623096，开始明显增加，但A仍错且没有明确视觉修复。四条全0–38帧contact sheet和Original/16/64同帧已静态复核；4步约20–22帧后重影雾化，8步后段模糊/透明。完整Original/4/8与Original/16/64的4/8学习曲线视频已归档，H264/yuv420p/24fps完整解码及第30帧标签检查通过。入口submission/reports/stage1_anyflow/03_anyflow_trials/shift12_duration64/FINAL_RESULTS.md。静态复核不是实时播放，未替换meeting主视频。

shift12新增32次含准备/验证5108.91秒、allocated40667.62MiB；endpoint32→64 A59.536766→61.048344变差，D22.905317→19.990650改善。完整固定0/16/32/64验证已归档，实际sigma/r/类型/权重一致，原日志未记录实际GPU noise哈希。AF64比FM32训练更多，不把0.623对0.414当AnyFlow优势证据。旧cached.json误记trainshift2.22的更正说明保留，真实training/checkpoint为12。

基于shift12开始移动的8步学习曲线，只增加一次有限64-update预算：保持loss/data/global batch4/LR/架构/anchor/噪声，使用已通过CPU多步恢复和33B只读等价检查的四卡同样本并行候选。先64→68核查真实恢复和全部replica weights/Adam/RNG，再96评测A/D8；数字gate通过则补4并停待视觉验收，否则最多128并评测4/8，无自动超128。不是SP，四张卡各自有完整33B副本；未使用diagonal shortcut，未改主trainer。

准备GPU0/3/4/5时，GPU0突然出现其它r2_real_models.py进程2253489，占约1.8GiB且高利用率，启动guard拒绝，未启动或打断任何任务。保留原输入清单，改用确认空闲GPU3–6，972项输入重新冻结。控制器2337864、torchrun2338965在13:19:51实际启动，训练state为running、resume64；真实optimizer结果仍待完成。GPU0不限25GiB的用户授权保留，独占配置reserve6，既有allocated实测约39.7GiB。新源目录outputs/2026-10-08-13/stage1_parallel_duration128/，报告入口submission/reports/stage1_anyflow/05_runtime/parallel_duration128/README.md。Stage1短片gate未过，Stage2暂缓。

## 2026-10-08 12:46：AnyFlow64的A/D4-step均未通过

D4已完成：flow−1.153995；A4为−1.000683，A−D=0.153312，A方向仍错误。两动作的全39帧contact sheet与Original/16/64同帧复核均显示约22帧后重影和严重雾化。Original/16/64三列、A/D两行的完整4-step诊断视频已生成并完整解码、检查标签：submission/reports/stage1_anyflow/03_anyflow_trials/full_history_duration64/original_anyflow16_anyflow64_4step_AD.mp4。属于静态完整帧复核，非真人实时播放。GPU0继续A8/D8；GPU1 shift12继续总64。所有训练/推理协议保持不变，Stage1未通过，Stage2暂缓。

## 2026-10-08 12:41：AnyFlow64首条A4仍失败；同batch四卡计算/恢复链路已验证

GPU0 shift2.22总64次训练完成，新增32次含准备/前后验证5241.37秒、GPU allocated峰值40668.18MiB。相同validation时间/权重下，endpoint32→64 A49.928→47.883、D24.234→18.441，diffusion raw也下降；不能用此代替视频验收。首条39f A4已完成，flow−1.000683；全39帧与Original/16/64对应帧复核显示约22帧起重影、后段严重雾化，A方向仍错。报告与原视频：submission/reports/stage1_anyflow/03_anyflow_trials/full_history_duration64/TRAIN64_RESULTS.md。D4/A8/D8继续；GPU1 shift12仍在总64训练。

独立diagonal省算力探针已正常结束：CPU20个小H3 cases loss/gradient/RNG/cache逐bit一致；真实33B同批损失一致，候选梯度差与重复原始反传差均约1.6e−4。当前forward16→10，本批约238→207秒（−13.19%），显存基本不变；单批不是正式吞吐benchmark。未合入主trainer，详见reports/stage1_anyflow/04_numerical_checks/diagonal_compute_probe/RESULTS.md。

进一步实现了隔离的四卡同样本并行候选：global logical batch仍4，每卡一个sample，原AnyFlow每样本4次forward保持不变；共享两个FM raw reference做原adaptive scale，loss已/4后梯度SUM。真实GPU3–6 step32 A/chunk2只读探针完成：四卡最慢82.27秒，对应串行225.98秒，约2.75×单批wall比（不同GPU资源、共享主机，不能当完整训练或推理加速）。实际noise、loss、RNG相同，四卡归约梯度hash相同；相对串行梯度cosine0.9999704、差范数1.738e−4，接近已有CUDA重复性波动；没有optimizer update。

四进程CPU六case以及集成trainer的串行4、并行4、并行2+恢复4、旧串行step2→并行4均完成。并行/串行bank最大差7.45e−9；并行连续与恢复的weights/Adam/RNG逐bit相等，四replica状态一致。rank0唯一写checkpoint的training_runtime及310项代码/测试收据已冻结；尚未启动真实33B多卡optimizer训练、未改主代码或GPU0/1实时runtime。完整证据在submission/reports/stage1_anyflow/05_runtime/sample_parallel_probe/RESULTS.md。仍先收齐既有64次视频，再决定更大训练预算；Stage1实际画质/action gate未过，Stage2暂缓。

## 2026-10-08 12:10：匹配FM32完整评测归档，未验证出AnyFlow少步收益

GPU3的FM32 A/D39f全部4/8-step完成：4步A=−1.058372、D=−1.298657、分离度0.240284；8步A=−0.943292、D=−1.356802、分离度0.413510。对应shift12 AnyFlow32为−0.019565/0.172691，两者A均错误。FM四条全39帧contact sheet及Original/AnyFlow同帧复核完成：4-step FM人物/停车场明显更完整，仍模糊/透明；AnyFlow4约20帧后严重雾化。8-step两者保留主体，但模糊、后段透明和动作偏差仍在，未见明确AnyFlow优势。静态完整帧复核不等同真人实时播放。

两条Original/FM32/AnyFlow32三列A/D诊断视频已归档submission/reports/stage1_anyflow/03_anyflow_trials/fm_full_history_shift12_32/FINAL_RESULTS.md，H264/yuv420p/24fps、完整39帧解码与展示标签检查通过。报告含时间/GPU/CPU KV/forwards/MAD/boundary、真实初始化与32更新匹配审计；相同更新数不是等算力，单次并行时间不宣称speedup。不替换meeting视频。

GPU0/1继续各自的AnyFlow总64；GPU0当前实占约42GiB，不限25GiB，无OOM。利用空闲GPU4启动独立只读探针（控制器1179085、子进程1179158）：检查r=t时省去三次零系数有限差分前向的等价性；CPU20个真实小H3 cases已确认loss/gradient/RNG/cache精确一致，真实33B结果尚待完成。331项输入冻结；没有修改主trainer、运行中runtime或任何checkpoint，没有optimizer update，不属于画质改善。Stage1未验收，Stage2暂缓。

## 2026-10-08 11:48：首条匹配FM32/A4画质优于AnyFlow32/A4，但动作仍失败

GPU3首条FM32 A/39f/4 steps per chunk完成：flow=−1.058372，A仍错误。全部39帧contact sheet及Original/AnyFlow32/FM32在12/24/30/38帧的对应画面已检查：FM仍有模糊/透明感，但人物与车库明显比AnyFlow4完整，后者约20帧后严重雾化。这条严格匹配初始化/容量/数据与sigma/更新数的结果不支持AnyFlow32的4步画质优势；不能扩展为尚未完成的D4/A8/D8结论。静态完整帧复核不等于真人实时播放。证据与原视频在submission/reports/stage1_anyflow/03_anyflow_trials/fm_full_history_shift12_32/GPU_RESULTS.md。

GPU3 D4正在生成（子进程991460）；GPU0/1继续各自总64训练，模型数学和训练协议未改，Stage1效果未验收，Stage2暂缓。

## 2026-10-08 11:44：匹配FM32训练完成，实际对照审计通过，视频评测开始

GPU3 full-history/train-shift12/FM32训练与验证结束：含准备/验证3215.65秒，allocated峰值41251.94MiB。真实初始adapter/三种RNG与AnyFlow匹配；两者32次action/chunk和全部128个current sigma逐项一致，共有训练配置与43,237,376参数一致。FM的r=t是其目标定义，不应和AnyFlow的off-diagonal r相等。实际训练forward计数：FM current128、AnyFlow512；两者各30 clean commits和120带梯度历史重建，不含验证/反向checkpoint重算。同更新数不是等算力。

公共每动作前两个r=t验证点，FM residual略低，但不能据此比较画质或声称显著收益。GPU3已开始FM step32 A4视频（子进程961255），后续D4/A8/D8按原队列继续；实际完整审计见submission/reports/stage1_anyflow/03_anyflow_trials/fm_full_history_shift12_32/GPU_RESULTS.md。GPU0/1的AnyFlow64继续，Stage1仍未验收，Stage2暂缓。

## 2026-10-08 11:40：shift12 AnyFlow32全部4/8步短片仍失败，双分支继续64

GPU1的full-history/train-shift12总32次训练完成，新增16次含准备/验证3012.30秒，allocated峰值40667.85MiB。相同noise公共endpoint A54.556→59.537变差、D24.329→22.905变好，其它raw residual变化很小。全32次更新的action/chunk与128个实际sigma/r对预先计划全部一致；未比较actual GPU noise hash，不把预测hash当实际采样证据。

同一step32的A/D39f评测全部完成：4 steps/chunk（GPU4补评测）A=−1.216895、D=−1.197330、分离度−0.019565；8 steps/chunk（GPU1）A=−1.305473、D=−1.478164、分离度0.172691。A均方向错误。四条全0–38帧contact sheet及Original/16/32的12/24/30/38帧对照已复核：4步约20帧后严重雾化/重影，8步主体保留但模糊、D后段有透明感；相比16次未见明确改善。静态完整帧复核不是实时播放评审。

完整Original/4-step/8-step、A/D两行H264/yuv420p/24fps诊断视频已归档submission/reports/stage1_anyflow/03_anyflow_trials/shift12_duration64/shift12_anyflow32_AD_diagnostic.mp4；STEP32_RESULTS.md含E2E/GPU/CPU KV/forwards/MAD/boundary与动作表。所有conditioning与原始H3逐张量相同；单次并行运行时间不作speedup比较。GPU4控制器正常完成退出，未覆盖meeting视频。

GPU1控制器334580已按原计划从step32继续总64；真实回载四套adapter、Adam、teacher/更新历史和CPU/CUDA/logical RNG逐项相同，见gpu_resume32_audit.json。GPU0控制器136179继续shift2.22的总64（快照43/64）；GPU1快照32/64；GPU3匹配FM32仍在训练（30/32），不提前宣称FM/AnyFlow优劣。匹配训练32的真实sigma/前向计数和三列评测整理入口已准备，尚未执行完成结果审计。本轮未改模型/训练数学，Stage1未通过，Stage2暂缓。

## 2026-10-08 11:22：补齐32次训练的AnyFlow/FM少步公平对照

检查发现GPU1的shift12 step32原计划只评测8 steps/chunk，而GPU3匹配FM32评测4/8。已利用空闲GPU4启动只读补评测控制器720483，outputs/2026-10-08-11/stage1_shift12_step32_4step/。它等待GPU1训练完整结束后读取同一step32，生成A/D39f、4 steps/chunk（12 noisy+3 commits）；不重新训练、不改anchor/action/noise/网格或模型结构。330项输入与控制器冻结，启动前GPU4占用0MiB，运行前再次检查；源训练、checkpoint与conditioning均有协议检查。

11:20实际进程核查：GPU0总64已到35次，GPU1总32已到30次，GPU3/FM32已到18次；均存活、日志继续推进。GPU4控制器存活等待训练完成，目前没有补评测的视频结果。匹配32次的三列视频整理入口已准备，会拒绝不完整评测；不把预备脚本当结果。目的仍是验证Stage1 AnyFlow是否在真实动作和画面上有效，Stage2暂缓。

## 2026-10-08 11:11：GPU0不限制25GiB；full-history32评测失败，64续训与匹配FM对照运行

GPU0独占继续reserve6，真实32次训练已完成：新增16次含准备/验证3024.73秒，allocated峰值40668.66MiB（39.72GiB），无OOM。初始化→16→32的公共noise/sigma验证已归档；A endpoint 59.625→61.344→49.928，D 22.805→22.865→24.234，不能用weighted total下降宣称视频改善。

32次A/D完整39f、8 steps/chunk均完成：A=−1.309516，D=−1.497935，A−D=0.188419（16次0.182509），A仍错误。两条0–38帧contact sheet和Original/16/32同帧检查完成：人物/车库主体保留，但模糊/重影、D后段透明感仍在，没有明确改善。这是静态全帧复核，非真人实时播放。可播放三列完整视频与E2E/GPU/CPU KV/forwards/MAD/boundary表见submission/reports/stage1_anyflow/03_anyflow_trials/full_history_duration64/STEP32_RESULTS.md，视频已完整解码H264/yuv420p/24fps。

GPU0控制器136179已按既定gate规则从step32精确续到总64，真实GPU回载的四套adapter/Adam/teacher与更新历史/三种RNG全部逐项相等（gpu_resume32_audit.json）；GPU1控制器334580继续自己的shift12总32训练（本快照27/32）；GPU3控制器515452运行匹配full-history/shift12/全覆盖rank8/FM32（本快照12/32）。三者都在运行。FM实际初始三套adapter、CPU/CUDA/logical RNG、公共diagonal验证与shift12 AnyFlow一致；预先CPU计划中的noise hash不是实际GPU逐样本hash证据。最终应将FM32与shift12 AnyFlow32比较，同更新数仍不等计算预算。

修正主源码和submission中benchmark的一处metadata记录：anyflow_training_shift现在优先读取training_timestep_shift，而非误用inference flow_shift。公式/训练/采样均未改；三个运行中的冻结runtime不改写，因此旧shift12 cached.json仍错误显示2.22，应以checkpoint/setup config及training.json中的12为准。三个legacy/None/explicit12元数据分支实际检查通过；此前84项完整测试收据保留，本轮未重跑完整套件。根README同时澄清旧FM、新AnyFlow和Stage2-lite的区别。Stage1效果未验收，Stage2暂缓。

## 2026-10-08 10:41：shift12/16全部评测失败，teacher-history定位完成；两组训练量对照运行

训练shift12的full-history AnyFlow16完整结束：含准备/验证2516.05秒，allocated41310.95MiB。真实初始四套adapter/三种RNG、公共验证和全部64个sample sigma/r审计通过。公共endpoint A59.625→54.556下降、D22.805→24.329上升，diffusion变化很小，不能用weighted total宣称改善。

完整39f自生成历史：4-step A=−1.227654、D=−1.232781、A−D=0.005126；8-step A=−1.312275、D=−1.492164、A−D=0.179888。四条全0–38帧contact sheet与Original/FM/shift2.22同帧复核：4-step后半段仍严重雾化，8-step主体保留但A方向错，没有明确shift12收益。完整mp4与指标见submission/reports/stage1_anyflow/03_anyflow_trials/training_shift12/FINAL_RESULTS.md。

GPU3同一shift12 step16的clean-history A/D8诊断也完成：A=−0.211312、D=−0.875811、分离度0.664499。人物/车库较完整，但teacher状态在17/34帧附近引入跳变，boundary MAD A14.823/D9.286；不是free-running改善，也未通过A正门槛。两动作的首5个latent在clean/generated运行中逐元素相同，max_abs=0，之后才分歧。A第0块[0,17) flow−1.207126，原始H3对应+0.083398；说明偏离在尚无生成历史时已出现，不能全归因于generated-history累积误差。teacher历史本身含action后的场景状态，后续正flow不能独立证明当前action binding。完整三列Original/generated/teacher视频与分段证据见submission/reports/stage1_anyflow/03_anyflow_trials/shift12_history_diagnostic/FINAL_RESULTS.md。

现在只做有限训练量对照，不加模块：GPU0 shift2.22从原step16续到32→最多64（控制器136179，子进程136859，已完成26/32）；GPU1 shift12源评测全部失败后也已从自己的step16续到32→最多64（控制器334580，子进程391767，快照完成16/32）。各自四套权重、Adam、teacher/更新历史、CPU/CUDA/logical RNG真实回载均逐项一致。两个step16之间三种RNG状态也相同，但已训练权重/Adam不同，绝不互换。训练/验证/推理shift各自固定，GPU0/1均reserve6，不限25GiB。32数字gate通过时停训等待视觉检查，否则继续到最多64；不自动判定Stage1通过。

本轮只新增诊断/队列与报告，主模型代码未改动，84项已完成检查不重复运行。会议主视频未替换，Stage1效果仍未验收，Stage2暂缓。

## 2026-10-08 10:09：full-history16全部失败已归档；GPU0同配置续训启动

完整A/D39f评测：4 steps/chunk A=−1.248505、D=−1.224589、A−D=−0.023916；8 steps/chunk A=−1.304568、D=−1.487077、A−D=0.182509。A均方向错误。四条全39帧contact sheet及Original/FM/detached同帧检查完成：4-step约20帧后严重雾化/ghosting，8-step人物/车库主体保留但无明确收益。完整39帧可播放诊断已放submission/reports/stage1_anyflow/03_anyflow_trials/full_history_candidate/full_history16_AD_diagnostic.mp4，指标与限制在FINAL_RESULTS.md；没有替换meeting。

GPU0按用户授权独占且不限制25GiB，继续reserve6；此前真实训练allocated峰值41310.60MiB（40.34GiB）。有限训练量对照已实际启动：控制器136179、子进程136859，outputs/2026-10-08-10/stage1_full_history_duration64/。从原full-history16的step16恢复完整权重/Adam/CPU-CUDA-logical RNG，先到32并评测A/D8；数字gate未过再到最多64并评测A/D4/8。仅改总更新数，训练/validation/inference shift继续2.22，anchor/架构/LR/数据/噪声不变；322项冻结依赖及保存state/config/hash预检查通过。启动前GPU0实占0MiB，未打断其它进程。真实GPU回载后step16快照核验通过：四套adapter、Adam、更新记录、teacher身份及CPU/CUDA/logical RNG全部与来源完全一致。

GPU1原shift12 fresh16对照继续运行（该快照11/16）；尚无视频结果。两支checkpoint与归因独立。16次更新的微小loss变化不足以证明训练充分，所以增加有限训练预算检查学习曲线；这不保证效果。Stage1尚未验收，Stage2继续暂缓。

## 2026-10-08 09:58：full-history16首条A4仍严重雾化；采样入口合入后84项检查通过

step16 A/39f/4 steps/chunk已完成，flow=-1.248505，A方向仍错。0–38帧全部静态复核与Original/FM16/detached16同帧对比显示约20帧后重影、雾化、人物/车库结构丢失；没有明确修复detached4的问题，FM4仍更完整。证据见submission/reports/stage1_anyflow/03_anyflow_trials/full_history_candidate/GPU_RESULTS.md。当前只完成此条step16视频，D4/A8/D8继续。

训练与validation时间分布的可选独立参数已从已验证候选合入H3-World与submission，默认旧数学保留；完整主源码84 passed（7.84秒）。GPU1仍用冻结runtime进行shift12训练，实测初始化/三种RNG/公共validation全部匹配；没有更改GPU0评测的推理shift、anchor、steps或其它协议。冻结CPU审计receipt的一项validation_before占位已另文明确不算一致性证据，其最终权重/Adam/RNG/逐sample/后验证比较均实际通过。

Stage1实际效果仍未通过，不将工程合入作为最终验收，Stage2暂缓。

## 2026-10-08 09:52：full-history AnyFlow16训练完成，公共验证无明确端点收益，开始step16视频

GPU0完成16次更新，含准备/验证2863.15秒，allocated峰值41310.60MiB（约40.34GiB）。四组LoRA更新，旧visual/time冻结；实际初始化/三种RNG/公共验证samples和16次action/chunk/sigma审计全部通过。额外带图历史前向合计56次。

同noise的公共验证：A weighted0.118051555→0.117942682、endpoint59.624653→61.344250；D weighted0.170266438→0.170181701、endpoint22.804770→22.865307。weighted下降不到0.1%，端点无明确改善，不能先宣称视频有效。完整数值见submission/reports/stage1_anyflow/03_anyflow_trials/full_history_candidate/GPU_RESULTS.md。

已自动开始step16 A/native4视频（PID4125388）；D4和A/D8继续串行。GPU1 training_shift12仍运行，实际初始weights/RNG/公共验证与原分支匹配，推理不改shift2.22。当前Stage1视觉/action gate未通过，Stage2暂缓。

## 2026-10-08 09:46：full-history step04 A/D未通过；官方训练shift12受控对照已启动

完整step04 A/D39f、8 steps/chunk均完成：A=-1.308945，D=-1.494894，A-D=0.185948。人物/车库主体保留但模糊，A方向仍错，相对detached16的8-step未见明显改善。两条全39帧静态检查、完整三列并排mp4和指标已归档submission/reports/stage1_anyflow/03_anyflow_trials/full_history_step04/FINAL_RESULTS.md；视频明确标注4与16 optimizer updates不同，不作等训练量消融结论。

发现训练分布与官方的一个差异：当前train shift2.22，官方H3 Stage1为12。按真实logical RNG和noise形状重放，本轮16次中A的8个endpoint样本无sigma≥0.9，D仅1个；改12则分别2/3个。A的diffusion/general-map仍有高噪声样本，不能说完全无监督，也尚未证明这是失败原因。覆盖证据见training_shift12/COVERAGE.md。

隔离实现training/validation/inference shift分离，仅训练采样及Gaussian权重改12，公共validation和inference继续2.22，其余full-history/初始化/rank8/FP32/LR/data/action/anchor/16updates均固定。17项CPU检查通过，默认新旧full4的参数/Adam/RNG/loss/gradient逐bit一致；shift12连续4与2+恢复4也一致，改变train shift不能伪装exact resume。主trainer尚未合入这些新参数。

GPU1新有限队列PID4006869、训练子进程4007285已运行（321项冻结输入）；真实step00四套adapter、CPU/CUDA/logical RNG和公共验证samples全部匹配，已观察训练sigma/r符合shift12计划。GPU0原队列3677838/3707254继续，已完成15/16更新，随后原计划A/D4/8评测。两条路线尚无通过Stage1的结果，不扩124f、不做Stage2。

## 2026-10-08 09:26：full-history step04首条A8仍未过动作gate

GPU1完成step04 A/39f/8 steps/chunk，flow=-1.308945，A仍朝错误方向。完整39帧静态复核显示人物/车库主体保留但模糊，与detached AnyFlow16的8-step没有明显改善；所有conditioning与原始H3逐项相同，真实bank/time/step/full-history元数据正确。耗时230.21秒、allocated38984.16MiB、CPU KV6484.13MiB、24 noisy+3 commits；与GPU0训练并行，不作speedup比较。

对照图与逐帧证据在submission/reports/stage1_anyflow/03_anyflow_trials/full_history_step04/VISUAL_REVIEW.md。D仍在生成；GPU0主训练已完成8/16，未更改配置。当前只得出step04 A尚未改善，不能提前判定16次结果。Stage1未验收，Stage2暂缓。

## 2026-10-08 09:20：完整历史梯度训练覆盖A/D三块，GPU1并行检查step04效果

GPU0独占训练已完成6/16次更新，A/D的chunk0/1/2均覆盖；history前向次数分别0/4/8，chunk2 CPU raw KV10806.88MiB。GPU0实占约42.6GiB，无OOM。保存的step04逐张量检查：原visual/action/time冻结，208个LoRA wrapper均有更新，history计数正确。实际step00四套adapter、CPU/CUDA/logical RNG及固定noise验证sample与detached参照完全一致。

同时发现独立GPU轨迹并非逐bit相同：chunk0尚无历史时梯度已出现微差。GPU1同模型/同noise重复反传（不更新参数）确认前向sample记录相同，detached两次梯度cosine0.9999106、差范数1.2636e-5；full-no-history为0.9999032、1.3149e-5，处于相近量级。它确认实际GPU反传重复性误差，具体kernel未定位；不将微小训练曲线差当因果效果证据。完整日志见submission/reports/stage1_anyflow/04_numerical_checks/gradient_repeatability/RESULTS.md。

GPU1现并行评测full-history step04的A/D 39f、8 steps/chunk（控制器3859336，A子进程3859754）；GPU0原16次训练及后续step16四条评测继续，实时源文件和冻结输入不变。step04视频存于原实验eval/anyflow/step_04，额外控制器在outputs/2026-10-08-09/stage1_full_history_step04_eval/。并行运行的时间不作公平speedup比较。尚无full-history视频结果，Stage1未验收，Stage2继续暂缓。

## 2026-10-08 09:07：FM/AnyFlow完整诊断归档，历史梯度可选入口合入

完整FM A4/D4/A8/D8的39帧contact sheet与Original/AnyFlow同帧对照已复核：FM4仍模糊/透明，但明显好于AnyFlow4后半段的重影雾化；8-step两者保留主体场景，均未恢复A方向。两条完整Original/FM/AnyFlow三列、A/D两行的H264诊断mp4已归档submission/reports/stage1_anyflow/03_anyflow_trials/fullscope_fm_control/FINAL_RESULTS.md，标记NOT PASSED；会议主视频未替换。

真实探针验证后的--history-gradient-mode full作为可选路径合入H3-World与submission，默认detached不变，主源码CPU完整77 passed（6.25秒）。实时训练runtime及314项冻结输入未改动。更改gradient mode不能伪装成exact resume；生成时仍使用原cache语义。fresh16训练已完成chunk0的A/D两次更新，后续带历史的chunk继续执行，当前尚无full-history视频。Stage1未验收，Stage2暂缓。

## 2026-10-08 09:02：GPU0完整历史梯度探针通过，fresh AnyFlow16已启动

同容量FM16四条39f评测已完成：4-step A=-1.076402、D=-1.290065、A-D=0.213663；8-step A=-1.135284、D=-1.456185、A-D=0.320901。A均为负，未过动作gate。完整FM/AnyFlow同帧视觉复核与诊断视频正在整理；不把单一光流或MAD当画质结论。

真实33B A/chunk2的detached/full-history探针已完成：四类sample raw loss逐项完全相同；梯度cosine=0.911813、差范数0.000651273，四组LoRA均有非零有限梯度。full模式额外8次带图历史前向，CPU raw KV峰值10806.88MiB；一次Adam更新后原visual/time/action不变。GPU allocated峰值40973.35MiB（约40.0GiB）、reserved43176MiB，无OOM，GPU0不再限25GiB。详见submission/reports/stage1_anyflow/04_numerical_checks/clean_history_gradient/GPU_RESULTS.md。

新控制器3677838已自动启动fresh16训练（子进程3707254），目录outputs/2026-10-08-08/stage1_anyflow39_full_history/。从原始初始化开始，不使用探针更新权重；唯一训练数学改动为history-gradient-mode full，继续同rank8/FP32/LR/data/noise/action/anchor协议。训练后核验初始化与采样序列，再评测A/D4/8 steps/chunk。历史梯度改变已证实，画质是否改善尚未证实。Stage1效果未验收，Stage2暂缓。

## 2026-10-08 08:43：首条同容量FM4画面优于AnyFlow4，但动作仍失败

首条FM16 A4完成，flow=-1.076402，A方向gate失败。同容量/初始化/训练量/FP32/causal协议下，0/6/12/18/24/30/38帧对照显示FM4人物与车库明显比AnyFlow4完整；仍有模糊，不能声称动作恢复。当前A4未显示AnyFlow少步画质收益，FM其余D4/A8/D8继续评测，不提前推广结论。证据见submission/reports/stage1_anyflow/03_anyflow_trials/fullscope_fm_control/FM16_A4_vs_AnyFlow_teacher.jpg及GPU_RESULTS.md。历史梯度probe等待控制器3542531仍待本队列完整退出；Stage1效果未验收。

## 2026-10-08 08:40：同容量FM16训练及真实初始化审计通过；历史梯度探针重新冻结排队

GPU0完成FM16训练，脚本含准备/验证1040.3秒，allocated峰值37970.6MiB。真实step00 visual/action/full-bank、CPU/CUDA/logical RNG与AnyFlow完全一致，16次训练action/chunk/sigma序列相同；四组LoRA更新且visual/action冻结。FM held-out A weighted0.126271→0.126175、D0.181725→0.181611，变化很小。当前A/D4/8视频评测已开始，不能先写FM效果优劣。见submission/reports/stage1_anyflow/03_anyflow_trials/fullscope_fm_control/GPU_RESULTS.md。

clean-history梯度候选在真实GPU启动前修正临时capture容器：每次前向后seal/clear，防止checkpoint backward重复复制CPU KV和通过容器保留自身计算图。修正后9项原专项及1项容器检查通过；新连续4次、从旧step2恢复到4次都与旧full4的权重/Adam/RNG/loss/gradient完全相同。前向/梯度数学未变，最终CPU fixture的history_protocol标签也已正确。

仅停止旧的等待控制器3477486，未中断FM GPU任务；新探针等待控制器PID3542531已启动，状态waiting_for_fullscope_fm16，冻结311项源代码/输入。完整FM四条评测退出后，只执行一次A/chunk2真实梯度、显存及单次更新探针；仍无full-history分支的33B或视频结果。Stage1未验收，Stage2暂缓。

## 2026-10-08 08:32：全覆盖AnyFlow四条视频均失败；同容量FM运行，clean-history梯度候选通过CPU检查

全覆盖rank8/native-FP32/16次更新的全部39f评测完成：4-step A=-1.198536、D=-1.242114、A-D=0.043578；8-step A=-1.305347、D=-1.496471、A-D=0.191124。两组A都不符合原始方向。4-step后段严重重影/雾化；8-step人物/车库较完整但仍模糊，与native tail16没有明确收益。完整39帧诊断mp4（Original/4/8，A/D两行）、同帧对照和数值已归档submission/reports/stage1_anyflow/03_anyflow_trials/fullscope_candidate/FINAL_RESULTS.md；不替换meeting，未通过Stage1 gate。

GPU0已自动转入相同容量/初始化/精度的FM16（当前12/16，PID3361205），后续A/D4/8按冻结协议串行运行。当前不能比较尚未完成的FM结果。

另核对官方Stage1的clean/noisy拼接计算图与本地detached历史KV差异，隔离实现--history-gradient-mode full：只给梯度prediction重建带计算图的clean history，目标前向仍no_grad；每块只读cache快照避免checkpoint反向重写历史。它保持现有cache前向语义，不是官方融合两流算子，未改变RGB/audio/action等其他差异，也没有证明是画质失败原因。

小H3 FM/AnyFlow、FP32/BF16、checkpoint/offload：前向和K/V逐元素相同，历史获得梯度、未来帧无梯度。参数数值差分0.00122190与full方向导数0.00121648吻合，detached为0.00020518；这是选定小模型方向，不是33B梯度缺失比例。完整套件72项通过后扩充FM专项，9项相关检查通过；full连续4次vs2+resume4的权重/Adam/RNG/loss/gradient完全相同。

新的有限探针队列PID3477486已确认存活，等待FM16四条评测完全退出后在GPU0/reserve6执行一次真实A/chunk2梯度、显存和单次更新检查。尚未执行33B full-history梯度，也无该分支视频。源目录outputs/2026-10-08-08/stage1_clean_history_gradient/；证据在submission/reports/stage1_anyflow/04_numerical_checks/clean_history_gradient/。候选未合入主源码，当前主源码仍是67项检查通过的detached默认路径。Stage2继续暂缓。

## 2026-10-08 08:03：全覆盖AnyFlow16完成，固定噪声验证未见明确收益，开始A/D视频

GPU0完成全50+2块rank8的总16次更新。step01→16脚本含准备/验证耗时1963.6秒，allocated峰值37386.2MiB；四组LoRA均更新，旧visual和时间MLP仍冻结。加上fresh1段，两段脚本共2415.2秒，包含重复启动/验证，不作连续训练速度推断。

同noise初始→16：A weighted 0.118052→0.118061，endpoint59.625→71.849；D weighted0.170266→0.170215，endpoint22.805→22.715。A端点变差、D变化很小，尚无明确验证收益。现已开始首条A/native4评测（PID3213786），完整4条视频仍待生成；同容量FM16继续等待。当前未通过Stage1动作/视觉验收，不能用训练完成替代效果通过。完整数值见submission/reports/stage1_anyflow/03_anyflow_trials/fullscope_candidate/GPU_RESULTS.md。

## 2026-10-08 07:55：全覆盖Stage1/FM入口合入主代码，67项检查通过

全50+2块Q/K/V/out/FFN LoRA、checkpoint/Adam/RNG恢复、FM与AnyFlow匹配评测入口已合入H3-World及submission。默认tail_qkv保留；全覆盖须显式--adapter-scope all_qkvo_ffn，推理须--stage1-lora。新增协议检查拒绝遗漏bank以及混用不同step/objective/precision/action/anchor；FM评测不再依赖AnyFlow模块。完整测试67 passed（6.59秒），此前真实小H3 FM连续4次vs2+resume已逐张量相同。主代码变更未触及两个实时实验runtime或311项冻结输入。

当前GPU0全覆盖AnyFlow完成14/16更新，尚无评测视频；同容量FM16队列继续等待。不能把工程合入和CPU测试称为画质/动作通过。Stage1 gate未通过，Stage2仍暂缓。复现入口见submission/REPRODUCE.md，证据见reports/stage1_anyflow/03_anyflow_trials/fullscope_fm_control/main_integration_tests.json。

## 2026-10-08 07:46：GPU0高显存全覆盖AnyFlow继续；同容量FM对照已排队

GPU0继续独占、reserve6，不限制25GiB；nvidia-smi实时占用约40GiB。当前全覆盖AnyFlow已完成9/16次更新，QKV/out/FFN/refiner梯度均有限；训练及视频评测尚未结束。

补齐同容量FP32/full-scope FM评测入口，避免将增加LoRA覆盖的作用误归因于AnyFlow。26项检查通过，FM真实小H3连续4次vs2+恢复4次的权重/Adam/RNG/逐sample指标一致。独立有限队列PID3099247已启动等待当前AnyFlow训练和四条视频全部完成，再串行FM16+A/D4/8。固定初始化、数据/噪声、训练参数量、LR和更新次数；目标时间条件与loss/sampler是对照差异，训练前向算力不同。源运行目录outputs/2026-10-08-07/stage1_fm39_fullscope_control/，证据镜像submission/reports/stage1_anyflow/03_anyflow_trials/fullscope_fm_control/。

当前仍无full-scope生成视频，不宣称画质改善；Stage1未过效果gate，Stage2继续暂缓。

## 2026-10-08 07:33：全覆盖rank8真实单次更新通过，已精确续训到16次

GPU0、reserve6完成全50主块+2refiner Q/K/V/out/FFN rank8的1次真实AnyFlow更新；43,237,376参数，update102.09秒，含准备/前后验证451.63秒，allocated峰值37,629.48MiB（36.75GiB），无OOM。QKV/out/FFN/refiner均有限非零梯度且实际更新，原visual adapter和时间MLP不变。峰值受驻留/offload状态影响，不把更低的单次allocated读数解释为新算法节省显存。

初始A/D固定noise8个样本与native旧初始化逐项相同。更新后A endpoint59.625→69.483、D22.805→21.893，weighted total略升；只证明硬件和梯度链路，不构成画质改善。当前尚无full-scope视频。

已从自身step01继续到总16次更新。独立真实GPU加载审计核验完整stage1_lora.pt、visual/time/action adapter、Adam、logical/CPU/CUDA RNG、更新历史和teacher身份全部一致。后续A/D4/8-step评测由有限队列完成；Stage1视觉/action验收仍未通过，Stage2暂缓。完整日志及限制见submission/reports/stage1_anyflow/03_anyflow_trials/fullscope_candidate/GPU_RESULTS.md。

## 2026-10-08 07:26：native-FP32完整评测失败；全覆盖rank8真实GPU测试启动

native-FP32/tail16总16次更新的A/D评测全部完成：4-step A=-1.251782、D=-1.212726、分离度-0.039056；8-step A=-1.333328、D=-1.478203、分离度0.144875。两组A方向均错。4-step后段严重雾化/重影；8-step人物/车库较完整，但与旧legacy8相比无明确视觉收益。完整39帧三列诊断mp4和抽帧在submission/reports/stage1_anyflow/04_numerical_checks/native_fp32_candidate/FINAL_RESULTS.md，标记未通过。汇总现含38条已完成视频、19组A/D。

当前GPU0已自动切换到全50主块+2refiner的Q/K/V/out/FFN rank8候选，trainable=43,237,376，单次真实更新链路运行中。新增零B LoRA的A/D固定noise前验证8个样本记录（包括raw loss和有限差分）与原生FP32初始化逐项完全相同，真实GPU初始函数审计通过；这不是视频改善或训练完成。显存/梯度组/冻结约束与1-update后的回载待该子进程结束检查。控制器仍只使用GPU0，reserve6，下一阶段为自身Adam/RNG恢复到16并评测A/D4/8。Stage1未通过，Stage2继续暂缓。

## 2026-10-08 07:10：native-FP32训练结束，首条step16 A4仍失败

native-FP32总16次更新完成；step01→16本次运行含准备/前后验证1436.1秒，allocated峰值38,652.9MiB。固定noise验证A endpoint raw74.741→55.934、D23.891→22.691；相对最初step00的59.625/22.805变化有限。QKV更新、time仍严格冻结。

首条训练后A/native4完整39帧已生成，flow=-1.251782，方向gate失败；约20帧起明显ghosting、后半段人物/车库严重雾化，未看到相对初始化的明确修复。输入逐张量与teacher相等，E2E226.71秒，GPU allocated39069.22MiB，CPU KV6484.13MiB。原始H3/FM16/native00/native16抽帧对照在submission/reports/stage1_anyflow/04_numerical_checks/native_fp32_candidate/native16_A4_vs_initial_FM_teacher.jpg。固定noise数值下降没有转化为这条视频的效果改善。

当前D/native4及A/D/native8继续串行评测，不能提前写成全部完成。后续full-scope rank8候选控制器仍存活等待；仅当前四条评测完整结束且两组数值gate均失败后才启动真实1-update/16-update测试。Stage1未验收，Stage2暂缓。

## 2026-10-08 07:04：native-FP32完成16次更新；全覆盖rank8候选准备完成

GPU0上的native-FP32 tail16已完成16次optimizer更新，正在后验证，4/8-step视频尚待完成，Stage1效果gate未通过。训练进程/等待控制器均核验存活，未重启现有任务。

准备了更接近官方Stage1可训练范围的独立候选：50个主块+2个refiner，Q/K/V独立rank8及out/FFN LoRA，共312个逻辑投影、43,237,376参数。原visual/action冻结，新增B零初始化；FP32/BF16实际小H3/offload/nonzero旧adapter初始输出与RNG逐bit保持，各组真实反传/回载/冻结检查通过。FP32参数+梯度+Adam约659.75MiB，实际GPU激活峰值仍须1次真实更新验证。alpha/rank从旧adapter的1/8改为新bank的1，因此是覆盖与参数化候选，不能把后续潜在收益严格归因于“只增加block数”。

38项集成及6项官方oracle通过；full-scope连续4次vs2+恢复4次的权重/Adam/RNG/loss/gradient逐bit一致，旧native/tail checkpoint通过新trainer续训也与原连续轨迹一致。scope/rank/alpha变化被精确恢复入口拒绝。代码保留在独立runtime，主训练源码未再次改变。

候选队列PID2666006已实际启动等待，不占GPU模型显存。只有当前native16四条评测全部完成且两组A/D数值gate均失败，才串行运行1次真实更新→自身精确续训16→A/D4/8评测；当前模型若通过数值gate则先等视觉审查。证据与限制见submission/reports/stage1_anyflow/03_anyflow_trials/fullscope_candidate/README.md。尚无full-scope GPU效果结果，不扩长视频、不开始Stage2。

## 2026-10-08 06:47：GPU0约38GiB运行，原生FP32单次更新完成，初始化仍失败

独立FP32输入诊断完成24项真实33B测量、8次逐bit回滚；耗时322.8秒，allocated峰值37,736.6MiB。与前一诊断重叠的16项legacy/boundary记录完全复现。结果混合：A/chunk2 endpoint146.207→67.016，但D的两个endpoint升高；没有导数真值或画质改善证据。已归档submission/reports/stage1_anyflow/04_numerical_checks/fp32_inputs/RESULTS.md。

native-FP32候选在独占GPU0、reserve6完成1次真实更新：update60.21秒，准备/前后验证合计308.71秒，allocated峰值38,556.3MiB（37.65GiB）。QKV实际更新、时间MLP严格冻结；A/D验证endpoint59.625→74.741、22.805→23.891，不能宣称改善。初始化step00的39f4-step A=-1.233794、D=-1.225326、A-D=-0.008468；推理allocated约39,069MiB，完整解码且输入逐张量公平。抽帧显示约20帧起重影、后段人物/场景雾化，与旧初始化无明确改善。

当前已从自身step01精确恢复Adam/RNG到总16次更新，快照完成5/16；真实加载审计确认QKV/time/action adapter、Adam、logical/CPU/CUDA RNG和teacher记录相同。后续4/8-step A/D由有限队列执行；训练后效果尚待验证，未扩124f或Stage2。当前legacy64次续训仍暂缓，未重启。

可选precision profile已合入主代码和submission，默认legacy保留；实时实验runtime/manifest未动。源测试41 passed，独立从base+导出patch重建模型/pipeline与源文件一致，17项集成及6项官方oracle检查通过。新提交补丁为code/diffsynth_native_fp32.patch；训练/推理必须一致声明--precision-profile h3_fp32，精度切换不是精确续训。汇总报告现含34条已完成视频、17个A/D pair；Stage1的动作与视觉gate仍未通过。

## 2026-10-08 06:24：legacy32仍失败，优先完整FP32输入与原生边界策略

GPU0高显存续训完成总32次更新，新增16次含准备/验证1316.6秒，allocated峰值38,619.2MiB（37.71GiB）。39f native8 A=-1.332148、D=-1.474533、A-D=0.142385，低于step16的0.229330，A仍方向错误；人物和车库尚可辨认，但后段模糊/重影没有明确修复。A验证weighted loss略降，endpoint raw反而146.207→186.948，D endpoint17.867→18.614。证据在submission/reports/stage1_anyflow/03_anyflow_trials/duration32_snapshot/。

基于这些结果与已确认的官方FP32扰动协议差异，06:21停止本实验自动64次分支（第33次更新尚未开始），保留完整step32/Adam/RNG及A/D评测。旧队列状态为completed32_deferred64；没有把计划中的64次写成已完成。随后重新安排等待控制器，FP32输入诊断已实际接管GPU0；其首条legacy数值复现上一轮，其他profiles运行中。停止/优先级调整记录在duration32_snapshot/precision_priority_receipt.json。

另外完成可训练/可评测的独立native-FP32副本：六组12个原始FP32权重恢复并记录哈希、FP32 clean/noise/有限差分输入、FP32时间混合和SiLU、BF16主block、持久offload dtype，以及精度/原生权重checkpoint校验。`--precision-profile h3_fp32`仅作用于新实验，旧legacy路径保留；精度改变不能假装是精确恢复。17项专项测试通过；小H3旧legacy checkpoint恢复与旧连续训练、以及FP32连续4次vs2+恢复到4次的权重/Adam/RNG/loss/gradient逐bit一致。CPU检查不是GPU或画质通过。

新候选位于outputs/2026-10-08-06/stage1_anyflow39_precision_training/，等待输入诊断完整退出后串行进行1次真实更新→step00 A/D4→保留自身Adam/RNG续训到16→step16 A/D4/8。fresh visual/action初始化保持不变，precision profile和FP32训练噪声舍入策略明确不同；未改变anchor、chunk、LoRA容量或action routing。当前尚无native-FP32的33B训练/视频结果。Stage1未通过，Stage2暂缓。

## 2026-10-08 05:59：冻结16-update的6条视频均未过gate；真实精度诊断完成，GPU0高显存续训中

冻结time-MLP分支的完整39f A/D评测已全部完成：native4 A=-1.218440、D=-1.221512、A-D=0.003072；native8 A=-1.250413、D=-1.479744、A-D=0.229330；uniform4 A=-2.632457、D=-2.310146、A-D=-0.322311。三组A方向均错误。native4/uniform4后段严重重影/雾化，native8人物和场景较完整但仍模糊/重影。所有6条视频完整解码、对应原始H3输入逐张量相等。冻结策略没有修复本轮画面/action。汇总在submission/reports/stage1_anyflow/03_anyflow_trials/frozen_time_snapshot/；合并报告现含30条视频、15个A/D pair，均不把解码完成当成质量通过。

GPU0的真实33B同状态计算精度诊断已完成24项测量和8次逐bit恢复检查：只切换time FP32或六组boundary FP32，固定旧clean KV、noisy state和有限差分方向。A/chunk2 endpoint raw146.207→59.420，但另外3个endpoint均上升；4个general-map raw小幅下降。结果混合，没有完整导数真值或画质改善证据。总计305.5秒，allocated峰值37,734.5MiB（36.85GiB），reserved峰值40,058MiB；reserve6实际可运行，无OOM。证据见submission/reports/stage1_anyflow/04_numerical_checks/precision_probe/RESULTS.md。

该诊断没有包含FP32输入扰动，也没有恢复原始F32权重值。进一步核对官方stage1.py/anyflow_loss.py确认其noisy/plus/minus到输入投影前都保留FP32；本地旧路径会先转BF16。新增完全隔离的runtime，在chunk_forward/_embed支持可选FP32输入。CPU真实小H3/offload测试：1+1e-4扰动在旧两条路径保留0/1920个原始值，新路径保留1920/1920；梯度/cache只读/逐bit回滚均通过，既有集成回归6 passed。这不代表真实训练所有扰动都丢失，也不是33B效果通过。真实输入精度诊断已在stage1_anyflow39_fp32_inputs/排队，等待当前有限32/64-update队列退出；原训练源码及共享DiffSynth未改变。

当前GPU0已从step16的Adam/RNG继续到总32次更新，截至本条已完成21次；reserve=6，最近完整更新约58–63秒。后续按既定gate决定是否继续64次。这些时间不是与旧GPU2/不同reserve的公平加速比。Stage1实际画质和动作gate仍失败，独立seed、切换动作、124f验收未满足，Stage2不启动。

## 2026-10-08 05:30：GPU0冻结时间MLP训练完成；首条A4评测失败，高显存队列继续

补充：frozen16 native4 A/D已全部完成，D=-1.221512，A-D=0.003072，两个视频后段都严重雾化；方向与视觉gate失败。native8/uniform4继续运行。

冻结time-MLP分支已完成总16次更新（GPU2完成1–8，GPU0精确恢复后完成9–16）。GPU0续训阶段含准备/前后验证用时933.1秒，PyTorch allocated峰值25,063.4MiB（约24.5GiB）；QKV实际变化、time参数严格不变。该数值不是整个1–16训练的合计耗时。step08→16相同noise验证weighted loss：A0.117044→0.116573，D0.169255→0.168760；A endpoint raw154.850→146.207，仍远未证明少步质量有效。

首条frozen16 A/native4已完整生成39帧，水平flow=-1.218440，A方向gate失败；与原始H3/旧trainable-time的同帧contact sheet显示约20帧开始严重重影/雾化，后段人物和车库仍丢失。冻结time参数没有修复这条视频，D/native4及native8/uniform4仍在评测；不提前推断尚未生成的结果。图片在outputs/2026-10-08-05/stage1_anyflow39_gpu0/report/frozen16_A4_vs_previous.jpg。

按用户允许GPU0独占使用的指示，后续保持reserve6（约38GiB模型驻留预算）。05:29在32/64-update续训前增加一个独立33B精度诊断：相同checkpoint、clean CPU KV、noisy state及有限差分方向，仅比较legacy/time_fp32/boundary_fp32计算。小型实际H3+offload wrapper的cache只读、反传和逐bit恢复已通过；真实33B诊断待6条视频评测完成后运行。它不恢复原生FP32权重、不改现有训练runtime，数值差异不能直接证明画质原因。只重启了尚未加载模型的续训等待控制器，当前GPU0评测未受影响。

诊断入口为outputs/2026-10-08-05/stage1_anyflow39_precision_probe/，后续队列为stage1_anyflow39_gpu0_extend64_fullmem/。顺序为frozen16六条评测→精度诊断→条件32/64续训；新增诊断实际显存及结果仍待测。Stage1效果验收未通过，Stage2保持暂缓。

## 2026-10-08 05 时段：按用户要求迁移GPU0；后续模型驻留预算提高至约38GiB

GPU2冻结time-MLP训练在step08完整保存后，于05:08停止本实验的训练与等待控制器；从step08在GPU0恢复继续到总16次更新。真实33B恢复审计确认QKV/time/action adapter权重、Adam、logical/CPU/CUDA RNG、更新历史及teacher身份逐项相等。GPU0已完成第9次更新（A/chunk1），说明恢复后真实反传/更新继续运行。该审计证明载入状态一致，不声称与未中断CUDA训练的整条轨迹完全相同。

当前训练/评测队列为outputs/2026-10-08-05/stage1_anyflow39_gpu0/。旧GPU2队列已标记migrated_to_gpu0，保留原状态和日志。当前16-update对照仍使用reserve20。

用户确认GPU0独占并允许更多显存，后续32/64-update条件续训及其评测改用ABOT_VRAM_RESERVE_GIB=6：L40约44.4GiB可见显存，对应约38.4GiB模型驻留预算（不是实际allocated峰值的保证）。新队列outputs/2026-10-08-05/stage1_anyflow39_gpu0_extend64_fullmem/已通过预检查并启动等待；旧reserve20续训等待队列已停止、标记superseded。仅运行设备/offload预算变更，训练目标、LR、data/noise、anchor、chunk、action routing及time冻结策略保持原协议；两种reserve的耗时/显存不直接混算算法收益。当前训练不中断，后续预算的实际峰值仍待测。

证据入口：submission/reports/stage1_anyflow/05_runtime/gpu0_migration/README.md。Stage1视频/action验收仍未通过，Stage2暂缓。

## 2026-10-08 05 时段：冻结参数策略真实更新正常；同协议32/64-update续训已条件排队

冻结时间MLP的33B训练已完成前4次更新，覆盖A/D chunk0/1；QKV梯度有限非零，time梯度始终0。逐张量核对step00/04确认16个QKV块全部变化，目标时间MLP权重严格不变，32个Adam参数状态及adapter SHA256校验通过。真实参数策略证据为submission/reports/stage1_anyflow/03_anyflow_trials/early_pilot/parameter_policy_gpu_step04.json。当前还没有这一分支的视频质量结果。

新增有限续训队列outputs/2026-10-08-04/stage1_anyflow39_frozen_extend64/run_extension.py，已确认控制器进程存活。它等待frozen16训练和6条A/D评测完整完成且父队列退出；若任一完整A/D配置满足A>0、D<0、A-D>1，则暂不加训，等待视觉审查。否则从step16的Adam/RNG/权重精确恢复到总32次更新，评测native8 A/D；仍失败才继续到总64次并评测native4/8。step48额外保存。仅改训练量，数据、noise序列、目标、LR、anchor、chunk、routing、time冻结策略保持一致；只串行使用GPU2，任何运行/解码/输入一致性异常都会停止。CPU preflight检查了真实step00状态兼容、源码/teacher输入哈希和gate分支；实际33B CUDA续训尚待触发。保存目录与计划已同步submission/reports/stage1_anyflow/07_protocols/continuation_plan/。

另核查官方Stage1 EMA包（约4.15GB）的可访问性：公开文件列表可读，但manifest下载返回GatedRepoError，现有HuggingFace凭据无权限；未下载该权重。本地训练不依赖这一外部资源。Stage1的短片动作与视觉验收仍未通过，Stage2暂缓。

## 2026-10-08 04:50：uniform4对照完成；冻结时间MLP的33B训练已实际启动

旧trainable-time AnyFlow的uniform4 step00：A=-2.711084、D=-2.140539、A-D=-0.570545；step16：A=-2.695826、D=-2.253384、A-D=-0.442442。四条39帧视频均完整解码、输入conditioning与对应原始H3逐张量一致，但抽帧都显示后半段严重雾化/人物和场景结构丢失。因此仅改官方均匀sigma网格没有修复该pilot，不继续扩solver/anchor sweep。证据汇总为submission/reports/stage1_anyflow/03_anyflow_trials/uniform_snapshot/README.md。

uniform进程退出后，冻结时间MLP的真实33B训练已自动在GPU2启动（训练PID1426752）。training.json明确记录trainable_parameters=3,440,640、target_time_trainable=false、target_time_trainable_parameters=0。当前处于初始验证/训练开始阶段，尚无该分支质量结论；将按计划完成16次更新，再做native4/native8/uniform4 A/D。进度入口为outputs/2026-10-08-04/stage1_anyflow39_frozen_time/run.json。Stage1未完成效果验收，Stage2暂缓。

## 2026-10-08 04 时段：旧AnyFlow pilot全部完成，均匀网格对照已启动

旧队列训练AnyFlow/FM各16次，完成18条causal评测（加2条archived original reference）。AnyFlow native4在step00/04/08/12/16的A-D分别为-0.0162/0.0390/0.0033/0.0361/-0.0909，无持续改善。clean-history对照A=-0.4053、D=-0.9188、A-D=0.5135，仍失败；teacher历史能恢复部分块首的人物/车库，但块内仍模糊/重影且块边界出现context重置。boundary RGB MAD A17.752/D12.307，不能把teacher-history重置当成部署时的视觉改善。抽帧证据为pilot/report/clean_vs_generated_4step.jpg。

原进程退出后，已把默认冻结目标时间MLP、显式uniform采样和精确续训CLI合入主源码与submission；36项测试通过。旧源码快照保存在pilot/original_source，历史结果不改标签。resume的连续4步 vs 2+恢复到4步仅经过小型H3 CPU验证，真实33B CUDA恢复仍待验证。新report明确记录时间MLP策略及sigma网格，不混合实验。

GPU2已进入旧step16/00的uniform4 A/D评测；冻结时间MLP16-update队列随后自动运行。两队列使用隔离代码和依赖哈希，只串行占用GPU2。结果汇总为submission/reports/stage1_anyflow/03_anyflow_trials/pilot_snapshot/REPORT.md；当前短片视觉和action gate未通过，Stage2继续暂缓。

## 2026-10-08 04 时段：FM对照完成；官方AnyFlow参数策略复核与受控修正

同初始化FM训练完成16次更新，训练脚本760.1秒，allocated peak=16,597.4 MiB；A/D固定noise验证loss从0.125927/0.181004到0.124712/0.179743。39帧generated-history评测完成：FM4-step A=-1.125998、D=-1.359751、A-D=0.233753；FM8-step A=-1.141958、D=-1.456845、A-D=0.314888。旧AnyFlow16对应分离度为-0.090887/0.304306，尚无AnyFlow动作收益证据。FM4画面比AnyFlow4明显完整；AnyFlow step00 A的抽帧也已有严重后段雾化，所以不能把全部画面失败归咎于训练时间MLP引起的参数漂移。原队列继续完成step00/04/08/12与clean-history诊断。

首块定位：只取前17个RGB帧的同一Farneback指标，原始H3 A=+0.0834、D=-0.8945；AnyFlow4 A=-1.3442、D=-1.6299；AnyFlow8 A=-1.2125、D=-1.5137。第一个latent chunk尚无generated history，其A/D endpoint差向量与teacher的cosine约-0.118（4-step）/-0.039（8-step）。后者只是endpoint轨迹诊断，不是同state score-field指标；但结合首段flow说明不能只用长时历史漂移解释当前问题。证据在pilot/report/first_chunk_flow_diagnostic.json及endpoint_chunk_diagnostic.json。

**重要更正：** 复核SolarWM固定版本ce1da4e后，runtime.py在enable_anyflow后执行整个transformer.requires_grad_(False)，再注入LoRA，optimizer仅使用self.lora.parameters；lora.py还断言all and only LoRA参数可训练。因此官方H3 Stage1的克隆目标时间MLP是冻结的。本地旧pilot额外训练该MLP，是参数策略变体；此前“time MLP必须实际更新”的官方验收理解不准确。数学loss/gradient一致不等于optimizer策略一致。原16-update结果保持不变，不重标为官方参数策略。

已在独立副本中添加默认冻结/显式train-target-time选项，小型真实H3两步训练验证：两分支初始time权重完全相同，QKV都更新，冻结分支time权重严格不变且梯度0，trainable分支time更新。证据在outputs/2026-10-08-04/stage1_anyflow39_frozen_time/parameter_policy_cpu_validation.json。正式39f冻结时间MLP16-update及评测已排队，但截至本条尚未开始占用GPU。

GPU2后续有限队列已启动并核验控制进程存活：uniform队列PID919787等待原pilot退出，再对旧AnyFlow step16/step00各跑A/D 4-step均匀sigma；frozen-time队列PID1033264等待uniform队列退出，再从同一旧visual初始化、同数据/噪声/协议训练16次，只改变时间MLP可训练性，然后native4/native8/uniform4分别评测A/D。当前只原pilot使用GPU；等待进程不加载模型。两个新实验使用各自runtime/代码副本，保留源码哈希和补丁，不改动旧队列冻结源码。原生/均匀采样helper的4项CPU检查通过；均匀4-step明确为[1,.75,.5,.25,0]，与训练shift分开记录。

冻结分支已实现optimizer/RNG/adapter SHA256状态保存与严格恢复CLI（--resume-from，--steps表示目标总更新数）。小型实际H3 CPU验证：冻结时间AnyFlow、可训练时间AnyFlow及FM三分支中，连续4步与2步后恢复到4步的adapter、Adam、RNG和loss/gradient记录完全相等；拒绝LR变化、权重错配和缺失optimizer状态。尚未验证真实33B CUDA恢复，不把CPU验证视为画质证据。详细更正及证据入口为submission/STAGE1_PARAMETER_POLICY.md。代码补丁暂保存在各实验目录和submission/reports/stage1_anyflow/，待旧pilot结束后统一合入主训练/benchmark；不要把准备完成和排队写成真实效果通过。Stage1质量gate仍未通过，Stage2不启动。

## 2026-10-08 03 时段：Stage1 AnyFlow 16-update 与4/8-step A/D完成，效果未通过

GPU 2 的真实 H3 33B AnyFlow 训练完成16次更新，step_00/04/08/12/16完整保存；训练脚本总计3043.0秒，更新及最终验证阶段的PyTorch allocated peak为17,439.4 MiB。QKV和目标时间MLP均确认变化，A/D三个chunk均实际反传，无NaN/OOM。

固定held-out noise的加权验证loss仅略降：A 0.117473→0.116278，D 0.169817→0.168317；但endpoint raw loss反而从A 133.788→144.185、D 18.083→19.440。AnyFlow自适应缩放会掩盖高endpoint残差，因此不将total loss下降当成少步学习成功。

step16的generated-history 39帧/4-step-per-chunk A/D已完成：A=-1.315323、D=-1.224436、A-D=-0.090887，方向gate失败。A全39帧contact sheet显示约20帧起明显重影/雾化，后段人物和车库结构严重退化；D的抽帧亦出现相同现象，视觉gate失败。它们是诊断结果，未替换会议主demo。保存的video/audio noise、prompt、initial image anchor、action text rows逐张量与对应原始H3 reference完全一致，见pilot/report/metrics.json。随后8-step/chunk A/D也已完成：A=-1.170865、D=-1.475171、A-D=0.304306；人物和车库比4-step明显完整，后段仍有模糊/重影，A方向仍错误，验收未通过。4-step A/D E2E=188.1/167.1秒，8-step=214.5/213.5秒；全部CPU KV peak=6484.1 MiB、allocated GPU peak=16247.2 MiB，不能与旧teacher不同offload的峰值作公平效率比较。三列原始H3/AnyFlow4/AnyFlow8诊断视频保存在pilot/report/anyflow16_AD_original_4step_8step_diagnostic.mp4，完整39帧，已解码检查；本轮没有新正式demo。GPU2串行队列已自动进入FM control训练，之后继续checkpoint和clean-history诊断。

另外完成可选的action RoPE位置契约原型：固定未来action内容长度时，改变未来embedding对当前chunk影响为0；只改变未来句子长度会通过原生text-length位置原点轻微影响当前输出。使用只依赖首个action和已知horizon的固定原点后，小型实际H3的FM/AnyFlow两项回归通过。当前A/D固定动作的坐标改动严格为0。这只是后续action-switching的因果性检查，未证明与画面崩坏有关；helper尚未接入正在运行的训练/benchmark，保持本轮协议冻结。源码为code/causal/position_contract.py，证据为pilot/future_action_layout_probe.json和position_contract_AD_noop.json，已同步submission。

## 2026-10-08 03 时段：Stage1 实验继续，输入公平性与验收证据补齐

已重新确认队列 PID 227188、训练 PID 227490 存活且使用 GPU 2。截至 03:09，AnyFlow 完成 9/16 次更新，step_00/04/08 已保存。A/D 的 chunk0、chunk1、chunk2（最后不足5 latent frames）均已有真实反传，QKV 和 target-time MLP 梯度有限且非零；这仍不构成视频质量通过。

逐张量对照 archived H3 teacher A 与 AnyFlow smoke A：initial video noise、audio noise、prompt embedding、image anchor 完全相等（最大绝对差0），action text rows 也相同。证据在 pilot/input_fairness_smoke.json。重算原始 H3 的 Farneback 参考：A=+1.181253，D=-0.842124，A-D=2.023377。新增 report_stage1_anyflow.py，统一输出已完成视频的指标和contact sheet；本轮表格/图片在 pilot/report/，不把尚未完成的视频列成结果。

官方源码复核发现：H3 Stage1 验证用均匀 sigma=[1,.75,.5,.25,0]，本轮保持之前的shift=2.22网格；官方也使用encoded silence加噪及音频时间调度，当前prototype固定audio noise。差异已列入 submission/STAGE1_ACCEPTANCE.md，尚未证明它们是失败原因。当前训练/采样源码保持冻结，先完成已有FM/AnyFlow对照，再单独检查采样网格，避免同时改变多个变量。旧teacher offload reserve未记录，不能把旧teacher和本轮不同的allocated peak当作公平显存收益。

目标继续保持“改善Stage1 AnyFlow/causal实际效果，再开始Stage2”；39帧动作与视觉、独立seed、切换动作、124帧验收仍未完成。已有1-update视频的后段分解/雾化和错误A方向仍然保留为失败证据。

## 2026-10-08 02 时段：GPU 2 真实 H3 AnyFlow 单次更新通过

用户指定 GPU 2。保留其他进程，使用 `CUDA_VISIBLE_DEVICES=2`、`ABOT_VRAM_RESERVE_GIB=20`、CPU raw KV 与 activation checkpoint/offload，运行 `outputs/2026-10-08-02/stage1_anyflow39_smoke/`。真实 H3 33B、A/D 39 帧 teacher 伪真值、RGB dual anchor，完成一次 optimizer update；QKV 3,440,640 参数和 target-time MLP 15,835,008 参数均实际更新。训练 loss=0.0938887、梯度范数=0.0370643；单 update 161.8 秒，包含数据准备及前后验证的脚本计时 499.7 秒，PyTorch allocated 峰值 17,221.8 MiB。

同一 held-out noise 的 A/chunk2 加权验证 loss 从 0.117473 到 0.118669，略升；高噪声 endpoint 样本 raw loss 从 133.788 到 42.081。只能据此确认训练和硬件链路可行，不能宣称质量改善。

回载后的首轮生成在第 2 个 solver step 暴露 FP32 轨迹与 BF16 H3 condition 的类型不匹配，失败日志保留在 `eval4/A/`。已修复 AnyFlow model-call 边界的输入转换，并让 RGB anchor 遵守显式 compute dtype，保持轨迹累积为 FP32；新增 BF16 H3 两步采样/clean commit 和 RGB anchor dtype 回归检查，专项测试 12 passed。修复后的回载视频 `eval4_verified/A/cached.mp4` 已完整解码 39 帧：12 noisy forwards + 3 clean commits，E2E 194.1 秒，GPU allocated peak 16,247.2 MiB，CPU KV 6,484.1 MiB。人工 contact-sheet 检查发现后半段人物/场景明显重影与雾化，A 水平光流 -1.296，未通过视觉或方向 gate。

已启动 `outputs/2026-10-08-02/stage1_anyflow39_pilot/run_pilot.py`，仅串行使用 GPU 2：AnyFlow 16 updates（每4步保存）→ step16 A/D 4/8-step → 同初始化 FM 16-update 对照及 A/D 4/8-step → AnyFlow 00/04/08/12 的 4-step 学习曲线 → step16 clean-history A/D 诊断。运行状态在 `run.json`，错误、解码/计数异常或源码变化都会停止队列。该队列不包含 Stage2 或长视频，也不自动判定质量通过。

## 2026-10-07 21 时段：补齐 Stage1 TF-AnyFlow 目标，33B 训练待 GPU

已从 GitHub fast-forward 同步到 d3d5844，保留用户精简后的提交包组织。根据用户要求暂停继续扩展 Stage2，先补 TF-AnyFlow。新增目标时间 MLP、(t,r) packed row 索引、SolarWM v1.5 有限差分目标、2 FM + 1 endpoint + 1 finite-map 的 logical batch，以及 benchmark 有限步采样。H3-World 的 scheduler velocity 已是 noise-clean，移植时未重复取负号；所有 loss 前向共享相同 clean history，更新参数后重建 CPU KV。

CPU 验证：源项目完整测试集 28 passed，与官方 loss/gradient 数值对照通过；实际 H3 小配置验证 target-time 作用、prefix 时间不污染、checkpoint 梯度、cache 只读与精确回载。随机小型 H3 跑完 8 次 AnyFlow 更新，held-out noise weighted loss 2.75052→2.74759，QKV 和目标时间 MLP 均独立确认实际更新；相同采样流程的 FM control 也完成。它们只证明实现可运行，不是 33B 画质或 action gate 通过。

实验目录为 `H3-World/outputs/2026-10-07-21/anyflow_stage1/`；新增源训练入口 `H3-World/code/causal/train_stage1_anyflow.py`，提交包实现与验收计划为 `submission/STAGE1_ANYFLOW.md`。提交补丁已补齐 cached attention 的梯度读取支持，并验证可从文档记录的 base + patches 重建为当前测试源码；独立临时目录复现的 AnyFlow 专项测试 10 passed。尚无新的 AnyFlow 预训练模型对比视频。检查时 8 张 L40 全部被占用（GPU 0–3 高负载、GPU 4–7 显存近满），已询问可用卡，没有停止或占用他人的训练。下一步先真实 33B 单 update 及回载，再 16-update 学习曲线与 39f A/D 的 FM/AnyFlow 4/8-step 对照，质量通过后才扩 124 帧或 Stage2。

## 2026-10-07：保留动作较强旧版，完成 10/20 秒长视频对照

按用户要求同时保留 fixed-mix 旧版与 RGB visual 稳定版，新增三列原始/旧 causal/稳定 causal 的 124 帧视频，路径为 `submission/meeting/action_vs_stability/`。旧版 action adapter 已独立备份到 `submission/checkpoints/legacy_fixed_mix/`；A=+0.1427、D=-0.3105、A-D=0.4532，方向符号较好但后段明显漂移。新稳定版 A=-0.7841、D=-1.0075、A-D=0.2233，结构较稳但动作失真。两套模型都保留，不把任何一套写成四方向完全保真。

长视频使用同一稳定 adapter、RGB dual、generated history、CPU raw KV、5 latent frames/chunk、5-chunk history、8 steps/chunk、W、seed=13、shift=2.22、832×480。实际长度为 243 帧/10.125 秒和 481 帧/20.042 秒，分别有 15/29 chunks、120/232 noisy forwards、15/29 clean commits。原始 H3 同长度 30-step 对照同时运行；同长度两分支的首帧、prompt、初始 video/audio noise SHA256 已验证一致。实验目录 `H3-World/outputs/2026-10-07-00/long_rgb_visual/`。

20 秒原始 H3 的 eager directed mask 构造申请 25.94 GiB 临时张量 OOM；保留失败日志，增加可选 `H3_COMPILE_BLOCK_MASK=1` 编译同一 mask 构造后重启。该优化不改变 attention predicate；2176 rows、padding、3 action rows、改变 action assignment 后的 dense mask 与 sparse BlockMask metadata 均逐元素完全一致。补丁与检查脚本归入 submission，未触碰 causal 模型架构。

长视频全部完成并通过完整 H.264/YUV420P、24 fps 解码检查；每个长度 original/causal 的输入指纹一致。真实单次结果如下：

| 帧数 | 方法 | E2E s | GPU peak MiB | CPU KV MiB | Noisy+commit | RGB MAD | Boundary MAD |
|---|---|---:|---:|---:|---:|---:|---:|
| 243 | original | 921.0 | 16425.8 | 0.0 | 30+0 | 3.802 | 3.876 |
| 243 | causal | 1773.8 | 18300.4 | 13508.6 | 120+15 | 2.713 | 3.642 |
| 481 | original | 2406.0 | 12394.8 | 0.0 | 30+0 | 3.252 | 3.022 |
| 481 | causal | 3659.3 | 30896.8 | 13508.6 | 232+29 | 2.759 | 4.666 |

10 秒后段存在模糊、重影和靠墙方向漂移；20 秒：**视觉稳定性明确失败。** 0–5 秒仍能辨认人物和车库；约 10 秒开始严重雾化和重影，15 秒后人物与场景结构难以辨认，20 秒末尾退化成模糊色块。原始 H3 的长时场景几何同样形变，但人物保持得明显更完整。这里没有裁掉失败后段；该视频应作为 generated-history 长时退化的负结果展示，不能作为“20 秒稳定性已解决”的证据。所谓稳定版只指其 124 帧表现相对旧 fixed-mix 较好，不是长时质量保证。

完整材料位于 `submission/meeting/long_horizon/`，三列动作/视觉对照位于 `submission/meeting/action_vs_stability/`；额外备份见 `submission/breakthrough/06_long_rollout_and_tradeoff/`。两边都保留完整后段，不声称动作控制已恢复或长视频质量 PASS。原始 H3 20 秒可通过编译 mask 与 CPU offload 运行，旧 OOM 不能用作原始模型的固有长度限制。单次并发时间和不同 reserve 不能作严格 speedup/memory 排名。


## 2026-10-07：会议视频主结果纠错与视觉协议更换

发现会议包第一版把 `outputs/2026-10-03-02/final_fixed_mix124_8step/` 作为主展示。该旧 fixed-mix 实验只有 action adapter，没有 visual tail16 QKV adapter，推理使用 `dynamic_last_frame_dual` / `global_retimed_latent_dual_v1` latent-only anchor，且 `action_prefix_mode=own`、`action_feedback=false`。它的 MP4 可解码，但 generated-history 后段出现人物透明、车库 tearing 和 temporal-VAE ghosting；“能播放”被错误地当成了“视觉合格”。这是选片/标注错误，不是播放器或编码器故障。旧视频、旧指标和审计已归档到 `submission/meeting/diagnostics/legacy_fixed_mix/`。

会议主入口已改为 `submission/meeting/annotated/h3world_rgb_stable_action_grid_124_timed.mp4` 及同目录 W/S/A/D 单动作视频。它们来自 `outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2/`，配置为 tail16 visual QKV + RGB-consistent dual anchor（generated tail 解码 RGB 后经过 H3 image branch）+ generated history + causal action prefix + action feedback + CPU raw KV。W/S/A/D 均为 124 帧、8 chunks、8 steps/chunk、64 noisy forwards + 8 clean commits，已完整解码为 H.264/YUV420P/24 fps。视觉结构比旧主视频稳定，但仍有 blur/ghosting，且 A/D action geometry 没有恢复。

当前 RGB visual-main 单次记录：原始 H3 30-step 为 441.5–454.2 s；causal 为 673.4–767.9 s；首块 41.8–57.4 s；CPU raw KV 13,508.6 MiB（约 13.19 GiB）；causal GPU 峰值 31,569.6–39,939.7 MiB（约 30.8–39.0 GiB）。由于 causal 使用 64 noisy forwards 且带 RGB anchor decode/re-encode，不宣称端到端加速。Farneback flow：原始 A=`+1.077`、D=`-1.602`、A-D=`2.679`；RGB-main A=`-0.784`、D=`-1.007`、A-D=`0.223`，严格 `A>0,D<0,A-D>1.0` gate 失败。

会议文档 `meeting/README.md`、`METRICS.md/csv`、`FAIRNESS.md`、`SLIDES.md`、`MEETING_SCRIPT.md` 已统一到 RGB visual-main；`meeting/source_metrics/rgb_visual/` 保存原始 JSON/flow/continuity。最终结论仍是：causal/KV 工程可行，RGB anchor 能修复主要视觉分解，但 generated-history 下 action-conditioned score geometry 尚未恢复，Stage2-lite 也不等于官方 SGF/DMD。

# GWM 项目进展

更新时间：2026-10-06


## 2026-10-06 16 时段：原始 H3 endpoint latent 监督的 action-QKV smoke 也未恢复动作

在 routing/still 诊断之后，新增了一个不同于单纯 score-delta 的监督来源：对 tail4 action-QKV
使用 full-attention H3 A/D counterfactual paired loss，并在 generated chunks 1、2 的四个
solver sigma 上加入原始 H3 30-step `baseline_latents.pt` 的 chunk endpoint `x0` MSE（权重
0.5）。视觉 tail16 RGB adapter、RGB dual anchor、generated history、CPU raw KV、8 steps/chunk、
seed 13 均保持不变；只做 1 个 optimizer update。

代码新增 `stage2_lite_dmd.py --latent-target-weight`，会从 A/D teacher dirs 读取并校验
`baseline_latents.pt`，因此目标是同一 initial image/prompt/noise 下的原始 H3 endpoint，而不
是新生成的伪标签。实验目录：

```text
H3-World/outputs/2026-10-06-16/action_qkv_latent_endpoint_pair_rgb_39_8step_1update/
```

结果：

| 配置 | flow(A) | flow(D) | A-D | latent A/D delta cosine |
|---|---:|---:|---:|---:|
| 原始 H3 30-step teacher | +1.181 | −0.842 | 2.023 | 1.000 |
| endpoint-QKV causal student | −0.8041 | −0.7047 | **−0.0994** | 0.0563 |

此前没有 endpoint target 的 RGB causal latent delta cosine 为约 `0.0160`；新目标只把它提高到
`0.0563`，仍接近正交，且图像空间 A/D 分离反而变差。因此“直接拟合原始 H3 chunk endpoint”
也不足以恢复 generated-history causal action geometry，不能继续扩大 update 或升级为正式
checkpoint。39 帧视频人物仍可解码，视觉稳定性没有重新出现早期 latent-anchor 的分解，但动作
gate `flow(A)>0, flow(D)<0, A-D>1.0` 明确失败。

可播放对照和报告：

```text
outputs/stage2_action_qkv_latent_endpoint_AD_39.mp4
outputs/2026-10-06-16/action_qkv_latent_endpoint_pair_rgb_39_8step_1update/ACTION_QKV_ENDPOINT_REPORT.md
```

至此，已用 routing upper bound、STILL counterfactual、score-field paired QKV、action-prefix
residual、released H3 action-LoRA、Stage2-lite DMD 和原始 endpoint latent supervision 分别
排除低层修补路径。后续若要再做，必须转为多状态/多 seed 的 rollout-distribution training 或
重新设计 causal action-token/attention topology；不再继续单场景单 state 的 adapter、gain、
anchor 或 endpoint 权重扫描。

需要修正上一段结论的协议边界：上一轮 endpoint paired 分支从同一个 A-generated state 计算
A/D score，却把独立 D teacher rollout 的 endpoint 当成 D target，存在 state mismatch。因此它
只能作为 naive mixed-target 负诊断，不能证明所有 endpoint supervision 都无效。随后新增
`--own-latent-target-weight`，让 A endpoint 只监督 A-generated history，D endpoint 只监督
D-generated history，同时保留 shared-state paired score loss。

修正后的实验目录为：

```text
H3-World/outputs/2026-10-06-16/action_qkv_own_endpoint_pair_rgb_39_8step_1update/
```

训练耗时 818.6 s，GPU peak 约 39.4 GiB，CPU KV peak 约 5.28 GiB，无 NaN/OOM。39-frame 自由
rollout 为 A=`-0.8086`、D=`-0.6893`、A-D=`-0.1194`，latent delta cosine=`0.0655`，仍未通过
`flow(A)>0, flow(D)<0, A-D>1.0`。这排除了同动作 own-history endpoint + 单场景 tail4 QKV 的
1-update 修复，但仍不替代多状态/multi-seed 或完整 rollout-distribution training 的结论。
可播放对照为 `outputs/stage2_action_qkv_own_endpoint_AD_39.mp4`，详细报告为
`outputs/2026-10-06-16/action_qkv_own_endpoint_pair_rgb_39_8step_1update/ACTION_QKV_OWN_ENDPOINT_REPORT.md`。


## 2026-10-06 16 时段：routing 上界与 still 反事实基线完成，action geometry 仍未恢复

在不改动 RGB 视觉适配器、action residual、KV、solver、seed 或视频协议的条件下，补做了一个
只改变 action-prefix 可见性的 routing 上界实验。固定协议为 39 RGB / 12 latent / 3 chunks、
5 latent frames/chunk、8 steps/chunk、flow shift 2.22、generated history、persistent CPU raw KV、
`dynamic_last_frame_rgb_dual`、action feedback、seed 13；视觉 checkpoint 使用
`outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2/`。本轮 `action_prefix_mode=all`，
表示对一段已经预先知道的固定动作暴露所有 action rows；它只是 topology upper-bound diagnostic，
不是最终 interactive causal protocol。

实验目录：

```text
H3-World/outputs/2026-10-06-16/routing_all_rgb_39_8step/
```

`all` routing 的 A/D 结果为：

| prefix policy | flow(A) | flow(D) | A-D | gate |
|---|---:|---:|---:|:---|
| all + feedback | -1.1535 | -1.4733 | 0.3198 | FAIL |

为了去除场景本身的共同漂移，又用同一协议生成 `STILL` 反事实：

| action | horizontal flow | 相对 STILL |
|---|---:|---:|
| STILL | -0.7129 | 0 |
| A | -1.1535 | -0.4406 |
| D | -1.4733 | -0.7604 |

因此两点都已排除：

1. 把 action rows 全部提前暴露只把 A-D 从 causal routing 的约 0.31 提到约 0.32，不能恢复
   左右方向；问题不是简单的 future action row 不可见。
2. STILL 反事实不能解释失败。相对 STILL 后 A、D 仍然都向同一负方向偏移，A/D 分离只有
   0.3198，说明共同相机/场景运动不是主要原因。

可播放的 A/D 对照：

```text
outputs/stage2_routing_all_rgb_AD_39.mp4
```

机器可读指标和完整协议见：

```text
outputs/2026-10-06-16/routing_all_rgb_39_8step/ROUTING_STILL_REPORT.md
outputs/2026-10-06-16/routing_all_rgb_39_8step/ROUTING_STILL_REPORT.json
outputs/2026-10-06-16/routing_all_rgb_39_8step/flow_with_still.json
```

本轮仍然没有产生新的正式 124-frame action grid。结合此前 frozen causal/teacher delta cosine
约 -0.015、student/teacher delta norm ratio 约 6.9x，以及 tail4 QKV 4-update learning
curve 未通过图像动作 gate，当前最合理结论是 action-conditioned score-field / causal routing
representation mismatch，而不是缺少一条 action edge 或单纯的 flow baseline 偏置。RGB anchor
修复的人物稳定性结论保持不变，但不能写成 action control 已保留。


## 2026-10-06：RGB anchor + generated-history endpoint adaptation 修复人物分解并通过 124 帧视觉验收

针对早期 `stage2_lite_39_8step_4updates` 视频中人物在后半段透明、分崩离析的问题，先确认了
根因不是 MP4 编码，而是 anchor protocol mismatch：旧 Stage2-lite 将上一 chunk 的 latent
tail 直接 patchify，而 tail16 causal visual adapter 的训练协议是 H3 原生 RGB image branch。

本轮在固定 causal mask、chunk、KV 和 solver 的条件下，只训练 tail16 causal visual QKV，并加入
原始 H3 30-step `baseline_latents.pt` 的 chunk endpoint MSE。协议为：39 RGB / 12 latent / 3
chunks，5 latent frames/chunk，history window=5，8 steps/chunk，flow shift=2.22，generated
history，persistent raw KV on CPU，`dynamic_last_frame_rgb_dual`，causal action rows + feedback，
seed=13；fixed-mix action residual 保持冻结。两个 optimizer update 依次使用 A、D，teacher replay
覆盖每个实际 solver sigma，endpoint loss 权重 0.5，boundary loss 权重 0.25。

实验目录：

```text
H3-World/outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2/
```

训练完成且无 NaN/OOM，峰值 allocated GPU 32,509 MiB；每份 student/teacher raw KV 约 6.33 GiB
CPU。A update 用时 553.6 s，D update 用时 522.5 s。39 帧评估均为 H.264/YUV420P，PyAV 完整
解码 39 帧，人物和车库结构到 frame 38 仍存在：

| action | horizontal flow | vertical flow | A-D | visual status |
|---:|---:|---:|---:|:---|
| A | -1.1448 | +0.3306 | — | stable through frame 38 |
| D | -1.4557 | +0.2567 | 0.3109 | stable through frame 38 |

这轮没有恢复 action gate（`flow(A)>0, flow(D)<0, A-D>1.0`），因此不能把它称为 action
geometry 修复；它明确证明 RGB-consistent anchor + endpoint regularization 能解决早期人物分解，
同时没有改变当前 action geometry 的结构性问题。

随后用同一 checkpoint 做 124-frame W 长时 rollout：

- 124 frames / 5.17 s，8 chunks；64 noisy denoiser forwards + 8 clean commits；
- generated history，CPU KV peak 13.19 GiB，GPU allocated peak 31,570 MiB；
- sampling 709.0 s，conditioning 后总耗时 752.9 s；
- H.264/YUV420P，PyAV 完整解码 124 帧；
- 抽帧 0、12、24、38、51、64、76、89、102、111、123 中人物和场景持续完整，没有此前
  latent-only Stage2 的后段分解。

随后在 GPU1/2 并行完成了同协议的 124-frame A、D rollout。两条视频也都完整解码 124 帧，
抽帧至 frame 123 仍保持人物和车库结构：

| action | horizontal flow | gray MAD | visual status |
|---:|---:|---:|:---|
| A | -0.7841 | 2.9972 | stable through frame 123 |
| D | -1.0075 | 2.8282 | stable through frame 123 |

A/D 的负水平 flow 仍是动作几何未恢复的证据，不是人物分解；因此这组长视频只用于视觉稳定
验收，不改变 action gate 的失败结论。新增并排视频为：

```text
outputs/stage2_rgb_endpoint_visual_stable_AD_124.mp4
```

可播放交付物已单独放在 `H3-World/outputs/` 根目录：

```text
outputs/stage2_rgb_endpoint_visual_stable_W_124.mp4
outputs/stage2_rgb_endpoint_vs_original_W_124.mp4
outputs/stage2_rgb_anchor_endpoint_visual_stability_comparison_39.mp4
outputs/stage2_rgb_endpoint_visual_stable_AD_124.mp4
```

其中第一项是新的 124 帧 causal W，第二项是原始 H3 30-step W 与新 causal W 的并排视频，第三项
是旧 latent-only Stage2 与 RGB/endpoint 版本的 A/D 四宫格对照。详细协议、指标和限制见：

```text
H3-World/outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2/VISUAL_DRIFT_REPAIR_REPORT.md
```

当前阶段的结论更新为：视觉 drift/人物分解已经有可复现的 RGB-anchor 修复，并在 124 帧上通过
结构稳定性检查；A/D action geometry 仍未通过 gate。后续若继续，应该把 RGB visual adapter 作为
稳定基线，专门处理 action-pathway 或真正的 rollout-distribution matching，不再混淆视觉修复与
动作控制结果。

随后把该新 visual adapter 接回真正的 `stage2_lite_dmd.py`，完成一轮完整 critic/DMD 集成：
A/D self-rollout，target chunks 1、2，四个 sigmas `0.94,0.79,0.57,0.24`，tail4 critic，
shared 33B backbone，RGB anchor。训练耗时 1111.6 s，GPU allocated peak 40,320 MiB，所有
critic/DMD 梯度有限；实验目录为：

```text
H3-World/outputs/2026-10-06-09/stage2_lite_rgb_endpoint_integrated_39_8step_chunks12_sigmas4_1update/
```

用保存的 student adapter 做同协议 39-frame A/D 验收：

| action | horizontal flow | vertical flow | A-D | visual status |
|---:|---:|---:|---:|:---|
| A | -1.1419 | +0.3277 | — | stable through frame 38 |
| D | -1.4550 | +0.2577 | 0.3131 | stable through frame 38 |

这证明真正的 Stage2-lite critic/DMD 链路不会重新触发人物分解，但单轮 DMD 仍没有恢复 action
geometry。可播放的集成 A/D 对照为：

```text
outputs/stage2_lite_rgb_endpoint_integrated_AD_39.mp4
```


## 2026-10-06 07 时段：Stage2-lite v2 已切换到 RGB-consistent multi-chunk/multi-sigma

为解决此前 Stage2-lite 中“visual tail16 adapter 用 RGB dual 训练、Stage2 rollout 却用 latent-only
dual anchor”造成的人物分解，本轮修改了
[`H3-World/code/causal/stage2_lite_dmd.py`](H3-World/code/causal/stage2_lite_dmd.py)：新增
`--anchor-mode latent|rgb`，默认 `rgb`。当使用 RGB 模式时，student self-rollout、critic
fake-score、frozen teacher 的 history cache 都从同一 generated prefix 解码最后 RGB 帧，再经 H3
image branch `process_image=True` 重编码为第二个 dual anchor；同一 rollout 内按 chunk 缓存，避免
同一边界重复 VAE 转换。latent 模式仍保留作便宜的历史对照。

新增实验目录：

```text
H3-World/outputs/2026-10-06-07/stage2_lite_v2_rgb_39_8step_chunks12_sigmas4_1update/
```

固定协议为 39 RGB frames / 12 latent frames / 3 chunks / chunk size 5 / history window 5 /
8 steps per chunk / generated history / CPU raw KV / seed 13；critic 覆盖 chunk 1、2 和
`sigma={0.94,0.79,0.57,0.24}`。仍然是共享一个 33B backbone 的 Stage2-lite feasibility
diagnostic，student hidden action residual 和 tail4 critic 分别只有约 0.77M/0.39M 可训练参数。

训练完成且没有 OOM/NaN：耗时 **872.9 s**，峰值 allocated GPU **39.37 GiB**，最大记录的 CPU
raw-KV cache **5.28 GiB**。A/D 每条评估使用 24 次 noisy denoiser forward + 3 次 clean commit；
RGB anchor 在 chunk 边界增加 VAE decode/re-encode 成本。训练曲线和完整指标见
[`REPORT.md`](H3-World/outputs/2026-10-06-07/stage2_lite_v2_rgb_39_8step_chunks12_sigmas4_1update/REPORT.md)。

| 39-frame RGB-dual Stage2-lite v2 | flow(A) | flow(D) | A−D | Gate |
|---|---:|---:|---:|---|
| 当前 checkpoint | −1.1369 | −1.4556 | 0.3187 | FAIL |

本轮的结论分成两部分：

1. **视觉稳定性改善。** A/D 抽帧中人物和停车场结构保留到第 38 帧，之前 latent-only Stage2
   样例中的人物透明/分解没有再出现。可播放 H.264 Constrained Baseline / YUV420P 对照为
   [`AD_rgb.mp4`](H3-World/outputs/2026-10-06-07/stage2_lite_v2_rgb_39_8step_chunks12_sigmas4_1update/AD_rgb.mp4)，
   抽帧为
   [`contact_sheet_rgb.jpg`](H3-World/outputs/2026-10-06-07/stage2_lite_v2_rgb_39_8step_chunks12_sigmas4_1update/contact_sheet_rgb.jpg)。
2. **动作几何没有恢复。** A 与 D 的水平 flow 都为负，A−D 只有 0.319，未达到短片 gate
   `flow(A)>0`、`flow(D)<0`、`A−D>1.0`，也明显低于原始 H3 teacher 的约 2.02。不能因为画面
   稳定就宣称 action control 已保留。共同的 forward/场景运动占据了水平 flow，当前 causal action
   pathway 仍没有复现 H3 的横向 score geometry。

因此 RGB anchor 修复了主要的视觉 conditioning mismatch，但没有解决 student/teacher action
delta 几乎正交的问题。Stage2-lite v2 的 multi-sigma/multi-chunk 覆盖也没有在一轮更新中恢复
A/D；不继续做 gain 或 anchor sweep，也不把本轮扩展为新的 124-frame 正式 grid。下一步应固定
RGB anchor、KV、solver 和 generated-history，做低容量 action-pathway alignment（以 full H3
counterfactual delta 为目标），然后再决定是否值得继续 DMD/SGF。详见
[`REPORT.json`](H3-World/outputs/2026-10-06-07/stage2_lite_v2_rgb_39_8step_chunks12_sigmas4_1update/REPORT.json)。

## 2026-10-06 08 时段：最小 action-pathway alignment smoke 已完成

在固定 RGB dual visual protocol 后，新增了只训练 full-attention H3 A/D counterfactual delta 的
alignment smoke，比较 hidden action residual 和零初始化 tail4 action-QKV refiner。两者都使用
39 帧 / 3 chunks / 8 steps per chunk / generated history / CPU raw KV / 四个 sigma / seed 13，
visual tail16 QKV 冻结；不是重新训练 visual adapter，也没有引入新的 anchor 或 solver。

实验目录：

```text
H3-World/outputs/2026-10-06-08/action_align_hidden_rgb_39_8step_pair1_final/
H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair1_final/
```

paired loss 只在同一个 generated A state 上匹配 student/teacher 的 A−D score delta。结果如下：

| Variant | paired delta cosine | student/teacher norm ratio | flow(A) | flow(D) | A−D | Gate |
|---|---:|---:|---:|---:|---:|---|
| hidden residual | 0.139 | 1.974 | −1.2499 | −1.4293 | 0.1794 | FAIL |
| tail4 action-QKV | **0.390** | **1.018** | −0.8359 | −0.7685 | −0.0675 | FAIL |

QKV refiner 的 score-field alignment 明显优于 hidden residual，但这一次 alignment 没有转化为
自由 generated-history 视频的图像空间 A/D 方向；两条视频都保留人物和停车场结构到第 38 帧，
所以本轮进一步排除了“只要把 teacher delta 对齐就能自动恢复 action geometry”的假设。可播放
对照和抽帧为：

- [hidden A/D](H3-World/outputs/2026-10-06-08/action_align_hidden_rgb_39_8step_pair1_final/AD.mp4)
- [tail4 QKV A/D](H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair1_final/AD.mp4)
- [contact sheet](H3-World/outputs/2026-10-06-08/action_alignment_contact_sheet.jpg)
- [完整报告](H3-World/outputs/2026-10-06-08/ACTION_ALIGNMENT_REPORT.md)

本轮说明 action representation 仍有更深的 causal routing/topology 问题；不能把 paired loss
下降或 delta cosine 上升直接写成 action control 恢复。已启动一个 tail4 QKV 的 4-update
learning curve，并在脚本中加入每轮 `update_XX/student_action_adapter.pt` 保存，完成后再用
同一 39-frame gate 选择是否继续。

## 最新结论：RGB-dual + FP32 gain 4-step 曲线已完成，39 帧 gate 未通过

当前需要先看这一节。下方保留按实验阶段累积的历史记录，历史中的“正在运行/下一步”不再
代表最新状态。最新完整报告和可播放对照为：

- [本轮报告与效率表](H3-World/outputs/2026-10-06-09/GAIN_CURVE_REPORT.md)
- [原始 H3 / 旧 latent-anchor 失败样例 / 当前 RGB-dual step_01 对照](H3-World/outputs/h3world_rgb_gain_diagnostic_AD_39.mp4)
- [A/D 抽帧图](H3-World/outputs/2026-10-06-09/contact_sheet.jpg)
- [完整训练记录](H3-World/outputs/2026-10-06-09/trainable_gain_fp32_full_teacher_rgb_8x4/training.json)

| 39-frame 配置 | flow(A) | flow(D) | A−D | 数值 gate |
|---|---:|---:|---:|---|
| 原始 H3 30-step reference | +1.181 | −0.842 | 2.023 | 参考 |
| RGB dual 手工 A64/D8 初始化 | +0.0474 | −0.7812 | 0.8286 | FAIL |
| FP32 gain step_01 | +0.0489 | −0.7782 | 0.8272 | FAIL |
| FP32 gain step_02 | +0.0445 | −0.7629 | 0.8075 | FAIL |
| FP32 gain step_03 | +0.0431 | −0.7782 | 0.8213 | FAIL |
| FP32 gain step_04 | +0.0350 | −0.7652 | 0.8002 | FAIL |

本轮在单卡 GPU0 完成 4 次 A/D 交替 optimizer update，GPU1/4 并行回载评估；其它用户
VLLM 进程没有改动。训练耗时 **2265.4 秒（37.8 分钟）**，峰值 allocated 显存
**40.06 GiB**，student / teacher 的 CPU raw KV **各 6.33 GiB**。

训练的是已有 action residual projection 和 9 个 gain（共 3,096,585 参数），不是只训
9 个数；visual tail16 QKV 和原 H3 LoRA 冻结。初始化仍是手工选出的 A64/D8，所以这轮
不能被描述成“从单位增益学会动作幅度”。每条评估保持 39 RGB frames / 12 latent frames /
3 chunks、8 steps/chunk、RGB dual、generated history、seed13；所有 checkpoint 的
initial noise、audio noise、初始 image anchor 和同一 action 的 prompt embedding 都和
保存的 teacher conditioning 逐元素核验。每条视频是 **24 noisy forwards + 3 clean commits**。

本轮修正了一个 gain 数值实现问题：此前 gain 保存为 FP32，但投影前转成 BF16，导致
63.9836/8.0092 在前向中又变成 64/8。现在只有很小的 action projection 用 FP32，完成的
residual 再转 BF16。checkpoint 保存 `gain_compute_dtype`，旧文件没有该字段时保持原计算
路径；新增精度、梯度和 checkpoint 兼容测试，相关测试 **9 passed**。修正前的
`2026-10-06-08/trainable_gain_full_teacher_rgb_8x4` 在 step_01 后主动中断并标为 interrupted。

修正后的 gain 最终为 A≈63.9344、D≈8.0369。虽然同一 rollout action 下 replay/magnitude
loss 有所下降，但 direction loss 仍约 1，说明 teacher/student delta 方向尚未对齐；rollout
的 A−D 没有超过初始化。A 的正向光流很小，不能仅以正负号宣称 action controllability 已恢复。
RGB dual 的视觉改善和动作几何恢复是两个不同结论，MAD/边界 MAD 也不能当视频质量分数。

抽帧复核：新 step_01 / step_04 人物主体保留到第 38 帧，不再像旧 latent-anchor Stage2-lite
样例那样基本消失；但 A 有明显向远处走/人物缩小的趋势，不能视为正确横移，D 末尾腿部仍
模糊。根目录三列对照为展示不同阶段结果，各列 adapter 不完全一致，并非 anchor-only
消融。视频已验证 39/39 帧解码、H.264 Constrained Baseline、YUV420P、24 fps。

推理效率也需如实报告：step_01 A/D 含加载/conditioning 总耗时约 246.0/224.7 秒，峰值
39.17 GiB，CPU KV 6.33 GiB；原始 30-step A/D 是 212.4/218.0 秒、38.99 GiB、无历史 KV。
时间存在并行训练/CPU offload 负载差异，这一短片 RGB-dual 配置没有证明速度或显存优势。

**决定：不继续 gain/anchor 扫描，不生成新的 124-frame 正式 grid。** 这只是一个场景、一个
seed、4 次更新的负结果，不能证明架构上限或证明必须使用 SGF/DMD。进一步训练前应先审计
full-teacher target：当前 teacher 是双向 generated-prefix/current-chunk 前向，并不是原始
完整 39-frame teacher 轨迹；要先确认同一 state 上 teacher delta 的方向、强度和 student
可学习梯度，再决定 action-path adaptation 或 RGB 一致的多 sigma critic/DMD。正式
124-frame 交付仍为旧的 `h3world_final_fixed_mix_action_grid_124.mp4`。

## 2026-10-06：full-teacher counterfactual delta audit

为避免把“关闭 causal mask 的 teacher”直接当成正确动作监督，新增了只读脚本
[`H3-World/code/causal/audit_teacher_action_delta.py`](H3-World/code/causal/audit_teacher_action_delta.py)。
它取 RGB-dual step_01 的 A generated-history latent，在同一个 chunk/current state 上分别
运行原始双向 H3 的 A、D conditioning，扫描 3 个 chunk 和 5 个 solver sigma；不训练任何
参数。原始数据和表格位于：

```text
H3-World/outputs/2026-10-06-10/teacher_delta_audit/audit.json
H3-World/outputs/2026-10-06-10/teacher_delta_audit/REPORT.md
```

teacher 的 A/D velocity 都有限且 delta 不为零，但 delta 相对平均 velocity norm 的比例只有
约 `0.0128–0.0336`，15 个 chunk/sigma 状态均值约 `0.0207`。因此 full teacher 确实给了可反传的动作信号，
但它是完整 score field 中很小的差异；4-step residual/gain 更新只能造成弱的 rollout 改变，
并不能从这个结果推出目标方向与图像空间 strafe 一致。audit 只使用一个 A generated state、
一个场景和一个 seed，也没有测 student 对齐或 optical flow，不能被描述为完整 action accuracy
结论。随后用 D-generated state 复核同一脚本，delta/velocity ratio 为
`0.0150–0.0424`、均值 `0.0206`；A-state 和 D-state 的均值都约 2%，说明“小而非零”的
监督信号不是只由某一个 rollout 方向造成。两组汇总见
[`COMBINED_REPORT.md`](H3-World/outputs/2026-10-06-10/teacher_delta_audit/COMBINED_REPORT.md)。
这仍不证明 target direction 与图像空间 strafe 一致；下一步应记录 student/teacher delta
cosine，再决定扩大 action pathway 还是做 RGB-consistent multi-sigma critic/DMD。

## 2026-10-06：student/teacher action-delta 对齐审计完成

进一步加载 RGB-dual gain curve 的 `step_01` causal student，在同一个 A generated latent、
同一份 detached causal raw-KV history、同一 chunk 和同一 sigma 上计算 student A/D delta，
并和 full-attention teacher delta 直接比较。15 个 chunk/sigma 状态的结果为：

```text
student delta norm mean: 69.70
teacher delta norm mean: 10.68
student / teacher norm ratio: mean 6.89x, median 7.31x, range 4.21x–9.91x
student-teacher delta cosine: mean -0.0087, median -0.0115, range -0.2257–+0.1178
```

原始逐状态数据和复现脚本位于：

```text
H3-World/outputs/2026-10-12/student_teacher_delta_audit.json
H3-World/outputs/2026-10-12/STUDENT_TEACHER_DELTA_REPORT.md
H3-World/code/causal/audit_student_teacher_delta.py
```

这比 teacher-only audit 更明确：teacher delta 虽小但非零，而 causal student 的 A/D score
差异约大 4–10 倍且几乎正交。换句话说，当前动作路径产生的是不同的 action-conditioned
score geometry；只调 gain 不能旋转这个方向，所以继续增加 gain optimizer steps 没有归因
价值。A/D flow 的正负号可以暂时正确，但不能据此声称已经保留原始 H3 的横移几何。下一步
应先适配/审计 action pathway 或 causal attention topology，使 student delta 对齐 teacher，
再考虑 critic/DMD；不应先用 distribution-level loss 掩盖表示/拓扑不匹配。

## 2026-10-05：真正的 SolarWM-style Stage2-lite 已完成首轮验证

本轮开始尝试用户提出的“真正 Stage2-lite”，与之前的 minimal two-pass replay
明确区分。新增脚本
[`H3-World/code/causal/stage2_lite_dmd.py`](H3-World/code/causal/stage2_lite_dmd.py)，在
同一个 H3-World 33B backbone 上轮换三个小 adapter：

```text
student causal action residual
critic/fake-score hidden residual
frozen teacher = backbone with causal/action residual disabled
```

训练链路为：

```text
student self-rollout (generated history, detached)
→ critic 在 student 生成的最后 chunk 上做 flow matching
→ frozen H3 teacher 和 critic 分别预测同一 noisy state
→ fake_x0 - real_x0 构造 DMD gradient
→ student replay 的 surrogate loss 反传到 action residual
```

H3-World 当前 causal DiT 返回的是 `noise-clean` velocity，故本实现使用
`x0 = noisy - sigma * velocity`。SolarWM 独立 SGF backend 在 adapter 后采用 data-ward
符号，不能直接把其正负号照搬到 H3-World。

### Smoke 与正式 39 帧结果

统一协议仍为 39 RGB frames / 12 latent frames / 3 chunks、chunk size 5、history window 5、
`flow_shift=2.22`、dynamic dual latent anchor、CPU raw KV、generated history、causal action
rows、action feedback、seed 13。Stage2-lite 第一版只在最后 generated chunk、`sigma=0.6` 做
critic 和 DMD 更新；critic 使用 tail4 hidden residual，student 与 critic 不复制 33B。

| 实验 | solver | 更新轮数 | flow(A) | flow(D) | A-D | 结论 |
|---|---:|---:|---:|---:|---:|---|
| fixed-mix Stage1 baseline | 8 | 0 | -0.01325 | -0.76891 | 0.75566 | 基线 |
| Stage2-lite smoke | 2 | 1 | -0.01205 | -0.18138 | 0.16933 | solver 不可与正式基线直接比较 |
| Stage2-lite | 8 | 1 | -0.01380 | -0.77279 | 0.75900 | 链路可行，未改善 |
| Stage2-lite | 8 | 4 | +0.00477 | -0.76839 | 0.77316 | A 符号恢复，未过 A-D>1.0 gate |

4 轮正式实验耗时约 1064 秒（约 17.7 分钟）。每个 action 都重新执行 3-chunk student
self-rollout、critic update 和 student DMD update。峰值 allocated GPU memory 约 39.4 GiB；
每个 CPU raw-KV cache 约 5.28 GiB，未加载第二份或第三份 33B backbone。所有 critic loss、
DMD surrogate 和梯度均为有限值，没有 OOM 或 NaN。

结果目录：

```text
H3-World/outputs/2026-10-05-03/stage2_lite_smoke39_2step/
H3-World/outputs/2026-10-05-04/stage2_lite_39_8step_1update/
H3-World/outputs/2026-10-05-05/stage2_lite_39_8step_4updates/
```

正式 4 轮的训练曲线、adapter、原始 flow JSON 和结论见
[`RESULTS.md`](H3-World/outputs/2026-10-05-05/stage2_lite_39_8step_4updates/RESULTS.md)、
[`summary.json`](H3-World/outputs/2026-10-05-05/stage2_lite_39_8step_4updates/summary.json) 和
[`stage2_lite.json`](H3-World/outputs/2026-10-05-05/stage2_lite_39_8step_4updates/stage2_lite.json)。
最终 A/D 并排视频为
[`stage2_lite_39_8step_4updates_AD.mp4`](H3-World/outputs/stage2_lite_39_8step_4updates_AD.mp4)。
抽帧 contact sheet
[`contact_sheet.jpg`](H3-World/outputs/2026-10-05-05/stage2_lite_39_8step_4updates/contact_sheet.jpg)
显示人物在约第 15–30 帧逐渐 ghost/雾化，末尾基本消失；因此 DMD 首轮没有解决视觉 drift。

### 播放兼容性修复

2026-10-05 发现 `eval/D/cached.mp4` 在部分播放器中无法播放。文件本身可以由 PyAV 解码
39 帧，但原始视频是 H.264 High Profile；现已在原路径重新编码为 H.264 Constrained Baseline、
YUV420P、24 FPS，并保留原文件为 `eval/D/cached_highprofile.mp4`。A 视频和根目录的并排视频
也统一转成相同的兼容编码；重新检查均为 39/39 帧可解码。现在应直接使用原路径
[`eval/D/cached.mp4`](H3-World/outputs/2026-10-05-05/stage2_lite_39_8step_4updates/eval/D/cached.mp4)
和根目录的
[`stage2_lite_39_8step_4updates_AD.mp4`](H3-World/outputs/stage2_lite_39_8step_4updates_AD.mp4)。

### 当前判断

这次实验已经证明的是：

```text
H3 causal student self-rollout
+ generated-history fake-score critic
+ frozen teacher
+ DMD surrogate
```

可以在现有单卡 46 GiB L40 上以共享 33B backbone 运行并完成真实反传。它还没有证明
Stage2-lite 能恢复 H3 的 action geometry：4 轮后 A-D 只有 0.773，低于短片验收线 1.0，
也远低于原始 H3 teacher 约 2.02。A 仅略高于 0，D 保持负向，因此不能把这个结果称为成功的
Stage2 checkpoint。

当前最合理的解释是训练信号仍然过窄：critic 只看最后 chunk 和单个 sigma，DMD 只对一次
counterfactual state 更新；它更像验证了训练机制和显存方案，而不是足够的 Stage2 distribution
matching。后续若继续，优先扩展到多个 solver sigma 和所有 generated chunks，并保留相同的
student/critic/teacher 共享 backbone；不要再把这轮结果覆盖正式 124-frame 主 demo。

## 2026-10-02 08 时后：为什么 W/S/A/D 还不能完全保真

当前四方向实验的准确结论不是“action control 已经完全恢复”，而是：动作条件仍能改变
causal 输出，但因果注意力、raw-KV 提交方式、teacher-forcing 与自由 rollout 的分布差异
共同削弱了方向响应。四方向的数值证据如下（同一首帧、prompt、seed 和初始噪声）：

| 条件 | A flow | D flow | A-D |
|---|---:|---:|---:|
| 原始 H3 teacher，39 帧，30 steps | +1.181 | -0.842 | +2.023 |
| causal，无 adapter，clean history，39 帧 | +0.391 | -0.565 | +0.956 |
| causal，无 adapter，generated history，39 帧 | -0.137 | -0.152 | +0.015 |
| causal，无 adapter，generated history，124 帧，8 steps/chunk | -0.018 | -0.026 | +0.008 |

这说明问题分成两个阶段。首先，即便历史使用 teacher latent，causal mask 也把 A-D 差异
从约 2.02 降到约 0.96：当前 chunk 看不到未来 video token，prefix/action token 在缓存
路径中不能像原始全序列 H3 那样与当前 video token 双向更新，且 action LoRA 原本是在
全序列、每个噪声时刻的注意力图上训练的。现在提交到 raw-KV 的是 sigma=0 clean chunk，
之后 noisy denoising 读取的是固定历史 K/V，和原始 H3 每一步重新计算所有历史 token 的
score field 不是同一个函数。

其次，free-running rollout 使用模型自己生成的历史。当前 Stage1 adapter 的训练主要是
clean-history teacher forcing，因此每个 chunk 的小误差会被写入下一次的 K/V 和 image
anchor；动作差异是相对于场景、首帧和 prompt 的较小 residual，进入未见过的 generated
history 后很容易被共同的场景先验覆盖。39 帧 generated-history 已经把 A-D 压到约
0.015，124 帧、8 steps/chunk 进一步降到约 0.008。增加 solver steps 不能消除这个训练/推理
分布差异；这正是需要 SolarWM Stage2 式 self-rollout distribution matching 的地方。

普通共享 MSE adapter 还存在目标冲突：四个动作共享 QKV 尾层，最容易降低平均 denoising
loss 的方式是学习共同的外观和运动，而不是保留 A/D 的符号。pair loss、base-output
regularization 和 schedule supervision 能提高动作差异，但没有建立显式的方向几何约束；
per-action adapter 虽能把 A-D 差异提高到约 0.323，却在约 30 帧后出现明显雾化和 temporal
ghosting，不能作为主结果。因此不能把“数值分开”误称为“视频质量和 action fidelity 同时
恢复”。

anchor 也有独立影响。latent dual anchor 只是把 temporal latent patchify 成 image-like
prefix，不等价于 H3 原生的 RGB decode → `process_image=True` encode 路径；RGB dual anchor
能显著改善后段结构连续性，但当前无 adapter 版本的 A/D flow 约为 -0.032/-0.028，方向
几乎没有水平流响应。RGB adapter 又会重新引入后段雾化。当前最佳视觉候选是
`H3-World/outputs/h3world_action_grid_rgbdual_124.mp4`，最佳动作分离候选是
`H3-World/outputs/h3world_action_grid_individual_reg_124.mp4`；二者代表质量与方向响应的
明确 trade-off，均不能宣称四方向完全保真。

最后，水平 optical flow 只适合直接判断 A/D。W/S 主要产生深度、尺度、主体位移和相机跟随
变化，原始 H3 的 W/S 水平 flow 本身也同号，因此不能用“W flow 必须为正、S flow 必须为负”
作为验收条件。W/S 应结合主体尺度/深度变化和视频对照，A/D 才用左右 flow 符号；当前报告
将 action sensitivity、directional fidelity、temporal continuity、visual quality 和
generated-history stability 分开记录。

因此当前阶段的严谨结论是：H3-World 的 action rows、causal chunk rollout 和 persistent
raw-KV 链路已经打通；clean-history 短时 rollout 保留部分动作响应，但 free-running history
会造成动作塌缩。要同时恢复四方向和长时质量，需要把 action-sensitive schedule supervision
与 generated-history self-rollout 训练结合起来，并重新设计能保持 H3 image-condition 语义
的局部 anchor；仅继续堆叠 mask、anchor 或增加 solver steps 不足以证明 Stage2 效果。

### W→A→D 时间切换验证

刚完成 `W:3,A:2,D:3` 的 124 帧 RGB dual-anchor rollout，结果目录为
`H3-World/outputs/2026-10-02-04/action_schedule_WAD_base_rgbdual_124/`，并排视频为
`H3-World/outputs/h3world_schedule_WAD_rgbdual_124.mp4`。因果分支使用 64 次 noisy
denoiser forward、8 次 clean commit、约 13.19 GiB CPU raw-KV、约 38.7 GiB allocated
GPU peak，采样时间约 590.0 s；这是无额外 causal adapter 的视觉连续性候选。

分段 Farneback central-flow proxy（W 0–50 帧、A 51–84 帧、D 85–123 帧）如下：

| 视频 | W flow x / magnitude | A flow x / magnitude | D flow x / magnitude |
|---|---:|---:|---:|
| Original H3 30-step | -0.033 / 1.108 | +1.564 / 2.072 | -1.278 / 2.291 |
| Causal RGB dual, 8-step/chunk | -0.102 / 0.345 | -0.003 / 0.301 | -0.002 / 0.147 |

原始 H3 在 A→D 切换处出现明显的正负水平流变化；RGB dual causal 视频保持结构但没有
复现这个方向切换，进一步验证“视觉连续性”和“action directional fidelity”是两个独立
问题。完整 JSON 为 `action_schedule_WAD_rgbdual_flow.json`。

## 2026-10-01 04–05 时段：修正 Stage1 rollout 调度并复测 anchor

本时段发现并修正了一个会直接造成“只闪两下”的 benchmark 错误：`benchmark.py` 原来在
每个 chunk 内调用了所有 timestep 的 denoiser，却把 `scheduler.step(...)` 放在 timestep
循环外。因此日志虽然记录了 8 次 denoiser forward，latent 实际只前进了一次，不能作为
8-step causal rollout 结果。现在 scheduler 更新已经放回循环内；每个 chunk 的 N steps
都会产生 N 次 latent 更新，chunk 结束后仍额外用 `sigma=0` 做一次 clean KV commit。
之前被终止的 `outputs/2026-10-01-04/fixed_base30_124/` 和
`fixed_base8_shift2_124/` 已标记为 `aborted`，不再作为实验结果。

### 修正版 22 帧 A/B

使用相同 seed 13、相同单末层 Stage1 adapter
`outputs/2026-10-01-01/stage1_long_retimed_train_seed13/adapter.pt`、8 steps/chunk、
shift=12、5-frame chunk。baseline 是同一条件下的原始 H3 30 steps。结果目录为
`H3-World/outputs/2026-10-01-05/`：

| 配置 | 采样时间 | denoiser / clean commit | GPU allocated peak | CPU raw-KV peak | 灰度 MAD 均值/p95 | 空间边缘差 |
|---|---:|---:|---:|---:|---:|---:|
| H3 original, 30 steps | 118.07 s | 30 / 0 | 39,922.7 MiB | 0 | 3.493 / 12 | 2.242 |
| causal, fixed first-frame anchor, 8 steps | 51.11 s | 16 / 2 | 39,923.5 MiB | 3,782.4 MiB | 2.779 / 9 | 2.093 |
| causal, previous-last-frame anchor + W action, 8 steps | 50.76 s | 16 / 2 | 39,923.5 MiB | 3,782.4 MiB | 3.341 / 11 | 2.280 |

修正后 8-step rollout 的输出已经是连续的多步更新，不再是旧 bug 造成的单步跳变。动态
末帧 anchor 的运动量比 fixed anchor 高，也更接近 30-step teacher；但这是 22 帧（约
0.92 秒）的短片，不能据此宣称长片质量已经解决。并排文件为
`H3-World/outputs/stage1_corrected_dynamic_anchor_8step_22.mp4`，fixed/dynamic 严格 A/B
为 `H3-World/outputs/stage1_corrected_fixed_vs_dynamic_22.mp4`。

“上一 chunk 最后一帧 + action”的实现协议是：当前 chunk 继续读取允许的历史 raw K/V，
同时把上一已生成 clean chunk 的最后一个 latent frame 替换到显式 H3 anchor 行，并把
anchor 的 temporal position retime 到真实的全局 frame index；action 文本仍按每个 latent
frame 的 W 按钮条件注入。当前 anchor 是 latent patchify 的最小原型，不等同于先解码 RGB
再用 H3 `process_image=True` 独立编码的原生 image anchor；后者需要单独的局部 anchor
layout 和 denoise mask，仍列为后续改造。

末层固定 anchor 的 4-block QKV LoRA 低学习率复测也已完成：训练 loss
`0.13966→0.11653`、validation `0.11173→0.09901`，但修正版短片 MAD 只有 `2.105/7`，
表现为运动被压平，暂不作为 Stage1 候选。它只保留在
`outputs/2026-10-01-04/stage1_fixed_tail4_lr1e4/` 作为负面 ablation。

### 124 帧修正版长片

同一单末层动态 anchor adapter 的 124 帧、8 steps/chunk rollout 已完成：64 次 denoiser
forward、8 次 clean commit，采样 `299.46 s`，VAE decode `12.73 s`，CPU raw-KV 峰值仍为
约 `13.5 GiB`。与同 seed 的 30-step teacher 比较，teacher 灰度 MAD 为 `4.295/15`、
边缘差 `2.226`，causal 为 `5.735/22`、边缘差 `4.714`。抽帧显示前几个 chunk 能保持人物
运动，但约第 30 帧后出现纹理重影、亮度漂移和场景结构破坏。因此修正后的 Stage1-style
链路可以稳定执行真实 N-step rollout，却仍未通过长片质量验收；根因仍是 clean-history
训练到 generated-history 自回归的分布差异，以及 latent anchor 与 H3 独立图片编码语义不
完全一致。长片并排视频为 `outputs/stage1_corrected_dynamic_anchor_8step_124.mp4`。

同一 adapter 的 16 steps/chunk 长片也已完成，用于区分积分步数误差和训练分布误差：采样
`586.63 s`，128 次 denoiser、8 次 clean commit；causal 灰度 MAD 为 `6.575/26`、边缘
差 `3.836`，仍明显劣于 30-step teacher。因此简单把 8 增到 16 步不能修复长片漂移，问题
主要在训练/rollout 分布而不是单纯积分误差。结果在
`outputs/2026-10-01-05/stage1_dynamic_corrected_124_steps16/`，并排文件为
`outputs/stage1_corrected_dynamic_anchor_16step_124.mp4`。无论 8 或 16 steps，都不称为
Stage2，因为没有 SGF/DMD student 蒸馏。

为对齐 H3 原生 scheduler，另在 GPU 3 启动了 `shift=2.22`、dynamic anchor、124 帧全部
chunk 的 Stage1-style 末层训练，目录为
`outputs/2026-10-01-05/stage1_dynamic_shift2_train_tail1_gpu3/`；此前 GPU 2 的同配置在
第 7 个最大 packed chunk 因显存临时 mask 分配失败，已保留为 failed 记录，不作结论。

该 shift=2.22 adapter 已完成 124 帧回载，使用完全相同的 8-step scheduler、dynamic
previous-last-frame anchor 和 W action：64 次 denoiser、8 次 clean commit，采样
`324.34 s`，CPU raw-KV 峰值约 `13.5 GiB`。它是当前最好的 Stage1-style 候选：teacher
的灰度 MAD/边缘差为 `4.295/15`、`2.226`，新 causal 为 `4.452/17`、`2.433`；相比
shift=12 adapter 的 `5.735/22`、`4.714` 有明显改善。训练指标为 train
`0.10522→0.08774`、validation `0.10618→0.09551`、replay error `0`。但抽帧仍显示约
第 30 帧后有重影和亮度漂移，所以当前结论是“协议对齐显著改善、长片质量仍未完全通过”，
不是 Stage2 成功。当前主对比视频为
`outputs/h3world_30step_vs_stage1_aligned_shift2_8step_124.mp4`（同内容的时段副本为
`outputs/stage1_stage1_aligned_shift2_dynamic_anchor_8step_124.mp4`），统计为
`outputs/2026-10-01-05/metrics_124_shift2.json`。

## 2026-10-01 Stage1 动态末帧 anchor 与 124 帧扩展

本轮继续围绕面试题主线“在 H3-World 中验证 SolarWM 的因果少步生成思路”，没有转去
完整复现 SolarWM 或下载其大规模数据。重点是把 Stage1 teacher-forcing 的条件协议和
causal rollout 对齐，并验证用户提出的想法：每个新 chunk 除了历史 K/V 外，再显式给模型
上一个 chunk 的最后一帧，作为类似 H3 image-to-video 首帧的视觉 anchor。

### 新增实现

- `H3-World/code/causal/h3_cached.py` 新增 `last_frame_anchor()`，把单个 clean latent
  frame 转成 H3 keyframe anchor rows。
- `H3-World/code/causal/benchmark.py` 新增 `--anchor-mode fixed|dynamic_last_frame`。
  动态模式从 chunk 1 开始使用上一已生成 clean chunk 的最后 frame；重复 anchor 只进入
  当前 prefix，不重复写入 video raw-KV cache。
- `H3-World/code/causal/train_pretrained_multichunk.py` 新增同样的动态 anchor 规则、H3
  shifted scheduler 噪声点、任意 latent 长度/target chunk 列表和训练/验证 noise seed
  参数；同时提高 Dynamo shape-cache 上限，支持长片不同 chunk 的变长 packed sequence。
- 当前动态 anchor 仍替换原首帧的 packed value，使用原首帧 position；这是诊断实现，不是
  最终的局部 anchor layout。正确实现仍需局部 anchor token、重复帧 denoise mask、局部或
  全局 RoPE 位置语义，以及只提交新 rows 的 cache 生命周期。

### 22 帧动态 anchor A/B

原始产物目录：`H3-World/outputs/2026-10-01-01/`。

同一个 seed 13、同一个旧 multi-chunk adapter、8 steps/chunk 的 fixed/dynamic A/B：

| 输出 | 采样时间 | denoiser forwards | clean commits | GPU allocated peak | CPU raw-KV peak |
|---|---:|---:|---:|---:|---:|
| fixed first-frame anchor | 75.80 s | 16 | 2 | 34,702.6 MiB | 3,782.4 MiB |
| dynamic previous-last-frame anchor | 66.96 s | 16 | 2 | 34,702.6 MiB | 3,782.4 MiB |

短片描述性统计：30-step teacher 的灰度帧间 MAD 均值/p95 为 `3.490/12`、空间边缘差
`2.242`；fixed 为 `2.425/11`、`2.277`；dynamic 为 `2.573/11`、`2.402`。动态 anchor
确实减少完全静止倾向，但也出现亮度/场景漂移；这些统计不是画质分数。直接 A/B 视频为
`H3-World/outputs/h3world_fixed_vs_dynamic_anchor8_22.mp4`。

### Stage1 训练与短片回载

使用 seed 13、22 帧 teacher、chunk 0/1、末层 rank-8 QKV LoRA、动态 anchor、H3 8 点
shift-12 schedule 训练 160 步：train `0.0862→0.0566`，validation `0.0548→0.0461`，
训练样本 16、验证样本 4、replay 最大误差 0，可训练参数 215,040。adapter 位于
`H3-World/outputs/2026-10-01-01/stage1_dynamic_anchor_seed13/adapter.pt`。

回载到 22 帧动态 anchor、8 steps/chunk 后：采样 `62.41 s`，GPU allocated peak
`34,702.6 MiB`，CPU raw-KV peak `3,782.4 MiB`，灰度帧间 MAD `3.020/14`，空间边缘差
`2.297`。它比旧 adapter 更有运动，但不足以证明质量等价。并排视频为
`H3-World/outputs/h3world_30step_vs_stage1_dynamic_anchor8_22.mp4`。

### 同 seed 124 帧对照

为满足面试题的长视频交付，使用相同 seed 13、首帧、prompt、action、832×480、124 帧
（5.17 秒）重新生成：

- 原始 H3 30 steps：采样 `368.77 s`，产物 `baseline30_seed13_124/baseline.mp4`；
- 22 帧训练 adapter + dynamic anchor 8 steps/chunk：采样 `478.52 s`，64 次 denoiser
  forward、8 次 clean commit，约 `13.5 GiB` CPU raw-KV；约第 40 帧后场景明显漂移。

原始产物在 `outputs/2026-10-01-01/baseline30_seed13_124/` 和
`stage1_dynamic_rollout_seed13_tail8_124/`，并排诊断视频为
`H3-World/outputs/h3world_30step_vs_stage1_dynamic_anchor8_124.mp4`。这证明短片 anchor
实验不能直接外推到长片。

### 124 帧多 chunk Stage1 teacher-forcing

为修复上面的训练/rollout 分布缺口，直接在 124 帧 teacher 的 chunk 0–7 上做 clean-history
训练（单 noise seed、每 chunk 4 个 H3 scheduler sigma、每 chunk 2 个 validation 样本）：

| 项目 | 结果 |
|---|---:|
| train/validation samples | `32 / 16` |
| optimizer steps | `240` |
| train loss | `0.09109 → 0.07251` |
| validation loss | `0.05029 → 0.04664` |
| replay max error | `0` |
| train allocated peak | `6,270.2 MiB` |
| feature extraction / total wall | `460.83 s / 601.35 s` |

adapter：`H3-World/outputs/2026-10-01-01/stage1_long_dynamic_anchor_seed13/adapter.pt`。
回载结果位于 `stage1_long_rollout_seed13_tail8_124/`：采样 `316.83 s`，64 次 denoiser
forward、8 次 clean commit、CPU raw-KV peak `约 13.5 GiB`。长片统计：teacher
灰度 MAD `4.295/15`、边缘差 `2.226`；long Stage1 causal 为 `5.887/23`、`4.343`。
抽帧显示运动量增加，但中段以后出现重影、亮度漂移和场景结构破坏。最终长片并排视频：
`H3-World/outputs/h3world_30step_vs_stage1_long_anchor8_124.mp4`。这是 Stage1 链路可运行
但质量仍失败的结果，不是 Stage2 或质量等价的少步模型。

当前最重要的结论：显式末帧 anchor 对短片运动有帮助，但直接把它放在首帧 packed position
不是长期稳定的实现；clean-history teacher-forcing 即便覆盖全部 8 个 chunk，也无法单独
解决 generated-history 分布偏移。下一步应优先实现真正局部 anchor layout，并加入 generated-
history/scheduled-sampling 训练；之后才有必要继续做 Stage2 SGF/DMD 的少步蒸馏。当前 124
帧最终视频、JSON 和训练 adapter 均已归档，根目录的并排视频保持可直接审阅。

## 2026-09-30 后续实验状态

上一阶段的 124 帧原始/causal raw-KV benchmark、并排 Demo、11 项测试和实验报告已经
完成。本轮继续实验前检查了机器状态：当前 GPU 1–5、7 仍有其他进程占用，GPU 6 约有
43 GiB 可用显存；没有终止其他用户进程。后续长实验将只使用明确空闲的 GPU，避免影响
同机任务。

本轮已完成短序列 cached/recompute 对照、CUDA 小模型训练和预训练 H3 末层 LoRA
训练及回载生成；本项目当前没有运行中的训练或 benchmark 进程。结果见下文。

### 后续实验结果（GPU 6）

当前 GPU 1–5、7 检查到有其他进程占用，因此没有终止或抢占这些任务；本轮只使用当时
约有 43 GiB 空闲的 GPU 6。

#### Cached 与 full-history recompute 对照

输出目录：`H3-World/outputs/2026-09-30-15/benchmark_cache_recompute_22_gpu6_latents/`。相同的 H3
权重、LoRA、prompt、首帧、seed 2、832×480 条件下，生成 22 帧、2 steps、5-frame
chunk、最多 5 个历史 chunk，并分别运行 raw-KV cached 和显式 full-history recompute：

| 路径 | denoiser forwards | clean commits | 采样时间 | raw-KV 峰值 | GPU allocated peak |
|---|---:|---:|---:|---:|---:|
| cached，CPU raw-KV | 4 | 2 | 30.56 s | 3,782.4 MiB | 39,601.9 MiB |
| full-history recompute | 4 | 0 | 25.63 s | 0 | 39,188.0 MiB |

两条路径都成功生成 22 帧、832×480、24 fps、H.264 视频。cached 额外执行 2 次 clean
commit，并将 100 个逐层 K/V 写入 CPU cache；在这个很短的实验上，CPU cache 搬运和
commit 开销使 cached 采样比 recompute 慢约 19%。这不是 GPU cache 加速结论，只量化了
当前实现的额外开销。

保存的最终 latent 直接比较结果为：平均绝对差 `0.010882`，95 分位绝对差 `0.03125`，
最大绝对差 `0.285156`。视频统计也接近：cached 灰度帧间 MAD 均值/p95 为 `2.054/7`，
recompute 为 `2.069/8`，空间边缘差为 `2.203` 与 `2.196`。默认混合后端运行的最终
latent 相对 L2 差异为 1.533%；完整对比保存在 `latent_comparison.json`。两条路径使用了
不同 attention 后端（SDPA/FlexAttention）、BF16 和不同计算形状，默认差异不能直接
归因为浮点误差，也不能视作真实 33B 严格等价的证据。后续统一 SDPA 诊断已将差异降到
0.502%，但仍需逐层定位剩余误差。小模型此前的 CUDA 检查约有 `1e-6` 最大误差，仅
适用于当时的小配置。

两种模式依次在同一进程执行，存在首轮编译、权重驻留与显存缓存状态差异；此时 GPU 6
也曾有其他用户小任务。表中时间属于这次运行的观测值，不用于声称稳定的 cache 加速比。

随后增加了诊断开关 `H3_CAUSAL_EAGER_SDPA=1`，让 full-history recompute 的 causal
mask 也使用 PyTorch CUDA SDPA，和 cached 路径采用相同 attention backend。该诊断运行
输出在 `outputs/2026-09-30-17/benchmark_cache_recompute_22_gpu6_sdpa/`：cached 采样 235.78 s，
recompute 采样 93.89 s；由于 dense SDPA mask 使整段路径显著变慢，这组时间不作性能
比较。latent 差异降为平均绝对差 `0.001938`、95 分位 `0.015625`、最大 `0.071777`、
相对 L2 `0.502%`。这说明前一组 1.533% 差异很大一部分来自 FlexAttention/SDPA 的
后端和归约差异，但仍有非零误差，尚不能宣称真实 H3 cached 与 recompute 严格等价。
该开关只用于正确性诊断，默认生产路径仍使用 FlexAttention。

#### Stage1 风格 teacher-forcing CUDA smoke

为了把训练验证从 8-step CPU 接口测试推进一步，`code/causal/train_smoke.py` 现在支持
`--device`，并把随机小 H3 的 attention head 改为 16 维，以满足 CUDA FlexAttention 的
最低 head dimension。GPU 6 上实际 patched `MiniMaxH3DiT` 配置为 2 层、hidden size 64、
4 个 head、head dim 16、LoRA rank 4、固定合成 latent batch，执行 256 次 clean-history
teacher-forcing flow loss：

```text
trainable parameters: 2,048
loss: 2.334223 -> 1.499979
relative decrease: 35.7%
parameters_changed: true
gradients: finite
wall time: 29.08 s
```

这证明最小 Stage1 风格 loss 在 CUDA 上可以持续优化，而不是只完成一次 forward/backward。
它仍然是随机小模型和固定合成数据，未加载预训练 33B 权重，不代表 H3 视频质量、AnyFlow
收敛或 Stage2 SGF/DMD 效果。

### 真实预训练 H3 末层 LoRA：最新结果

本轮新增 `H3-World/code/causal/train_pretrained_tail.py` 和 `pretrained_lora.py`。
使用原始 H3 30-step 生成的单个 22 帧片段作为 teacher，保存原始 latent 与条件在
`H3-World/outputs/2026-09-30-15/teacher_baseline30_22/`。冻结前 49 层和原有 action LoRA，只在
第 50 层 QKV 新增 rank-8 LoRA，共 215,040 个可训练参数，训练 80 步。

训练目标为第二个 chunk（2 个 latent frame），前 5 个 latent frame 为 clean history。
训练使用 2 个噪声 seed × 3 个 sigma（0.3/0.6/0.9），验证使用同片段的 2 个新噪声样本；
不是独立视频或跨场景验证。固定前 49 层的特征只提取一次，末层重放与整模型原始输出
最大误差为 0。

| 项目 | 结果 |
|---|---:|
| 平均训练 loss | 0.052538 → 0.040543（下降 22.8%） |
| 同片段新噪声验证 loss | 0.042231 → 0.036073（下降 14.6%） |
| 参数更新和梯度 | 参数确实改变，梯度有限 |
| 加载及特征提取 | 41.18 s |
| 80 步优化时间 | 11.90 s |
| 完整训练脚本墙钟 | 78.39 s |
| 仅末层优化阶段 allocated peak | 2,454.1 MiB |

2,454.1 MiB 不包含完整 33B 模型提取特征所需显存，不能说完整模型训练只需约 2.4 GiB。
详细记录为 `H3-World/outputs/2026-09-30-15/pretrained_tail_stage1/training.json`，adapter 为该目录
的 `adapter.pt`（约 843 KiB）。adapter 已分别回载到 22 帧、2/4 steps per chunk 的
真实 raw-KV 推理，输出在 `benchmark_trained_tail_22/` 与 `benchmark_trained_tail4_22/`。

新增可播放对照：`H3-World/outputs/h3world_30step_vs_trained_tail4_22.mp4`，左侧原始
30-step，右侧已训练的末层 causal LoRA 4-step，22 帧、1664×528、24 fps。主 124 帧
Demo 继续保留，主 Demo 的 causal 分支仍是未训练模型。

视觉观察中 2-step 的训练后输出仍明显模糊，4-step 的低层视频统计也不能支持质量改善
结论。当前验证了真实预训练权重上的“训练→保存→加载→生成”，尚未验证等质量加速。
新推理使用 `ABOT_VRAM_RESERVE_GIB=10`，GPU 外部占用随运行变化，权重驻留量也不同；
这些运行的显存和时间差不能归因为 LoRA 本身的性能收益。

最终统一测试为 `12 passed in 5.99s`，全部 causal Python 文件及相关 patched DiT/
pipeline 的语法检查通过。新增测试覆盖零初始化等价、仅末层 LoRA 有梯度及保存/加载
结果一致。上述 12 项为 CPU 测试，之前的 CUDA 小模型等价检查是另一次手动实验。

当前待解决的优先问题：真实 BF16 cached/recompute 在混合后端下有 1.533% 相对 L2，
统一 SDPA 诊断下仍有 0.502%，还需要逐层比较才能确认剩余差异来源；随后需要多片段、
首个及后续 chunk、独立验证片段上的 teacher-forcing，才能评估质量是否改善。仍未实现
AnyFlow 或 Stage2 SGF/DMD。

## 项目目标（实习生面试题）

题目是：**在 H3-World 中验证 SolarWM 的因果少步生成思路**。

需要阅读 H3-World 和 SolarWM 的 MiniMax-H3 实现，重点理解 SolarWM 的：

- 因果分块生成（causal chunks）；
- 滑动窗口局部注意力；
- 分层 KV Cache；
- Stage2 少步生成和其训练来源。

然后在 H3-World 中设计并实现一个最小 causal 模型训练/推理原型，验证这些思路
是否有机会改善长视频生成效率。最终交付：

- 一个并排对比视频：左侧为 H3-World 原始推理（目标配置例如 30 steps），右侧为
  借鉴 SolarWM 的 causal 原型；
- 相同输入、分辨率、帧数和硬件下的推理耗时与峰值显存；
- 视频质量和时间连续性比较；
- 简短实验报告，说明方法、实验、结果、可行性和局限；
- 可运行代码，以及最小 causal attention/KV-cache 和训练接口原型。

面试题明确不要求：

- 完整复现 SolarWM Stage2；
- 训练完整的 33B 模型；
- 下载或处理 SolarWM 官方约 14.45 TB latent-WDS 数据集；
- 把 SolarWM Stage2 LoRA 直接套到 H3-World 上。

重点考察能否读懂两个项目、准确定位 causal attention/KV Cache、将论文/仓库方法
迁移成可运行原型，并通过实验说明借鉴是否有效。建议周期为 1–2 周。

### 方法说明

- **SolarWM Stage0.5**：双向 flow matching。模型在完整视频片段上学习基础的视频、
  文本和相机条件表示，建立后续因果训练使用的 backbone/初始化。
- **SolarWM Stage1**：teacher forcing + AnyFlow。用干净历史 chunk 条件化当前 noisy
  目标 chunk，同时学习去噪和有限步 flow map，使模型先具备自回归因果 rollout 能力。
- **SolarWM Stage2**：DMD/self-gradient forcing。学生模型使用自己的自回归 rollout，
  frozen teacher 和 critic 提供分布匹配梯度，进一步把多步扩散/flow 轨迹蒸馏成少数几次
  denoiser evaluation。
- **Stage2 能减少采样步数的原因**：Stage2 学到的是接近数据分布的有限步 flow map，
  每次 evaluation 可以跨越更大的噪声区间，近似原来需要多次积分的小步更新。因果分块
  和 KV Cache 解决的是长视频历史条件的计算组织与复用；它们本身不等于 Stage2 蒸馏，
  不能仅凭 causal mask/KV cache 声称实现了 DMD 少步模型。

### H3-World 与 SolarWM 的关键差别

- H3-World 当前流程使用 FL2VA 的完整视频 latent，在每个 denoising step 对整段目标视频
  做一次双向 DiT attention，官方示例为 50 steps；LoRA 主要注入动作文本/attention
  条件，推理时历史 chunk 没有以 SolarWM 的 raw KV 形式跨 chunk 持久化。
- SolarWM H3 将目标视频切为 5 个 latent-frame 的 chunk，使用最多 6 个 chunk 的滑动
  窗口；当前 chunk 查询条件、音频和窗口内历史，前面 chunk 的每层 raw K/V 写入 cache，
  后续 rollout 复用；Stage1/Stage2 再分别学习 teacher-forced 和 SGF 少步更新。
- 因此本题的迁移重点是 H3-World attention 的因果可见性、chunk 边界、每层 K/V 生命周期
  和 rollout 调度，而不是复制 SolarWM 的完整数据和训练规模。

### H3-World 最小原型边界

原型应优先完成以下可验证层次：

1. 用固定 chunk 长度和窗口长度构造 H3-World 的 causal attention 可见性 mask；
2. 定义每层 raw K/V cache 的写入、读取、滑动淘汰和一致性检查；
3. 提供 teacher-forcing/flow-map 训练接口，能在少量 latent 或合成小 batch 上运行 smoke
   loss/backward；
4. 在相同输入上跑 causal inference，记录它与完整双向 baseline 的速度、显存和视频指标；
5. 明确区分“causal/KV-cache 原型”和“经过 SGF/DMD 训练的 Stage2 少步模型”。

## 当前目录结构

```text
GWM/
├── H3-World/                 # H3-World 源码仓库
├── SolarWM/                  # SolarWM 源码仓库
├── models -> /work/lpeng/qma/GWM_models
└── progress.md
```

当前用户为 `qma`。项目的用户可见路径是 `/home/lpeng/code/mq_PubDataset/GWM`，该路径是
`/home/lpeng/code/mq_PubDataset/GWM` 的符号链接；两者指向同一个工作树，
不是两个项目副本。后续命令统一使用 `/home/lpeng/code/mq_PubDataset/GWM`，共享模型链接仍然
指向 `/work/lpeng/qma/GWM_models`。

模型权重保存在共享模型目录中，没有复制大文件：

- `models/MiniMax-H3/`：约 135 GB，MiniMax-H3 FL2VA 权重；
- `models/H3-World/step-10000.safetensors`：约 131 MB，H3-World LoRA；
- `models/SolarWM/`：SolarWM H3 base 和 Stage2 checkpoint。

H3-World 内的路径链接：

- `H3-World/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3` → 共享 MiniMax-H3 权重；
- `H3-World/checkpoints/H3-World/step-10000.safetensors` → 共享 H3-World LoRA。

## 已完成

### H3-World

- H3-World 源码仓库已准备完成，包含 `code/abot/infer.py`、训练脚本和示例图片。
- 当前 `main` 与官方仓库同步，HEAD 为 `f0c7be2acbbde8256b2473f6e7b088f58272acae`。
- DiffSynth-Studio 已初始化为本地 Git checkout。
- 已固定到 H3-World 指定 revision：`300e3e4da76e881d5e6bd97d897810c18f6e4893`。
- 已应用 `H3-World/code/diffsynth_h3_action.patch`。
- 静态预检通过：
  - patched DiffSynth 从本地 `DiffSynth-Studio-h3-v2` 加载；
  - directed attention marker `leak_out` 存在；
  - per-latent action marker `action_text_spans_local` 存在；
  - MiniMax-H3 `FL2VA/model_index.json` 可见；
- LoRA checkpoint 可读取，共 208 个 tensor，全部为 LoRA 权重。
- `code/abot/infer.py` 已加入 DiffSynth `vram_limit`，默认保留 5 GiB 显存，
  通过 `ABOT_VRAM_RESERVE_GIB` 可调整；否则 48-GiB L40 会在文本编码器阶段 OOM。
- 2-step smoke inference 已成功：124 帧、832x480、seed 2，耗时 2:41.48，
  输出 `H3-World/outputs/2026-09-29-20/smoke_2step.mp4`，视频可正常解码。
- 官方 50-step baseline 已成功：相同输入和分辨率，耗时 12:12.52，GPU 0 峰值
  42179 MiB（监控最低空闲 3280 MiB），输出 `H3-World/outputs/2026-09-29-20/baseline_50step.mp4`。
- 已实现并通过独立 causal 原型单测（3 passed）：
  - `H3-World/code/causal/prototype.py`：chunk visibility、分层 raw-KV cache、
    Tiny teacher-forcing flow-map loss/backward；
  - `H3-World/tests/test_causal_prototype.py`：验证未来 chunk 不可见、窗口淘汰、
    chunk index 连续性、detach/no-grad commit 和有限 loss/梯度。
- H3 patched DiT 已增加可选 mask-only causal 开关：
  `--causal-chunk-size 5 --causal-window-chunks 5`。它复用 H3 的
  FlexAttention block mask，在 `[text | condition | audio | video | pad]` 布局上
  让视频 chunk 只能读取静态条件、当前 chunk 和窗口内历史 chunk；目前仍是
  full-sequence QKV，不是增量 KV 复用，也不是 Stage2 SGF/DMD。
- 2-step causal mask smoke 已成功：输出 `H3-World/outputs/2026-09-29-21/causal_mask_2step.mp4`，
  124 帧、832x480、24 fps、可正常解码；完整脚本墙钟约 2:03，帧间绝对差均值
  2.33、p95 8（与原始 2-step smoke 使用相同统计口径）。
- 统一输入/seed/GPU 的 30-step baseline 已完成：输出
  `H3-World/outputs/2026-09-29-21/baseline_30step.mp4`；30 次采样阶段约 5:59，VAE 解码随后完成。
  30-step causal mask 对照已完成：输出 H3-World/outputs/2026-09-29-21/causal_mask_30step.mp4；30 次采样阶段约 5:20，VAE 解码随后完成。
- 两次 30-step 采样均为 124 帧、832x480、24 fps、seed 2。GPU 0 的 nvidia-smi 采样峰值分别为 baseline 43,607 MiB、causal 43,583 MiB；差异仅 24 MiB，不能声称 causal mask 节省了显存。采样进度阶段约减少 39 秒（约 11%），但本次命令没有单独记录模型加载/采样/VAE 的结构化 timer，因此这是量级记录。
- 统一帧间绝对差统计已完成：baseline 30-step 均值 3.395、p95 12；causal mask 30-step 均值 5.355、p95 21。causal 视频连续性指标更差，说明未经 causal 训练时，直接屏蔽未来信息有质量代价。
- 已生成可播放左右并排 Demo：H3-World/outputs/h3world_30step_vs_causal_mask.mp4，1664x480、124 帧、24 fps、H.264；生成脚本为 H3-World/code/causal/make_side_by_side.py。
- 已补充实际 H3 DiT 训练可行性 smoke：随机初始化的 2-layer MiniMaxH3DiT 小配置、QKV LoRA rank 4、固定合成 latent、CPU 8 optimizer steps；loss 由 2.33863 降至 2.33549，1,536 个 LoRA 参数更新且梯度有限。该 smoke 只验证 forward/backward 接口和冻结逻辑，不代表预训练 H3 视频质量。
- causal 原型测试已扩展为 12 个：包含 mask oracle、窗口淘汰、detach/no-grad、cache 与显式 clean history 一致性、实际 H3 DiT 的未来帧隔离和 LoRA 梯度测试、cache eviction、末尾不足完整 chunk、CPU full-history recompute、adapter 梯度及保存加载；pytest -q 四组 causal 测试已通过（12 passed）。
- 已写实验报告：docs/causal_prototype_report.md，包含 Stage0.5/Stage1/Stage2 解释、H3/SolarWM 差异、实现边界、统一视频结果和局限性。
- 已阅读并定位 SolarWM MiniMax-H3 的因果实现：`H3RawKVCache`、5-frame chunk、
  最多 5 个历史 chunk 的滑动窗口、rollout/replay attention，以及 Stage1
  teacher-forcing/AnyFlow 和 Stage2 SGF/DMD 的调用边界。
- 已阅读并定位 H3-World patched DiT 的 `MiniMaxH3Attention`、FlexAttention
  `block_masks`、`MiniMaxH3DiT.forward` 和 `model_fn_minimax_h3`。当前 H3-World
  的默认路径仍是整段视频在每个 denoising step 做 full-sequence attention；没有
  自动复用跨 chunk 的 raw K/V。

### 真实 chunk + raw-KV 实验（已完成）

在上述 mask-only 原型之后，已经完成真正按 chunk rollout、逐层 raw K/V 提交和窗口
淘汰的 124 帧 H3 实验。公共条件保持一致：`examples/first_frame.png`、同一 prompt、
`forward` action、seed 2、832×480、24 fps、124 帧；三次运行分别使用空闲的 NVIDIA
L40 GPU 3、GPU 1、GPU 2，结果保存在 `H3-World/outputs/`：

| 配置 | 输出 | denoiser forwards | 采样时间 | VAE prepare/decode | 峰值显存 | raw-KV 峰值 | 总墙钟（conditioning 后） |
|---|---|---:|---:|---:|---:|---:|---:|
| 原始 H3，30 steps | `benchmark_baseline_124/baseline.mp4` | 30 | 374.57 s | 15.87 / 11.66 s | 34,958 MiB allocated；37,820 MiB reserved | 0 | 424.09 s |
| 原始 H3，4 steps | `benchmark_baseline4_124/baseline.mp4` | 4 | 55.10 s | 25.24 / 10.37 s | 37,748 MiB allocated；40,452 MiB reserved | 0 | 116.11 s |
| causal chunk + raw-KV，4 steps/chunk | `benchmark_cached_cpu_124/cached.mp4` | 32；另有 8 次 clean commit | 231.69 s | 19.32 / 11.90 s | 37,748 MiB allocated；38,782 MiB reserved | 13,508.6 MiB（CPU） | 284.89 s |

cached 配置为 5 latent-frame chunk、最多 5 个历史 chunk、8 个 chunk、固定 audio noise、
全局 H3 RoPE。每个 chunk 的 4 次采样完成后，额外以 `sigma=0` 做一次 clean forward
写入 cache；不能把最后一个 noisy step 的 K/V 当作历史。8 个 chunk × 50 个 Transformer
层得到 400 次 per-layer commit。cached 视频和两个 baseline 均为 124 帧、5.1667 秒、
H.264，可正常解码。

这次实验证明 33B H3 的 causal chunk 调度和 raw-KV 生命周期在工程上可以跑通，但当前
CPU cache 不是性能优化：它比原始 4-step 全序列路径多出 chunk 级 forward 和 cache 搬运，
总墙钟为 284.89 s；与 30-step 原始路径相比采样阶段减少到 231.69 s，但这不能归因于
Stage2，也不能作为同等质量的加速结论。cached 视频存在背景模糊、场景变形和人物细节
丢失，原因是模型没有经过 Stage1 causal teacher-forcing 或 Stage2 SGF/DMD 训练。

曾在 GPU 2 尝试把 raw-KV 放到 CUDA：第一个 chunk 后 cache 约占 2.64 GiB，下一层
权重加载时仅剩约 443 MiB，申请约 498 MiB 失败并 OOM。因此单张 48-GiB L40 在当前
33B 逐层加载策略下不能同时容纳完整 GPU cache；后续需要 CPU offload、KV 压缩、权重/
cache 分层或多 GPU 才能评估 GPU cache 的真实收益。当前不能宣称已经实现 GPU KV-cache
加速。

质量/连续性统计写入 `H3-World/outputs/2026-09-30-14/benchmark_video_metrics.json`：原始 30-step
的灰度帧间 MAD 均值/p95 为 3.322/12，原始 4-step 为 2.666/9，causal raw-KV 4-step
为 2.986/11；空间边缘差分别为 2.166、2.114、2.939。MAD 较低不等于质量更好，必须
结合视频观察；当前 cached 输出的视觉退化是明确的失败边界。

最终面试题 Demo 已生成：
`H3-World/outputs/h3world_30step_vs_cached4step.mp4`，左侧是原始 H3 30 steps，右侧
是 causal chunk + raw-KV prototype 4 steps/chunk，1664×528、124 帧、24 fps、H.264，
带顶部标签栏。生成脚本仍为 `H3-World/code/causal/make_side_by_side.py`。

原型测试目前共 12 项，新增覆盖 cache eviction、末尾不足完整 chunk、adapter
保存加载和末层梯度约束，以及 CPU
full-history recompute 与 CUDA cached attention 的数值一致性；最近一次完整结果为
`12 passed`（包含 adapter 测试）。

三份 benchmark 视频均为 124 帧、24 fps、832x480、约 5.17 秒；baseline 和 causal
raw-KV 的帧间差、空间边缘差均已使用相同脚本统计。

本项目不下载 SolarWM 官方完整 latent-WDS。该发布包约 14.45 TB，面向完整
SolarWM Stage0.5/Stage1/Stage2 训练和官方数据集评测；当前目标是把因果 chunk、
局部注意力和 KV-cache 思路移植到 H3-World 的最小原型，不要求完整复现 Stage2。
因此不需要 ModelScope 登录、TB 级 latent 数据或官方训练数据集。

### Python 环境

已创建两个隔离环境，避免 H3-World 与 SolarWM 的 CUDA/PyTorch 依赖冲突。
环境由当前用户 `qma` 创建，均使用 uv 管理的 CPython 3.10.21，实际路径为
`/home/lpeng/code/mq_PubDataset/GWM/.venvs/`；系统 `/usr/bin/python` 的 Python 3.12 不用于这两个环境。
两个 venv 中都额外安装了 `pip==26.2.1`，既可以使用 uv，也可以在激活后使用
`python -m pip`。

#### `h3world`

- 路径：`/home/lpeng/code/mq_PubDataset/GWM/.venvs/h3world`
- Python 3.10.21
- PyTorch 2.10.0+cu128
- torchvision 0.25.0+cu128
- torchaudio 2.10.0+cu128
- H3-World `requirements.txt` 已安装，包括 Diffusers 0.37.1、Transformers 4.57.3、Accelerate 1.14.0、PEFT 0.20.0 等。
- PyTorch CUDA 可用，检测到 8 张 NVIDIA L40。
- patched DiffSynth 静态 preflight 通过，实际从 `H3-World/DiffSynth-Studio-h3-v2` 加载。

#### `solarwm-h3`

- 路径：`/home/lpeng/code/mq_PubDataset/GWM/.venvs/solarwm-h3`
- Python 3.10.21
- PyTorch 2.6.0+cu124
- Diffusers 0.40.0
- Transformers 5.12.1
- PEFT 0.20.0
- FlashAttention 2.8.3 已安装
- SolarWM 已 editable 安装。
- `solarwm environment probe` 已通过，CUDA/NCCL/核心包版本可识别。
- 当前 `main` 与官方仓库同步，HEAD 为 `ce1da4e7705391eda8eeda6016c0fd3f614b975e`。

## 已知问题

SolarWM 环境中的 FlashAttention 2.8.3 扩展目前无法在 PyTorch 2.6.0+cu124 下直接导入，报 C++ ABI 错误：

```text
undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_string...
```

SolarWM 的 environment probe 已检测到该问题并回退到 PyTorch native attention，因此核心适配器可以导入；本机没有 `nvcc`，无法直接从源码重编 CUDA 扩展。正式 Stage2 性能实验前，需要在有匹配 CUDA toolkit 的环境中修复，或明确记录 native attention 基线，否则不能声称使用了 FlashAttention 加速。

SolarWM 的 `decord==0.6.0` 可以正常 import，但其 PyPI wheel 的旧平台标签会使
`python -m pip check` 报告 `built for a different platform`；这是 wheel 元数据问题，
不影响当前 import。由于 PyPI 没有对应的 source distribution，暂不在本机重编 decord。

## 尚未完成

- 尚未完成 Stage1 AnyFlow/完整数据规模的 teacher-forcing 训练和 Stage2 SGF/DMD 少步
  蒸馏；当前 raw-KV 视频仍不能称为 Stage2 模型。已完成随机小模型 CUDA smoke，以及
  单片段预训练 H3 末层 QKV LoRA 的有限可行性实验。
- 尚未实现可在单张 48-GiB L40 上运行的 GPU raw-KV 版本；CPU cache 已完成完整长视频，
  GPU cache 的 OOM 边界已记录。
- 尚未证明 causal 少步模型在相同视频质量下优于原始 H3；当前 cached 视频的视觉质量
  明显下降，音频输出也固定为静音实验路径。

## 下一步计划

1. 在少量 H3 latent 上做 Stage1 风格 clean-history teacher forcing/AnyFlow，再比较
   4-step/8-step student 与 30-step baseline；不要把当前 raw-KV 结果称为 Stage2。
2. 优化 cache CPU offload、KV 压缩、权重/cache 分层或多 GPU，重新测量 GPU cache 的
   实际收益和显存占用。
3. 若需要 SolarWM runtime 对照，再处理 FlashAttention ABI；当前 native attention
   限制必须在报告中保留，且不是 H3 causal 原型的必需条件。

### 预训练 adapter 的 4-step 回载验证

同一 `outputs/2026-09-30-15/pretrained_tail_stage1/adapter.pt` 又在 4 steps/chunk、22 帧上完成回载
验证，输出为 `H3-World/outputs/2026-09-30-15/benchmark_trained_tail4_22/cached.mp4`：8 次 denoiser
forward、2 次 clean commit，采样 78.99 s，conditioning 后总墙钟 122.24 s，GPU
allocated peak 25,895.4 MiB，CPU cache 峰值 3,782.4 MiB。它证明 adapter 在多步
cached rollout 中可以稳定加载；这仍然不是 Stage2 少步蒸馏，也没有质量优越性结论。

## 运行环境注意事项

- 当前用户可见工作目录：`/home/lpeng/code/mq_PubDataset/GWM`，物理路径为 `/home/lpeng/code/mq_PubDataset/GWM`；二者是同一工作树。
- 两个 Python 环境：`/home/lpeng/code/mq_PubDataset/GWM/.venvs/h3world` 和 `/home/lpeng/code/mq_PubDataset/GWM/.venvs/solarwm-h3`。
- 不要把代码路径写成 `/work/lpeng/qma/GWM`；该位置只用于共享模型存储的真实目标路径。
- H3-World 必须加载固定 revision 加 patch 的 DiffSynth，不能使用未修改的普通 DiffSynth。
- H3-World 不要 editable install DiffSynth；脚本会检查实际加载的 patched checkout。
- 已完成 2-step/30-step H3 推理、CPU/GPU 小模型训练 smoke、单片段预训练 H3 末层
  LoRA smoke、124 帧 raw-KV benchmark 和 12 项测试；不要在无明确需要时重新启动长时间生成任务。

### 2026-09-30 本轮新增实验

12 项测试仍为通过状态；`code/causal/*.py` 已通过 `py_compile`。为避免损坏的 teacher
被训练脚本静默使用，`code/causal/benchmark.py` 新增 finite 检查：conditioning 完成后
会检查 prompt embedding、video/audio latent 和首帧 anchor，任一 NaN/Inf 都立即失败。

在 seed 2 的 22 帧 teacher clip 上新增 `target_chunk=0` 训练：冻结前 49 层和原有
action LoRA，只训练第 50 层 rank-8 QKV LoRA，共 215,040 个参数、80 步。frozen-feature
replay 最大误差为 0；train loss `0.102748 → 0.082926`（下降 19.3%），validation
`0.083897 → 0.071582`（下降 14.7%）；特征提取 74.17 s，优化 10.83 s，优化阶段
allocated 峰值 2,213.9 MiB。adapter 为
`H3-World/outputs/2026-09-30-22/pretrained_tail_chunk0/adapter.pt`。结合此前 seed 2 的 chunk 1
结果（train 下降 22.8%、validation 下降 14.6%），说明接口同时覆盖无历史首 chunk 和
有 clean 历史的后续 chunk，但仍只是单片段、有限噪声样本。

为做最小独立随机复核，在空闲 GPU 1 上生成了 seed 11 的 22 帧、30-step teacher：
`H3-World/outputs/2026-09-30-22/teacher_baseline30_22_seed11/`。conditioning/latent 均 finite，
采样 146.72 s，allocated 峰值 34,701.8 MiB。对该 clip 的 chunk 1 训练结果为：train
`0.058336 → 0.044018`（下降 24.5%），validation `0.045028 → 0.038609`（下降
14.3%），特征提取 53.53 s，优化 6.64 s，优化阶段 allocated 峰值 2,454.6 MiB，
replay 最大误差 0；adapter 为
`H3-World/outputs/2026-09-30-22/pretrained_tail_seed11_chunk1/adapter.pt`。这只是同一场景不同随机
teacher 的有限复核，尚未构成独立场景泛化或质量提升证据。

一次 GPU 3、seed 7 的 teacher 运行出现 prompt/video NaN，mp4 仅几 KB，已判为无效并
未用于训练；该失败促成上面的 finite 保护。当前没有本项目后台进程，外部 GPU 任务未被
终止。

随后把 seed 11 adapter 回载到 4-step/chunk causal rollout，输出
`H3-World/outputs/2026-09-30-22/benchmark_trained_seed11_tail4_22/cached.mp4`，并生成并排 Demo
`H3-World/outputs/h3world_30step_vs_trained_seed11_tail4_22.mp4`。该 rollout 为 22 帧、
8 次 denoiser forward、2 次 clean commit，采样 47.03 s，conditioning 后总墙钟 89.53 s，
GPU allocated 峰值 34,702.6 MiB，CPU raw-KV 峰值 3,782.4 MiB，100 个逐层 commit。它
证明独立随机 teacher 的 adapter 可加载到多步 causal 调度；仍没有质量等价或 Stage2
少步蒸馏结论。

seed 11 teacher 与 causal 输出的描述性统计也已写入
`H3-World/outputs/2026-09-30-23/seed11_video_metrics.json`：teacher 帧间灰度 MAD 均值/p95 为
`3.578/13`、空间边缘差 `2.233`；causal 为 `3.149/11`、`2.045`。这些指标不能替代
VMAF、光流或人工评价，低 MAD 也可能来自模糊或运动不足。

### 下一阶段路线（2026-09-30）

下一步先做 243 帧（约 10.1 秒）的 paired diagnostic：同一 prompt、首帧、seed 下分别
运行原始 H3 30-step 和 causal raw-KV 4-step，并记录 chunk 边界附近的帧间 MAD、长期
漂移、denoiser 次数、CPU KV 峰值和 GPU 峰值。243 帧对应 72 个 latent frame、15 个
5-frame chunk；4-step causal 预计为 60 次 noisy denoiser forward 加 15 次 clean commit。
先用 243 帧而不是 481 帧，是为了在成本可控的情况下确认 5 秒视频中观察到的跳变是否
在更长时间上累积。

243 帧实验完成后，进入真正的 Stage1 风格多 chunk teacher-forcing：训练数据至少覆盖
首 chunk、带 clean history 的后续 chunk 和一个 held-out clip；比较 4-step/8-step
causal rollout 与 30-step teacher 的边界连续性和画质。只有 student 能在少步下稳定跟随
teacher 后，才考虑 Stage2 风格的轨迹蒸馏；当前 mask/cache inference 仍不能称为 Stage2。

本次资源检查发现 GPU 1、2、4 有外部任务，GPU 3 只有约 28 GiB 空闲，未启动长任务，
也未终止任何外部进程。等到有至少约 40 GiB 可用显存的空闲 L40 后再运行 243 帧配对
benchmark。

### 2026-10-01 00 时段：多 chunk Stage1 风格实验

用户提供了三张空闲 L40 后，新增脚本
`H3-World/code/causal/train_pretrained_multichunk.py`。它在真实预训练 H3 上冻结前
49 层和已有 action LoRA，只训练第 50 层 rank-8 QKV LoRA；同一个 adapter 同时使用
chunk 0（无历史）和 chunk 1（前 5 个 latent frame 为 clean history），每个 chunk 有
6 个训练噪声样本和 2 个验证噪声样本，共 120 步。该实验严格标记为 Stage1-style
clean-history flow matching，不包含 AnyFlow、Stage2 SGF/DMD 或少步 student 蒸馏。

本时段所有原始产物统一放在
`H3-World/outputs/2026-10-01-00/`，目录说明见其中的 `README.md`。三张卡并行完成：

| teacher | train loss | validation loss | replay 最大误差 | adapter |
|---|---:|---:|---:|---|
| seed 2 | `0.077643 → 0.062125`（-20.0%） | `0.063064 → 0.053967`（-14.4%） | 0 | `pretrained_multichunk_seed2/adapter.pt` |
| seed 11 | `0.083957 → 0.066981`（-20.2%） | `0.067879 → 0.058830`（-13.3%） | 0 | `pretrained_multichunk_seed11/adapter.pt` |
| seed 13（本时段新生成） | `0.077194 → 0.061457`（-20.4%） | `0.063288 → 0.054377`（-14.1%） | 0 | `pretrained_multichunk_seed13/adapter.pt` |

三次训练均参数发生改变、梯度有限、没有 OOM；每次可训练参数 215,040，优化阶段
allocated 峰值约 2,453 MiB，完整脚本墙钟约 98–109 s（不含同一进程的完整模型驻留
解释）。seed 13 adapter 已回载到真实 cached rollout，原始结果位于
`2026-10-01-00/benchmark_multichunk_seed13_tail4_22/`：22 帧、4 steps/chunk、8 次
denoiser forward、2 次 clean commit，采样 37.98 s，conditioning 后总墙钟 77.29 s，
GPU allocated 峰值 34,702.6 MiB，CPU raw-KV 峰值 3,782.4 MiB。

本时段最终需要查看的对比视频单独放在 `H3-World/outputs/` 根目录：
`h3world_30step_vs_multichunk_seed13_tail4_22.mp4`。其他最终 Demo（例如
`h3world_30step_vs_cached4step.mp4`）也继续只放在根目录；训练、teacher、benchmark
原始数据不再直接堆在根目录。

需要明确标记：seed 13 multi-chunk 视频不是 Stage1 成功效果。该视频只有 22 帧（约
0.92 秒），右侧 causal 输出肉眼几乎停在原地；teacher 的灰度帧间 MAD 均值/p95 为
`3.490/12`，causal 为 `2.905/11`，与运动被压平的观察一致。三次多 chunk 训练的
loss 下降、replay=0 只证明冻结特征上的单次 clean-history flow prediction 可以优化，
没有证明 free-running rollout 能保持运动。

失败原因已写入实验报告：当前每个 clip 只有 12 个训练噪声和 4 个验证噪声，只训练最后
一个 DiT block 的 QKV LoRA；训练用固定 clean history 和 `sigma={0.3,0.6,0.9}`，推理
用生成历史和 4-step scheduler；训练没有对多步自由运行、动作保持或 chunk 边界施加约束。
因此这不是 SolarWM Stage1 的完整效果，而是 Stage1-style 接口的负结果。下一轮应先统一
8/16-step 的训练和推理时间表、扩大多 clip 数据、训练所有 causal QKV block，并先以
8-step free-running rollout 作为验收，再考虑 4-step。

为隔离采样步数影响，同一个 seed 13 multi-chunk adapter 又做了 8 steps/chunk 回载：
原始数据在 `2026-10-01-00/benchmark_multichunk_seed13_tail8_22/`，采样 62.02 s，
16 次 denoiser forward、2 次 clean commit，GPU allocated 峰值 34,702.6 MiB，CPU
raw-KV 峰值 3,782.4 MiB。8-step 输出的帧间灰度 MAD 均值/p95 为 `2.422/11`，低于
4-step 的 `2.905/11`，抽帧仍近似静止；增加采样步数没有恢复运动，说明主因是训练目标
与 free-running causal rollout 的分布不匹配，而非单纯 4 步不足。8-step 诊断视频
`h3world_30step_vs_multichunk_seed13_tail8_22_diagnostic.mp4` 已放在 outputs 根目录，
同样标记为负结果。

随后完成了整个 `H3-World/outputs/` 的归档整理：历史原始目录和单独的 teacher/benchmark
文件也按香港本地修改时间移动到 `YYYY-MM-DD-HH/` 二级目录（例如
`2026-09-30-15/`、`2026-09-30-22/`），根目录现在只保留 `README.md` 和
`h3world_*.mp4` 最终并排对比视频。每个时间目录都有自己的 `README.md`，根目录索引为
`H3-World/outputs/README.md`；报告中的历史路径已同步更新。

### 2026-10-01 05–06 时段：Stage1 双 anchor 原型和 tail4 容量复测

本时段继续围绕面试题的 Stage1 主线，加入了用户提出的“上一 chunk 最后一帧作为图片条件，同时保留 action”的实现。单独替换原始首帧会丢失全局场景锚点，因此最终采用双 anchor 协议：slot 0 永远保留 H3 原始首帧 image anchor，slot 1 在第 0 个 chunk 先放首帧副本，后续 chunk 替换成上一 chunk 的最后一个 latent frame；slot 1 的 H3 temporal RoPE position 会被 retime 到真实的全局前置帧位置。action text 仍只绑定当前 latent frame，历史 action 不会被当前 video query 读取。

代码改动：

- `H3-World/code/causal/h3_cached.py` 增加并完善 `expand_packed_two_anchors()`，将旧的单 image-anchor packed layout 扩展为双 slot，并保持 audio/text/video index、padding 和 action metadata 的一致性；chunk 0 的副本现在与首帧使用同一个时间位置，后续调用 `retime_anchor_position(..., anchor_slot=1)`。
- `H3-World/code/causal/benchmark.py` 新增 `--anchor-mode dynamic_last_frame_dual`，cached/recompute rollout 和 clean KV commit 使用同一套双 anchor 规则；baseline 被明确禁止使用该 causal-only 模式。
- `H3-World/code/causal/train_pretrained_multichunk.py` 同步支持双 anchor，训练时使用 clean previous-history tail，推理时使用同样的第二 slot，避免训练和 rollout 的 packed layout 不一致。
- `H3-World/tests/test_h3_cached.py` 新增双 anchor index remap、原始 slot 保持不变和 slot 1 全局 retime 测试。

验证结果：`py_compile` 通过，完整测试为 **14 passed**；GPU1 上的双 anchor 22 帧 smoke rollout 也成功完成（1 step/chunk、2 次 clean commit、100 个逐层 cache commit），结果在 `H3-World/outputs/2026-10-01-08/dual_smoke_22_corrected/`。

完成了两个 124 帧（5.17 秒）tail4 Stage1-style 复测，均为 8 steps/chunk、scheduler shift 2.22、rank-8 QKV LoRA、160 optimizer steps、chunks 0–7：

| 配置 | train loss | validation loss | rollout 灰度 MAD 均值/p95 | 空间边缘差 | adapter |
|---|---:|---:|---:|---:|---|
| 4 blocks，dynamic last frame | `0.10522 → 0.08495` | `0.10618 → 0.09324` | `3.786 / 15` | `2.013` | `outputs/2026-10-01-06/stage1_dynamic_shift2_tail4/adapter.pt` |
| 双 anchor，4 blocks，dynamic last frame | `0.09880 → 0.07969` | `0.10231 → 0.09009` | `4.225 / 16` | `2.074` | `outputs/2026-10-01-08/stage1_dual_shift2_tail4/adapter.pt` |
| H3 原始 30-step teacher | — | — | `4.295 / 15` | `2.226` | `outputs/2026-10-01-01/baseline30_seed13_124/baseline.mp4` |

完整 dual-anchor rollout 位于 `H3-World/outputs/2026-10-01-08/stage1_dual_shift2_tail4_rollout_124/cached.mp4`，三者统计位于 `H3-World/outputs/2026-10-01-08/metrics_124_shift2_dual_tail4.json`；单 adapter 的统计位于 `H3-World/outputs/2026-10-01-06/metrics_124_shift2_tail4.json`。根目录的最终并排视频尚未替换，避免把仍有明显长程漂移的 dual 结果误标成最终成功结果。

当前结论：双 anchor 的架构是合理的，且 clean-history Stage1 训练链路、cache、位置重映射已经真实跑通；但在单场景、少量样本、仅 4 个 tail blocks 的条件下，free-running rollout 仍在约第 30 帧以后出现重影、亮度漂移和 chunk 边界突变，dual 版本没有证据表明它已经优于单 anchor。tail4 单 anchor 的整体 MAD 较低但在第 34、85、102 帧附近有较大边界峰值，dual 版本保留了更多运动幅度却在长程上仍然模糊。这说明瓶颈仍是 clean-history teacher forcing 与 generated-history rollout 的分布差异，以及当前 raw latent patchify 并非真正的 RGB decode/re-encode image anchor；不能把该结果称为 SolarWM Stage2，也不能声称已经实现少步质量等价。

下一步应保持这套双 anchor 作为可复现实验分支，优先做 scheduled/noisy-history 或多 clip 训练，并对 chunk boundary 加入显式 continuity loss；如果继续使用双 anchor，应再比较真正的 RGB decode/re-encode anchor，而不是修改随机种子或把 ordinary 8-step solver 误称为 Stage2。

### 2026-10-01 06–07 时段：scheduled/noisy-history 诊断

为直接检查 generated-history 分布偏移，使用相同 124 帧 teacher、双 anchor、4 个 tail blocks、shift 2.22 和 160 步训练，给 clean history 加 `history_noise_std=0.08`，结果在 `H3-World/outputs/2026-10-01-09/stage1_dual_shift2_tail4_historynoise08/`。训练本身成功，train loss `0.10213 → 0.08516`、validation `0.10841 → 0.09908`、replay error `0`；但 8-step free-running 回载的 MAD/边缘差为 `5.362/21`、`2.773`，明显差于 clean-history dual adapter 的 `4.225/16`、`2.074`，并且第 102、119 帧边界峰值更大。

这次结果不支持“简单给历史 latent 加高斯噪声就能解决 rollout drift”。它仍然是有效的 Stage1 负诊断：scheduled history 需要模拟真实多步 student/generated history 的结构性误差，而不是只增加独立像素噪声。当前保留 clean-history dual adapter 作为主架构示例，noisy-history adapter 作为失败对照；下一轮应考虑 chunk-level rollout unroll、边界 continuity loss 和真实 RGB image anchor。

### 2026-10-01 07 时段：persistent cached-history Stage1 修正与双 anchor 长片复测

上一轮尝试把训练端改为持久化 causal KV cache 时，在回放训练特征的阶段发现了一个实现错误：
尾部 4 个 DiT block 都复用了第一个尾 block 的 `layer_index`，使第 47–49 层读取了第 46
层的历史 K/V。该轮输出位于 `outputs/2026-10-01-11/`，已标记为 `INVALID.md`，adapter
和任何 rollout 都不作为实验结论。

修正内容：`causal/pretrained_lora.py` 新增 `replay_tail()`，按实际 block index 逐层回放；
训练脚本恢复严格的 zero-initialized adapter replay 检查。小模型回归和完整测试均通过，训练
端的 replay 最大误差重新为 `0`。这一步很关键：Stage1 的历史 KV 必须按层保存和读取，不能
把“cache 能运行”误认为“cache 语义正确”。

修正后的真实 H3 训练产物在
`outputs/2026-10-01-12/stage1_causal_cache_dual_tail4/`：124 帧 teacher、chunk 0–7、
4 个末端 DiT blocks、rank-8 QKV LoRA、160 optimizer steps、H3 scheduler `shift=2.22`、
1 个训练噪声 seed 和 2 个验证 seed。结果为：

| 指标 | 结果 |
|---|---:|
| train loss | `0.10318 → 0.08088` |
| validation loss | `0.10897 → 0.09149` |
| train/validation samples | `32 / 16` |
| replay max error | `0` |
| trainable parameters | `860,160` |
| optimizer wall time | `84.4 s` |

回载 rollout 位于
`outputs/2026-10-01-12/stage1_causal_cache_dual_tail4_rollout_124/cached.mp4`。它使用
8 steps/chunk、8 个 5-frame chunk，执行 `64` 次 noisy denoiser 和 `8` 次 clean KV commit；
采样时间 `323.55 s`，GPU allocated 峰值约 `39.6 GiB`，CPU raw-KV 峰值约 `13.2 GiB`。
与同 seed 的原始 H3 30-step teacher 对比：

| 输出 | 灰度帧间 MAD 均值/p95 | 空间边缘差 |
|---|---:|---:|
| H3 30-step teacher | `4.295 / 15` | `2.226` |
| corrected cached-history Stage1-style | `4.277 / 16` | `2.275` |

整体运动幅度已经接近 teacher，但 chunk 边界相邻帧仍有明显峰值（例如 decoded frame 34、
68、85、102 附近），所以这仍是 Stage1-style feasibility result，不是质量等价结果，也
不是 SolarWM Stage2。根目录可直接播放的并排视频是
`outputs/h3world_30step_vs_stage1_cached_history_dual_tail4_8step_124.mp4`。

### 上一末帧作为图片条件：latent 与真实 RGB 两条路径

用户提出的“每个新 chunk 额外给上一 chunk 最后一帧，并继续给 action”已经落成双 anchor
协议：slot 0 保留 H3 原始首帧，slot 1 在 chunk 0 用首帧副本，后续 chunk 使用上一段可用
历史的末帧；action text 仍按当前 latent frame 绑定，因此不会因为增加图片 anchor 而丢掉
动作条件。现有 Stage1 adapter 训练的是 normalized temporal-latent patchify 版本。

另外新增了 `last_frame_image_anchor()` 和 benchmark 的
`--anchor-mode dynamic_last_frame_rgb_dual`。它严格走 H3 图片条件语义：把当前已经生成的
latent prefix 解码为 RGB，取 prefix 的最后一个可见 RGB frame，再用
`encode_video(process_image=True)` 重编码并 patchify；不会把 temporal latent 直接冒充图片
latent。由于 H3 temporal VAE 需要上下文，不能只 decode 一个 latent token 后取第一张补帧。

22 帧、4 steps/chunk 的 RGB-prefix smoke 位于
`outputs/2026-10-01-12/rgb_prefix_smoke_22/`，VAE anchor 额外耗时约 `25.4 s`。相对于
同一 22 帧 30-step teacher，它的描述性统计是 MAD `2.573/8`、空间边缘差 `2.754`；这只是
未训练的 causal anchor ablation，较低 MAD 不能解释为更高质量。它证明了“上一末帧 + action”
可以用 H3 原生 image branch 实现，下一步若要验证质量，必须把同样的 RGB-prefix 协议纳入
Stage1 teacher forcing，而不能训练 latent anchor 后再只在推理时替换编码方式。

本时段最后的代码检查为：`pytest -q H3-World/tests` **16 passed**，
`python -m py_compile H3-World/code/causal/*.py` 通过。当前主线仍是：先完成严格对齐的
Stage1 causal training 和长视频连续性，再考虑 Stage2 的 SGF/DMD trajectory distillation；
普通 8-step solver、causal mask 或 KV cache 本身都不应被称为 Stage2。

### 2026-10-01 08–09 时段：RGB prefix anchor 的正式 Stage1 训练与长片对比

为真正验证“上一 chunk 最后一帧作为图片条件”的想法，而不是只在推理时替换条件，训练脚本
新增 `--anchor-mode dynamic_last_frame_rgb_dual`。对于每个 clean-history chunk，训练端先
解码当前可用 latent prefix，取 prefix 的最后 RGB frame，再用 H3 的
`encode_video(process_image=True)` 重新编码 slot 1；rollout 端使用完全相同的 prefix 规则。
slot 0 仍保留原始 H3 首帧，action text 仍只绑定当前 latent frame。

RGB-prefix Stage1 训练产物在
`outputs/2026-10-01-13/stage1_causal_cache_rgb_dual_tail4/`：

| 指标 | latent dual anchor | RGB prefix dual anchor |
|---|---:|---:|
| train loss | `0.10318 → 0.08088` | `0.09810 → 0.08012` |
| validation loss | `0.10897 → 0.09149` | `0.10255 → 0.09040` |
| replay max error | `0` | `0` |
| trainable parameters | `860,160` | `860,160` |

RGB 124 帧 free-running rollout 位于
`outputs/2026-10-01-13/stage1_causal_cache_rgb_dual_tail4_rollout_124/cached.mp4`，
对应并排视频为根目录的
`h3world_30step_vs_stage1_rgb_cached_history_dual_tail4_8step_124.mp4`。它执行同样的
64 次 noisy denoiser 和 8 次 clean commit，但由于每个后续 chunk 要切换 VAE 并编码图片，
采样时间增加到 `674.72 s`；7 次 RGB anchor 的额外 VAE 处理合计约 `202.8 s`，CPU raw-KV
峰值仍约 `13.2 GiB`。

与原始 30-step teacher 的统计：

| 输出 | 灰度帧间 MAD 均值/p95 | 空间边缘差 |
|---|---:|---:|
| H3 30-step teacher | `4.295 / 15` | `2.226` |
| latent dual cached-history | `4.277 / 16` | `2.275` |
| RGB-prefix dual cached-history | `4.200 / 16` | `2.138` |

RGB 版本整体运动幅度和边缘差略低于 teacher，且没有证据显示它解决长期边界漂移；后段边界
相邻帧 MAD 仍可达到 `13–17`（frame 85、102 附近）。因此当前实验回答是：**上一末帧作为
图片条件在接口和训练语义上可行，但在单场景、小数据、只训练 4 个尾 blocks 的 Stage1
原型中没有带来可确认的质量或连续性提升；它还显著增加了推理 VAE 开销。** 这不是 RGB
方案无效的最终结论，因为还没有做 generated-history unroll、多 clip 数据或 continuity loss。

本时段 RGB 训练和 rollout 均完成，所有新增 smoke/long-run 文件按小时归档；根目录保留两条
可播放长片，便于直接比较 latent dual 与 RGB dual 的差异。

## 2026-10-01 08–09 时段：mixed generated-history Stage1 诊断与当前主 Demo

本时段继续完善 Stage1 训练协议，直接检查“clean-history teacher forcing 与 free-running
generated-history 不一致”是否可以通过按比例混合历史来缓解。`train_pretrained_multichunk.py`
新增：

```text
--history-latents PATH
--history-mix 0..1
```

其中 `history-mix=0` 是 teacher clean history，`history-mix=1` 是给定 detached causal
rollout history；当前 chunk 的 supervised flow target 仍然来自原始 30-step teacher，没使用
未来 chunk，也没有 SGF/DMD 或 Stage2 student。这个开关只用于 scheduled-history 诊断。

历史来源使用 2026-10-01 16 时段的双 anchor、blend=0.35、overlap=2 rollout latent。两套
训练都覆盖 124 帧的 chunk 0–7，末 4 个 DiT block 使用 rank-8 QKV LoRA，H3 scheduler
shift=2.22，160 optimizer steps：

| 目录 | history mix | train loss | validation loss | replay max error |
|---|---:|---:|---:|---:|
| `outputs/2026-10-01-17/stage1_mixed_history025_dual_tail4/` | 0.25 | `0.11228 → 0.08230` | `0.11542 → 0.09854` | `0` |
| `outputs/2026-10-01-17/stage1_mixed_history050_dual_tail4/` | 0.50 | `0.14743 → 0.10088` | `0.13406 → 0.11759` | `0` |

两个 adapter 均通过逐层 zero-initialized replay 检查，并回载到同 seed 13、124 帧、8
steps/chunk 的真实 cached rollout。结果如下：

| 输出 | 灰度 MAD 均值/p95 | 空间边缘差 | sampling |
|---|---:|---:|---:|
| H3 30-step teacher | `4.295 / 15` | `2.226` | 原始 teacher |
| mixed history 0.25 | `5.207 / 20` | `4.100` | `315.84 s` |
| mixed history 0.50 | `6.479 / 24` | `4.318` | `278.99 s` |

抽帧接触图显示两套 mixed-history 分支运动比之前近似静止的负结果更明显，但停车场结构
在后续 chunk 更快偏移，不能作为质量改善。结论是：把一条已有 generated rollout 以固定
比例混入单场景 teacher history 没有解决分布偏移，反而放大了 drift；这两套结果保留为
Stage1 负对照。它们都不是 Stage2，因为没有 trajectory distillation、AnyFlow 或 SGF/DMD。

### 边界诊断（已由后续 tail16 主 Demo 超越）

该边界诊断视频仍保留用于分析 soft overlap，但当前根目录主 Demo 已在后续 09–10 时段
切换为 tail16 causal 适配版本：

```text
H3-World/outputs/h3world_30step_vs_stage1_cached_history_dual_anchor_softoverlap2_8step_124.mp4
```

它对应 `outputs/2026-10-01-16/stage1_cached_dual_boundaryblend035_overlap2_latest_124/`：
8 steps/chunk、双 anchor、blend `0.35`、前两帧衰减 soft overlap。统计为 MAD `3.665/13`、
边缘差 `2.274`，teacher 为 `4.295/15`、`2.226`。该 overlap 使接触帧不再出现整段场景
的硬瞬移，但 frame 85、102 一带仍有 temporal VAE ghosting；更低 MAD 不能单独证明质量
更好。因此它是当前 Stage1-style 可行性主候选，不是质量等价或 Stage2 Demo。

本时段还完成：

- `outputs/README.md` 增加 2026-10-01-15/16/17 的索引和负结果说明；
- `README.md`、`docs/causal_prototype_report.md` 同步记录 overlap=2 主候选和 mixed-history
  负结果；
- 原始日志已放入 `outputs/2026-10-01-17/`，没有继续堆在 outputs 根目录；
- 新增 `history-mix` 后，`.venvs/h3world/bin/pytest -q H3-World/tests` 为 **16 passed**，
  `py_compile H3-World/code/causal/*.py` 通过。

下一步主线仍然是：如果要继续提高 Stage1，必须使用多 clip 的 generated-history unroll、
更接近 SolarWM AnyFlow 的 flow-map 训练和明确的 temporal continuity objective；不能把
soft overlap、causal mask、KV cache 或普通 8-step solver 称为 Stage2。当前用户提出的
“上一 chunk 最后一帧 + 当前 action”已经有 latent dual 和 RGB-prefix dual 两条可运行路径，
但单场景原型还没有证明它能稳定改善长片质量。

## 2026-10-01 09–10 时段：扩大 causal 适配容量，生成无硬瞬移主 Demo

mixed-history 结果表明，直接把一条旧 generated rollout 混入单场景训练不能解决漂移。本时段
改做架构容量对照：保持同一 seed、prompt、首帧、124 帧 teacher、双 anchor、persistent
cached-history、H3 scheduler `shift=2.22` 和 8 steps/chunk，只把尾部 QKV LoRA 从 4 个
DiT block 扩展到 8/16 个 block。没有修改随机种子，没有读取未来 chunk，也没有用输出后处理
作为唯一修复。

| 配置 | 可训练参数 | train loss | validation loss | replay | 124 帧 rollout MAD/p95 | edge diff |
|---|---:|---:|---:|---:|---:|---:|
| tail4（此前主候选） | `860,160` | `0.10318→0.08088` | `0.10897→0.09149` | `0` | `4.277/16` | `2.275` |
| tail8 | `1,720,320` | `0.10087→0.06886` | `0.10414→0.08553` | `0` | `4.508/16` | `3.157` |
| tail16 | `3,440,640` | `0.10087→0.06384` | `0.10414→0.08219` | `0` | `4.786/18` | `2.563` |

原始产物：

```text
outputs/2026-10-01-18/stage1_clean_dual_tail8/
outputs/2026-10-01-18/stage1_clean_dual_tail16/
outputs/2026-10-01-18/rollout_tail8_124/
outputs/2026-10-01-18/rollout_tail16_124/
```

tail16 的并排主 Demo 已放到 outputs 根目录：

```text
H3-World/outputs/h3world_30step_vs_stage1_tail16_dual_anchor_8step_124.mp4
```

抽帧检查 frame 17、34、51、68、85、102、119：tail16 右侧的停车场和人物轨迹在 chunk
边界处保持连续，没有 tail4 + soft overlap 版本那种大范围 temporal-VAE ghosting 或整段
场景瞬时切换。tail16 的长期路径仍会逐渐偏离 30-step teacher（例如后段进入不同的坡道
方向），所以当前结论是“causal chunk 的硬瞬移问题已通过扩大适配容量显著缓解”，不是
“teacher 质量已经复现”。

对同一 tail16 adapter 另跑了 `blend=0.2`、overlap=1 的后处理对照：MAD `5.206/20`、edge
`2.923`，比无后处理 tail16 的 `4.786/18`、`2.563` 更模糊，因此没有把它设为主 Demo。
2026-10-01-16 的 blend=0.35、overlap=2 仍保留用于边界约束说明，不取代 tail16 架构版本。

文档已同步更新：根目录 README 现在把 tail16 并排视频列为当前主 Demo，并明确说明它解决的是
chunk 边界硬切换而非长期语义 drift；`docs/causal_prototype_report.md` 和
`H3-World/outputs/README.md` 均加入 tail8/tail16 对照、指标和负结果。

## 2026-10-01 09–10 时段：tail16 的 RGB-prefix image anchor 复测

为直接验证用户提出的“上一 chunk 最后一帧作为图片条件”，在 tail16 adapter 上使用
`dynamic_last_frame_rgb_dual` 做了完整 124 帧训练和 rollout。slot 1 的条件严格走：当前
prefix temporal VAE decode → 取最后可见 RGB frame → `encode_video(process_image=True)` →
patchify；slot 0 仍为原始首帧，当前 action 绑定不变。

训练目录：

```text
outputs/2026-10-01-19/stage1_rgb_dual_tail16/
```

配置为 124 帧、8 chunks、末 16 blocks、rank-8 QKV LoRA、160 步、shift=2.22。训练结果：

```text
train loss:       0.09810 -> 0.06211
validation loss:  0.10255 -> 0.08152
replay max error:  0
trainable params: 3,440,640
```

rollout：

```text
outputs/2026-10-01-19/rollout_rgb_tail16_124/cached.mp4
```

统计为 MAD `4.585/17`、空间边缘差 `2.368`，比 latent tail16 的 `4.786/18`、`2.563` 更接近
teacher `4.295/15`、`2.226`。采样耗时 `567.41 s`，latent tail16 为 `304.52 s`；7 次
RGB-prefix anchor 处理约 `202.8 s`，CPU KV 峰值约 `13.2 GiB`。根目录新增并排视频：

```text
H3-World/outputs/h3world_30step_vs_stage1_rgb_tail16_dual_anchor_8step_124.mp4
```

该结果说明“上一末帧 + action”在 H3 原生图片分支中可训练、可自由 rollout，且连续性指标有
轻微改善；但代价是几乎翻倍的采样时间，因此效率主 Demo 仍采用 latent tail16，RGB tail16
作为用户方案的直接对照。

## 2026-10-01 10 时段：243 帧（10.125 秒）长时段验证

为检查 5 秒之后的行为，使用相同 seed 13、双 latent anchor、tail16 adapter、H3 shift=2.22
和 8 steps/chunk 生成了 243 帧 causal 视频：

```text
outputs/2026-10-01-10/tail16_dual_243_verified/cached.mp4
```

它包含 15 个 chunk、120 次 noisy denoiser forward、15 次 clean KV commit，采样
`633.52 s`，GPU allocated 峰值约 `39.0 GiB`，CPU raw-KV 峰值约 `13.2 GiB`。视频长
`10.125 s`，灰度 MAD 为 `5.044/19`，空间边缘差 `3.400`，最大相邻帧 MAD 约 `9.73`；
抽帧和周期性边界检查显示没有整段场景瞬时切换，但最后几秒出现明显的停车场几何 drift。
这证明 causal 分块和 CPU KV cache 可以把序列延长到 10 秒以上，不代表长时段质量已经通过。

同一 seed、相同 prompt、首帧和 resolution 的 H3 30-step full-sequence baseline 也尝试生成
243 帧，但在第一次 denoiser 调用申请额外 874 MiB 时 OOM（44.4-GiB L40 只剩约 579 MiB）。
失败记录在：

```text
outputs/2026-10-01-10/baseline30_243_verified/
```

该 OOM 是显存边界证据，不能作为画质对照；它同时说明当前分块 + CPU raw-KV 路径对长视频
有实际工程价值。124 帧 tail16 仍然是面试主 Demo，243 帧作为长时段和显存报告附件。

官方 SolarWM 对照路径也已核对并写入根 README：MiniMax-H3 的
`stage0p5.py`、`stage1.py`、`stage2.py` 以及对应 YAML 配置分别明确了 bidirectional FM、
`causal_mode: teacher_forcing` + `objective: anyflow_forward_map`、以及
`causal_mode: self_gradient_forcing` + frozen teacher/trainable critic。当前 H3 原型只声称
Stage1-style causal teacher-forcing，不把普通 flow loss、KV cache 或 8-step solver 称作
AnyFlow/Stage2。

## 2026-10-01 20 时段：固定 action intervention 已完成

本时段完成面试题最后一个关键验收：验证 SolarWM-style causalization 是否保留 H3-World
原有的 action interface。benchmark 新增 `--action-preset`，支持 W/S/A/D 别名，分别解析为
H3 的 `forward`、`back`、`strafe-left`、`strafe-right`，并把 action preset 写入每个
`setup.json` 和运行 JSON。没有修改随机种子。

固定条件：同一首帧、同一 prompt、seed `13`、832×480、124 RGB frames（5.17 秒）、同一
H3 action LoRA、同一初始 video/audio noise。对 W/S/A/D 各跑了两条完整视频：

1. 原始 H3-World：30 full-sequence denoiser steps，fixed first-frame image condition；
2. causal Stage1-style：tail16 rank-8 QKV LoRA、5 latent-frame chunk、5-chunk sliding
   history、persistent per-layer raw KV、CPU offload、latent dual anchor、8 steps/chunk。

四个 causal case 均完成 `64` 次 noisy denoiser forward、`8` 次 clean commit，adapter replay
 误差仍为 `0`；采样时间为 W/S/A/D=`342.7/347.5/348.5/344.5 s`，GPU allocated peak
 `38.6–39.0 GiB`，CPU raw-KV peak `13.2 GiB`。原始 30-step case 的采样时间为
 `359.2/362.3/373.3/371.2 s`，没有 KV cache。因而这组实验不是把 64 次 forward 声称成
 8 次 denoiser；它仍然是 8 steps/chunk × 8 chunks，并通过短 active sequence 与历史 KV
 复用获得因果分块效率。

新增产物：

```text
H3-World/outputs/2026-10-01-20/action_W_baseline30_124/baseline.mp4
H3-World/outputs/2026-10-01-20/action_S_baseline30_124/baseline.mp4
H3-World/outputs/2026-10-01-20/action_A_baseline30_124/baseline.mp4
H3-World/outputs/2026-10-01-20/action_D_baseline30_124/baseline.mp4
H3-World/outputs/2026-10-01-20/action_W_tail16_124/cached.mp4
H3-World/outputs/2026-10-01-20/action_S_tail16_124/cached.mp4
H3-World/outputs/2026-10-01-20/action_A_tail16_124/cached.mp4
H3-World/outputs/2026-10-01-20/action_D_tail16_124/cached.mp4
H3-World/outputs/h3world_stage1_tail16_action_intervention_grid_124.mp4
```

四行 4×2 review grid 的左列是 Original H3-World、右列是 causal Stage1-style，行顺序为
W/S/A/D。`action_summary.json` 统一记录 inference time、GPU peak、CPU KV、denoiser
forwards、clean commits、frame MAD、boundary MAD 和 action response。causal 124 帧 MAD
均值/p95 分别为：W=`4.785/18`、S=`4.914/19`、A=`4.925/19`、D=`4.803/18`；RGB
boundary score（帧 17、34、51、68、85、102、119 的相邻帧 MAD 均值）分别为
`5.69/6.14/6.00/5.72`。这些是运动/连续性描述，不是视频质量分数。

action intervention 结论：W/S causal 视频均值 RGB 差异 `7.50`，A/D 差异 `6.97`，接触图中
人物和停车场轨迹确实随 action 改变，说明 action 文本行和 packed action rows 没有因 causal
mask/KV cache 而被抹掉。原始 H3 的全画面 Farneback horizontal flow 对 A/D 给出
A=`+1.08`、D=`−1.60` 的明显方向差；causal 全画面 flow 受背景和长期 drift 影响，方向
proxy 不够稳定，所以报告为 action-response 辅助证据，不能夸大为严格 action accuracy。

需要明确的限制：tail16 adapter 的训练 teacher 是单场景 W action clip，这次 S/A/D 是同一
checkpoint 的跨 action robustness/intervention，不是多 action causal fine-tuning。当前结果
已经足够支持面试题的最小结论：H3-World 可以做 SolarWM-style causal chunk rollout，且 action
interface 仍然有效；剩余主要问题是 generated-history distribution shift 和长期 drift，
不是 action binding 消失。后续不再扩展 SGF/DMD、更多 anchor 或新的 fancy metric；最终 demo
固定为主 tail16 causal 视频、RGB-anchor ablation、generated-history mix 负对照，action grid
作为 action-control 证据。

本时段新增代码：

- `H3-World/code/causal/evaluate_action_control.py`：中心区域 Farneback optical-flow
  action-response proxy，明确输出方向约定和局限；
- `H3-World/code/causal/make_action_grid.py`：生成 W/S/A/D 四行 Original-vs-Causal 网格；
- `H3-World/code/causal/benchmark.py`：`--action-preset W/S/A/D` 和 action metadata。

验证：四 action benchmark 均 `status=complete`、每个 MP4 124 帧；
`.venvs/h3world/bin/python -m py_compile H3-World/code/causal/*.py` 通过。已有测试仍为
`16 passed`，没有残留 benchmark 进程。

## 2026-10-02 00 时段：多 action teacher-forcing 与 action-control 复核

上一节的单 W teacher adapter 只能说明 action 行没有被 causal mask 完全抹掉，不能说明
W/S/A/D 的方向控制仍然准确。本时段补齐了四个固定条件的原始 H3 teacher clip，并训练
了一个共享的 multi-action causal adapter。四个 teacher 都使用 seed `13`、同一首帧、同一
prompt、同一初始 video/audio noise 和 124 帧；只改变 H3 action：W=`forward`、S=`back`、
A=`strafe-left`、D=`strafe-right`。

### shared adapter 训练

四个 teacher latent 位于：

```text
outputs/2026-10-01-21/action_W_teacher_latents/
outputs/2026-10-01-21/action_S_teacher_latents/
outputs/2026-10-01-21/action_A_teacher_latents/
outputs/2026-10-01-21/action_D_teacher_latents/
```

tail8 训练成功，目录为 `outputs/2026-10-01-21/stage1_multiaction_tail8/`；它有
1,720,320 个可训练参数，train `0.1588316→0.1419430`，validation
`0.1908010→0.1823134`，replay error `0`。它的四个固定 action 和 W:3,A:2,D:3、
D:2,S:3,A:3 schedule rollout 已经完整结束。

为检验容量限制，使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 在 GPU1 重跑
末 16 个 DiT block。目录为：

```text
outputs/2026-10-02-00/stage1_multiaction_tail16_retry/adapter.pt
```

这次不是 OOM：adapter 已保存且格式为 `h3_causal_tail_qkv_v2`，16 个 block、
3,440,640 个可训练参数，replay error `0`，train `0.1588316→0.1385778`，validation
`0.1908010→0.1801894`，训练 allocated peak 约 `40.86 GiB`。之前
`outputs/2026-10-01-21/stage1_multiaction_tail16/training.json` 的 OOM 记录仍保留，说明
该配置需要 allocator 处理和空闲显存；不能把那次失败解释为算法失败。

### tail8 固定 action 结果

tail8 rollout 目录为 `outputs/2026-10-01-21/action_*_multiaction_tail8/`。每条视频均为
124 帧、5.17 秒、64 次 noisy denoiser forward、8 次 clean KV commit，GPU allocated
峰值约 `39.2 GiB`，CPU raw-KV 峰值 `13.19 GiB`。中心区域 Farneback 水平 flow proxy 为：

| Action | Original H3 flow | Causal tail8 flow | Causal gray MAD | Causal boundary RGB MAD |
|---|---:|---:|---:|---:|
| W | `-1.111` | `-0.751` | `3.904` | `6.010` |
| S | `-1.084` | `-0.671` | `3.859` | `6.142` |
| A | `+1.077` | `-0.575` | `3.838` | `5.891` |
| D | `-1.602` | `-0.676` | `3.839` | `6.029` |

tail8 causal 的 W/S 和 A/D mean RGB 差异分别为 `3.80` 和 `7.05`。因此 action 条件会
改变输出，但水平 flow 仍然同号；不能写成 W/S、A/D 方向控制已经保真。

### tail16 shared adapter 结果

tail16 四条固定 action 和两个 schedule 已经在 GPU0/1/2/3/4/6 完整完成，目录为：

```text
outputs/2026-10-02-00/action_W_multiaction_tail16_retry/
outputs/2026-10-02-00/action_S_multiaction_tail16_retry/
outputs/2026-10-02-00/action_A_multiaction_tail16_retry/
outputs/2026-10-02-00/action_D_multiaction_tail16_retry/
outputs/2026-10-02-00/action_schedule_WAD_multiaction_tail16_retry/
outputs/2026-10-02-00/action_schedule_DSA_multiaction_tail16_retry/
```

每条仍是 `124` 帧、`64 + 8` 次 noisy/commit、CPU KV `13.19 GiB`，GPU allocated 峰值
约 `38.6–39.0 GiB`。新的 4×2 对比视频为：

```text
outputs/h3world_stage1_multiaction_tail16_action_grid_124.mp4
```

统一指标表和 JSON 为：

```text
outputs/2026-10-02-00/action_multiaction_tail16_summary.md
outputs/2026-10-02-00/action_multiaction_tail16_summary.json
```

tail16 的全画面 flow proxy 为 W=`-0.540`、S=`-0.623`、A=`-0.594`、D=`-0.579`；W/S
和 A/D 的 causal mean RGB 差异分别只有 `4.10` 和 `3.56`。相比原始 H3 的 A=`+1.077`
和 D=`-1.602`，方向分离没有恢复，扩大 QKV LoRA 容量也不能解决这个问题。causal MAD
约为 `3.18–3.38`，低于 teacher 不能当作质量更好；它可能表示运动减弱或生成漂移。

### chunk-level schedule 结果

W:3,A:2,D:3 的 tail16 逐段 flow proxy（W=`0:51`、A=`51:85`、D=`85:124`）为：

```text
W -0.746, A -0.603, D -0.214
```

原始 H3 对应为 `-0.033, +1.564, -1.278`。D:2,S:3,A:3 的 tail16 逐段 flow（D=`0:34`、
S=`34:85`、A=`85:124`）为：

```text
D -0.823, S -0.648, A -0.302
```

原始 H3 对应为 `-1.212, -1.635, -1.210`。新增带输入动作标注的并排 schedule 视频：

```text
outputs/h3world_schedule_WAD_multiaction_tail16_124.mp4
outputs/h3world_schedule_DSA_multiaction_tail16_124.mp4
```

视频中的 action label 是输入时间表，不是模型预测结果；这些 schedule 证明 action rows
确实可以按 chunk 进入 causal pipeline，但当前 Stage1 clean-history 训练没有维持原始 H3
的方向响应。

### 当前结论边界

多 action teacher-forcing 比单 W robustness intervention 更接近题目要求，但 tail8 和
tail16 都得到同一个结论：

1. causal mask、persistent raw KV、clean commit、124 帧 rollout 和 action schedule 机械链路可运行；
2. 变更 action 会改变 causal 输出，说明 action conditioning 没有完全消失；
3. W/S/A/D 的方向性以及 chunk-level 方向切换没有保真，不能宣称“action control 完全保留”；
4. 主要剩余问题是 generated-history distribution shift、单场景训练容量和 Stage1 目标本身，
   而不是把 KV cache 再扩展一倍；
5. 这组负结果反而说明为什么 SolarWM 需要 Stage2 self-gradient forcing / distillation。

因此面试报告的最终表述应改为：

> We successfully causalized H3-World with chunk-wise attention and persistent KV caching.
> Multi-action clean-history teacher forcing keeps the action rows active and produces
> action-dependent videos, but the current Stage1 prototype does not yet preserve the
> original H3 directional action response under generated-history rollout. Stable action
> grounding and few-step long-horizon quality require generated-history distribution matching,
> motivating SolarWM Stage2-style training.

本时段新增/修改：`train_pretrained_multiaction.py` 已用于四 action shared adapter，新增
`summarize_action_experiment.py` 统一 timing/memory/MAD/boundary/flow 表，
`make_side_by_side.py` 支持在 schedule 视频中标出每帧输入 action。README、
`docs/causal_prototype_report.md`、`H3-World/outputs/README.md` 和本文件均已同步这个
结论。最后验证已完成：所有 6 条 tail16 rollout 为 `status=complete`、124 帧、64 noisy
forwards、8 clean commits；所有 causal Python 文件通过 `py_compile`，测试为 `16 passed`。
不再增加 anchor trick 或 SGF/DMD；Stage1 prototype 可以收工，后续若继续应进入
generated-history distribution matching / SolarWM Stage2，而不是继续扩展当前诊断模块。

## 2026-10-02 04 时段：action fidelity decomposition 与保守 adapter 修复

根据建议文件 `pasted-text-1.txt`，本时段不再把“动作失真”归因于单一模块，补做了
causal function shift 与 generated-history shift 的分解。

### 39 帧、30 steps/chunk、无 adapter 的 generated-history 对照

新增目录：

```text
outputs/2026-10-02-04/action_A_base_generated30_39/
outputs/2026-10-02-04/action_D_base_generated30_39/
```

两条运行使用同一首帧、prompt、seed=13、同一初始噪声、dynamic dual latent anchor，
只把历史来源设为模型生成结果；每条都是 39 RGB frames、12 latent frames、3 chunks、
90 次 noisy denoiser forwards 和 3 次 clean commit。结果为：

```text
A horizontal flow: -0.136920
D horizontal flow: -0.152110
A-D difference:     +0.015190
```

此前 clean teacher history、无 adapter、30 steps/chunk 的 A/D 结果为
`+0.390690/-0.564994`，差值 `+0.955683`。因此动作方向在短片段的 causal mask 阶段
仍保留一部分，但换成 generated history 后已经明显塌缩；这比单纯增加 solver steps
更直接地支持 generated-history distribution shift 是主要问题之一。

合并诊断证据如下：

| 条件 | A flow | D flow | A-D |
|---|---:|---:|---:|
| 原始 H3 teacher，39f，30 steps | +1.181253 | -0.842124 | +2.023377 |
| causal，无 adapter，clean history，39f | +0.390690 | -0.564994 | +0.955683 |
| causal，普通 MSE tail16，clean history，39f | -0.572779 | -0.877738 | +0.304958 |
| causal，pair-loss tail8，clean history，39f | -0.346916 | -0.614203 | +0.267288 |
| causal，无 adapter，generated history，39f，30 steps | -0.136920 | -0.152110 | +0.015190 |
| causal，无 adapter，generated history，124f，8 steps | -0.017959 | -0.025754 | +0.007795 |

水平光流仍只是动作响应 proxy；A/D 比较最适合该指标，W/S 需要结合人物位移、尺度或
深度方向 proxy，不能要求四个动作在水平光流上完全对称。

### 新增 base-output regularization

修改：`code/causal/train_pretrained_multiaction.py`。

新增参数：

```text
--base-output-reg-weight FLOAT
```

该项约束训练后的 causal adapter 输出不要偏离零 adapter causal 基线：

```python
L = L_flow + lambda_base * MSE(adapter_output, zero_adapter_causal_output)
```

它在输出空间约束动作几何，比单纯惩罚 LoRA 参数更直接；同时保留 action-pair loss
来约束 W/S、A/D 的差分。新增训练任务：

```text
outputs/2026-10-02-04/stage1_multiaction_generated_reg_tail8_lr1e4/
```

配置为 generated-history、tail8、rank8、lr=1e-4、base-output-reg=0.5、
action-pair=0.5，运行在 GPU2。它与 GPU0 的 generated-history pair-loss 训练、GPU1 的
schedule-aware clean-history 训练并行，不打断已有任务。

### 历史状态（已被后续 09 时实验覆盖）

当时记录的训练任务：

```text
GPU0 stage1_multiaction_generated_pair_tail8
GPU1 stage1_multiaction_schedule_pair_tail8
GPU2 stage1_multiaction_generated_reg_tail8_lr1e4
```

GPU3/GPU4/GPU6 已用于补齐 39 帧 generated-history 30-step action decomposition；A/D 已
完成，W/S 正在完成。训练完成后必须分别复测 clean/generated 39f 和 generated 124f，
再决定哪个 checkpoint 用于最终 W/S/A/D action grid。不能仅凭 train/validation flow
loss 低就宣称 action fidelity 恢复。

本时段验证：

```text
.venvs/h3world/bin/python -m py_compile \
  H3-World/code/causal/train_pretrained_multiaction.py
```

通过；这些任务状态属于当时记录，后续 09 时实验已经覆盖本轮动作残差诊断。

### 2026-10-02 09 时：动作残差隔离实验

上一节中的 GPU0/GPU1/GPU2 训练状态是旧记录，不能作为当前状态；本节更新到本次
动作保真拆解后的实际结果。

`stage1_action_residual_only39` 已完成：只训练 action-dependent Q/K/V residual，冻结
通用 QKV adapter，8 个尾部 DiT block，39 帧、3 chunks、clean-history teacher forcing。
训练 metadata 为 train `0.2411168 -> 0.2383919`、validation `0.2668816 -> 0.2653980`、
`replay_max_error=0`。在相同 seed、首帧和 teacher latent、30 steps/chunk 下：

| 条件 | A flow | D flow | A-D |
|---|---:|---:|---:|
| causal 无 adapter | `+0.390690` | `-0.564994` | `+0.955683` |
| causal action QKV residual（冻结通用 QKV） | `+0.195898` | `-0.619446` | `+0.815343` |

该 residual 保留了 A/D 的相反符号，但没有恢复原始 H3 teacher 的 `+2.023377` 差异，且
A 响应反而减弱。因此“在当前 chunk 加一个动作 residual”不能等价于恢复原始全序列
attention 的 action geometry。

`stage1_action_hidden_only39` 也已完成训练：只训练 hidden FiLM residual、冻结通用
QKV，8 个尾部 block，`replay_max_error=0`；训练 train `0.2411168 -> 0.2411083`、
validation `0.2668816 -> 0.2647181`，目前正在等待其 A/D benchmark 完成，不能提前把它
当作修复结果。

该 benchmark 现已完成，输出为：

```text
outputs/2026-10-02-08/actionhidden_only_A_clean30_39/
outputs/2026-10-02-08/actionhidden_only_D_clean30_39/
outputs/2026-10-02-08/actionhidden_only_clean30_39_flow.json
```

hidden-only 的 A=`+0.313006`、D=`-0.658585`、A-D=`+0.971591`。它比 qkv-only 的
`+0.815343` 稍稳定，也保留了 A/D 相反符号，但仍明显低于原始 H3 teacher 的
`+2.023377`，所以不能作为“四方向完全保真”的 checkpoint。

### 当前诊断结论

动作保真损失不是单个开关造成的：

1. causal mask 改变了 H3 原 action LoRA 所适应的全序列交互图；缓存路径中 action prefix
   不再和当前 video hidden 做同样的双向层间反馈。
2. raw KV 在每个 chunk 的 clean sigma=0 状态提交，后面的 noisy solver 读取固定历史；
   原始 H3 则在每个 denoising 时刻重算整段历史，两个 score field 不同。
3. adapter 的主要训练协议是 clean teacher forcing，而自由 rollout 将生成误差写入下一
   chunk 的 KV 和 latent anchor；39 帧 generated-history 已把 A-D 从 `0.956` 降到
   `0.015`，所以仅增加 solver steps 或 action residual 不能解决长期动作塌缩。
4. 共享 MSE/QKV 目标更容易学习四个动作共同的场景和外观，action residual 的 pair loss
   只能部分拉开 A/D，不能自动学习 schedule transition 或 generated-history 分布。
5. latent dual anchor 是 temporal latent 的 image-like 近似，不等价于 H3 原生 RGB image
   condition；它改善 handoff 时也可能把场景先验压过动作残差。

另外，`action_prefix_mode=all` 的上界诊断并没有恢复动作：39 帧 generated-history 的
A-D=`-0.010542`。因此问题不能简化为“当前 chunk 没看到 action token”。W/S 也不能用
水平 flow 符号验收，原始 H3 的前进/后退本身可能产生同号的相机/深度运动；A/D 才适合
用水平 flow 直接判断左右。

### 2026-10-02 10 时：generated-history 与 schedule-aware 修复实验

根据建议文件的优先级，开始训练真正使用 causal generated history 的 action adapter，
而不是继续只用 clean teacher history。已有的四个 124 帧固定动作 teacher latent 位于：

```text
outputs/2026-10-01-21/action_{W,S,A,D}_teacher_latents/baseline_latents.pt
```

对应的 124 帧、8-step/chunk、无 adapter causal generated latent 位于：

```text
outputs/2026-10-02-04/action_{W,S,A,D}_base_causal124_latents/cached_latents.pt
```

新增训练任务：

```text
outputs/2026-10-02-10/stage1_action_generated_hidden124/
```

它只训练 8 个尾部 block 的 hidden action residual，冻结通用 QKV，使用 generated-history
target、A/D 与 W/S pair loss 和弱 base-output regularization，目标 chunk 覆盖完整 37 个
latent frames（124 RGB frames）。

同时补充了 `train_pretrained_multiaction.py --action-specs`。它现在能接受：

```text
W:3,A:2,D:3
D:2,S:3,A:3
```

并把 schedule 展开为每个 latent interval 的 action one-hot，而不是把一个 action 粗略地
广播给整段视频。该解析已经用 37 latent frames 的 one-hot 检查验证。

为训练 schedule-aware residual，先生成了真实的 124 帧 generated-history：

```text
outputs/2026-10-02-10/action_schedule_WAD_generated124/
outputs/2026-10-02-10/action_schedule_DSA_generated124/
```

两条 rollout 都是 124 frames、8 chunks、64 noisy forwards、8 clean commits，CPU KV 峰值
约 13.19 GiB。对应的 schedule hidden residual 训练为：

```text
outputs/2026-10-02-10/stage1_schedule_generated_hidden124/
```

此外启动了 39 帧快速诊断和一个强 action-pair loss 变体：

```text
outputs/2026-10-02-10/stage1_action_generated_hidden39/
outputs/2026-10-02-10/stage1_action_generated_hidden39_pairstrong/
```

### 2026-10-02 10 时训练结果（均已完成）

五项训练的最终状态如下。所有 adapter 均采用 8 个尾部 DiT block、rank 8、`action_residual_mode=hidden`、`dual_anchor_protocol`，可训练参数 774,144。

| 实验 | 帧数 | 步数 | history 类型 | A/D pair loss | 训练完成 | wall 时间 | 峰值显存 |
|---|---:|---:|---|---:|---|---:|---:|
| `stage1_action_generated_hidden39` | 39 | 160 | generated | 4.0 / 8.0 | ✅ | 823 s | 25.4 GiB |
| `stage1_action_generated_hidden39_pairstrong` | 39 | 160 | generated | 强化版 | ✅ | 823 s | — |
| `stage1_action_generated_feedback_hidden39` | 39 | — | generated | — | ✅ | 926 s | — |
| `stage1_schedule_generated_hidden124` | 124 | 160 | generated | 无 pair | ✅ | 1165 s | 20.8 GiB |
| `stage1_action_generated_hidden124` | 124 | 160 | generated | 4.0 / 8.0 | 🔄 运行中 | — | — |

`stage1_action_generated_hidden124` 截至本记录时仍在运行（`status: running`），特征提取已完成（768 s），进入训练阶段，尚无最终 checkpoint。

### 2026-10-02 10 时 cached inference 评估

对上述已完成的 adapter 进行了 7 次 cached inference 评估，均 `status: complete`，使用 `attention_backend: PyTorch SDPA`，`trained_for_causal: false`：

| 实验 | 帧数 | 步数 | history | action_feedback | adapter | 采样时间 | VRAM 峰值 | KV cache 峰值 |
|---|---:|---:|---|---|---|---:|---:|---:|
| `action_feedback_A_clean30_39` | 39 | 30 | clean | True | 无 | 322 s | 33.5 GiB | 6.5 GiB |
| `action_feedback_D_clean30_39` | 39 | 30 | clean | True | 无 | — | — | — |
| `action_generated_hidden39_eval/A` | 39 | 30 | generated | False | `hidden39` | 357 s | 39.6 GiB | 6.5 GiB |
| `action_generated_hidden39_eval/D` | 39 | 30 | generated | False | `hidden39` | — | — | — |
| `pairstrong_hidden39_eval/A` | 39 | 30 | generated | False | `pairstrong` | 355 s | 39.6 GiB | 6.5 GiB |
| `pairstrong_hidden39_eval/D` | 39 | 30 | generated | False | `pairstrong` | — | — | — |
| `schedule_generated_hidden124_eval_DSA` | 124 | 8 | generated | False | `schedule_hidden124` | 336 s | 39.9 GiB | 13.5 GiB |
| `schedule_generated_hidden124_eval_WAD` | 124 | 8 | generated | False | `schedule_hidden124` | — | — | — |

**A-D horizontal flow 对比**（`evaluate_action_control.py` / `evaluate_action_schedule.py` 输出）：

| 条件 | A flow x | D flow x | A-D |
|---|---:|---:|---:|
| clean-history，无 adapter，clean30，39 帧 | +0.392 | −0.603 | **+0.994** |
| generated-history，`hidden39` adapter，30 steps | −0.221 | − | — |

`action_feedback_clean30_39` 的 clean-history A-D 差值 +0.994 是本轮最好的短时方向分离结果，优于上一轮的 +0.972（2026-10-02-08）。但 `action_generated_hidden39_eval` 的 generated-history A 水平流为 −0.221，接近零，继续确认 generated-history 分布偏移导致动作塌缩的结论。

这进一步验证：clean-history 短时 rollout 能保留方向响应，free-running generated history 无论加什么 generated-history adapter，动作方向仍会在 39 帧内接近消失。修复路径仍指向 generated-history self-rollout 训练分布匹配（Stage2-style），而不是继续堆叠 adapter 变体。

---

## 2026-10-02 23:44 - Task 2: Scheduled-Sampling 实验启动

### 实验设计

为验证 `--history-mix-schedule` 修复 generated-history 动作塌缩的效果，启动了 3 组并行对比实验（39 帧，W/A/D 三动作）：

| 实验 | GPU | 配置 | 目的 |
|---|---|---|---|
| baseline_clean_only | 2 | 无 history-latent-dirs，纯 clean history | 对照组，验证 clean-history 性能上界 |
| fixed_mix_0.5 | 3 | `--history-mix 0.5`，固定 50% generated history | 验证固定混合比例效果 |
| curriculum_0.0_to_0.5 | 4 | `--history-mix-schedule 0.0:0.5` | 验证课程学习效果（核心方案） |

### 实验参数

```bash
--steps 160
--tail-blocks 8
--rank 8
--lr 1e-3
--anchor-mode dynamic_last_frame_dual
--history-protocol cached
--scheduler-steps 8
--scheduler-shift 2.22
--action-residual
--action-residual-mode hidden
--no-qkv-adapter
```

### 数据路径

- **Teacher latents**: `outputs/2026-10-02-03/action_{W,A,D}_teacher_39/`
- **Generated-history latents**: `outputs/2026-10-02-10/generated_history39/action_{W,A,D}_generated39/`
- **输出目录**: `outputs/2026-10-02-scheduled-sampling/`

### 进程状态

- baseline PID 1369208 (GPU 2)
- fixed_mix PID 1371090 (GPU 3)  
- curriculum PID 1372925 (GPU 4)

所有进程已在 23:44 启动，当前处于特征提取阶段。

### 监控工具

创建了两个脚本：
1. `monitor_scheduled_sampling.sh` - 手动检查实验进度
2. `wait_and_benchmark.sh` - 自动等待完成并运行 benchmark

### 验收标准

预期：
- **baseline (clean only)**: A-D flow ≈ +0.956（已知上界）
- **fixed_mix 0.5**: A-D flow > 0.015（如果有效）
- **curriculum 0.0→0.5**: A-D flow ≥ fixed_mix（课程学习应该更好或至少持平）

如果 curriculum 在 generated-history rollout 下能将 A-D 从 +0.015 提升到 >0.1，则确认 scheduled-sampling 有效，可以扩展到 124 帧。


## 2026-10-03 02 时：Scheduled-Sampling 公平 A/D 评估完成

上一轮的三个 scheduled-sampling checkpoint 已经用同一套推理条件重新评估，避免把不同 action、不同 chunk 或不同 step 数的单条 benchmark 误当成动作对照。评估配置为：

```text
seed=13；39 RGB frames（13 latent frames）
causal chunk=5 latent frames；3 chunks；history_chunks=5
30 denoiser steps/chunk；flow_shift=2.22
history_source=generated；anchor_mode=dynamic_last_frame_dual
action_prefix_mode=own；action_feedback=false；cache_device=cpu
每条视频 denoiser forwards=90，clean commits=3
```

所有 A/D 使用相同首帧、prompt、seed 和初始 noise，只替换 action；输出集中在：

```text
outputs/2026-10-03-02/scheduled_sampling_fair39_30steps/
```

### A/D horizontal flow 结果

| checkpoint | A mean flow | D mean flow | A-D | sampling time（A/D） | peak GPU | CPU KV peak |
|---|---:|---:|---:|---:|---:|---:|
| baseline_clean_only | −0.184 | −1.029 | **0.845** | 310/358 s | 39,924 MiB | 6,484 MiB |
| fixed_mix_0.5 | −0.412 | −1.415 | **1.003** | 350/349 s | 39,924 MiB | 6,484 MiB |
| curriculum_0.0_to_0.5 | −0.379 | −1.265 | **0.886** | 389/359 s | 39,924 MiB | 6,484 MiB |
| 原始 H3 teacher（39 帧） | +1.181 | −0.842 | **2.023** | — | — | — |
| causal 无 adapter（39 帧 generated） | −0.137 | −0.152 | **0.015** | — | — | — |

三组 adapter 都明显超过无 adapter generated-history 的动作塌缩；fixed 50% mix 当前短时 A/D 分离最好，但仍只有原始 H3 teacher 的约一半，且 A 的绝对方向仍受 generated-history 漂移影响。三个模型不是用不同 action 比较，而是在同一个 checkpoint 内分别生成 A 和 D；原始完整 JSON 分别位于各 checkpoint 目录的 `AD_flow.json`，每条视频的 `cached.json` 记录完整耗时、峰值显存、KV 峰值和 90 次 denoiser forwards。

这一步支持的结论是：scheduled-history 训练能恢复一部分 action sensitivity，但还不能宣称四方向保真或已经完成 SolarWM Stage2。下一步将选择 `fixed_mix_0.5/action_adapter.pt` 作为当前最佳诊断 checkpoint，执行统一的 W/S/A/D、124 frames、8 steps/chunk rollout；若长时动作仍衰减，将把结果报告为 generated-history distribution shift，而不是继续堆叠普通 adapter。

## 2026-10-03 03 时：固定 scheduled-sampling checkpoint 的 124 帧 W/S/A/D 验收

根据 39 帧公平 A/D 诊断，选择当前短时方向分离最好的 `fixed_mix_0.5` action residual checkpoint 做最终长视频验收：

```text
outputs/2026-10-02-scheduled-sampling/fixed_mix_0.5/action_adapter.pt
```

四个 action 使用同一张首帧、prompt、seed=13、初始 video/audio noise 和同一 checkpoint，只改变 W/S/A/D。统一配置：

```text
124 RGB frames = 5.17 s；37 latent frames；8 causal chunks
5 latent-frame chunk；history_chunks=5；generated history
8 denoiser steps/chunk；64 noisy forwards + 8 clean KV commits
flow_shift=2.22；dynamic_last_frame_dual；action_prefix_mode=own
action_feedback=false；persistent raw KV on CPU
```

原始 H3 对照是已有的同 seed、同首帧、同 prompt 的 30-step full-sequence 视频：

```text
outputs/2026-10-01-20/action_{W,S,A,D}_baseline30_124/baseline.mp4
```

本轮 causal 原始输出位于：

```text
outputs/2026-10-03-02/final_fixed_mix124_8step/{W,S,A,D}/cached.mp4
```

### 长视频指标

| Action | Original time (s) | Causal time (s) | Causal GPU peak | Causal CPU KV | Original mean RGB MAD | Causal mean RGB MAD | Original boundary MAD | Causal boundary MAD | Original horizontal flow | Causal horizontal flow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W | 441.5 | 450.2 | 39,927 MiB | 13,509 MiB | 4.32 | 4.60 | 4.97 | 6.29 | −1.111 | −0.624 |
| S | 444.8 | 451.9 | 39,927 MiB | 13,509 MiB | 4.13 | 4.62 | 4.70 | 7.76 | −1.084 | −0.096 |
| A | 454.2 | 438.8 | 39,948 MiB | 13,509 MiB | 4.52 | 7.10 | 5.14 | 13.68 | +1.077 | +0.143 |
| D | 450.4 | 383.4 | 39,948 MiB | 13,509 MiB | 4.46 | 5.39 | 5.39 | 9.47 | −1.602 | −0.311 |

`boundary MAD` 是 RGB 相邻帧在 chunk 边界帧 17、34、51、68、85、102、119 的均值；`MAD` 和 boundary MAD 都是连续性/运动描述，不是质量分数。causal rollout 仍然保持完整 124 帧和 cache lifecycle 正确，但 A/D 后半段出现明显 generated-history 漂移，边界 MAD 高于原始 H3。

A/D horizontal-flow action response：

```text
Original H3 A-D = +2.679
Causal fixed_mix A-D = +0.453
```

因此 causal branch 仍保留了 A>D 的正确符号，并且 W/S/A/D 会生成不同视频，但方向分离显著弱于原始 H3，不能写成“四方向完全保真”。W/S 的水平 flow 不作为前进/后退严格指标，因为本场景的前后运动可能投影成同号相机/深度运动；W/S 以并排视频作为 qualitative evidence。

### 最终可审阅产物

四行 4×2 对比网格（左：原始 H3 30 steps；右：scheduled-sampling causal 8 steps/chunk）：

```text
outputs/h3world_final_fixed_mix_action_grid_124.mp4
```

单 action 并排视频：

```text
outputs/h3world_final_W_original_vs_causal.mp4
outputs/h3world_final_S_original_vs_causal.mp4
outputs/h3world_final_A_original_vs_causal.mp4
outputs/h3world_final_D_original_vs_causal.mp4
```

统一指标和复现实验说明：

```text
outputs/2026-10-03-02/final_fixed_mix124_8step/final_action_summary.md
outputs/2026-10-03-02/final_fixed_mix124_8step/final_action_summary.json
outputs/2026-10-03-02/final_fixed_mix124_8step/action_flow.json
outputs/2026-10-03-02/final_fixed_mix124_8step/compact_continuity_metrics.json
```

本轮最终结论仍需保持克制：scheduled-history action residual 明显比无 adapter generated-history（39 帧 A-D≈0.015）更能保留动作敏感性，但长时生成分布偏移尚未解决；因此这验证了 `causal rollout + action interface` 的可行性，同时也验证了只靠 Stage1-style causal/teacher-forcing 或固定 history mixing 不能达到 SolarWM Stage2 的稳定少步长时 rollout。没有实现或声称 SGF/DMD。

补充的 pairwise action intervention 检查显示，causal 视频的 W-vs-S 平均 RGB 差异为 `29.92`，A-vs-D 为 `8.75`（`outputs/2026-10-03-02/final_fixed_mix124_8step/action_pair_differences.json`）。这只能说明动作干预产生了不同视频，不能替代方向正确性指标；A/D flow 分离和后半段视觉漂移仍应按上表报告。

## 2026-10-03：SolarWM 是否处理过 generated-history drift 的源码核对

已核对本地 `SolarWM/` 的 README、MiniMax-H3 Stage1/Stage2 配置和 SGF 实现。结论如下：

1. SolarWM 的公开训练路线明确把问题分成 Stage1 TF-AnyFlow 和 Stage2 SGF：Stage1 用 clean history 做 teacher-forced causal initialization；Stage2 则让 student 在自己的 autoregressive rollout 上训练，并用 frozen teacher + trainable critic 做 distribution matching。README 的原文是 `Stage2 performs DMD via self-gradient forcing (SGF), training the causal student on its own autoregressive rollout with a frozen teacher and a trainable critic`。
2. 因此 SolarWM 已经在方法层面处理了与本项目相同的 teacher-history / generated-history 分布差异和误差累积问题，但源码/README 没有专门公布 H3-World W/S/A/D action-collapse 的 ablation；不能说他们公开报告过同一个键盘动作失败案例。
3. SolarWM Stage2 不是简单把 clean latent 和 generated latent 做静态混合。`sgf_rollout.py` 中 student 先以 5-latent chunks 自回归生成，使用 detached raw KV cache；随后 `h3_sgf_replay` 用 rollout 的 noisy exit 和 generated clean target 重新 replay。`stage2_runtime.py` 再用 frozen Stage0.5 teacher 和 trainable critic 对 student output 做 score comparison，计算 SGF/DMD gradient，并更新 student；critic 每 5 次更新后才更新一次 student。
4. SolarWM 的 H3 action/control 语义与 H3-World 当前 W/S/A/D 不完全相同。SolarWM 的 `H3SGFInputs` 和 `H3SGFAttention` 主要接收每帧 `camera_viewmats`/`camera_K`，源码把 camera controls 视为 global immutable metadata；没有 H3-World 的 keyboard action rows/action_script 路径。因此 SolarWM Stage2 checkpoint 的成功不能直接证明 H3-World 的 W/S/A/D direction fidelity。
5. raw KV cache 和 sliding window 只保证历史上下文复用及可扩展的 causal window，不会自动纠正语义漂移；纠正漂移的是 Stage2 self-rollout + teacher/critic distribution matching。我们当前的 `history-mix` 是 detached static generated-history proxy，没有当前 adapter 的在线 self-rollout，也没有 SGF teacher/critic gradient，所以只能解释 39 帧 A-D 从 0.015 部分恢复，不能消除 124 帧后半段 drift。

对应源码：

```text
SolarWM/README.md
SolarWM/src/solarwm/backends/minimax_h3/stage1.py
SolarWM/src/solarwm/backends/minimax_h3/stage1_sampling.py
SolarWM/src/solarwm/backends/minimax_h3/sgf_rollout.py
SolarWM/src/solarwm/backends/minimax_h3/stage2_runtime.py
SolarWM/src/solarwm/backends/minimax_h3/sgf.py
SolarWM/configs/examples/minimax_h3/stage2-158f-lora384-w6-sp4.yaml
```

对本项目的直接启示是：如果继续做 SolarWM-inspired follow-up，核心应是在线 self-rollout 的 SGF-like 训练，并在 student、teacher、critic 三条路径中都保留 H3 action rows；继续堆叠普通 causal mask、KV cache、anchor 或静态 history mixing 不能等价替代 Stage2。
## 2026-10-03 17 时段：最小 action-aware online self-rollout teacher-replay

本轮按 `next_plan.md` 的收敛方案实现了一个小型 Stage2-inspired 诊断，目标是验证
SolarWM Stage2 的核心工程路径是否能迁移到 H3-World，而不是复现 SGF/DMD：

```text
student 自己 rollout 39 帧 / 3 causal chunks
→ 每个 chunk clean KV commit 时 detach 到 CPU
→ 用同一 generated history 建 frozen H3 teacher cache
→ student/teacher 在当前 chunk 上 replay
→ teacher velocity matching + 小 boundary loss
```

W/S/A/D action rows 在 student rollout、student commit、teacher commit、student replay 和
teacher replay 中都保留；teacher 只关闭 student 的 action residual，H3 的 packed action
condition 没有删除。新增脚本为
`H3-World/code/causal/train_online_selfrollout.py`。为支持当前 chunk 在读取 detached
KV 时保留梯度，`h3_cached.py` 增加 `allow_grad_read` 和 gradient-checkpoint 参数；DiT 的
causal control 允许 `allow_grad_read=True` 的 checkpoint；`CausalActionResidual` 增加运行时
`enabled` 开关，用于在同一 H3 实例上切换 student/frozen teacher。

### Smoke test

GPU 1 上完成 A 动作、39 帧、3 chunks、1 solver step、1 optimizer step：

- teacher-replay loss `0.0012899`，gradient norm `0.0121`，action adapter 参数发生更新；
- student forward 有梯度，KV commit 在 `no_grad` 中执行，缓存 history 校验通过；
- allocated GPU peak `42,981.9 MiB`，student/teacher CPU raw-KV 各约 `6.33 GiB`；
- 结果目录：`outputs/2026-10-03-16/online_selfrollout_smoke_A_ckpt2/`。

初版直接 GPU 反向在 4-step solver 上触发显存不足；随后启用 CPU activation checkpoint
offload，保留了 3 chunks 的真实 generated-history 训练路径。

### A/D online 训练

在 GPU 1 上从现有 fixed-mix action residual 初始化，A/D 交替做 4 个 optimizer steps，
每个 optimizer step 使用 4 solver steps/chunk：

| 项目 | 结果 |
|---|---:|
| latent/RGB frames | `12 / 39` |
| causal chunks | `3`（5+5+2 latent frames） |
| optimizer steps | `4`（A,D,A,D） |
| trainable parameters | `774,144` hidden action residual |
| replay loss | `0.001078, 0.001216, 0.001043, 0.001193` |
| wall time | `590.2 s` |
| allocated GPU peak | `41,344.8 MiB` |
| each CPU raw-KV cache | `6,799,104,000 bytes` |

adapter 和 trace 位于
`outputs/2026-10-03-17/online_selfrollout_AD_39_4x4_offload/`，其中
`training.json` 明确记录了 `history_source=online_student`、`history_detached=true`、
`student_forward_grad=true` 和 `commit_no_grad=true`。

同一时段的 `outputs/2026-10-03-17/online_selfrollout_AD_39_4x4/` 是未启用 CPU activation
offload 的失败尝试（显存不足），不作为结果；可复现实验以带 `_offload` 后缀的目录为准。

### 严格同条件回载对照

为了避免把 solver 或 anchor 差异误当作训练收益，fixed-mix 和 online adapter 都用：

```text
39 frames, seed=13, flow shift=2.22,
dynamic_last_frame_dual, 4 steps/chunk, generated history
```

| adapter | A horizontal flow | D horizontal flow | A−D |
|---|---:|---:|---:|
| fixed-mix 0.5 | `-0.0199` | `-0.3829` | **`+0.3630`** |
| online teacher replay | `-0.0210` | `-0.3703` | **`+0.3493`** |

online 的 A−D separation 比 fixed-mix 低 `0.0137`，属于基本持平，没有达到预设的
“online ≥ fixed-mix”成功标准。连续性 proxy 也基本持平：fixed A/D 的灰度 MAD 均值为
`2.963/2.895`，online 为 `2.933/2.893`；空间边缘差 fixed 为 `2.161/1.959`，online
为 `2.173/1.968`。这些是描述性指标，不是视频质量分数。

对应的 39 帧并排视频已放到输出根目录：

- `outputs/h3world_online_selfrollout_vs_fixedmix_A_39_dynamicdual.mp4`
- `outputs/h3world_online_selfrollout_vs_fixedmix_D_39_dynamicdual.mp4`

原始 JSON 指标：

- `outputs/2026-10-03-17/fixedmix_rollout_eval39_4step/AD_flow.json`
- `outputs/2026-10-03-17/online_rollout_eval39_dual/AD_flow.json`
- `outputs/2026-10-03-17/online_rollout_eval39_dual/continuity_comparison.json`

### 本轮结论

这一步证明了最小 online self-rollout 的**工程链路**可行：student 能在自身生成历史上继续
rollout，raw KV 可以 detach/复用，frozen H3 teacher 可以在相同 generated history 上
replay，action condition 仍然进入两条路径。它没有证明少步质量或 action fidelity 已经
改善：4 个 optimizer steps 的 hidden action residual teacher replay 没有超过已有 static
fixed-mix adapter，因此没有扩展到 124 帧，也没有把它称为 SolarWM Stage2。

当前更可信的面试结论是：

```text
causalization + persistent KV + action-aware online replay is runnable,
but a tiny teacher-replay update is insufficient to recover the 30-step
directional response. Stable long-horizon improvement needs a fuller
generated-history distribution-matching objective (SolarWM SGF/DMD-like
training), rather than another inference-only anchor trick.

### 为什么 online 结果仍然差

这次失败不是单一的 KV cache bug，而是目标、solver 和 H3 action 路径同时不匹配：

1. **4 steps 本身没有被蒸馏。** 当前 fixed-mix 在严格相同的 dynamic-dual、4-step 条件下
   A−D 只有 `+0.363`；同一类 30-step causal benchmark 的 A−D 约为 `+1.003`。online
   训练并没有实现 AnyFlow、SGF 或 DMD，因此不能把 4-step 当作已经具备 30-step score
   field 的少步模型。
2. **teacher replay loss 与 action fidelity 不一致。** student 只在 rollout 结束后的
   `sigma=0` 当前 chunk 上匹配 frozen teacher velocity，没有在 `sigma=1, 0.869, 0.689,
   0.425` 四个 solver 点上匹配 teacher map，也没有 A/D pair loss 或 action-margin loss。
   该目标更容易学共同的外观/运动，不能主动增大 A−D separation。
3. **student teacher 切换时关闭了 causal action residual。** 这使 teacher target 成为
   “没有 student causal residual 的 H3 replay”，而不是带有明确 action-direction target 的
   Stage0.5/Stage1 teacher。小 residual 的更新幅度也很小，4 个 optimizer steps 后各 block
   权重范数变化约为千分之几，说明这轮更接近 diagnostic smoke，而不是有效蒸馏。
4. **causal mask 改变了 H3 原有 action routing。** 当前训练使用
   `action_prefix_mode=own`、`action_feedback=False`。视频 query 可以读自己的 action
   row，但 action row 不能读回当前 video latent；H3 原始 directed action path 有这条
   feedback，released action LoRA 也是在 full-sequence bidirectional 条件下训练的。仅用
   hidden residual 很难补回这个被 causal cut 掉的反馈路径。
5. **dynamic latent dual anchor 不是 H3 原生 image anchor。** 生成的上一 chunk 尾帧经过
   latent patchify 后直接放入 image-like prefix，和 H3 的 RGB decode →
   `process_image=True` encode 语义不同；一旦前一 chunk 有漂移，anchor 会把漂移继续传给
   下一 chunk。

因此，“replay loss 下降”不能解释为“视频动作变好”。当前实验实际证明的是 cache/梯度/teacher
replay plumbing 可执行，而不是 causal model 已学会少步 action-conditioned score field。
若继续实验，优先级应是：先在固定 39 帧验证 `action_feedback=True` 的 action routing；再
把 teacher matching 放到每个 solver sigma，并加入 A/D pair or margin loss；最后才考虑
SGF-like critic。不要先扩展到 124 帧，也不要再用更多 anchor trick 掩盖这个目标错位。
```

## 2026-10-03 21--22 时段：action feedback、per-sigma replay 与 8-step 对齐实验

本轮目标是先解决当前视频“人物逐渐淡出、动作变弱”的问题，并把训练目标和最终
`8 steps/chunk` 推理对齐。所有 39 帧 A/D 对照都固定同一首帧、prompt、seed=13、初始
noise、`dynamic_last_frame_dual`、CPU raw-KV 和 H3 checkpoint；只替换 action 或 adapter。

### 1. Causal action routing ablation

新增的严格对照目录为：

```text
outputs/2026-10-03-21/action_feedback_ablation39/
```

在同一 `fixed_mix_0.5/action_adapter.pt`、4 steps/chunk 下：

| routing | A horizontal flow | D horizontal flow | A−D |
|---|---:|---:|---:|
| `action_prefix_mode=own`, feedback off | −0.0181 | −0.3834 | **0.3653** |
| `action_prefix_mode=causal`, `action_feedback=true` | −0.0496 | −0.4289 | **0.3793** |

恢复 action row 到当前 video latent 的 causal-safe feedback 后，A/D separation 有小幅提升，
但提升不足以解释全部画面退化。因此 action feedback 是必要条件，但不是完整解决方案。
原始结果位于 `own_nofb_flow.json` 和 `causal_fb_flow.json`。

### 2. Online teacher replay 改为匹配每个实际 solver sigma

`H3-World/code/causal/train_online_selfrollout.py` 现在支持：

```text
student 在自身 generated history 上 rollout
→ 每个 solver sigma 都用 frozen H3 teacher replay
→ local sigma loss 立即反向传播，solver state detach
→ clean sigma=0 replay 与 boundary loss
→ clean KV commit 后 detach 到 CPU
```

新增参数为 `--action-prefix-mode`、`--action-feedback`、`--sigma-replay-weight`。之前的
诊断只在每个 chunk 的 `sigma=0` 匹配，无法约束 4/8-step solver 使用的 noisy score field。
另外新增了可选的 `--causal-adapter --train-causal-adapter` 路径：student 可以训练已有的
tail QKV LoRA；teacher 在同一 DiT 中临时切换到冻结的 QKV snapshot，不复制第二份约 40 GiB
模型。QKV 训练结果同时保存为 `causal_adapter.pt`。

### 3. 与最终 8-step 推理对齐的 online 训练

输出目录：

```text
outputs/2026-10-03-21/online_sigma_feedback_39_8x4/
outputs/2026-10-03-21/online_sigma_feedback_39_8x4_eval/
```

配置为 39 RGB 帧、12 latent 帧、3 chunks、8 solver steps/chunk、4 optimizer steps，动作
顺序 A→D→A→D，`action_prefix_mode=causal`、`action_feedback=true`。结果：

| 项目 | 结果 |
|---|---:|
| replay loss（4 steps） | 0.005338, 0.005293, 0.005115, 0.005213 |
| online training wall time | 1,045.9 s |
| allocated GPU peak | 39,707.9 MiB |
| 每个 CPU raw-KV cache | 6,799,104,000 bytes（约 6.33 GiB） |
| A horizontal flow | −0.0131 |
| D horizontal flow | −0.8110 |
| A−D horizontal-flow separation | **0.7979** |

严格同条件的未再训练 `fixed_mix_0.5`、8 steps/chunk、feedback on 对照为 **0.7557**。
因此这轮 online per-sigma replay 在 39 帧短片上有小幅正向作用，但仍远低于原始 H3 teacher
的 A/D separation；它仍然是 Stage2-inspired diagnostic，不是 SGF/DMD。

### 4. 通用 tail QKV online smoke 与负结果

目录：

```text
outputs/2026-10-03-21/online_qkv_smoke39b/
outputs/2026-10-03-21/online_qkv_eval1_39_8step/
```

使用已有 tail16 causal QKV adapter，同时训练 3,440,640 个 QKV 参数和 774,144 个 action
residual 参数，1 optimizer step、4 solver steps/chunk：

- wall time `180.9 s`；
- allocated GPU peak `39,827.8 MiB`；
- student/teacher cache 各约 `6.33 GiB`；
- 训练链路和 `causal_adapter.pt` 保存均通过。

但只训练 1 step 后用 8-step 推理得到 A−D=`0.1674`，抽帧仍有明显淡出，因此不能把 smoke
当成质量改进。另一个冻结的旧 tail16 QKV adapter 与 fixed-mix action residual 叠加时，
39 帧 8-step 的 A−D 只有 `0.1596`。这两个结果说明通用 QKV adapter 需要更充分、与
generated-history 对齐的训练，不能直接作为最终 checkpoint。

### 5. 当前判断

8 steps/chunk 本身是本轮最明显的影响因素：同一 fixed-mix、feedback on 条件下，4-step
A−D=`0.3793`，8-step A−D=`0.7557`；8-step online per-sigma 训练进一步到 `0.7979`。
但是 39 帧视频抽帧仍可看到人物和场景随 chunk/history 逐渐淡出，说明核心瓶颈仍是
generated-history distribution shift 和少步 score field，而不是 KV cache lifecycle 或
显存不足。

目前不要把 39 帧 8-step online 结果扩展成 124 帧训练，也不要称为完成 SolarWM Stage2。
124 帧最终验收视频仍是：

```text
outputs/h3world_final_fixed_mix_action_grid_124.mp4
```

它对应的是之前的 `action_prefix_mode=own`、`action_feedback=false` fixed-mix checkpoint，
A/D 长时 separation=`0.453`；本轮 feedback/per-sigma 改动尚未重新生成 124 帧四方向网格。
下一步应先对 39 帧训练增加稳定的 action-pair/score matching 约束，并观察画面淡出是否
真正下降；只有短片画面稳定后，才重新制作 124 帧最终 demo。

## 2026-10-03 23 时段：N1 paired A/D teacher-delta objective

按照 `next_plan.md` v3，`code/causal/train_online_selfrollout.py` 已加入可选的
`--paired-action-loss`。在每个实际 solver sigma 上，A/D 两个 counterfactual 使用同一
generated latent、同一 detached student/teacher raw-KV history 和同一 sigma，分别计算：

```text
L_replay = 1/2 (MSE(vS_A,vT_A) + MSE(vS_D,vT_D))
L_dir    = 1 - cosine(vS_A-vS_D, vT_A-vT_D)
L_mag    = |norm(ΔS)-norm(ΔT)| / (norm(ΔT)+eps)
```

训练目标为 `L_replay + 0.1 L_dir + 0.1 L_mag`，继续保留 clean replay、boundary loss、
causal action prefix、action feedback、8 steps/chunk、dynamic dual latent anchor、CPU raw KV。
新增 `--checkpoint-every`，N2 可保存 `step_04/08/12/16/action_adapter.pt`。A/D 的
`prompt_embeds` 允许动作文本行不同；initial video noise 和 audio noise 要求逐元素一致，避免
把两个不同世界状态误当成 paired counterfactual。

N1 运行目录：

```text
outputs/2026-10-03-23/online_paired_sigma_39_8x4/
outputs/2026-10-03-23/online_paired_sigma_39_8x4_eval/
```

N1 配置是 4 optimizer steps、A→D→A→D、39 RGB frames、12 latent frames、3 chunks、8
solver steps/chunk、seed 13。训练 wall time `1,948.1 s`，allocated GPU peak `39,793.9 MiB`，
每份 student/teacher CPU KV 为 `6,799,104,000 bytes`（约 6.33 GiB）。训练中的均值如下：

| step | action | total | L_replay | L_dir | L_mag |
|---:|:---:|---:|---:|---:|---:|
| 1 | A | 0.066101 | 0.005902 | 0.366680 | 0.316511 |
| 2 | D | 0.075440 | 0.005681 | 0.394522 | 0.395598 |
| 3 | A | 0.060499 | 0.005746 | 0.346872 | 0.274806 |
| 4 | D | 0.071995 | 0.005648 | 0.388332 | 0.363436 |

N1 step-04 的 A/D 39-frame rollout 使用空闲 GPU1/GPU4 并行生成，采样各约 100 s，24 noisy
denoiser forwards、3 clean commits、CPU KV peak 约 6.33 GiB。动作 flow 结果：

| model | flow(A) | flow(D) | A−D |
|---|---:|---:|---:|
| online per-sigma baseline | −0.0131 | −0.8110 | **0.7979** |
| N1 paired step-04 | −0.0063 | −0.7750 | **0.7687** |

A 仍未满足 `flow(A)>0`，A−D 也没有超过原 baseline，因此 N1 未通过 PASS gate。配对损失
本身已正常生效且训练没有 NaN/OOM，但 4 步对 action residual 的参数改变量很小，不能将
该负结果解释为 paired objective 已经无效。N2 的 16-step learning curve 已从统一
`fixed_mix_0.5/action_adapter.pt` 初始化启动，输出目录为：

```text
outputs/2026-10-03-23/online_paired_sigma_39_8x16/
```

在 N2 完成前不制作新的 124-frame 正式 grid；现有正式视频和 PASS gate 仍保持不变。

## 2026-10-04：N2 paired A/D 16-step learning curve

N2 目录：

```text
outputs/2026-10-03-23/online_paired_sigma_39_8x16/
outputs/2026-10-03-23/online_paired_sigma_39_8x16_eval/
```

从同一 `fixed_mix_0.5/action_adapter.pt` 开始，严格使用 Phase 0 协议和 A→D 交替共 16 个
optimizer steps；每 4 步保存 `step_04/08/12/16/action_adapter.pt`。训练 wall time
`6608.7 s`，allocated GPU peak `39,793.9 MiB`，student/teacher CPU KV 各约 `6.33 GiB`。
paired score 的数值目标确实下降：最后 A 步 `L_dir=0.2937, L_mag=0.1503`，最后 D 步
`L_dir=0.3019, L_mag=0.1999`；这说明优化器能匹配 teacher 的局部 A/D score delta。

但每个 checkpoint 用同一 39-frame、8-step、generated-history rollout 测得的动作流为：

| checkpoint | flow(A) | flow(D) | A−D | 39f gate |
|---|---:|---:|---:|---|
| step_04 | +0.0070 | −0.7898 | 0.7968 | A/D 方向通过，separation 不足 |
| step_08 | −0.0106 | −0.7832 | 0.7726 | 未通过 |
| step_12 | −0.0039 | −0.7894 | 0.7855 | 未通过 |
| step_16 | −0.0487 | −0.7898 | 0.7411 | 未通过 |

`learning_curve.md/json` 位于 `online_paired_sigma_39_8x16_eval/`。A/D 视频的 frame MAD 和
edge proxy 在四个 checkpoint 间没有出现能解释方向恢复的视觉改善。结论是：paired teacher
delta loss 在当前 774,144 参数 action residual 上可以降低局部 score-field loss，却不能把
这种局部几何传递到自身 generated-history 的 rollout；继续增加 paired steps 没有价值。

这满足“39 帧方向 gate 未通过”的分支判断：暂不做 124-frame 新 grid，也不做 SGF/DMD。下一
个最小实验改为 action-pathway adaptation，固定 causal mask、KV、anchor、chunk、solver 和
generated-history 协议，只解冻 H3 原 action LoRA 或 action-token refiner 的小参数集，并用
同样的 A/D teacher replay 重新测试；若仍不能达到 `flow(A)>0`、`flow(D)<0`、A−D>`1.0`，
再报告为 action representation 在 causal 拓扑下的限制。

## 2026-10-04：N3 action-token prefix residual smoke

为验证 action 文本行是否因 causal 拓扑变化而失配，新增了零初始化的
`CausalActionPrefixResidual`。它只向当前 chunk 的 action text prefix rows 注入由 9 维
action one-hot 产生的 hidden residual，参数量为 `387,072`；已有 video Q/K/V action
residual 冻结，causal mask、persistent CPU raw KV、dynamic dual anchor、8-step solver 和
generated-history 协议全部不变。

实验目录：

```text
outputs/2026-10-04-15/action_prefix_online_pair39_8x4/
outputs/2026-10-04-15/action_prefix_online_pair39_8x4_eval/
```

训练使用 A→D→A→D 四步 paired per-sigma teacher replay，学习率 `1e-5`，总耗时约
`1973 s`，allocated GPU peak 约 `39,925 MiB`，student/teacher CPU raw KV 各约 `6.33 GiB`。
4 步均正常完成并保存 `action_adapter.pt`、`action_prefix_adapter.pt` 及 step-04 checkpoint。
Rollout 的 paired loss 没有 NaN，最终 39-frame flow 为：

| 配置 | flow(A) | flow(D) | A−D | gate |
|---|---:|---:|---:|---|
| online per-sigma baseline | −0.0131 | −0.8110 | 0.7979 | 未通过 |
| prefix residual step-04 | **+0.0055** | **−0.7668** | 0.7724 | 方向通过，分离不足 |

prefix residual 让 A 恢复了正号，但没有达到 `A−D>1.0`，且分离度低于原始 online
baseline。因此 action text rows 不是唯一瓶颈，不能据此制作新的 124-frame grid，也不进入
SGF/DMD。下一步开始 H3 原始 action LoRA 的小范围低学习率适配：先只解冻 tail8 的原始
`qkv_proj/out_proj` LoRA，在 teacher replay 时交换回冻结的原始 LoRA，避免 teacher 随 student
漂移。

## 2026-10-04：H3 原始 LoRA 适配代码已接入

`train_online_selfrollout.py` 新增 `--train-h3-lora`、`--h3-lora-blocks` 和
`--h3-lora-lr`，可将指定 H3 block 的 released LoRA 矩阵作为 student 参数训练，同时在
teacher forward 中恢复原始矩阵。新增 `h3_lora_adapter.pt` 保存格式和 benchmark 的
`--h3-lora-adapter` 加载选项，避免把该实验误当成正式 H3 checkpoint。代码通过 py_compile，
`tests/test_pretrained_lora.py tests/test_h3_cached.py` 共 7 项通过。

当前仍未满足 39-frame PASS gate：

```text
flow(A) > 0, flow(D) < 0, A−D > 1.0
```

所以新的 124-frame 视频、action switching grid 和 SGF/DMD 均继续冻结。prefix smoke 的原始
视频和指标保留在上述时间目录中，正式 124-frame deliverable 仍为
`outputs/h3world_final_fixed_mix_action_grid_124.mp4`。

## 2026-10-04：原始 H3 LoRA tail8 单步 smoke

为检查 causal action representation 是否需要直接适配 H3 已发布的 action LoRA，新增了
一个单步实验：只训练 blocks `42--49` 的 `qkv_proj/out_proj` LoRA（共 `10,092,544`
参数，学习率 `1e-6`），已有 causal video action residual 冻结；teacher forward 每次恢复
原始 H3 LoRA 矩阵。代码路径和 adapter 文件为：

```text
outputs/2026-10-04-17/h3_lora_tail8_online_pair39_8x1/
outputs/2026-10-04-17/h3_lora_tail8_online_pair39_8x1_eval/
```

单步训练耗时约 `693 s`，allocated GPU peak 约 `41,576 MiB`，没有 NaN/OOM，成功写出
16 个 tail8 LoRA module 的 `h3_lora_adapter.pt`。但同一 39-frame rollout 的 flow 为：

| 配置 | flow(A) | flow(D) | A−D |
|---|---:|---:|---:|
| online per-sigma baseline | −0.0131 | −0.8110 | 0.7979 |
| tail8 H3 LoRA 1-step | −0.0134 | −0.7956 | 0.7822 |

单步更新没有改善 A 的正向符号，分离度也略低于 baseline；抽帧仍显示约第 20 帧后人物
雾化和 ghosting。因此暂不投入昂贵的 4-step tail8 训练，先把该结果作为 action-pathway
负诊断保存。当前证据更支持：问题主要来自 causal generated-history 的 score-field/
distribution shift，而不是仅靠解冻 tail LoRA 就能修复的 action token 映射。

本轮新增的 `h3_online_lora_v1` adapter 只用于实验回载，benchmark 通过
`--h3-lora-adapter` 加载，不能称作新的 H3 checkpoint。39-frame gate、124-frame grid、
action switching 和 SGF/DMD 仍未启动。

## 2026-10-04：action-prefix `all` routing 上界

为区分 action-row routing 与 generated-history drift，固定 fixed-mix action adapter、8
steps/chunk、dynamic dual anchor、CPU KV 和 generated history，只把
`action_prefix_mode` 从 `causal` 改为 `all`，并行生成 A/D 39 帧。输出位于：

```text
outputs/2026-10-04-18/action_mode_all_fixedmix39/
```

结果为：

| action prefix mode | flow(A) | flow(D) | A−D |
|---|---:|---:|---:|
| causal（fixed-mix 参考） | 约 +0.02 | 约 −0.74 | 约 0.756 |
| all（本次上界诊断） | +0.0513 | −0.1508 | 0.2021 |

`all` 并没有恢复动作，反而让 D 的响应接近消失，说明把未来 action rows 暴露给当前视频
查询会产生跨时间控制污染；问题不是简单的“causal mask 少看了未来 action”。因此不采用
`all` 作为模型方案，继续固定 `action_prefix_mode=causal`。当前证据将重点收敛到
generated-history distribution shift 与少步 score field，而不是继续放宽 action mask。

## 2026-10-04：generated chunk endpoint matching smoke

为直接测试生成历史是否可以通过 teacher latent anchor 稳定，给 online paired replay 增加了
可选的 `--latent-target-weight`。它在每个 chunk 的最后一个 solver step，把 student 的
最终 latent 与同 action、同 initial noise 的原始 H3 30-step `baseline_latents.pt` 做 endpoint
MSE；不改变 causal mask、KV cache、anchor、solver 或 action routing。新增的日志字段为
`latent_target_loss_mean`。

单步、权重 `0.1` 的 smoke 输出为：

```text
outputs/2026-10-04-18/online_target_pair39_8x1_retry/
outputs/2026-10-04-18/online_target_pair39_8x1_retry_eval/
```

训练正常完成，`latent_target_loss_mean=0.2469`，无 NaN/OOM；但 39-frame rollout 为：

| 配置 | flow(A) | flow(D) | A−D |
|---|---:|---:|---:|
| online per-sigma baseline | −0.0131 | −0.8110 | 0.7979 |
| endpoint target 1-step | −0.0110 | −0.7569 | 0.7459 |

动作分离没有改善，短片后段 ghosting 也没有消失，因此不继续扫描 endpoint 权重。结合
paired score、prefix residual、tail8 H3 LoRA 和 `action_prefix_mode=all` 的结果，当前廉价
action-pathway 修复均未通过 gate；问题更像 generated-history distribution shift 与少步
score field，而不是单个 action token 或未来 action-row 可见性。

## 2026-10-04：最小 Stage2-inspired two-pass replay

为直接测试 generated-state 分布匹配，`train_online_selfrollout.py` 新增了
`--final-replay-weight` / `--final-replay-sigma`。每个 optimizer step 先完成 detached 的
3-chunk student rollout，再重建前两 chunk 的 student/teacher raw KV，并在最终生成 chunk
的自身 state 上用固定 sigma 做一次 student/teacher replay。这是 SolarWM Stage2 的最小
诊断版本，没有 critic、DMD、SGF KL gradient 或第二个 score role。

单步 smoke（weight=`0.1`、sigma=`0.6`）目录：

```text
outputs/2026-10-04-19/stage2_minimal_final_replay39_8x1_retry2/
outputs/2026-10-04-19/stage2_minimal_final_replay39_8x1_retry2_eval/
```

训练正常完成，`final_replay_loss_mean=0.000637`，耗时约 `456 s`；39-frame rollout 为：

| 配置 | flow(A) | flow(D) | A−D |
|---|---:|---:|---:|
| online per-sigma baseline | −0.0131 | −0.8110 | 0.7979 |
| minimal two-pass replay | −0.0008 | −0.7551 | 0.7543 |

动作分离没有改善，视频后段 ghosting 仍然存在。因此这个两阶段 replay 不能替代
SolarWM Stage2 的 critic/score-role 分布匹配，不能称为 Stage2 成功，也不继续做无意义的
权重扫描。当前实验链已经覆盖 action prefix、原始 H3 tail LoRA、未来 action-row 上界、
endpoint latent matching 和最小 two-pass replay；39-frame PASS gate 仍未通过。

## 2026-10-04：当前实验归档

为便于面试演示和复核，最新几轮结果已汇总到：

```text
outputs/2026-10-04-19/final_report/REPORT.md
outputs/2026-10-04-19/final_report/summary.json
outputs/2026-10-04-19/final_report/action_ablation_contact_sheet.jpg
```

报告统一列出 online baseline、prefix residual、tail8 H3 LoRA、`action_prefix_mode=all`、
endpoint target 和 minimal two-pass replay 的 A/D flow。当前没有任何新 checkpoint 达到
`flow(A)>0`、`flow(D)<0`、A−D>`1.0`，因此唯一正式 124-frame 对照仍是
`outputs/h3world_final_fixed_mix_action_grid_124.mp4`。最终结论应把 Stage1 causal/KV
feasibility 与 Stage2 generated-history quality limitation 分开陈述。

## 2026-10-06：anchor 作用范围、full-teacher action geometry 与 visual/action 折中

本轮继续围绕“人物分解”和 A/D action collapse 做了严格的 39-frame 诊断。所有结果均固定
12 latent frames、3 chunks、chunk size 5、8 steps/chunk、flow shift 2.22、seed 13、
generated history、CPU raw KV、`action_prefix_mode=causal` 和 `action_feedback=true`；没有
扩展到 124 帧。

首先给 `benchmark.py` 增加了 `--causal-adapter-scope {all,commit,last_step_commit}`。目的是
让训练得到的 visual tail16 QKV adapter 只负责 clean KV commit，而不改变当前 chunk 的
高噪声 action score field。用原始 W-trained tail16 adapter 做 A/D 对照时：

| scope | flow(A) | flow(D) | A−D | 观察 |
|---|---:|---:|---:|---|
| `commit` | −0.0266 | −0.0494 | 0.0228 | 人物仍会后段分解，动作基本消失 |
| `last_step_commit` | −0.0260 | −0.0616 | 0.0356 | 与 `commit` 相同，不能恢复动作 |

随后把同一个 scope 传入 `train_online_selfrollout.py`，分别做 4-step paired replay。两个
训练都正常结束，没有 NaN/OOM；`commit` 的第 4 步 `L_dir=0.1415`，`last_step_commit` 为
`0.1317`，但对应 rollout 的 A/D separation 只有约 0.022/0.012。说明训练/推理 scope
一致本身不能解决 causal action geometry。

一个更关键的修正是：此前的 paired delta loss 比较的是“同一个 causal mask 下的 teacher”，
而这个 teacher 的 A/D delta 已经随 causal 拓扑塌缩，不能作为原始 H3 action geometry 的
监督。因此 `train_online_selfrollout.py` 新增了 `--paired-full-teacher` 和
`full_teacher_forward`：在同一 generated state 上关闭 causal controller，使用原始 H3
双向 attention 计算 A/D teacher velocity，再对齐 student 的 action delta。1-step smoke
的目标确实不同（`L_dir=0.9566`，而 causal-teacher 约 0.14），但 rollout 仍只有
`flow(A)=0.0438`、`flow(D)=-0.5871`、A−D=`0.6309`。把高幅 action residual 与 tail16
QKV 联合更新 1 step 也没有改变这一结果。该链路证明了监督方向正确，但 1 step 不足以把
full-teacher geometry 迁移到 causal rollout，不能称为 Stage2 成功。

本轮最终定位了人物分解的主要工程原因：tail16 visual adapter 是按
`dynamic_last_frame_rgb_dual` 训练的，而最近诊断使用了便宜的 `dynamic_last_frame_dual`
latent anchor。换回训练一致的 RGB dual 后，人物和停车场结构在 39 帧内明显稳定：

| anchor / adapter | flow(A) | flow(D) | A−D | 视觉观察 |
|---|---:|---:|---:|---|
| latent dual + fixed visual | −0.698 | −0.691 | −0.008 | 人物后段透明/分解 |
| RGB dual + fixed visual | −0.881 | −0.756 | −0.124 | 人物全程基本完整，但动作同向 |
| RGB dual + action residual ×64 | +0.047 | −0.613 | 0.660 | A 方向恢复，D 开始 ghosting |

对 action residual 做了 8/16/32/48/64 倍幅度扫描。单一 global gain 不能同时保持 A/D 的
稳定性：A 需要很大幅度才翻到正号，而 D 在相同幅度下会分解。作为机制诊断，构造了一个
单文件的 per-action gain 原型（A 列 ×64、D 列 ×8），仍使用 RGB dual 和同一 checkpoint
加载路径。它的 39-frame 结果为：

```text
flow(A) = +0.0474
flow(D) = -0.7812
A-D     =  0.8286
```

抽帧中 A、D 两条人物都保持完整，动作方向也正确；但 separation 仍低于严格 gate
`A-D>1.0`，而且 per-action gain 目前只是 inference ablation，不是训练得到的最终模型。
完整指标和 contact sheet 位于：

```text
H3-World/outputs/2026-10-06-06/action_gain_sweep/RESULTS.md
H3-World/outputs/2026-10-06-06/action_gain_sweep/summary.json
H3-World/outputs/2026-10-06-06/action_gain_sweep/rgb_gain_A64_D8/eval/contact_sheet.jpg
```

这一目录下的 MP4 已用 PyAV 验证为 H.264、YUV420P、832×480、39 帧，可正常解码。训练一致
RGB dual 的代码参数也已经保留在 `benchmark.py`，并不会触发 ModelScope 下载。

当前结论更新为：RGB dual 修复了主要的视觉崩坏，per-action gain 证明 action geometry
可以被重新激活，但尚未达到统一 checkpoint 的 39-frame PASS gate。因此不制作新的
124-frame action grid，也不把该 inference gain ablation 称为 SolarWM Stage2。下一步若
继续，应该把 per-action gain 作为可训练参数并在 full-attention teacher 的 counterfactual
目标下优化；在此之前继续扫 solver、anchor 或重复 causal-teacher paired loss 没有归因价值。

## 2026-10-06：per-action gain 已接入可训练路径

为避免把 A×64/D×8 的手工缩放误当成模型能力，`CausalActionResidual` 现在包含一个可选的
FP32 `action_gain[9]` 参数。旧版 `h3_causal_action_residual_v1` checkpoint 没有这个字段时
自动使用全 1，因此所有已有 adapter 保持兼容。`train_online_selfrollout.py` 新增：

```text
--action-gain-init A=64,D=8
--action-gain-lr 1e-2
--anchor-mode {latent,rgb}
--paired-full-teacher
```

`--anchor-mode rgb` 会在每个 generated chunk 边界执行真实的 RGB decode/re-encode，和
tail16 visual adapter 的训练协议一致。用 RGB anchor、full-attention counterfactual teacher、
A=64/D=8 初始化做了 1-step smoke：训练约 10.5 分钟，峰值约 40.1 GiB，无 NaN/OOM；gain
参数从 `[A=64,D=8]` 只发生了很小变化（A≈63.98，D≈8.01），这说明 optimizer 和保存/回载
路径已经生效，但 1 step 不足以改变 rollout。记录位于：

```text
H3-World/outputs/2026-10-06-07/gain_lr_smoke_rgb_8x1/training.json
H3-World/outputs/2026-10-06-07/gain_lr_smoke_rgb_8x1/step_01/action_adapter.pt
```

相关代码通过 `py_compile`，已有 `tests/test_pretrained_lora.py tests/test_h3_cached.py`
共 7 项测试通过。当前最佳可播放短片仍是 RGB dual + A64/D8 inference ablation（A−D≈0.829）；
gain 训练需要继续做 4-step learning curve 才能判断是否能超过 1.0，之前不生成新的 124-frame
grid。

## 2026-10-06：开始 per-action gain 4-step learning curve

在完成 1-step RGB-anchor/full-attention-teacher smoke 后，启动了严格固定协议的 4-step 训练：

```text
output: outputs/2026-10-06-08/trainable_gain_full_teacher_rgb_8x4/
GPU: 0（GPU 2/3/5/7 上的其它 VLLM 进程未触碰）
39 RGB frames / 12 latent frames / 3 chunks
chunk=5 / history=5 / 8 solver steps/chunk
flow shift=2.22 / seed=13
generated history / CPU raw KV / action_prefix_mode=causal
action_feedback=true / RGB dual anchor
full-attention H3 A/D counterfactual teacher
trainable action gain init: A=64, D=8
action-gain-lr=1e-2 / base lr=5e-5
checkpoint every step
```

该实验只改变 action gain 的训练状态，不再改变 causal adapter、anchor、solver、chunk 或
KV 配置。第一次启动在 `step_01` 写出后发现旧前向把 FP32 gain 提前转换成 BF16（例如
8.0092 实际变成 8.0），因此已中断并标记为实现诊断，不把它当作完整曲线。现在
`CausalActionResidual.project_action` 对小型 action projection 使用 FP32，只有完成的残差
才转回 DiT 的 BF16；旧 checkpoint 仍按 legacy projection 路径兼容加载。修正后的 4-step
曲线将放在新的独立目录，仍会逐步保存 `step_01` 至 `step_04` 并统一生成 A/D 39 帧，计算
horizontal flow、A−D、boundary/frame MAD 与 contact sheet。只有同时满足 `flow(A)>0`、
`flow(D)<0`、A−D>`1.0` 且人物完整，才会进入 124-frame 阶段。若曲线仍停在约 0.8，则
接受当前 action geometry 受 causal/generated-history 分布限制的结论，不把 gain ablation
或普通 replay 误称为 SolarWM Stage2。

修正前实验目录：

```text
outputs/2026-10-06-08/trainable_gain_full_teacher_rgb_8x4/
```

其中 `step_01` 可作为“旧 BF16 gain path”诊断证据，训练状态为 interrupted；它不会被提升为
最终模型。修正同时新增了 gain 精度/旧 checkpoint 兼容测试，相关测试总数为 9 项。

---

## 第 3 轮（2026-09-23）：(A) 构念实验收官 + (B) 证伪复现 + (C) 论文改造

### （A）构念实验：从"弱"变成"弱但可复现"

**新增 k=11 独立样本（r68-1B，45 条），并改用与分布匹配的统计量（r76）。**

| 样本 | n | 平均 gap | 95% CI | 符号秩 p | top-3 率 | 平均秩 |
|---|---|---|---|---|---|---|
| 1B k=5 三个池 | 20/44/70 | +0.449/+0.650/+0.741 | 见附表 | 0.33/0.019/0.0048 | 45/50/53% | 6.05/5.39/5.07 |
| 1B k=5 合并 | 134 | +0.667 | [+0.326, +1.002] | 1.5e-4 | 51% | 5.32 |
| 1B k=11（独立样本） | 45 | +0.584 | [-0.030, +1.170] | 0.089 | 49% | 5.33 |
| 1B 两样本合并 | 179 | +0.646 | [+0.364, +0.943] | 3.3e-5 | 50% | 5.32 |
| 2B k=5 | 20 | +0.055 | [-0.967, +1.043] | 0.31 | 65% | 4.95 |

- **新的主统计量是"答案区域进入前 3 的比例"（50% vs 随机 25%）和平均秩（5.32 vs 零假设 6.5）**，
  而不是符号检验：item gap 分布强烈右偏，符号检验把效应所在的样本点丢掉。同一批数据
  符号检验 p=0.012、符号秩检验 p=1.5e-4。
- 2B 的**幅度**仍未分辨，但**排序**是锚定的（top-3 65%，平均秩 4.95，p=0.045）——
  之前写的"2B 完全测不到"要改成"未分辨的是幅度，不是定位"。

**机制操作（对比度降到 40%）**：两个样本的边际均值差都是 +0.14（+0.139 / +0.142），
但**配对检验不显著**（+0.141，CI [-0.102, +0.413]，p=0.47，见 D30）。
论文按配对检验写，并明确写出"两个边际均值一致到 0.003"正是让它看起来可信的原因。

### （B）C1 冗余感知探针：两个解码器上都证伪（r60 重跑，两个解码器）

- 配对增益 1B +0.0576 [-0.0581, +0.1733] p=0.329；2B +0.0162 [-0.0421, +0.0746] p=0.586。
- 可靠性更差：目标臂 split-half τ 0.140 / 0.210，已发表的边际探针 0.376 / 0.370。
- 池化敏感度也更低（1B 0.0351 vs 0.0470；2B 0.0188 vs 0.0217）。
- **结论：冗余不是可靠性天花板背后的机制，这个设计在两个解码器上都被支配。**
- 顺带补上了一个此前的漏洞：2B 的"已发表探针 0.370"以前在 results 里没有来源，
  r60 原来只写了 1B；重跑后 0.3762 / 0.3695 都有了出处。

### （C）论文改造：8 页合规 + 建设性产物

- 正文新增 §4.9 的 k=11 复现、top-3/平均秩、四套对照规则的稳健性、对比度机制的**诚实写法**。
- §5 新增 **"What a reliance claim should report"**（那就是用户要的"建设性产物"）：
  ① 写明干预方式（region 级结论只在给定操作下可复现，但**跨模型分歧**不依赖它）；
  ② 报告信度与它蕴含的噪声地板并做衰减校正；③ 检查构念，不能只看信度。
- 补充材料新增 **§S16 冗余感知探针**，并把 §S15 扩成完整表格（三个池 / 合并 / k=11 / 2B）+ 机制配对表。
- **页数恢复合规**：正文 8 页、参考文献从第 9 页开始、0 overfull、0 undefined。

### 机械门（全部本地 + 远端各跑一遍）

| 检查 | 结果 |
|---|---|
| final_gate（页数/串栏/警告/未定义） | **PASS** |
| check_s11（S11/S13/S15/S16 逐格对账） | **148/148 一致** |
| reverse_audit（正文每个小数是否可溯源） | **415 个中 0 个无法溯源**（4 个派生量已列明操作数） |
| number_audit（22 个关键数字） | 20 FOUND，2 NEAR-ONLY（四舍五入形式） |
| check_pages（含新的"正文不得越页"判据） | COMPLIANT，并用注入式回归测试验证会 FAIL |

### 仍未完成

- **r68-2B（k=11，70 条）** 进行中（37/70）。它决定 2B 的幅度是否只是样本量不足；
  落地后要把 2B 的 k=11 行加进 §S15 表格，并按结果决定正文那句"only the rank-level effect is resolved"是否要改。

## 2026-10-06：tail4 action-QKV paired alignment 4-update curve 已完成

在 RGB-consistent anchor、generated history、persistent CPU raw KV、8 steps/chunk、3 chunks
和 seed=13 全部冻结的条件下，完成了 `stage2_lite_dmd.py --paired-delta --paired-only`
的 4 个 optimizer update。只训练 tail4 action-conditioned Q/K/V refiner；每轮都保存了
`student_action_adapter.pt`，并用同一个 checkpoint 做 39 帧 A/D 自回归评估。训练使用
同一 generated state 的 full-attention H3 A/D counterfactual delta，记录四个 sigma 和两个
target chunks 的 `L_dir`、`L_mag`、student/teacher delta norm。

结果汇总如下。flow 是固定中心裁剪上的水平 Farneback flow，只是动作响应代理；严格 gate 为
`flow(A)>0`、`flow(D)<0`、`A-D>1.0`，并要求人物/车库结构保持稳定。

| update | flow(A) | flow(D) | A-D | mean paired direction cosine | mean norm ratio | gate |
|---:|---:|---:|---:|---:|---:|:---|
| 1 | -0.8563 | -0.7790 | -0.0774 | 0.3901 | 1.018 | FAIL |
| 2 | -0.8886 | -0.7536 | -0.1350 | 0.3474 | 0.973 | FAIL |
| 3 | -0.8703 | -0.7785 | -0.0918 | 0.3448 | 1.019 | FAIL |
| 4 | -0.8506 | -0.7684 | -0.0822 | 0.4242 | 1.045 | FAIL |

RGB dual anchor 继续有效：四轮视频中的人物和停车场结构都保持到第 38 帧，没有此前 latent-only
anchor 的透明/分解。可是 A/D 仍然表现为几乎相同的共同场景运动，paired score-field 的内部
对齐（norm ratio 接近 1、cosine 约 0.34–0.42）没有传递到自由 generated-history 的图像空间
动作方向。也就是说，这一轮排除了“tail4 paired update 不够多”这一简单解释，进一步支持
`action rows/feedback -> causal video-token routing` 与原始 H3 拓扑不一致的判断。

完整逐视频指标、训练 loss 曲线和 contact sheet：

```text
H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair4_final/ACTION_ALIGNMENT_LEARNING_CURVE.md
H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair4_final/ACTION_ALIGNMENT_LEARNING_CURVE.json
H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair4_final/action_alignment_learning_curve_contact_sheet.jpg
```

当前决策：不再继续增加 tail4 paired updates，不再做新的 gain/anchor/solver sweep，也不扩展
到新的 124 帧 action grid。正式 124 帧主 demo 仍是旧的 fixed-mix 结果；本轮作为“内部 score
alignment 改善但自由 action geometry 未恢复”的严谨负结果保留。若要继续，下一次有归因价值的
改动应是直接检查/重构 causal action routing（例如保留原始 action-to-token 路径或显式 action
token refiner），而不是继续扩大同一 QKV loss。

## 2026-10-06：action-routing probe 证明 feedback edge 存在但贡献很小

新增 `H3-World/code/causal/probe_action_routing.py`，不训练模型，只在同一 generated A history、
同一 chunk 1 noisy state、同一 RGB dual anchor、sigma=0.6 上比较 action prefix visibility 与
`action_feedback`。A/D score delta norm 为：

| variant | A/D delta norm | 相对 causal/no-feedback |
|---|---:|---:|
| own + feedback off | 5.502 | 0.748x |
| causal + feedback off | 7.352 | 1.000x |
| causal + feedback on | 7.642 | 1.039x |
| all + feedback on | 8.338 | 1.134x |

`causal_fb1` 明确不同于 `causal_fb0`，说明 action-row 到 current-video 的反馈边确实生效；
因此不能把当前失败简单归因于 causal mask 完全切断 action rows。另一方面，约 643 的总 velocity
norm 对应的动作 delta 只有约 1.2%，与 teacher delta audit 的弱信号一致。当前更合理的解释是
动作 representation/score geometry 在 generated-history causal rollout 下被压弱或旋转，而不是
缺少一条边。

详细记录：

```text
H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair4_final/ACTION_ROUTING_PROBE.md
H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair4_final/action_routing_probe.json
```

下一步保持 mask、KV、RGB anchor 和 solver 不变，先测原始 H3 action LoRA 的 action-row hidden
与 video output sensitivity；只有确认单 chunk 有正确 signal、但多 chunk 后 signal 消失时，才
进入 generated-history distribution matching。当前不实现未经证实的 bypass，也不生成新的
124-frame grid。

## 2026-10-06：冻结 causal 与原始 H3 的 action geometry 直接对照

在同一个 generated A history（chunk 0）、同一个 chunk 1 noisy latent、RGB dual anchor、prompt、
audio noise 和 sigma=0.6 上，比较冻结 visual causal adapter（causal prefix + action feedback）
与原始双向 H3 teacher 的 A/D counterfactual velocity。未安装 student action-QKV adapter，以排除
paired refiner 的影响。

| quantity | frozen causal | original H3 teacher |
|---|---:|---:|
| A/D delta norm | 7.642 | 8.704 |
| causal / teacher norm ratio | 0.878 | — |
| A velocity norm | 642.857 | 599.640 |
| D velocity norm | 643.615 | 598.936 |
| delta cosine | **-0.015** | — |

动作 delta 的幅度接近 teacher，但方向几乎正交。因此当前失败不是 action signal 完全消失，也
不是简单增大 gain 可以解决；causal chunk attention 加 generated-history 改变了 action-conditioned
score field 的方向。这与 routing probe 的结论一致：feedback 边存在，但它没有把 action delta
旋转回原始 H3 的 image-space geometry。

详细报告：

```text
H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair4_final/ACTION_GEOMETRY_PROBE.md
H3-World/outputs/2026-10-06-08/action_align_qkv_tail4_rgb_39_8step_pair4_final/action_geometry_probe.json
```

当前不再实现未经证实的 action bypass，也不继续 gain/paired-QKV sweep。若继续实验，唯一有归因
价值的方向是 action representation/score-field adaptation，且必须先在相同 39-frame gate 上
验证，再考虑任何长视频或 Stage2 扩展。


## 2026-10-06：action-prefix representation 1-step smoke

为区分“video-token QKV 不足”和“action row 表示本身需要重映射”，冻结现有 video action residual
与 RGB-consistent visual adapter，只训练 tail8 的 `CausalActionPrefixResidual`，目标仍是 full-attention
H3 的 A/D counterfactual delta。配置保持 39 帧、3 chunks、8 steps/chunk、RGB dual、generated
history、CPU raw KV、seed=13。

训练约 504.5 秒，峰值 allocated GPU 约 40.1 GiB，387,072 个 prefix 参数，无 OOM/NaN。paired
loss=1.0807、direction loss=0.9756、magnitude loss=0.9608。用 step_01 checkpoint 做同协议 A/D
rollout：

| variant | flow(A) | flow(D) | A-D | gate |
|---|---:|---:|---:|:---|
| tail8 action-prefix, update 1 | -1.1091 | -1.4603 | +0.3513 | FAIL |

抽帧中人物和车库结构保持到第 38 帧；A/D 有轻微区分，但两个 flow 仍为负，separation 远低于
1.0。该结果说明 action-prefix 表示适配可能比 tail4 QKV 更有方向性，但一个 update 尚不足以
恢复 image-space geometry，也不支持直接扩大训练预算。

报告和视频：

```text
H3-World/outputs/2026-10-06-08/action_prefix_align_full_teacher_rgb_39_8step_1update/ACTION_PREFIX_ALIGNMENT_REPORT.md
H3-World/outputs/2026-10-06-08/action_prefix_align_full_teacher_rgb_39_8step_1update/action_flow.json
H3-World/outputs/2026-10-06-08/action_prefix_align_full_teacher_rgb_39_8step_1update/contact_sheet.jpg
```

当前不把该 checkpoint 作为主方案，不生成 124-frame grid；继续 prefix 训练前需要新的训练
目标或多状态监督，而不是简单把 optimizer steps 增大。


## 2026-10-06：原始 H3 action LoRA tail8 1-step smoke

最后做了一个更接近原始 H3 action pathway 的适配：冻结 visual RGB causal adapter 和 video action
residual，只微调 released H3 attention LoRA 的 tail8 QKV/out matrices，目标仍为 full-attention
H3 A/D counterfactual delta。训练 526.8 秒，10,092,544 个 LoRA 参数，峰值约 40.14 GiB GPU、
CPU raw KV 6.33 GiB，无 OOM/NaN。

39-frame rollout：

| variant | flow(A) | flow(D) | A-D | gate |
|---|---:|---:|---:|:---|
| H3 action LoRA tail8, update 1 | -1.1249 | -1.4151 | +0.2902 | FAIL |

A/D 仍同向负 flow，且 separation 略低于 action-prefix smoke 的 0.3513。至此 tail4 action-QKV、
action-prefix hidden residual、原始 H3 action LoRA 三类 action-path adaptation 都没有在短片上
恢复 `A>0,D<0,A-D>1.0`。这轮作为最终结构性负结果保留，不再扩 optimizer steps 或生成
124-frame grid。

报告：

```text
H3-World/outputs/2026-10-06-08/action_h3_lora_align_full_teacher_rgb_39_8step_1update/ACTION_H3_LORA_ALIGNMENT_REPORT.md
H3-World/outputs/2026-10-06-08/action_h3_lora_align_full_teacher_rgb_39_8step_1update/action_flow.json
```

## 2026-10-06：action 实验收敛与最终诊断报告

已把 RGB Stage2-lite、tail4 action-QKV 4-update、tail8 action-prefix 1-update、released H3
action-LoRA tail8 1-update 统一收集到：

```text
H3-World/outputs/2026-10-06-08/FINAL_ACTION_DIAGNOSTIC.md
```

共同结论是：RGB-consistent anchor 已解决早期人物分解，8 个新增 39-frame 视频都能以
H.264/YUV420P 正常解码并保持场景结构；但所有 action-path adaptation 都未同时满足
`flow(A)>0`、`flow(D)<0`、`A-D>1.0`。冻结 geometry probe 显示 causal A/D delta 幅度接近
teacher、方向却几乎正交（cosine=-0.015），routing probe 显示 feedback 边存在但影响很小。

因此 Stage1/Stage2-lite 的最终边界已经清楚：causal/KV 工程可行，generated-history 下的
H3 action geometry 尚未恢复。后续若要突破，需要多状态/多 seed action supervision 或真正的
SolarWM Stage2 rollout-distribution matching；不再继续单场景单 state 的 LoRA/gain/anchor
sweep，也不生成新的正式 124-frame action grid。

## 2026-10-06：面试题提交包整理完成

已在 `/home/qma/work/GWM/submission`（当前源项目真实路径为 `/home/lpeng/code/mq_PubDataset/GWM/submission`）整理干净提交包，包含：

- `INTERVIEW_ANSWER.md`：SolarWM Stage0.5/Stage1/Stage2、KV cache、H3-World 流程和迁移结论；
- `EXPERIMENT_REPORT.md`：39/124 帧协议、视觉稳定性、Stage2-lite 和 A/D geometry 结果；
- `REPRODUCE.md`：外部权重、DiffSynth patch、因果 benchmark 和视频验证命令；
- `code/causal`、`code/abot` 和 DiffSynth patches；
- 小型 RGB visual、Stage2-lite 和 action diagnostic adapters；
- H.264/YUV420P 可解码的原始-vs-causal、视觉修复、Stage2-lite 和 action geometry 视频；
- `breakthrough/01` 至 `breakthrough/05`：每个关键突破的问题、方案、证据视频和限制。

提交包没有复制 33B 基础权重、H3-World 基础 LoRA、数据集、`.cache`、`__pycache__`、latent 或 conditioning 中间文件。23 个收录视频已用 PyAV 验证为完整可解码的 H.264/YUV420P、24 fps；25 个 Python 源文件通过语法检查。当前包约 87 MB。

提交包最终结论保持诚实：causal chunk rollout、persistent KV 和长时视觉稳定性已验证；generated-history 下的原始 H3 A/D action geometry 仍未通过 `flow(A)>0, flow(D)<0, A-D>1.0` gate，因此不声称已经完整保留 H3-World action control。

## 2026-10-06：新增会议展示包

已在 `/home/qma/work/GWM/submission/meeting` 建立现场展示材料：

- `annotated/h3world_final_action_grid_124_timed.mp4`：W/S/A/D 四行总览，左侧原始 H3 30 steps，右侧 causal 8 steps/chunk，并标注每个动作的 recorded end-to-end 时间；
- `annotated/h3world_final_{W,S,A,D}_original_vs_causal_timed.mp4`：带步数、总耗时、124 帧/5.17 秒和 causal forward/commit 口径的并排视频；
- `METRICS.md`、`METRICS.csv`：逐动作耗时、首块/平均块延迟、峰值显存、CPU raw KV、相邻帧 MAD、块边界 MAD 和 Farneback 水平光流；
- `FAIRNESS.md`：同首帧、prompt、动作、seed、初始 noise、分辨率和帧数的公平性说明，以及 30 full-horizon steps 与 8 steps/chunk 的计数口径；
- `SLIDES.md`、`MEETING_SCRIPT.md`：8 页展示提纲和约 5 分钟讲稿。

会议材料明确注明当前耗时是每个动作一次 recorded run，不是 warmup 后多次均值；也明确说明 formal fixed-mix causal grid 的 64 noisy forwards 高于原始 30 forwards，因此当前结论是 causal feasibility、历史复用和视觉修复，而不是端到端加速或完整 action preservation。

<!-- END SNAPSHOT progress.md -->


## 2026-10-10 root_cleanup: root document snapshots

下列原文为根目录清理前快照；历史状态、路径和运行建议不再构成当前授权。README 将重写为导航，report 将绑定 EXP-001；其余五份历史 Markdown 从根目录删除，内容保存在本节。

### root_cleanup snapshot: README.md

Bytes: 20418; SHA-256: `d0f10ebbcce621ea4f2da37b4ba4a41188fc46a8974dbdac0ffa0c44c1881bb7`

<!-- BEGIN ROOT_CLEANUP README.md -->
# H3-World × SolarWM 因果少步生成验证

2026-10-09 17:14：已冻结E2并完成提交收尾。独立venv＋新源码的15项KV/因果测试、小H3训练smoke、真实33B/39f推理均通过。会议主片固定旧RGB checkpoint，质量No-Go不变。[最终验收](submission/reports/final_acceptance/README.md) · [5分钟答辩](submission/meeting/MEETING_SCRIPT.md) · [checkpoint来源](submission/meeting/DEMO_PROVENANCE.md)。

2026-10-09 16:16：E2两臂各4更新、六组局部视频评测及48次held-out诊断全部完成。停车场两份历史的A/D符号均保留，但A分支重影仍在，FM+action没有一致优于FM-only；局部动作＋结构联合gate仍为No-Go。本轮不自动扩训，不进入AnyFlow/Stage2。 [完整结果与视频](submission/reports/stage1_anyflow/01_real_video/real_transition_windows/FINAL_RESULTS.md)。

这个项目对应面试题“在 H3-World 中验证 SolarWM 的因果少步生成思路”。目标不是复现
SolarWM 的 33B 全量训练，而是把它的因果分块、窗口化注意力、raw KV cache 和 Stage1
训练接口迁移到 H3-World，做出可以运行、可以测量、可以解释的最小原型。

当前代码已经完成了 H3-World 的 causal visibility mask、5 latent-frame chunk、滑动历史
窗口、逐层 raw K/V cache、clean commit、CPU KV offload、teacher-forcing flow loss、预训练
H3 末层 QKV LoRA，以及“上一 chunk 最后一帧作为显式 anchor”的可切换实验。训练端现在
使用持久化的 per-layer cached history，并对尾部 block 的 layer index 做严格 replay 校验。

2026-10-01 的复核还修正了 causal benchmark 的 scheduler 调度：此前 scheduler 更新误放在
每个 chunk 的 timestep 循环外，导致所谓 8 steps 实际只推进一次。现在每个 timestep 都会
推进 latent，日志中的 denoiser 次数与真实 latent 更新一致。修正版 22 帧对照已经归档在
`H3-World/outputs/2026-10-01-05/`，并排结果在
`H3-World/outputs/stage1_corrected_dynamic_anchor_8step_22.mp4`。动态末帧 anchor + W
action 会提高短片运动量，但当前仍是 latent patchify 的最小原型，不能等同于 H3 原生的
独立 RGB 图片条件。

## SolarWM 三个阶段的作用

### Stage0.5：建立双向视频 backbone

Stage0.5 在完整视频片段上使用双向 attention 做 flow matching。模型可以同时读取片段中
前后的视觉 token、文本/动作和其他条件，因此先学习基础的视频外观、场景结构、动作响应
和连续 latent 表示。这一阶段的模型仍然是普通的非因果视频生成 backbone，不能直接用于
自回归 chunk rollout。

### Stage1：让 backbone 学会因果分块生成

Stage1 使用 teacher forcing：训练当前 noisy chunk 时，只提供干净的历史 chunk、静态条件、
音频和当前动作。模型学习的不是“看完整视频后一次性预测”，而是：

1. 当前 chunk 只能看到允许的过去窗口；
2. 历史 chunk 的信息可以通过 causal attention 和 KV cache 复用；
3. 在指定 flow 时间点上，把当前 noisy chunk 推向 clean target；
4. 在少量步数的 flow map 上仍能保持 chunk 边界和动作连续性。

早期原型只有 clean-history 普通 flow matching。2026-10-08 已补齐并在真实 H3 上运行
TF-AnyFlow：目标时间条件、有限差分训练目标和有限区间采样；另有全 block QKVO/FFN LoRA
与完整历史梯度的受控实验。公式与梯度检查通过，但视频效果尚未验收，也没有完整复现官方
训练规模及融合两流算子。实现和最新证据见 [STAGE1_ANYFLOW](submission/STAGE1_ANYFLOW.md)。

### Stage2：把多步轨迹蒸馏成少步 student

Stage2 使用 student 自己的 causal rollout，并由冻结 teacher/critic 提供分布匹配梯度，
在 SolarWM 中对应 SGF/DMD 一类的少步蒸馏。训练目标不再只是“给定干净历史时一次预测
teacher target”，而是让 student 在自己产生的历史和中间状态上也回到真实视频分布。

本项目已做过 shared-backbone、独立 fake-score critic 的 Stage2-lite 诊断，但没有恢复
动作门槛。目前按用户要求先恢复 Original H3 的局部动作信息流，再验证 AnyFlow 的画质和动作。早期 FM 的 4/8-step
诊断、新 AnyFlow 的有限区间采样和 Stage2-lite 结果必须分别标注；步数本身不能辨认训练阶段。

以上阶段划分对应仓库中的实现与配置：`SolarWM/src/solarwm/backends/minimax_h3/stage0p5.py`、
`stage1.py`、`stage2.py`，以及
`SolarWM/configs/examples/minimax_h3/stage0p5-158f-lora384-sp2.yaml`、
`stage1-158f-lora384-w6-sp2.yaml`、`stage2-158f-lora384-w6-sp4.yaml`。H3 的 Stage1 配置明确
使用 `causal_mode: teacher_forcing` 和 `objective: anyflow_forward_map`；Stage2 配置明确
使用 `causal_mode: self_gradient_forcing`、4-step rollout、frozen teacher 和 trainable critic。
本项目的 `train_pretrained_multichunk.py` 是早期普通 FM 接口；新增的
`train_stage1_anyflow.py` 实现 TF-AnyFlow 与匹配 FM 对照。源码运行不等于官方 Stage1
效果复现，当前验收情况见 [STAGE1_ACCEPTANCE](submission/STAGE1_ACCEPTANCE.md)。

## Stage2 为什么可以减少采样步数

普通 flow/diffusion 推理把从高噪声到低噪声的连续轨迹切成很多小积分步。每一步只移动一小段，
所以需要多次 denoiser evaluation 才能到达 clean latent。

Stage2 student 通过 teacher/critic 的分布匹配训练，直接学习数据分布附近的“大步 flow map”：
一次 denoiser evaluation 可以跨过更大的噪声区间，同时保持最终结果接近 teacher 轨迹。于是
推理时可以把原本的 30/50 个小步压缩成 4/8 个大步。

因果 chunk 和 KV cache 解决的是历史条件的组织与复用，减少的是重复的历史 attention 计算；
它们本身不会把 30 个积分步变成 4 个积分步。Stage1 AnyFlow 已可训练有限区间的 flow map，
Stage2 再在 student 自生成分布上施加分布匹配信号；两者的少步效果都需要真实视频验证，
不能仅凭配置中的 4/8 steps 宣称质量或效率提升。

## H3-World 与 SolarWM 因果生成的主要区别

| 项目 | H3-World 当前原始流程 | SolarWM H3 因果流程 |
|---|---|---|
| 视频组织 | FL2VA 把整段视频 latent 一起送入 DiT | 5 个 latent frame 为一个 causal chunk |
| attention | 每个 denoising step 对完整目标序列做双向 attention | 当前 chunk 只能看静态条件、当前 chunk 和滑动窗口历史 |
| 历史复用 | 默认没有跨 chunk 持久化 raw K/V | 每层保存历史 chunk 的 raw K/V，后续 rollout 读取并淘汰旧窗口 |
| 训练 | 官方 H3-World LoRA 主要学习动作/文本条件 | Stage0.5 backbone 后再做 Stage1 teacher forcing/AnyFlow，Stage2 SGF/DMD |
| 首帧条件 | image-to-video 路径显式接收 keyframe | causal rollout 还需要处理 chunk 边界和历史 cache 生命周期 |
| 少步依据 | 原始 H3 依赖普通 scheduler steps | Stage2 student 通过轨迹蒸馏学习有限步 flow map |

本项目还验证了一个额外方向：新 chunk 可以显式接收上一 chunk 的最后一帧，类似 H3 最初
的 image-to-video anchor。默认双 anchor 实现保留原始首帧作为 slot 0，并在 slot 1 放上一
chunk 的末 latent，同时把 slot 1 retime 到真实的全局 frame grid；第 0 个 chunk 用首帧副本
占位。action text 仍绑定当前 latent frame。另有 `dynamic_last_frame_rgb_dual` 分支，会把
当前已生成 prefix 解码到 RGB，取最后可见 RGB frame，再通过 H3
`encode_video(process_image=True)` 重编码，真正走图片条件分支。temporal latent 不能直接
等价替代 image latent，二者必须分别训练和评测。

## 当前可审阅结果

最终 124 帧、5.17 秒并排视频：

[H3-World 30-step vs Stage1 causal long-anchor 8-step](/home/lpeng/code/mq_PubDataset/GWM/H3-World/outputs/h3world_30step_vs_stage1_long_anchor8_124.mp4)

经过 scheduler 调度修正后，当前可复核的同 seed 长片在：

[修正版 30-step vs causal 8-step + 上一末帧 anchor + action](/home/lpeng/code/mq_PubDataset/GWM/H3-World/outputs/stage1_corrected_dynamic_anchor_8step_124.mp4)

当前最完整的 persistent cached-history Stage1-style 结果是：

[30-step teacher vs cached-history dual-anchor 8-step/chunk](/home/lpeng/code/mq_PubDataset/GWM/H3-World/outputs/h3world_30step_vs_stage1_cached_history_dual_tail4_8step_124.mp4)

它在真实 H3 权重上覆盖 124 帧的 8 个 chunk，训练 replay 最大误差为 `0`，训练 loss
`0.10318→0.08088`，validation `0.10897→0.09149`。回载执行 64 次 noisy denoiser 和
8 次 clean commit；teacher 与 causal 的帧间 MAD 分别为 `4.295/15`、`4.277/16`，空间
边缘差为 `2.226`、`2.275`。chunk 边界仍有峰值，因此这是 Stage1-style 可行性结果，不能
称为 Stage2 质量等价模型。

使用同一训练协议将 slot 1 改成真实 RGB prefix image anchor 后，得到另一条 124 帧长片：

[30-step teacher vs RGB-prefix cached-history 8-step/chunk](/home/lpeng/code/mq_PubDataset/GWM/H3-World/outputs/h3world_30step_vs_stage1_rgb_cached_history_dual_tail4_8step_124.mp4)

RGB 版本的 train/validation loss 为 `0.09810→0.08012`、`0.10255→0.09040`，replay
error 为 `0`。它的 MAD/边缘差为 `4.200/16`、`2.138`，整体略低于 teacher 的
`4.295/15`、`2.226`，但 frame 85/102 附近仍有边界峰值。每个后续 chunk 都要做一次
prefix temporal decode 和 image re-encode，7 次 anchor 额外耗时约 `202.8 s`，总采样约
`674.7 s`；因此当前 RGB 条件路径在语义上成立，但没有在这个小规模 Stage1 原型中证明
质量或连续性提升。

当前用于审阅的主视频是扩大 causal QKV 适配容量后的版本：

[30-step teacher vs Stage1 causal tail16 + dual anchor](/home/lpeng/code/mq_PubDataset/GWM/H3-World/outputs/h3world_30step_vs_stage1_tail16_dual_anchor_8step_124.mp4)

它位于 `outputs/2026-10-01-18/rollout_tail16_124/`，使用末 16 个 DiT block 的 rank-8
QKV LoRA、双 anchor、8 steps/chunk，不使用 boundary blend。与 teacher 的 MAD `4.295/15`、
边缘差 `2.226` 相比，causal 分支为 `4.786/18`、`2.563`。抽帧显示 chunk 边界不再出现
整段画面的瞬间切换，人物运动和场景变化保持连续；但长期轨迹会偏离 teacher，仍是 Stage1
可行性结果，不是 Stage2 质量等价结果。

之前的双 anchor + soft overlap 版本仍保留为边界约束对照：

[30-step teacher vs dual anchor + two-frame soft overlap](/home/lpeng/code/mq_PubDataset/GWM/H3-World/outputs/h3world_30step_vs_stage1_cached_history_dual_anchor_softoverlap2_8step_124.mp4)

它能降低硬边界峰值，但会引入明显 temporal-VAE ghosting；因此最终主 Demo 使用 tail16
架构适配版本，而不是依赖后处理平滑。

针对“上一帧作为真正图片条件”的直接复测也已经完成：

[30-step teacher vs Stage1 causal tail16 + RGB last-frame anchor](/home/lpeng/code/mq_PubDataset/GWM/H3-World/outputs/h3world_30step_vs_stage1_rgb_tail16_dual_anchor_8step_124.mp4)

这个版本把 slot 1 的上一段末帧先经 temporal VAE decode，再经 H3 `process_image=True`
重新编码。它的统计为 MAD `4.585/17`、边缘差 `2.368`，比 latent tail16 的
`4.786/18`、`2.563` 更接近 teacher，但采样从 `304.5 s` 增加到 `567.4 s`，其中 7 次
RGB anchor 编码约占 `202.8 s`。所以 RGB 条件在连续性和场景约束上更有希望，但当前原型
仍以 latent tail16 作为效率主 Demo，RGB tail16 作为用户提出方案的直接对照。

随后新增的 `--history-mix` 用上一轮 overlap rollout 的 detached latent history 与 teacher
history 混合，直接诊断 generated-history 分布偏移。mix `0.25` 和 `0.50` 的 124 帧
free-running 结果分别为 MAD `5.207/20`、边缘差 `4.100`，以及 `6.479/24`、`4.318`，
均比 teacher 和 soft-overlap 主候选更差。这个负结果保留在
`outputs/2026-10-01-17/`，说明单条生成历史的简单混合会放大漂移，不能冒充 Stage2。

修正版 124 帧 causal 分支真实执行 64 次 denoiser forward 和 8 次 clean commit，采样
299.46 秒；前几个 chunk 的运动连续，但约第 30 帧后仍出现纹理重影和亮度漂移。把步数增
到 16 也没有解决长期漂移（对应视频在
`outputs/stage1_corrected_dynamic_anchor_16step_124.mp4`），因此当前最可信的结论是工程链路
已打通，Stage1-style 训练仍不足以保证 generated-history 长片质量。

随后把训练和推理都对齐到 H3 原生 `shift=2.22`，并覆盖 124 帧的全部 8 个 chunk。这个
版本的训练 loss 为 `0.10522→0.08774`，validation 为 `0.10618→0.09551`；8-step 回载
的 124 帧统计为 MAD `4.452/17`、边缘差 `2.433`，已经明显优于 shift=12 版本，但第 30
帧后仍有重影。该视频保留为 shift 对齐基线，当前主视频见上面的 soft-overlap 版本：
[原始 30-step vs shift=2.22 对齐 Stage1-style 8-step](/home/lpeng/code/mq_PubDataset/GWM/H3-World/outputs/h3world_30step_vs_stage1_aligned_shift2_8step_124.mp4)。

这段视频的右侧已经由 124 帧 teacher 的 8 个 chunk 做 clean-history Stage1-style 训练，
但中段以后仍出现重影和场景漂移，因此它是当前的负面质量诊断，不是质量等价的最终模型。
原始实验目录：

[outputs/2026-10-01-01](/home/lpeng/code/mq_PubDataset/GWM/H3-World/outputs/2026-10-01-01)

关键训练结果：

- 124 帧 teacher，chunk 0–7，240 optimizer steps；
- train loss `0.09109 → 0.07251`；
- validation loss `0.05029 → 0.04664`；
- frozen-feature replay 最大误差 `0`；
- 训练阶段 allocated 峰值约 `6.27 GiB`；
- 长片 rollout：64 次 denoiser forward、8 次 clean commit、CPU raw-KV 峰值约 `13.5 GiB`。

当前的工程结论是：causal mask、chunk 调度、raw KV cache 和 Stage1 teacher-forcing 接口
已经在真实 H3 权重上打通；但要达到“视频不瞬移”的验收标准，还必须加入 generated-history
或 scheduled-sampling 训练，并完成局部 anchor layout 的位置/去噪语义修正。Stage2 SGF/DMD
应在这两个问题解决后再开始。

## 代码与验证

- [causal/h3_cached.py](/home/lpeng/code/mq_PubDataset/GWM/H3-World/code/causal/h3_cached.py)：chunk rollout、raw KV cache、dynamic anchor、anchor retime。
- [causal/benchmark.py](/home/lpeng/code/mq_PubDataset/GWM/H3-World/code/causal/benchmark.py)：baseline/cached/recompute benchmark，支持 `dynamic_last_frame_dual` 和 `dynamic_last_frame_rgb_dual`。
- [causal/train_pretrained_multichunk.py](/home/lpeng/code/mq_PubDataset/GWM/H3-World/code/causal/train_pretrained_multichunk.py)：真实 H3 权重的 Stage1-style teacher-forcing，训练和回载使用同一双 anchor layout 与持久化 cached-history 协议。
- `train_pretrained_multichunk.py --history-latents ... --history-mix 0.25`：用于检查生成历史分布偏移的混合历史诊断接口。
- [docs/causal_prototype_report.md](/home/lpeng/code/mq_PubDataset/GWM/docs/causal_prototype_report.md)：完整实验记录和局限。
- [progress.md](/home/lpeng/code/mq_PubDataset/GWM/progress.md)：按时间归档的项目进展。

当前测试：`16 passed`，所有 causal Python 文件通过 `py_compile`。项目使用两个隔离环境：
`.venvs/h3world` 和 `.venvs/solarwm-h3`，不需要下载 SolarWM 官方约 14.45 TB 数据集。

## Action controllability intervention (2026-10-01 20)

The required fixed-seed intervention is now complete. The same initial image,
prompt, seed `13`, H3 action LoRA, and 124-frame noise are used for W/S/A/D;
only the held action changes (`forward`, `back`, `strafe-left`,
`strafe-right`). Each action has both an original 30-step output and a causal
tail16 output using 5-latent-frame chunks, persistent CPU raw-KV cache, latent
dual anchor, and 8 steps/chunk.

The review artifact is
[`h3world_stage1_tail16_action_intervention_grid_124.mp4`](H3-World/outputs/h3world_stage1_tail16_action_intervention_grid_124.mp4).
Each row is one action; the left tile is original H3-World and the right tile
is the causal Stage1-style rollout. The raw per-action outputs and JSON are in
[`outputs/2026-10-01-20`](H3-World/outputs/2026-10-01-20), including
`action_summary.json`, `action_video_metrics.json`, and the optical-flow
diagnostics.

The causal runs completed all 124 frames with 64 noisy denoiser forwards and 8
clean commits. Sampling time was `342.7–348.5 s`, GPU allocated peak was
`38.6–39.0 GiB`, and CPU raw-KV peak was `13.2 GiB`; the corresponding original
30-step runs took `359.2–373.3 s` with no KV cache. Causal frame MAD means were
`4.785 (W)`, `4.914 (S)`, `4.925 (A)`, and `4.803 (D)`; boundary scores (mean
MAD at frames 17, 34, ..., 119) were `5.69`, `6.14`, `6.00`, and `5.72`.
These are motion/continuity diagnostics, not quality scores.

Changing the action changes the resulting causal videos: paired W/S and A/D
RGB differences are `7.50` and `6.97` mean absolute intensity units. The
qualitative grid shows different character trajectories for all four actions.
Central whole-frame Farneback flow is retained as an action-response proxy in
the JSON; it is reliable for the original A/D camera intervention but is less
directionally stable for the causal clips because background/camera drift can
dominate the average. Therefore the defensible conclusion is that the H3
action conditioning path remains active after causalization, while exact
action-direction accuracy requires a multi-action causal training set. The
current adapter was trained on the original W teacher clip, so this experiment
also measures its cross-action robustness rather than claiming multi-action
fine-tuning.

## Multi-action shared causal checkpoint (2026-10-02)

The single-W intervention was followed by a stricter four-teacher experiment. W/S/A/D teacher
clips share the same image, prompt, seed `13`, initial noise, and 124-frame length; only the H3
action changes. A shared clean-history causal QKV LoRA was trained from all four clips, then
evaluated with 5-latent-frame chunks, persistent CPU raw-KV cache, latent dual anchor, and 8
steps/chunk.

The successful tail16 checkpoint is
[`stage1_multiaction_tail16_retry/adapter.pt`](H3-World/outputs/2026-10-02-00/stage1_multiaction_tail16_retry/adapter.pt).
It has 16 trainable tail blocks and 3,440,640 parameters, replay error `0`, train loss
`0.1588316→0.1385778`, and validation loss `0.1908010→0.1801894`. The W/S/A/D review grid is
[`h3world_stage1_multiaction_tail16_action_grid_124.mp4`](H3-World/outputs/h3world_stage1_multiaction_tail16_action_grid_124.mp4).
All four causal videos complete 124 frames with 64 noisy denoiser forwards and 8 clean commits;
peak allocated GPU is about `38.6–39.0 GiB`, and CPU raw-KV peak is `13.19 GiB`.

The fixed-action horizontal-flow proxy is W=`-0.540`, S=`-0.623`, A=`-0.594`, D=`-0.579`, while
the original H3 A/D teacher gives `+1.077` and `-1.602`. The causal W/S and A/D mean RGB
differences are `4.10` and `3.56`; those differences show that action changes the output, but
they do not establish that the requested direction is correct. W:3,A:2,D:3 and D:2,S:3,A:3
chunk schedules likewise do not reproduce the original H3 direction switches. Their annotated
comparisons are [`h3world_schedule_WAD_multiaction_tail16_124.mp4`](H3-World/outputs/h3world_schedule_WAD_multiaction_tail16_124.mp4)
and [`h3world_schedule_DSA_multiaction_tail16_124.mp4`](H3-World/outputs/h3world_schedule_DSA_multiaction_tail16_124.mp4).

The defensible conclusion is narrower than “action control is preserved”: causal masking,
persistent KV and action-conditioned scheduling run on H3-World, and changing action keeps the
conditioning path active, but this Stage1-style clean-history prototype does not preserve H3's
directional action response under generated-history rollout. The remaining work is generated-history
distribution matching / SolarWM Stage2-style training rather than more KV-cache or anchor variants.
The unified table and JSON are in [`outputs/2026-10-02-00`](H3-World/outputs/2026-10-02-00), especially
`action_multiaction_tail16_summary.md` and `action_multiaction_tail16_summary.json`.

<!-- END ROOT_CLEANUP README.md -->

### root_cleanup snapshot: SCHEDULED_SAMPLING_EXPERIMENTS.md

Bytes: 6838; SHA-256: `b428ea00cc0f193cba2343faf67448b3ff7c1eac819d8cf88f84be5c869a1ae1`

<!-- BEGIN ROOT_CLEANUP SCHEDULED_SAMPLING_EXPERIMENTS.md -->
# Scheduled-Sampling Experiments Tracker

**Created**: 2026-10-02 23:44  
**Status**: Running

---

## Objective

Validate whether **scheduled-sampling / curriculum learning** can fix the generated-history action collapse problem in the causal world model.

### Problem Statement

Current causal models show severe action collapse when using generated history:

| Condition | A-D Horizontal Flow |
|---|---:|
| Original H3 teacher, 39f, 30 steps | **+2.023** |
| Causal, clean history, 39f | +0.956 |
| Causal, **generated history**, 39f | **+0.015** ⚠️ |
| Causal, generated history, 124f, 8 steps/chunk | +0.008 |

**Root cause**: Train/test distribution mismatch
- **Training**: Model sees clean teacher history
- **Inference**: Model gets its own generated history with accumulated errors

---

## Experimental Design

### Three Parallel Experiments (39 frames, W/A/D actions)

| Experiment | GPU | Configuration | Purpose |
|---|:---:|---|---|
| **baseline_clean_only** | 2 | No history mixing, 100% clean | Control: establish clean-history upper bound |
| **fixed_mix_0.5** | 3 | `--history-mix 0.5` | Test fixed 50/50 mixing |
| **curriculum_0.0_to_0.5** | 4 | `--history-mix-schedule 0.0:0.5` | Test curriculum learning (our main hypothesis) |

### Shared Parameters

```bash
--steps 160
--tail-blocks 8
--rank 8
--lr 1e-3
--anchor-mode dynamic_last_frame_dual
--history-protocol cached
--scheduler-steps 8
--scheduler-shift 2.22
--action-residual
--action-residual-mode hidden
--no-qkv-adapter
```

### Data Paths

- **Teacher latents**: `H3-World/outputs/2026-10-02-03/action_{W,A,D}_teacher_39/`
- **Generated-history latents**: `H3-World/outputs/2026-10-02-10/generated_history39/action_{W,A,D}_generated39/`
- **Output directory**: `H3-World/outputs/2026-10-02-scheduled-sampling/`

---

## Process Status

| Experiment | PID | GPU | Status | Started |
|---|---:|:---:|---|---|
| baseline_clean_only | 1369208 | 2 | 🟡 Running | 2026-10-02 23:44 |
| fixed_mix_0.5 | 1371090 | 3 | 🟡 Running | 2026-10-02 23:44 |
| curriculum_0.0_to_0.5 | 1372925 | 4 | 🟡 Running | 2026-10-02 23:44 |

---

## Expected Results

### Hypothesis

Curriculum learning should gradually expose the model to generated history during training, teaching it to maintain action response even when history quality degrades.

### Success Criteria

✅ **Success**: Curriculum achieves A-D flow > 0.1 on generated-history rollout
- Significantly better than baseline generated-history (0.015)
- Demonstrates that scheduled-sampling bridges the train/test distribution gap

❌ **Failure**: A-D flow remains near 0.015
- Would indicate scheduled-sampling alone is insufficient
- May need Stage2-style self-rollout or other approaches

### Detailed Predictions

| Experiment | Expected A-D Flow (generated-history rollout) | Rationale |
|---|---:|---|
| baseline_clean_only | ~+0.956 | Known upper bound from clean-history training |
| fixed_mix_0.5 | 0.015 to 0.5 | If mixing helps, should be between baseline generated and clean |
| curriculum_0.0_to_0.5 | ≥ fixed_mix | Curriculum should match or beat fixed mixing |

---

## Monitoring Commands

```bash
# Check experiment progress
bash /home/qma/work/GWM/monitor_scheduled_sampling.sh

# View summary of all results
bash /home/qma/work/GWM/summarize_results.sh

# Auto-wait and run benchmarks (background)
nohup bash /home/qma/work/GWM/wait_and_benchmark.sh > /tmp/wait_and_benchmark.log 2>&1 &
```

---

## Implementation Details

### What We Implemented

Added `--history-mix-schedule START:END` parameter to `train_pretrained_multiaction.py`:

1. **Feature Extraction Phase**:
   - Generate 3 history mix variants per action clip:
     - variant 0: mix=START (e.g., 0.0 = 100% clean)
     - variant 1: mix=(START+END)/2 (e.g., 0.25)
     - variant 2: mix=END (e.g., 0.5 = 50% generated)

2. **Training Phase**:
   - Dynamically select variant based on training step progress
   - Early steps use variant 0 (mostly clean history)
   - Middle steps use variant 1 (balanced mix)
   - Late steps use variant 2 (mostly generated history)

3. **Metadata Tracking**:
   - Each sample tagged with `history_mix_variant` index
   - Training loop uses `get_history_mix_for_step(step)` to pick appropriate variant

### Modified Files

- ✅ `H3-World/code/causal/train_pretrained_multiaction.py`
  - Added argument parsing for `--history-mix-schedule`
  - Modified feature extraction to generate 3 variants
  - Added curriculum sampling logic in training loop
  - Verified with `python -m py_compile`

---

## Next Steps

### If Successful (A-D > 0.1)

1. **Scale to 124 frames**:
   - Use same curriculum approach (0.0→0.5 or 0.0→1.0)
   - Train with 124-frame teacher clips
   - Benchmark with 8 steps/chunk rollout

2. **Compare with Task 1**:
   - Task 1: `stage1_action_generated_hidden124` (clean-history trained, 124f)
   - New: scheduled-sampling trained, 124f
   - Determine best approach for final deliverable

3. **Make final demo videos** (Task 4)

### If Unsuccessful (A-D ≈ 0.015)

1. **Investigate why scheduled-sampling failed**:
   - Check if variants are being correctly selected
   - Verify generated-history latents are actually different from clean
   - Examine per-step loss curves for curriculum effect

2. **Consider alternatives**:
   - More aggressive mixing schedule (0.0→1.0)
   - Stage2-style self-rollout during training
   - Different curriculum strategies (exponential rather than linear)

---

## Related Tasks

- **Task 1**: `stage1_action_generated_hidden124` (124 frames, still finalizing)
- **Task 3**: Latent dual anchor position semantics (independent, can parallelize)
- **Task 4**: Final demo videos (blocked on Tasks 1/2 results)
- **Task 5**: SolarWM Stage2 inference (low priority)

---

## Log Files

- `H3-World/outputs/2026-10-02-scheduled-sampling/baseline_clean_only.log`
- `H3-World/outputs/2026-10-02-scheduled-sampling/fixed_mix_0.5.log`
- `H3-World/outputs/2026-10-02-scheduled-sampling/curriculum_0.0_to_0.5.log`

---

## Results

*To be filled in after experiments complete (~1-2 hours)*

### Training Metrics

| Experiment | Status | Train Loss (before→after) | Val Loss (before→after) | Time (s) | Peak Mem (MiB) |
|---|---|---|---|---|---|
| baseline_clean_only | - | - | - | - | - |
| fixed_mix_0.5 | - | - | - | - | - |
| curriculum_0.0_to_0.5 | - | - | - | - | - |

### Benchmark Results (Generated-History Rollout)

| Experiment | A-D Horizontal Flow | Interpretation |
|---|---:|---|
| baseline_clean_only | - | - |
| fixed_mix_0.5 | - | - |
| curriculum_0.0_to_0.5 | - | - |

---

## References

- **Progress log**: `/home/qma/work/GWM/progress.md`
- **Next plan**: `/home/qma/work/GWM/next_plan.md`
- **Task summary**: `/home/qma/work/GWM/TASK_SUMMARY.md`
- **Code**: `H3-World/code/causal/train_pretrained_multiaction.py`

<!-- END ROOT_CLEANUP SCHEDULED_SAMPLING_EXPERIMENTS.md -->

### root_cleanup snapshot: STATUS_REPORT.md

Bytes: 6418; SHA-256: `0d22314d8bdb3167a6164db7c15dbfd548ab2e4cb05138a3a8d0ceb6afe75448`

<!-- BEGIN ROOT_CLEANUP STATUS_REPORT.md -->
> 本文件为2026-10-02历史快照。当前Stage1 AnyFlow状态以[README](README.md)、[progress](progress.md)和[next plan](next_plan.md)为准。

# GWM Project Status Report
**Generated**: 2026-10-02 23:46  
**Session**: Next Plan Task Completion

---

## ✅ Completed Tasks

### Task 2: Scheduled-Sampling Implementation (COMPLETE)

**Status**: ✅ Implementation complete, experiments running, auto-monitoring active

#### What Was Done

1. **Implemented `--history-mix-schedule` in trainer**:
   - Added curriculum learning parameter to `train_pretrained_multiaction.py`
   - Feature extraction generates 3 history mix variants (start, middle, end)
   - Training loop dynamically selects variants based on step progress
   - Syntax verified with `python -m py_compile`

2. **Launched 3 parallel validation experiments** (39 frames, W/A/D):
   - `baseline_clean_only` (GPU 2, PID 1369208): Control, 100% clean history
   - `fixed_mix_0.5` (GPU 3, PID 1371090): Fixed 50/50 mixing test
   - `curriculum_0.0_to_0.5` (GPU 4, PID 1372925): Curriculum learning (main hypothesis)

3. **Created monitoring infrastructure**:
   - `monitor_scheduled_sampling.sh`: Manual progress checker
   - `wait_and_benchmark.sh`: Auto-waits for completion and runs benchmarks
   - `summarize_results.sh`: Results aggregator
   - `SCHEDULED_SAMPLING_EXPERIMENTS.md`: Full experiment documentation
   - Background monitor running (PID 1395686)

#### Files Modified/Created

**Core Implementation**:
- ✅ `H3-World/code/causal/train_pretrained_multiaction.py` (modified)

**Documentation**:
- ✅ `/home/qma/work/GWM/next_plan.md` (updated)
- ✅ `/home/qma/work/GWM/progress.md` (updated)
- ✅ `/home/qma/work/GWM/TASK_SUMMARY.md` (created)
- ✅ `/home/qma/work/GWM/SCHEDULED_SAMPLING_EXPERIMENTS.md` (created)

**Automation Scripts**:
- ✅ `/home/qma/work/GWM/monitor_scheduled_sampling.sh` (created)
- ✅ `/home/qma/work/GWM/wait_and_benchmark.sh` (created)
- ✅ `/home/qma/work/GWM/summarize_results.sh` (created)

#### Expected Timeline

- **Feature extraction**: ~10-15 minutes (in progress)
- **Training**: ~30-45 minutes (160 steps)
- **Auto-benchmark**: ~10-15 minutes per experiment
- **Total**: ~1-2 hours from start (23:44)

#### Success Criteria

✅ **Success**: Curriculum achieves A-D flow > 0.1 on generated-history rollout
- Confirms scheduled-sampling bridges train/test distribution gap
- Next: Scale to 124 frames

❌ **Failure**: A-D flow remains ≈ 0.015
- Need to investigate why curriculum learning didn't help
- May require Stage2-style self-rollout

---

## ⏳ In Progress

### Task 1: `stage1_action_generated_hidden124` (124 frames)

**Status**: ⏳ Training complete (160/160 steps), finalizing

- Training finished but `status: running` (saving final checkpoint)
- Once complete, will run 124-frame W/A/D benchmark
- Expected to show similar A-D collapse as 39-frame version (~0.008)

---

## 📋 Next Steps (Automatic)

The following will happen automatically via background monitoring (PID 1395686):

1. ⏳ Wait for all 3 experiments to complete
2. 🔄 Auto-run benchmarks on each trained adapter
3. 📊 Generate flow.json with A-D horizontal flow measurements
4. ✅ Log results to `/tmp/wait_and_benchmark.log`

---

## 📋 Next Steps (Manual Decision Points)

### After Task 2 Results Arrive (~2 hours)

**If curriculum learning succeeds** (A-D > 0.1):
1. Scale scheduled-sampling to 124 frames
2. Compare with Task 1's 124-frame baseline
3. Choose best checkpoint for final demo (Task 4)

**If curriculum learning fails** (A-D ≈ 0.015):
1. Investigate why (check logs, loss curves, variant selection)
2. Try more aggressive schedule (0.0→1.0)
3. Consider Stage2-style self-rollout approach

### Other Tasks

**Task 3** (can run in parallel):
- Fix latent dual anchor position semantics
- Independent of Tasks 1/2 results

**Task 4** (blocked on results):
- Create final side-by-side demo videos
- Write experiment report
- Needs best checkpoint from Tasks 1/2

**Task 5** (low priority):
- SolarWM Stage2 inference
- Only if Tasks 1/2 show promise

---

## 🖥️ System Status

### Running Processes

| Process | PID | GPU | Status | Purpose |
|---|---:|:---:|---|---|
| baseline_clean_only | 1369208 | 2 | 🟡 Running | Task 2 experiment 1/3 |
| fixed_mix_0.5 | 1371090 | 3 | 🟡 Running | Task 2 experiment 2/3 |
| curriculum_0.0_to_0.5 | 1372925 | 4 | 🟡 Running | Task 2 experiment 3/3 |
| wait_and_benchmark | 1395686 | - | 🟢 Monitoring | Auto-benchmark orchestrator |

### Available Resources

- GPUs 2, 3, 4: In use (scheduled-sampling experiments)
- GPU 6: Free (45GB available)
- Auto-monitoring active, will find free GPU for benchmarks

---

## 📊 Monitoring Commands

```bash
# Quick status check
bash /home/qma/work/GWM/monitor_scheduled_sampling.sh

# Full results summary
bash /home/qma/work/GWM/summarize_results.sh

# Check auto-benchmark progress
tail -f /tmp/wait_and_benchmark.log

# Check if experiments are done
cat H3-World/outputs/2026-10-02-scheduled-sampling/*/training.json | jq '.status'
```

---

## 🎯 Key Hypothesis Being Tested

**Problem**: Clean-history training → generated-history inference causes severe action collapse (A-D: +0.956 → +0.015)

**Root Cause**: Train/test distribution mismatch
- Training: Model only sees perfect teacher history
- Inference: Model gets its own imperfect generated history

**Hypothesis**: Scheduled-sampling (curriculum learning from clean→generated) will:
1. Expose model to generated history during training
2. Teach it to maintain action response despite history quality degradation
3. Bridge the distribution gap and recover action separation

**Test**: Compare 3 approaches on 39-frame W/A/D:
- Baseline (clean only): Expected A-D ≈ +0.956
- Fixed mix 0.5: Test if any mixing helps
- Curriculum 0.0→0.5: Test if gradual exposure is better

**Timeline**: Results in ~2 hours (started 23:44, expect completion ~01:30-02:00)

---

## 📝 Documentation Trail

All work is documented in:
1. `/home/qma/work/GWM/next_plan.md` - Updated task status
2. `/home/qma/work/GWM/progress.md` - Chronological log with experiment details
3. `/home/qma/work/GWM/TASK_SUMMARY.md` - Task 2 implementation summary
4. `/home/qma/work/GWM/SCHEDULED_SAMPLING_EXPERIMENTS.md` - Full experiment tracker
5. This file: `/home/qma/work/GWM/STATUS_REPORT.md` - Current state overview

---

**End of Report**

<!-- END ROOT_CLEANUP STATUS_REPORT.md -->

### root_cleanup snapshot: TASK_COMPLETION.md

Bytes: 6632; SHA-256: `49d9d032db6c6b2c342722beab1be260469e8e8b42dc6678368d41a13f996f4a`

<!-- BEGIN ROOT_CLEANUP TASK_COMPLETION.md -->
# Task Completion Summary - 2026-10-02

## 🎉 任务完成情况

### ✅ Task 2: Scheduled-Sampling 实现与实验启动 (完成)

**完成时间**: 2026-10-02 23:44-23:47

**核心成就**:
1. 成功实现 `--history-mix-schedule` 课程学习功能
2. 启动 3 组并行对比实验（39 帧，W/A/D 三动作）
3. 建立完整的自动化监控和评估流程

---

## 📊 当前实验状态

### 运行中的实验

| 实验 | GPU | PID | 状态 | 进度 |
|---|:---:|:---:|---|---|
| baseline_clean_only | 2 | 1369208 | ✅ 运行中 | 特征提取阶段 (case 2/3) |
| fixed_mix_0.5 | 3 | 1371090 | ✅ 运行中 | 特征提取阶段 (case 2/3) |
| curriculum_0.0_to_0.5 | 4 | 1372925 | ✅ 运行中 | 特征提取阶段 (case 0/3) |
| wait_and_benchmark | - | 1395686 | ✅ 监控中 | 自动等待并评估 |

**GPU 使用情况**:
- GPU 2: 8% 利用率, 41.6GB/46GB
- GPU 3: 5% 利用率, 41.7GB/46GB  
- GPU 4: 99% 利用率, 41.7GB/46GB (正在计算)

---

## 🔬 实验设计回顾

### 核心假设

**问题**: Clean-history 训练导致 generated-history 推理时动作塌缩
- Clean history: A-D flow = +0.956
- Generated history: A-D flow = +0.015 ⚠️

**根本原因**: 训练/推理分布不匹配
- 训练时：模型只见过完美的 teacher history
- 推理时：模型得到自己生成的不完美 history

**解决方案**: Scheduled-sampling (课程学习)
- 训练早期：100% clean history (易学)
- 训练后期：逐步增加 generated history (逼近推理分布)
- 目标：让模型学会在 history 质量下降时保持动作响应

### 三组对比实验

1. **Baseline (clean only)**: 
   - 配置：纯 clean history，无混合
   - 目的：确认 clean-history 性能上界
   - 预期：A-D ≈ +0.956

2. **Fixed mix 0.5**:
   - 配置：固定 50% clean + 50% generated
   - 目的：测试固定混合是否有效
   - 预期：如果有效，A-D 应该在 0.015 和 0.956 之间

3. **Curriculum 0.0→0.5**:
   - 配置：从 0% generated 线性增长到 50% generated
   - 目的：测试课程学习是否优于固定混合
   - 预期：应该 ≥ fixed mix，理想情况 > 0.1

---

## 📁 已创建/修改的文件

### 代码实现
- ✅ `H3-World/code/causal/train_pretrained_multiaction.py`
  - 新增 `--history-mix-schedule` 参数
  - 特征提取生成 3 个 history 变体
  - 训练循环动态选择变体
  - 语法验证通过

### 文档
- ✅ `/home/qma/work/GWM/next_plan.md` - 任务状态更新
- ✅ `/home/qma/work/GWM/progress.md` - 实验详情记录
- ✅ `/home/qma/work/GWM/TASK_SUMMARY.md` - Task 2 实现总结
- ✅ `/home/qma/work/GWM/SCHEDULED_SAMPLING_EXPERIMENTS.md` - 完整实验追踪
- ✅ `/home/qma/work/GWM/STATUS_REPORT.md` - 项目状态总览
- ✅ `/home/qma/work/GWM/TASK_COMPLETION.md` (本文件)

### 自动化脚本
- ✅ `/home/qma/work/GWM/monitor_scheduled_sampling.sh` - 手动进度检查
- ✅ `/home/qma/work/GWM/wait_and_benchmark.sh` - 自动等待并运行 benchmark
- ✅ `/home/qma/work/GWM/summarize_results.sh` - 结果汇总

---

## ⏰ 时间线

- **23:44**: 启动 3 个训练实验
- **23:46**: 启动自动监控和 benchmark 流程
- **23:47**: 任务完成，实验运行中
- **预计 01:30-02:00**: 实验完成，自动 benchmark 开始
- **预计 02:30**: 所有 benchmark 完成，结果可用

---

## 🎯 验收标准

### Task 2 实现 (✅ 已完成)
- ✅ `--history-mix-schedule` 参数实现
- ✅ 多变体特征提取实现
- ✅ 课程学习训练循环实现
- ✅ 语法验证通过
- ✅ 实验启动并正常运行

### 实验结果评估 (⏳ 等待中)
将在实验完成后评估：

**成功标准**:
- Curriculum 在 generated-history rollout 下达到 A-D > 0.1
- 明显好于 baseline generated-history (0.015)
- 证明 scheduled-sampling 有效

**失败标准**:
- A-D 仍然接近 0.015
- Scheduled-sampling 没有改善分布偏移问题
- 需要更激进的策略或 Stage2-style self-rollout

---

## 📋 后续步骤

### 自动进行 (无需干预)
1. ⏳ 实验继续运行 (~1-2 小时)
2. 🔄 `wait_and_benchmark.sh` 自动检测完成
3. 🚀 自动在空闲 GPU 上运行 benchmark
4. 📊 生成 flow.json 结果文件

### 需要手动决策 (实验完成后)

**如果成功** (A-D > 0.1):
1. 扩展 scheduled-sampling 到 124 帧
2. 与 Task 1 的 124 帧结果对比
3. 选择最佳 checkpoint 制作最终 demo (Task 4)

**如果失败** (A-D ≈ 0.015):
1. 分析日志，检查为什么课程学习无效
2. 尝试更激进的 schedule (0.0→1.0)
3. 考虑 Stage2-style self-rollout 方法

---

## 🔍 监控命令

```bash
# 快速检查进度
bash /home/qma/work/GWM/monitor_scheduled_sampling.sh

# 查看完整结果汇总
bash /home/qma/work/GWM/summarize_results.sh

# 查看自动 benchmark 日志
tail -f /tmp/wait_and_benchmark.log

# 检查实验是否完成
cat H3-World/outputs/2026-10-02-scheduled-sampling/*/training.json | jq '.status'

# 查看 GPU 使用情况
nvidia-smi

# 检查进程
ps aux | grep train_pretrained_multiaction
```

---

## 🎓 技术要点

### 实现的课程学习算法

```python
def get_history_mix_for_step(step, total_steps, start_mix, end_mix):
    """线性插值从 start_mix 到 end_mix"""
    progress = step / total_steps
    return start_mix + progress * (end_mix - start_mix)

# 在特征提取时生成 3 个变体
variants = [start_mix, (start_mix + end_mix)/2, end_mix]

# 在训练时选择最接近当前进度的变体
current_mix = get_history_mix_for_step(step, total_steps, start_mix, end_mix)
variant_index = argmin(|variants - current_mix|)
```

### 为什么是 3 个变体？

- 平衡离散采样与计算成本
- 3 个点足以近似线性 schedule
- 特征提取成本增加 3 倍，但训练时无额外计算

### 为什么是 0.0→0.5 而不是 0.0→1.0？

- 保守策略：先验证部分混合是否有效
- 如果 0.5 有效，可以扩展到更高比例
- 避免在 100% generated history 时训练不稳定

---

## ✅ 总结

**Task 2 已完成**：
- ✅ Scheduled-sampling 算法实现完毕
- ✅ 3 组对比实验正在运行
- ✅ 自动化评估流程已就绪
- ✅ 完整文档已建立

**下一个里程碑**：
等待实验结果（~2 小时），根据 A-D flow 数据决定是否扩展到 124 帧或调整策略。

**项目进展**：
- Task 1: ⏳ 训练完成，等待最终保存
- Task 2: ✅ **实现完成，实验运行中**
- Task 3: 📋 待启动（可并行）
- Task 4: ⏸️ 等待 Task 1/2 结果
- Task 5: ⏸️ 低优先级

---

**完成时间**: 2026-10-02 23:47  
**状态**: ✅ Task 2 实现完成，等待实验结果

<!-- END ROOT_CLEANUP TASK_COMPLETION.md -->

### root_cleanup snapshot: TASK_SUMMARY.md

Bytes: 4110; SHA-256: `2325b26b09fa54c0b85647994d7c2f8cc367c0573beb02393f085b9ad44fba91`

<!-- BEGIN ROOT_CLEANUP TASK_SUMMARY.md -->
# Task Summary: Scheduled-Sampling Implementation

**Date**: 2026-10-02  
**Status**: ✅ Implementation Complete, Awaiting Experimental Validation

---

## What Was Accomplished

### Task 2: Implemented `--history-mix-schedule` for Curriculum Learning

Successfully added scheduled-sampling / curriculum learning to the multi-action trainer. This addresses the core problem: **clean-history Stage1 training cannot handle generated-history free-running rollout** (A-D flow collapses from +0.956 to +0.015 in 39 frames).

#### Implementation Details

**New Parameter**:
```bash
--history-mix-schedule START:END
```

Example: `--history-mix-schedule 0.0:0.5` starts training with 100% clean history (mix=0.0) and linearly increases to 50% generated / 50% clean (mix=0.5) by the final step.

**Technical Approach**:
1. **Feature Extraction**: Generate 3 history mix variants per action clip:
   - variant 0: mix=START
   - variant 1: mix=(START+END)/2  
   - variant 2: mix=END

2. **Training Loop**: Dynamically select the appropriate variant based on training progress:
   - Early steps use variant 0 (mostly clean)
   - Middle steps use variant 1 (balanced mix)
   - Late steps use variant 2 (mostly generated)

3. **Metadata**: Each sample stores `history_mix_variant` index for curriculum sampling

**Modified File**: `H3-World/code/causal/train_pretrained_multiaction.py`

**Verification**: 
```bash
✅ python -m py_compile H3-World/code/causal/train_pretrained_multiaction.py
```
Syntax validation passed.

---

## Why This Matters

The root cause of action collapse is **train/test distribution mismatch**:

- **Training**: Model sees clean teacher history
- **Inference**: Model gets its own generated history with accumulated errors

**Evidence**:
| Condition | A-D horizontal flow |
|---|---:|
| Original H3 teacher, 39f, 30 steps | **+2.023** |
| Causal, clean history, 39f | +0.956 |
| Causal, **generated history**, 39f | **+0.015** ⚠️ |

Scheduled-sampling bridges this gap by **gradually exposing the model to generated history during training**, teaching it to maintain action response even when the history is imperfect.

---

## Next Steps

### Immediate: Quick 39-Frame Validation

Run a 39-frame experiment with 3 conditions:

1. **Baseline**: clean history only
2. **Fixed mix**: `--history-mix 0.5`
3. **Curriculum**: `--history-mix-schedule 0.0:0.5`

**Command template**:
```bash
python H3-World/code/causal/train_pretrained_multiaction.py \
  --teacher-dirs outputs/.../action_W_teacher_latents \
                 outputs/.../action_A_teacher_latents \
                 outputs/.../action_D_teacher_latents \
  --action-specs W A D \
  --history-latent-dirs outputs/.../action_W_base_causal39_latents \
                        outputs/.../action_A_base_causal39_latents \
                        outputs/.../action_D_base_causal39_latents \
  --history-mix-schedule 0.0:0.5 \
  --out-dir outputs/2026-10-02-XX/stage1_scheduled_sampling_39 \
  --steps 160 \
  --tail-blocks 8 \
  --device cuda:X
```

**Success Criterion**: A-D flow on 39-frame generated-history rollout improves from ~0.015 to measurable directional separation (e.g., >0.1).

### If Successful: Scale to 124 Frames

Repeat the experiment with 124-frame teacher clips and 8-step/chunk rollout.

### Task 1: Still Waiting

`stage1_action_generated_hidden124` is still training (status: running). Once complete, benchmark its 124-frame W/A/D output to compare against the new scheduled-sampling approach.

---

## Files Modified

- ✅ `H3-World/code/causal/train_pretrained_multiaction.py`: 
  - Added `--history-mix-schedule` parameter
  - Implemented 3-variant feature extraction
  - Added curriculum-based sample selection in training loop
  - Added `get_history_mix_for_step()` helper function

- ✅ `/home/qma/work/GWM/next_plan.md`: Updated to mark Task 2 complete

---

## References

- **Progress log**: `/home/qma/work/GWM/progress.md` (2026-10-02 sections)
- **Original plan**: `/home/qma/work/GWM/next_plan.md`
- **Action control evidence**: `H3-World/outputs/2026-10-02-04/` (39-frame A/D decomposition)

<!-- END ROOT_CLEANUP TASK_SUMMARY.md -->

### root_cleanup snapshot: WORKSPACE.md

Bytes: 2024; SHA-256: `167bbd457e0043fa004c40f61c99ffed281e62354c253949b9e8b4c126bea561`

<!-- BEGIN ROOT_CLEANUP WORKSPACE.md -->
# GWM workspace

## Layout

- `H3-World/`: upstream H3-World source repository.
- `SolarWM/`: upstream SolarWM source repository.
- `models/`: existing symlink to `/work/lpeng/qma/GWM_models`; weights stay on that storage volume.
- `H3-World/checkpoints/H3-World/step-10000.safetensors`: relative symlink to the shared H3-World LoRA.
- `H3-World/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3`: relative symlink to the shared MiniMax-H3 FL2VA weights.

The DiffSynth directory currently contains only the model-path scaffold, NOT an
installed source checkout. Follow the in-place initialization instructions in
`H3-World/README.md` (ordinary `git clone` into this non-empty directory will fail),
use `code/diffsynth_base_commit.txt`, and apply `code/diffsynth_h3_action.patch`.
Do not install DiffSynth in editable mode.

## Intended deliverable and current status

Investigate SolarWM's MiniMax-H3 causal chunks, KV cache, and Stage2 few-step
generation, then build a minimal causal training/inference prototype for H3-World.
Deliver code, a short experiment report, and a playable side-by-side video:
original H3-World (e.g. 30 steps) versus the causal prototype, reporting runtime,
peak GPU memory, visual quality, and temporal continuity. Full 33B training and a
full Stage2 reproduction are not required.

The repository now contains the minimal runnable prototype and review artifacts. The current primary
video is `H3-World/outputs/h3world_30step_vs_stage1_tail16_dual_anchor_8step_124.mp4`; the direct
RGB-prefix last-frame comparison is
`H3-World/outputs/h3world_30step_vs_stage1_rgb_tail16_dual_anchor_8step_124.mp4`. The prototype
has real H3 causal masking, per-layer raw KV cache, clean commits, Stage1-style teacher-forcing
QKV LoRA and dual last-frame anchors. Full SolarWM AnyFlow and Stage2 SGF/DMD are intentionally
not implemented; causal masking, KV caching and ordinary few-step flow must not be presented as
Stage2 distillation. The reports record the remaining long-horizon drift and memory limits.

<!-- END ROOT_CLEANUP WORKSPACE.md -->

### root_cleanup snapshot: report.md

Bytes: 2916; SHA-256: `d3a8466905b5d9ce4f9f50c774e0b163a135c4bd6119b9e00521a850bdd3a912`

<!-- BEGIN ROOT_CLEANUP report.md -->
# 当前任务执行报告

> 初始化日期：2026-10-10（Asia/Hong_Kong）。由 Judge 创建报告入口；实际执行记录由 Exp Worker 维护。
> 本文件是尚未绑定新任务的空报告，不是对历史实验的补报，也不改变旧任务的实际执行状态。

## 1. 任务与状态

| 字段 | 当前值 |
| --- | --- |
| Task ID | 未分配；等待符合 guideline 的任务书 |
| Plan Version | 未分配；现存 next_plan.md 为历史 v5，不能据此虚构 EXP 编号 |
| Worker Status | not_started（仅指本空报告对应的新任务） |
| Judge Acceptance | pending |
| Research Track / Parent Version | 待任务书确定 |
| 授权入口 | [next_plan.md](next_plan.md) |
| 规则与正式进展 | [guideline.md](guideline.md)、[progress.md](progress.md) |
| 历史与接管记录 | [archive.md](archive.md) |

现存任务书保持研究冻结。此模板不授权训练、模型推理、预算扩展或新实验。

## 2. 历史已知证据

待 Worker 按本轮任务引用已有证据，标明历史来源和验收范围。历史结果不能记作本轮新发现。

## 3. 本轮实际执行

尚无 Worker 执行记录。执行时逐项填写：

- 实际命令、起止时间、退出状态、环境与依赖版本。
- Git revision、工作区改动或冻结源码 hash。
- checkpoint、released LoRA、新 adapter 路径及 hash。
- 数据、场景、随机种子、初始噪声与历史来源。
- attention topology、action routing/feedback、anchor、timestep/RoPE。
- sampling steps、sigma schedule、chunk partition、KV prefill/commit/reuse 和解码协议。
- GPU 型号及编号、前向次数、训练更新次数、耗时与峰值显存；预算消耗与剩余量。
- 原始实验目录、配置、日志、指标、视频和 MANIFEST 路径。

## 4. 本轮观察结果

尚无本轮结果。实际填写时保留负结果、异常、失败轮次及协议偏离。

| 能力 | 本轮状态 | 证据与范围 |
| --- | --- | --- |
| Action Fidelity | NOT_TESTED | 未执行 |
| Visual Stability | NOT_TESTED | 未执行；正式验收需要完整视频序列及必要原分辨率细节 |
| Efficiency | NOT_TESTED | 未执行；需说明硬件、采样与计时口径 |

## 5. 可以支持的结论

暂无。执行 completed 与能力 PASS 分开记录；数值诊断改善不能替代动作和画质验收。

## 6. 尚未验证的假设与限制

待填写。明确混杂因素、缺失证据、历史范围及泛化边界。

## 7. 下一步建议

待填写建议及其正负结果对研究决策的影响。建议不构成自动执行授权。

## 8. 提交检查与 Judge 入口

Worker 完成本轮后记录实际预算与交付路径，将 Worker Status 更新为 completed、blocked 或 failed，并等待 Judge 审计。Judge 的正式评价写入 next_plan.md；结束任务的任务书、完整报告和验收结论归入 archive.md。

<!-- END ROOT_CLEANUP report.md -->



## 2026-10-10T03:17:05+08:00：EXP-001 plan_version 2 资源与监督更新

用户授权本夜09:00前最多8GPU，09:00后最多3GPU，并要求Judge监督Worker报告、完成验收后更新next_plan。此修订只调整并发与交接规则，保留612次denoiser、8 GPU-hours、8小时wall和零训练限制。v1下已执行的运行保留当时配置与hash，不追溯改写；Worker后续确认v2。

旧任务书SHA-256：`2b9245a31061ebc8de7315d1d98f88c2e4f23ab4d07a0cc20f02f1c244b7864a`。以下为修订前完整原文：

<!-- BEGIN EXP-001 PLAN_V1 -->
# EXP-001：V2b 多窗口续写与 Original H3-World 对比

发布日期：2026-10-10（Asia/Hong_Kong）。发布者：Judge。

## 1. Task ID / Version / Status

| 字段 | 内容 |
| --- | --- |
| Task ID | EXP-001（新编号体系首项；历史 experiments/01–11 不改号） |
| Plan Version | 1 |
| Research Track | Mainline / V2b capability validation |
| Parent Version | V2b `C12_then5_N_30step_selfhistory`；Original H3 + released action LoRA，零新增 adapter |
| Worker Status | not_started |
| Judge Acceptance | pending |
| 授权范围 | 用户要求将V2b长视频与Original对比列为当务之急；本任务书限定为分阶段124帧验证 |
| 当前结果 | NOT_TESTED；本次仅发布计划，尚未运行 |

**当前唯一可执行研究任务为EXP-001。** 旧v5交付冻结任务书已原文归档，由Judge按用户新优先级行政关闭（cancelled / superseded）；其历史质量No-Go结论保持。当前空报告已绑定本任务，无未完成Worker报告被覆盖。

## 2. Research Question / Hypothesis

**核心问题：V2b在不重置自身生成历史、保持既有推理协议的条件下，能否从已验证的56帧续写到124帧，并在持续动作与一次动作切换中保留方向响应和人物结构？**

假设：Single I0、native time、Same-σ history和12→5分块的局部正结果可延续到更多窗口；该假设可能被累计漂移、动作响应衰减或解码边界破坏否定。

这次唯一主要变化是rollout长度及其自然增长的自生成历史。Original对照属于完整协议的能力比较，存在attention、音频处理、历史和输出方式差异，不作严格单变量归因。

124帧@24fps约5.17秒，是第一轮多窗口目标。10/20秒要在此结果后另定任务与预算。

## 3. Baseline / Controlled Variables / Changed Variables

### Baseline

1. **Parent对照**：已有V2b四条56帧结果及其6份生成latent endpoint；续写前保持前两块配置和历史来源可追溯。
2. **Original对照**：Original H3-World + 同一released action LoRA，124帧整段双向30-step生成，A、D、A→D、D→A四种动作。
3. 优先检查[Original manifest](submission/mainline/V0_original_bidirectional/manifest.json)中的既有A/D124帧；只有权重、初图、prompt/action、noise、时间/位置、精度/backend等来源可核对才复用为匹配对照。仅seed相同不能证明noise相同；无法核对时可在下述4条总预算内重生成。
4. Original保留原生联合audio/video去噪；V2b保留既有固定audio noise与native时间条件。记录该已知混杂，不为匹配而改动V2b，也不将改过audio协议的窗口重算器标作原生Original。Original可见完整未来动作/视频，V2b仅可见截至当前块的信息。

### Controlled Variables

- Original底座、released action LoRA；LoRA SHA-256：`ddd9187b920b1e52c2d090f4e264fd83d8d433efc2a5b159e58883aeaf96e526`。基础权重使用冻结runtime既有来源/hash，Worker记录完整引用。
- 停车场初图832×480、seed13、full37 latent noise/layout、原prompt、Single I0、native text/action time、全局RoPE、30 steps/chunk、native schedule、flow shift2.22。
- V2b沿用旧`h3_fp32`精度策略及冻结backend；该名称不代表所有算子均FP32，应记录实际dtype/attention设置。不要换成P0审计专用canonical backend。
- V2b历史为各路径自身generated endpoints，每步用同一原始noise切片加到当前sigma；仅积分当前chunk。保存的history、已发布RGB保持不变。
- 使用原Original directed action mask，T2每步重算可见prefix/history/current；未来动作/video在refiner前移除。**全部历史为[0:start]，不新增滑窗截断或eviction。**
- 无persistent hidden KV、无新增adapter、无GT/teacher重置、无输出平滑。

### Changed Variables

续写区间从既有[12,17)扩展至[17,22)、[22,27)、[27,32)、[32,37)。历史自然增长，其他协议保持。

| 轨迹ID | latent [0,12)动作 | latent [12,37)动作 | 展示 |
| --- | --- | --- | --- |
| AA | A | A | 持续A |
| DD | D | D | 持续D |
| AD | A | D | 在latent12切为D，之后持续D |
| DA | D | A | 在latent12切为A，之后持续A |

完整分块为[12,5,5,5,5,5]，共37 latent；累计RGB预期39、56、73、90、107、124。切换以latent/action-span边界为准，展示RGB39为发布边界；VAE存在时间影响，不声称物理动作精确从第39帧立即发生。

## 4. Inputs

- [旧冻结协议](H3-World/outputs/2026-10-09-22/chunk_partition_cb/protocol.json)：输入hash与推理配置。
- [6份endpoint manifest](submission/experiments/11_causal_12_then5_selfhistory/manifest.json)：`states/first12_A.pt`、`first12_D.pt`及四条`history{A,D}_current{A,D}_next5.pt`。
- 四条起始17-latent历史由对应first12与next5拼接，不可误将next5当完整历史或训练adapter。逐文件与tensor hash核对；任何缺失/不一致先停止，不在本预算内重跑前两块。
- 输入conditioning/noise：`H3-World/outputs/2026-10-09-05/stage1_coarse_window12/inputs/parking_{A,D}.pt`；旧solver状态与冻结runtime见protocol/source_coarse引用。
- 冻结参考源码：[run.py](H3-World/outputs/2026-10-09-22/chunk_partition_cb/run.py)、[interval_forward.py](H3-World/outputs/2026-10-09-22/chunk_partition_cb/interval_forward.py)。旧runner只支持第二块和B首窗，需要新增多块续写入口，不能直接把旧脚本当作已有长片生成器。

## 5. Execution Plan

### S0：CPU准备、冻结与最小回放

1. 在`submission/experiments/EXP-001_v2b_124/`建立README、config.yaml、MANIFEST、logs、artifacts；原始大tensor可放`H3-World/outputs/EXP-001_v2b_124/`，MANIFEST记录路径/hash。检查目录不存在冲突后创建，不覆盖历史目录。
2. 基于冻结源码实现隔离的`run_rollout.py`与评估入口，支持显式全局区间、四条路径、逐块checkpoint、恢复与预算计数。完整保存过去动作序列：AD/DA不能因当前动作改变而改写前12 latent动作。
3. 在CPU小模型上验证新范围[17,22)、[22,27)、[27,32)、[32,37)的action spans/RoPE/未来信息隔离、历史只读、同sigma加噪、旧区间等价、解码发布帧数与续接恢复。新增测试针对这些真实风险，不只验证CLI。
4. 保存代码hash、checkpoint/hash、完整sigma序列、输入/noise hash、环境与预算配置。核查空闲GPU后，使用不超过12次真实denoiser回放验证旧first12/second5协议一致；目标relative RMS≤1e-6，历史hash不变。不匹配则停止，不能放宽门槛后继续。
5. 解码旧prefix并核对已有56帧历史的发布方式；不能用一次性decode完整17 latent所得的前56帧替换原先分两次发布的39+17帧。保存未压缩已发布RGB及hash用于后续精确不回改检查。MP4解码差异需与有损压缩误差分开。

现有CPU检查命令（项目根目录，沿用已有环境）：

```bash
.venvs/h3world/bin/python -m pytest -q H3-World/outputs/2026-10-09-22/chunk_partition_cb/test_intervals.py
```

以下是**Worker需实现的入口约定，当前脚本尚不存在**；Worker在report中填写实际可运行命令与GPU分配：

```bash
.venvs/h3world/bin/python submission/experiments/EXP-001_v2b_124/run_rollout.py --config submission/experiments/EXP-001_v2b_124/config.yaml --stage gate90
.venvs/h3world/bin/python submission/experiments/EXP-001_v2b_124/run_rollout.py --config submission/experiments/EXP-001_v2b_124/config.yaml --stage extend124
```

### S1：第三／第四块门槛，56→90帧

对AA、DD、AD、DA四条已存自身历史，各续写[17,22)、[22,27)，共8个新chunk、240次sampling forward。逐块解码、冻结已发布RGB，仅追加17帧，保留过去末5帧重新解码差异的诊断值。

每生成一块就检查完整新增帧、边界和人物原分辨率细节。任何路径出现明确动作方向失败或严重结构崩坏时，保存证据并停止后续扩展。四条均通过至90帧后，将逐路径gate写入metrics和report，才能在既定预算内执行S2；不需要额外等待Judge，但判定不确定时停交Judge复核。

### S2：通过门槛后，90→124帧

从S1各自保存的历史继续[27,32)、[32,37)，再8个chunk、240次sampling forward。逐块执行同样门槛；不从头重跑、不换seed、不重新选择“最好”的路径。

### S3：Original匹配对照与报告

1. 通过124帧后复用合格Original视频；缺失的匹配A/D/AD/DA最多补4条，每条整段30 forwards，共≤120。若V2b提前失败，先交付失败报告及已有参考，停止自动补生成Original。
2. 优先复用已保存的相同初始video/audio noise，记录原生joint音频积分差异；无法确认匹配输入时停止该对照并报告缺失，不悄悄降低比较标准。
3. 制作四条左右并排完整124帧对比（左Original，右V2b），加一个四路径索引/总览。标明动作、24fps、帧数、30 full-horizon steps vs 30 steps/chunk、history/anchor/topology、音频协议差异。
4. 本轮交付全部正负结果与原始视频；可截取演示摘要，但摘要不能替代完整原片。Judge验收后才同步升级mainline/report代表片。

## 6. Resource Budget

| 项目 | 硬上限 |
| --- | --- |
| 训练 | 0 optimizer updates |
| 场景/seed/路径 | 1停车场、seed13、4固定路径 |
| V2b新sampling | S1 240 + S2 240 = 480次forward；前两块复用 |
| Original补生成 | 最多4条124帧 × 30 = 120次forward |
| 回放/必要诊断 | 最多12次完整denoiser forward；不作额外机制扫描 |
| 总denoiser预算 | ≤612次，失败调用和排错调用也计入；不能把prefill或重试藏在预算外 |
| VAE | 每条每块最多一次prefix decode，另加初始化；V2b≤24次、Original≤4次，共≤28次decode、0次新RGB re-encode |
| GPU | 最多同时2张；运行时确认空闲，沿用已有offload；单卡allocated≤44GiB并预留系统空间 |
| 时间 | 模型任务累计≤8 GPU-hours且首个GPU进程启动后≤8小时；计入加载、诊断、VAE和失败重试，先触及任一上限即停 |
| 输出长度 | 每条≤124帧；不生成10/20秒，不增加额外路径 |

不要固定沿用历史GPU编号。不能抢占其他任务；OOM、显存阈值或协议异常触发停止，不自动换精度、分辨率、历史窗口来迁就资源。

## 7. Evaluation / Acceptance

执行状态与能力状态分别报告。任务completed允许包含失败结果；只有达到下述条件才可给相应能力PASS。

### Action Fidelity

- 使用旧`evaluate_action_control`同一中心裁剪/Farneback参数，对每个新增17帧区间和全片分别记录水平flow及有效帧数；边界flow单独记录，不能用整片均值掩盖局部失败。
- 四路径每一新块的符号须符合当前动作：A>0、D<0；同时人工完整帧序列确认与场景约定的左右响应一致，动作切换后不持续沿用旧方向。接近零或视觉与flow冲突时记不确定，停交Judge，不依赖符号单独判PASS。
- 继承的第二块同history A/D正结果属于历史证据；第三块后四轨迹的history已分叉，跨路径差分不能描述为同状态counterfactual。该任务验证持续控制/切换的rollout表现，不声称新增固定状态因果证明。

### Visual Stability

- 全部124帧逐帧覆盖检查及正常24fps播放；每个边界前后至少3帧查看原分辨率人物细节。
- 每块人物保持单一可辨身体、场景连续；无严重ghosting、透明化、肢体分解、人物消失或突变。记录普通模糊/细节形变，不能用“视频可解码”替代画质判断。
- 已发布RGB在本轮续写过程中的hash必须保持；记录每块boundary MAD、块内MAD及过去末5RGB重解码差异。MAD是诊断量，无预先依据时不临时编造通过阈值。
- 报告首次异常RGB/latent区间和持续范围，保留失败原片；不得裁掉失败后宣称全长通过。

### Efficiency

逐块记录forward次数、可见history长度、sampling latency、VAE/搬运/编码耗时、allocated/reserved peak、CPU模型offload与latent内存。persistent hidden KV应为0。

四条V2b都复用前两块，因此本轮测得的是增量耗时；如引用旧first12/second5时间，必须标为跨次累计估算，不冒充同次124帧E2E。Original30整段forward与V2b完整六块180forward不等工作量，不直接宣传公平speedup。

Efficiency本轮只验收测量完整性；加速能力没有预设通过门槛，不能因表格齐全标作模型效率PASS。

### 总体交付判定

四路径均达到124帧动作与结构门槛、Original四路径对照来源匹配且完整、协议和测量可追溯时，Judge可接受“V2b在单场景/seed的124帧多窗口能力”。失败或对照缺失按FAIL/PARTIAL/INVALID注明具体范围。任何结果均不自动升级V3，也不代表10/20秒或跨场景泛化通过。

## 8. Stop Conditions / Out of Scope

出现任一路径明确质量失败、数值异常、OOM、hash/回放不一致、未来信息泄漏、已发布帧回改、需要改协议或预算耗尽，即保存现场停止并提交Judge。充分否定假设后不继续延长剩余路径以消耗预算。

范围外：训练、换seed/场景、gain/anchor/sigma sweep、clean-history消融、KV迁移、strict causal架构修复、AnyFlow、DMD、延长到10/20秒、为性能排名重跑完整V2b。

## 9. Deliverables

- `submission/experiments/EXP-001_v2b_124/`：README、冻结config.yaml、metrics.json、MANIFEST.md、日志、CPU检查、独立续写入口与评估代码。
- 原始checkpoint/输入引用与hash、每块生成latent、已发布RGB hash、逐块性能、动作与视觉评审、完整正负原片。
- 四条Original vs V2b对比及元数据；未达到生成阶段时交付明确的失败/缺失清单。
- 根目录[report.md](report.md)：历史证据、本轮执行、观察、可支持结论、未验证假设、实际预算分别记录。

## 10. Judge 决策与验收栏

| 项目 | 当前记录 |
| --- | --- |
| Decision | continue：发布本有限任务，Worker尚未启动 |
| Acceptance | pending |
| 执行合规 | 待报告 |
| Action / Visual / Efficiency | NOT_TESTED / NOT_TESTED / NOT_TESTED（本轮） |
| 验收证据与问题 | 待Worker提交；不得用旧56帧结果填写本轮PASS |

本实验若成功，会提供下一步长时/KV研究所需的多窗口正控；若失败，会确定V2b局部正控的长度边界并阻止过早推进少步生成。已有证据只到第二块，尚未回答本问题。复用6份endpoint并在90帧设门槛，是本轮降低成本的方式。

<!-- END EXP-001 PLAN_V1 -->


## 2026-10-10T03:23:39+08:00：按用户ROI要求修订EXP-001 v3

用户要求不要钻牛角尖、审核不要卡得过死，重视研究ROI与大局。Judge复核：第三块AA/DD/AD符号满足，DA flow −0.175372未过原预登记符号门槛，但人物结构仍保持；仅此proxy不支持停止一切长视频能力验证，也不支持“动作信号完全消失”。撤回全盘停止及追加专项诊断方向。

新的有限任务只续AA/DD持续动作到124帧并与Original A/D对比，AD/DA保留73帧观察。原四路径联合门槛未通过的事实保留；新范围明确为持续动作，不回写旧结果为PASS。重用现有73帧、零新增训练、不扩大总预算，优先完成有研究价值的交付。

旧v2任务书SHA：`9718de3b6020fc722b8fd64a33f50e180cd2b75de8846bd1dd3162f9a0410a05`；当时在途Worker报告SHA：`9a8c96b4fe38cbc399f0b3af8f806f2073bf88670f6d82ea0cfc489c3e9e1988`。报告仍由Worker维护，此快照不是最终报告或正式验收。

<!-- BEGIN EXP-001 PLAN_V2 -->
# EXP-001：V2b 多窗口续写与 Original H3-World 对比

发布日期：2026-10-10（Asia/Hong_Kong）。发布者：Judge。

## 1. Task ID / Version / Status

| 字段 | 内容 |
| --- | --- |
| Task ID | EXP-001（新编号体系首项；历史 experiments/01–11 不改号） |
| Plan Version | 2 |
| Research Track | Mainline / V2b capability validation |
| Parent Version | V2b `C12_then5_N_30step_selfhistory`；Original H3 + released action LoRA，零新增 adapter |
| Worker Status | running；已启动第三块，见report与原始日志 |
| Judge Acceptance | pending |
| 授权范围 | 用户要求将V2b长视频与Original对比列为当务之急；本任务书限定为分阶段124帧验证 |
| 当前结果 | 执行中；能力验收pending，不能将中间结果当作通过 |

**当前唯一可执行研究任务为EXP-001。** 旧v5交付冻结任务书已原文归档，由Judge按用户新优先级行政关闭（cancelled / superseded）；其历史质量No-Go结论保持。Worker正在执行本任务；本次只修订资源与交接规则，保留在途运行及其v1配置证据。

## 2. Research Question / Hypothesis

**核心问题：V2b在不重置自身生成历史、保持既有推理协议的条件下，能否从已验证的56帧续写到124帧，并在持续动作与一次动作切换中保留方向响应和人物结构？**

假设：Single I0、native time、Same-σ history和12→5分块的局部正结果可延续到更多窗口；该假设可能被累计漂移、动作响应衰减或解码边界破坏否定。

这次唯一主要变化是rollout长度及其自然增长的自生成历史。Original对照属于完整协议的能力比较，存在attention、音频处理、历史和输出方式差异，不作严格单变量归因。

124帧@24fps约5.17秒，是第一轮多窗口目标。10/20秒要在此结果后另定任务与预算。

## 3. Baseline / Controlled Variables / Changed Variables

### Baseline

1. **Parent对照**：已有V2b四条56帧结果及其6份生成latent endpoint；续写前保持前两块配置和历史来源可追溯。
2. **Original对照**：Original H3-World + 同一released action LoRA，124帧整段双向30-step生成，A、D、A→D、D→A四种动作。
3. 优先检查[Original manifest](submission/mainline/V0_original_bidirectional/manifest.json)中的既有A/D124帧；只有权重、初图、prompt/action、noise、时间/位置、精度/backend等来源可核对才复用为匹配对照。仅seed相同不能证明noise相同；无法核对时可在下述4条总预算内重生成。
4. Original保留原生联合audio/video去噪；V2b保留既有固定audio noise与native时间条件。记录该已知混杂，不为匹配而改动V2b，也不将改过audio协议的窗口重算器标作原生Original。Original可见完整未来动作/视频，V2b仅可见截至当前块的信息。

### Controlled Variables

- Original底座、released action LoRA；LoRA SHA-256：`ddd9187b920b1e52c2d090f4e264fd83d8d433efc2a5b159e58883aeaf96e526`。基础权重使用冻结runtime既有来源/hash，Worker记录完整引用。
- 停车场初图832×480、seed13、full37 latent noise/layout、原prompt、Single I0、native text/action time、全局RoPE、30 steps/chunk、native schedule、flow shift2.22。
- V2b沿用旧`h3_fp32`精度策略及冻结backend；该名称不代表所有算子均FP32，应记录实际dtype/attention设置。不要换成P0审计专用canonical backend。
- V2b历史为各路径自身generated endpoints，每步用同一原始noise切片加到当前sigma；仅积分当前chunk。保存的history、已发布RGB保持不变。
- 使用原Original directed action mask，T2每步重算可见prefix/history/current；未来动作/video在refiner前移除。**全部历史为[0:start]，不新增滑窗截断或eviction。**
- 无persistent hidden KV、无新增adapter、无GT/teacher重置、无输出平滑。

### Changed Variables

续写区间从既有[12,17)扩展至[17,22)、[22,27)、[27,32)、[32,37)。历史自然增长，其他协议保持。

| 轨迹ID | latent [0,12)动作 | latent [12,37)动作 | 展示 |
| --- | --- | --- | --- |
| AA | A | A | 持续A |
| DD | D | D | 持续D |
| AD | A | D | 在latent12切为D，之后持续D |
| DA | D | A | 在latent12切为A，之后持续A |

完整分块为[12,5,5,5,5,5]，共37 latent；累计RGB预期39、56、73、90、107、124。切换以latent/action-span边界为准，展示RGB39为发布边界；VAE存在时间影响，不声称物理动作精确从第39帧立即发生。

## 4. Inputs

- [旧冻结协议](H3-World/outputs/2026-10-09-22/chunk_partition_cb/protocol.json)：输入hash与推理配置。
- [6份endpoint manifest](submission/experiments/11_causal_12_then5_selfhistory/manifest.json)：`states/first12_A.pt`、`first12_D.pt`及四条`history{A,D}_current{A,D}_next5.pt`。
- 四条起始17-latent历史由对应first12与next5拼接，不可误将next5当完整历史或训练adapter。逐文件与tensor hash核对；任何缺失/不一致先停止，不在本预算内重跑前两块。
- 输入conditioning/noise：`H3-World/outputs/2026-10-09-05/stage1_coarse_window12/inputs/parking_{A,D}.pt`；旧solver状态与冻结runtime见protocol/source_coarse引用。
- 冻结参考源码：[run.py](H3-World/outputs/2026-10-09-22/chunk_partition_cb/run.py)、[interval_forward.py](H3-World/outputs/2026-10-09-22/chunk_partition_cb/interval_forward.py)。旧runner只支持第二块和B首窗，需要新增多块续写入口，不能直接把旧脚本当作已有长片生成器。

## 5. Execution Plan

### S0：CPU准备、冻结与最小回放

1. 在`submission/experiments/EXP-001_v2b_124/`建立README、config.yaml、MANIFEST、logs、artifacts；原始大tensor可放`H3-World/outputs/EXP-001_v2b_124/`，MANIFEST记录路径/hash。检查目录不存在冲突后创建，不覆盖历史目录。
2. 基于冻结源码实现隔离的`run_rollout.py`与评估入口，支持显式全局区间、四条路径、逐块checkpoint、恢复与预算计数。完整保存过去动作序列：AD/DA不能因当前动作改变而改写前12 latent动作。
3. 在CPU小模型上验证新范围[17,22)、[22,27)、[27,32)、[32,37)的action spans/RoPE/未来信息隔离、历史只读、同sigma加噪、旧区间等价、解码发布帧数与续接恢复。新增测试针对这些真实风险，不只验证CLI。
4. 保存代码hash、checkpoint/hash、完整sigma序列、输入/noise hash、环境与预算配置。核查空闲GPU后，使用不超过12次真实denoiser回放验证旧first12/second5协议一致；目标relative RMS≤1e-6，历史hash不变。不匹配则停止，不能放宽门槛后继续。
5. 解码旧prefix并核对已有56帧历史的发布方式；不能用一次性decode完整17 latent所得的前56帧替换原先分两次发布的39+17帧。保存未压缩已发布RGB及hash用于后续精确不回改检查。MP4解码差异需与有损压缩误差分开。

现有CPU检查命令（项目根目录，沿用已有环境）：

```bash
.venvs/h3world/bin/python -m pytest -q H3-World/outputs/2026-10-09-22/chunk_partition_cb/test_intervals.py
```

以下是**Worker需实现的入口约定，当前脚本尚不存在**；Worker在report中填写实际可运行命令与GPU分配：

```bash
.venvs/h3world/bin/python submission/experiments/EXP-001_v2b_124/run_rollout.py --config submission/experiments/EXP-001_v2b_124/config.yaml --stage gate90
.venvs/h3world/bin/python submission/experiments/EXP-001_v2b_124/run_rollout.py --config submission/experiments/EXP-001_v2b_124/config.yaml --stage extend124
```

### S1：第三／第四块门槛，56→90帧

对AA、DD、AD、DA四条已存自身历史，各续写[17,22)、[22,27)，共8个新chunk、240次sampling forward。逐块解码、冻结已发布RGB，仅追加17帧，保留过去末5帧重新解码差异的诊断值。

每生成一块就检查完整新增帧、边界和人物原分辨率细节。任何路径出现明确动作方向失败或严重结构崩坏时，保存证据并停止后续扩展。四条均通过至90帧后，将逐路径gate写入metrics和report，才能在既定预算内执行S2；不需要额外等待Judge，但判定不确定时停交Judge复核。

### S2：通过门槛后，90→124帧

从S1各自保存的历史继续[27,32)、[32,37)，再8个chunk、240次sampling forward。逐块执行同样门槛；不从头重跑、不换seed、不重新选择“最好”的路径。

### S3：Original匹配对照与报告

1. 通过124帧后复用合格Original视频；缺失的匹配A/D/AD/DA最多补4条，每条整段30 forwards，共≤120。若V2b提前失败，先交付失败报告及已有参考，停止自动补生成Original。
2. 优先复用已保存的相同初始video/audio noise，记录原生joint音频积分差异；无法确认匹配输入时停止该对照并报告缺失，不悄悄降低比较标准。
3. 制作四条左右并排完整124帧对比（左Original，右V2b），加一个四路径索引/总览。标明动作、24fps、帧数、30 full-horizon steps vs 30 steps/chunk、history/anchor/topology、音频协议差异。
4. 本轮交付全部正负结果与原始视频；可截取演示摘要，但摘要不能替代完整原片。Judge验收后才同步升级mainline/report代表片。

## 6. Resource Budget

| 项目 | 硬上限 |
| --- | --- |
| 训练 | 0 optimizer updates |
| 场景/seed/路径 | 1停车场、seed13、4固定路径 |
| V2b新sampling | S1 240 + S2 240 = 480次forward；前两块复用 |
| Original补生成 | 最多4条124帧 × 30 = 120次forward |
| 回放/必要诊断 | 最多12次完整denoiser forward；不作额外机制扫描 |
| 总denoiser预算 | ≤612次，失败调用和排错调用也计入；不能把prefill或重试藏在预算外 |
| VAE | 每条每块最多一次prefix decode，另加初始化；V2b≤24次、Original≤4次，共≤28次decode、0次新RGB re-encode |
| GPU | Asia/Hong_Kong时间2026-10-10 09:00前最多同时8张，09:00起最多3张；沿用已有offload，单卡allocated≤44GiB |
| 时间 | 模型任务累计≤8 GPU-hours且首个GPU进程启动后≤8小时；计入加载、诊断、VAE和失败重试，先触及任一上限即停 |
| 输出长度 | 每条≤124帧；不生成10/20秒，不增加额外路径 |

### 分时GPU调度与09:00释放要求

- 以`Asia/Hong_Kong`和绝对截止时刻`2026-10-10T09:00:00+08:00`判断；截止前上限8张，截止时刻起上限3张。后续夜间不会自动恢复8张，除非用户重新授权。
- 这是本项目所有在途模型任务合计占用的不同GPU上限，包括加载、采样、VAE、预留/空闲驻留模型；不能只限制新进程启动而让旧进程越界。可用卡数多于独立路径数时，不扩展实验来占满卡。
- 每次启动、恢复和块间调度都检查实时GPU与项目进程身份。08:30后不再启动会使总占卡超过3的新chunk；提前保存安全检查点、释放额外模型，确保09:00已降到≤3。
- Worker必须在调度器或独立守护检查中落实截止释放；不能仅在Markdown写约束。若额外任务不能在截止前正常结束，应在安全步保存当前latent、solver进度、noise/RNG、已发布RGB和预算后退出/释放GPU，之后在≤3卡内恢复。不能恢复时报告blocked，不擅自从头重跑。
- 截止守护只操作本EXP登记且PID/进程启动标识匹配的进程；不得终止其他用户任务。到点后暂停而仍占用显存不算释放。
- 8卡并发不增加总预算：所有活跃进程已消耗时间与已结束进程时间一起计入8 GPU-hours，wall deadline从本任务第一个GPU进程开始计算，不能在重启时重置。
- 若Worker先前按用户另一消息记录了“09:00前6卡／之后4卡”，该调度规则由本次最新明确授权8／3替代；旧运行保留原配置快照和hash，在下一启动前更新调度与报告。

不要固定沿用历史GPU编号。不能抢占其他任务；OOM、显存阈值或协议异常触发停止，不自动换精度、分辨率、历史窗口来迁就资源。

## 7. Evaluation / Acceptance

执行状态与能力状态分别报告。任务completed允许包含失败结果；只有达到下述条件才可给相应能力PASS。

### Action Fidelity

- 使用旧`evaluate_action_control`同一中心裁剪/Farneback参数，对每个新增17帧区间和全片分别记录水平flow及有效帧数；边界flow单独记录，不能用整片均值掩盖局部失败。
- 四路径每一新块的符号须符合当前动作：A>0、D<0；同时人工完整帧序列确认与场景约定的左右响应一致，动作切换后不持续沿用旧方向。接近零或视觉与flow冲突时记不确定，停交Judge，不依赖符号单独判PASS。
- 继承的第二块同history A/D正结果属于历史证据；第三块后四轨迹的history已分叉，跨路径差分不能描述为同状态counterfactual。该任务验证持续控制/切换的rollout表现，不声称新增固定状态因果证明。

### Visual Stability

- 全部124帧逐帧覆盖检查及正常24fps播放；每个边界前后至少3帧查看原分辨率人物细节。
- 每块人物保持单一可辨身体、场景连续；无严重ghosting、透明化、肢体分解、人物消失或突变。记录普通模糊/细节形变，不能用“视频可解码”替代画质判断。
- 已发布RGB在本轮续写过程中的hash必须保持；记录每块boundary MAD、块内MAD及过去末5RGB重解码差异。MAD是诊断量，无预先依据时不临时编造通过阈值。
- 报告首次异常RGB/latent区间和持续范围，保留失败原片；不得裁掉失败后宣称全长通过。

### Efficiency

逐块记录forward次数、可见history长度、sampling latency、VAE/搬运/编码耗时、allocated/reserved peak、CPU模型offload与latent内存。persistent hidden KV应为0。

四条V2b都复用前两块，因此本轮测得的是增量耗时；如引用旧first12/second5时间，必须标为跨次累计估算，不冒充同次124帧E2E。Original30整段forward与V2b完整六块180forward不等工作量，不直接宣传公平speedup。

Efficiency本轮只验收测量完整性；加速能力没有预设通过门槛，不能因表格齐全标作模型效率PASS。

### 总体交付判定

四路径均达到124帧动作与结构门槛、Original四路径对照来源匹配且完整、协议和测量可追溯时，Judge可接受“V2b在单场景/seed的124帧多窗口能力”。失败或对照缺失按FAIL/PARTIAL/INVALID注明具体范围。任何结果均不自动升级V3，也不代表10/20秒或跨场景泛化通过。

## 8. Stop Conditions / Out of Scope

出现任一路径明确质量失败、数值异常、OOM、hash/回放不一致、未来信息泄漏、已发布帧回改、需要改协议或预算耗尽，即保存现场停止并提交Judge。充分否定假设后不继续延长剩余路径以消耗预算。

范围外：训练、换seed/场景、gain/anchor/sigma sweep、clean-history消融、KV迁移、strict causal架构修复、AnyFlow、DMD、延长到10/20秒、为性能排名重跑完整V2b。

## 9. Deliverables

- `submission/experiments/EXP-001_v2b_124/`：README、冻结config.yaml、metrics.json、MANIFEST.md、日志、CPU检查、独立续写入口与评估代码。
- 原始checkpoint/输入引用与hash、每块生成latent、已发布RGB hash、逐块性能、动作与视觉评审、完整正负原片。
- 四条Original vs V2b对比及元数据；未达到生成阶段时交付明确的失败/缺失清单。
- 根目录[report.md](report.md)：历史证据、本轮执行、观察、可支持结论、未验证假设、实际预算分别记录。

## 10. Judge 决策与验收栏

| 项目 | 当前记录 |
| --- | --- |
| Decision | continue：Worker执行中；采用v2分时GPU上限，Judge监督交付 |
| Acceptance | pending |
| 执行合规 | 待报告 |
| Action / Visual / Efficiency | NOT_TESTED / NOT_TESTED / NOT_TESTED（本轮） |
| 验收证据与问题 | 待Worker提交；不得用旧56帧结果填写本轮PASS |

本实验若成功，会提供下一步长时/KV研究所需的多窗口正控；若失败，会确定V2b局部正控的长度边界并阻止过早推进少步生成。已有证据只到第二块，尚未回答本问题。复用6份endpoint并在90帧设门槛，是本轮降低成本的方式。

## 11. Worker交接与Judge监督

1. Worker执行期间维护report与机器可读metrics；确认当前任务版本为2，保留v1运行的实际配置/hash。质量失败或阻塞时立即提交，不等预算耗尽。
2. 完成时更新根report的Worker Status与最终manifest/metrics/视频/日志路径，将源代码和配置快照冻结；明确实际预算、阶段门槛、未完成项。通知Judge同一工作区的报告已就绪。
3. Judge读取完整报告并核对任务版本、输入/源码/checkpoint、每块原始指标、所有相关视频序列及原分辨率细节、停止条件与资源。独立区分执行合规、Action、Visual、Efficiency，不因completed自动接受模型能力。
4. 证据不完整则needs_revision，保留EXP-001并递增版本，只补必要证据/修复；不能借返工重跑失败模型或扩大预算。若本身是有效负结果，可接受任务执行并将模型能力记FAIL，停止延长。
5. 验收评价先写本任务书，随后将完整任务书、完整Worker报告、结论及证据归档，再更新progress；全部完成后才发布EXP-002或明确停止。正在运行/未验收时不覆盖当前任务报告。
6. 用户已授权Judge在充分验收后直接更新下一任务书，无需重复请求一般性交接确认。任何新任务仍须明确问题、预算、正负结果如何改变决策，并遵守本次GPU分时约束；不可自动扩大训练或长期队列。

<!-- END EXP-001 PLAN_V2 -->

<!-- BEGIN EXP-001 INTERIM_REPORT_BEFORE_V3 -->
# 当前任务执行报告

> 初始化日期：2026-10-10（Asia/Hong_Kong）。由 Judge 创建报告入口；实际执行记录由 Exp Worker 维护。
> 当前绑定 EXP-001 / plan_version 1。Worker 已开始 S0；以下记录随本轮执行更新。

## 1. 任务与状态

| 字段 | 当前值 |
| --- | --- |
| Task ID | EXP-001 |
| Plan Version | 1 |
| Worker Status | running |
| Judge Acceptance | pending |
| Research Track / Parent Version | Mainline / V2b C12_then5_N_30step_selfhistory |
| 授权入口 | [next_plan.md](next_plan.md) |
| 规则与正式进展 | [guideline.md](guideline.md)、[progress.md](progress.md) |
| 历史与接管记录 | [archive.md](archive.md) |

当前执行范围以 next_plan.md 的分阶段124帧任务与硬预算为准；Worker 正在进行输入/源码校验和隔离入口实现；尚未启动模型实验。

## 2. 历史已知证据

待 Worker 按本轮任务引用已有证据，标明历史来源和验收范围。历史结果不能记作本轮新发现。

## 3. 本轮实际执行

S0 已核对六份生成endpoint与冻结原始文件的 SHA-256；均与历史 manifest 一致。已建立 `submission/experiments/EXP-001_v2b_124/`，冻结配置和来源清单；原始大张量输出放在 `H3-World/outputs/EXP-001_v2b_124/`。S0 旧区间8项CPU检查和多窗口4项CPU检查通过；13份H3 transformer shard哈希已保存。用户更新GPU共享上限为09:00前最多6卡、之后最多4卡，其他预算不变。

已启动第三块[17,22)的AA/DD/AD/DA四条路径，分别使用空闲GPU2/4/5/7。每条只运行一个新chunk，下一块要先核对本块动作、人物、边界和预算。加载前输出均写入独立日志；模型回放与采样结果见实验目录。训练更新0。

执行项记录要求：

- 实际命令、起止时间、退出状态、环境与依赖版本。
- Git revision、工作区改动或冻结源码 hash。
- checkpoint、released LoRA、新 adapter 路径及 hash。
- 数据、场景、随机种子、初始噪声与历史来源。
- attention topology、action routing/feedback、anchor、timestep/RoPE。
- sampling steps、sigma schedule、chunk partition、KV prefill/commit/reuse 和解码协议。
- GPU 型号及编号、前向次数、训练更新次数、耗时与峰值显存；预算消耗与剩余量。
- 原始实验目录、配置、日志、指标、视频和 MANIFEST 路径。

## 4. 本轮观察结果

尚无本轮结果。实际填写时保留负结果、异常、失败轮次及协议偏离。

| 能力 | 本轮状态 | 证据与范围 |
| --- | --- | --- |
| Action Fidelity | NOT_TESTED | 未执行 |
| Visual Stability | NOT_TESTED | 未执行；正式验收需要完整视频序列及必要原分辨率细节 |
| Efficiency | NOT_TESTED | 未执行；需说明硬件、采样与计时口径 |

## 5. 可以支持的结论

暂无。执行 completed 与能力 PASS 分开记录；数值诊断改善不能替代动作和画质验收。

## 6. 尚未验证的假设与限制

待填写。明确混杂因素、缺失证据、历史范围及泛化边界。

## 7. 下一步建议

待填写建议及其正负结果对研究决策的影响。建议不构成自动执行授权。

## 8. 提交检查与 Judge 入口

Worker 完成本轮后记录实际预算与交付路径，将 Worker Status 更新为 completed、blocked 或 failed，并等待 Judge 审计。Judge 的正式评价写入 next_plan.md；结束任务的任务书、完整报告和验收结论归入 archive.md。

<!-- END EXP-001 INTERIM_REPORT_BEFORE_V3 -->


# EXP-001 / v3正式结案：2026-10-10


## 完整任务书

来源：submission/experiments/EXP-001_v2b_124/accepted_taskbook.md；SHA-256 `974b78cde561372485e9cdf507b99b5d532c20b55b72fe3104fa2ea7dad993ed`。

# EXP-001：V2b 持续动作124帧与 Original 对比

更新：2026-10-10（Asia/Hong_Kong）。发布者：Judge。

## 1. Task ID / Version / Status

| 字段 | 内容 |
| --- | --- |
| Task ID / Plan Version | EXP-001 / 3 |
| Research Track | Mainline / V2b capability validation |
| Parent Version | V2b `C12_then5_N_30step_selfhistory`；Original H3 + released action LoRA，无新增adapter |
| Worker Status | completed；AA/DD124帧，AD/DA73帧 |
| Judge Acceptance | accepted |
| Decision | accept / close：持续A/D124帧可行性通过，切换/效率限制保留 |

**本文件是唯一当前任务书。** v1、v2和在途报告原文已存入[archive.md](archive.md)。Worker保留每次运行对应配置、源码和hash，后续使用v3；不要追溯改写已执行记录。

原四路径第三块结果为AA +0.347111、DD −0.878582、AD −0.972243、DA −0.175372。DA未过v1/v2预登记的正flow门槛，该事实保留。用户随后明确要求审核重ROI、避免为微小指标钻牛角尖；Judge改为完成持续动作长片对照，切换能力作为已知限制。此修订不把原四路径联合结果改成PASS。

## 2. Research Question / Hypothesis / Baseline

**研究问题：保持既有V2b协议，持续A和持续D能否在自身历史下生成124帧、维持可用人物与场景结构，并形成与Original有解释价值的完整对比？**

假设：当前持续动作的第三块可用结果能够延续；若明显结构崩坏或持续控制失效，则确定该路线的可用长度边界。

- Parent baseline：已有AA/DD73帧；前两块为历史复用，第三块为本轮结果。
- Original baseline：Original H3-World +同released action LoRA，持续A/D、124帧、整段30步。优先复用[输入匹配审计](submission/experiments/EXP-001_v2b_124/original_source_audit.json)指向的已有视频；仅必要且源输入不能匹配时补生成，最多2条。
- 对比是协议能力比较。Original原生联合audio/video去噪，V2b固定audio noise与native时间；attention、历史与发布方式不同，明确混杂，不宣传严格单变量归因或公平speedup。
- 124帧@24fps≈5.17秒。本轮不扩到10/20秒，不继续AD/DA，不追加专项动作诊断、训练或消融。

## 3. Controlled Variables / Changed Variables / Inputs

保持Original权重、released action LoRA、停车场初图832×480、seed13、full37 noise/layout、Single I0、native time、全局RoPE、30 steps/chunk、shift2.22、Same-σ history、旧`h3_fp32`实际精度/backend、T2全部可见历史重算、无persistent hidden KV、无新增adapter。保持历史动作序列，不使用GT/teacher重置，不截断历史，不改变anchor或用平滑掩盖断裂。

唯一模型实验变化为AA/DD从73继续到124帧：latent[22,27)、[27,32)、[32,37)，累计RGB90、107、124。完整partition为[12,5,5,5,5,5]。

输入入口：

- [旧协议](H3-World/outputs/2026-10-09-22/chunk_partition_cb/protocol.json)与[6份历史endpoint](submission/experiments/11_causal_12_then5_selfhistory/manifest.json)。
- [EXP-001源码与输入manifest](submission/experiments/EXP-001_v2b_124/source_manifest.json)、[当前配置](submission/experiments/EXP-001_v2b_124/config.yaml)。
- 新历史为各自first12+second5+third5，使用`H3-World/outputs/EXP-001_v2b_124/{AA,DD}/state.json`、`chunk_17_22.pt`和`published.npy`续接。逐项检查hash，不重跑已完成块。

## 4. Execution Plan

### S0：最小必要调度与交接修订

保全v1运行源码/config、四条第三块结果及原停止记录；新增版本快照/manifest，不覆盖证据。后续配置标明plan_version=3、active_paths=[AA,DD]；门槛状态文件不得继续要求四路径都PASS，也不能伪造四路径PASS以绕过旧runner。

修复已发现的资源执行缺口：GPU上限按绝对09:00截止判断；总GPU-hours计入活跃进程；wall deadline从最早GPU运行计时；提前释放多余GPU。只做必要CPU检查（版本/路径选择、截止和预算边界、恢复不重置预算），不重跑已经通过的模型机制实验或额外GPU回放。

### S1：AA/DD第四块至90帧

各续写一个chunk，共60次forward。检查新增帧、人物原尺寸与边界；若出现明确严重崩坏或持续动作语义明显错误，停止受影响路径。普通模糊、细节形变或单项flow小幅反号记录为限制，不机械停止另一条有价值的对照。

### S2：可用路径继续至124帧

AA/DD各续两块，最多120次forward。每块保存latent和已发布RGB，继续只追加新17帧；可见历史始终为[0:start]，每步重算。若一个路径严重失败，保留失败长度；另一条仍可完成，以避免损失独立能力证据。无法通过两条时整体报告PARTIAL/FAIL，不冒称两条成功。

### S3：完整对照、报告与收敛

优先使用已审计Original A/D原片。生成两条左右并排124帧对比及一个简洁A/D总览；左Original、右V2b，标明动作、步数、history/anchor、协议差异及计时范围。失败路径展示已完成的真实范围并明确长度，不补帧、不裁掉失败冒充成功。

AD/DA已有73帧和DA疑点纳入同一报告的限制栏，不扩实验、不要求先解释全部失败机制才交付本轮对照。Worker完成后通知Judge，等待验收；不得自动进入下个GPU实验。

当前runner入口已存在，Worker需按v3调整配置和active-path gate后执行：

```bash
.venvs/h3world/bin/python submission/experiments/EXP-001_v2b_124/run_rollout.py --config submission/experiments/EXP-001_v2b_124/config.yaml --stage gate90 --path AA --gpu <idle_gpu>
.venvs/h3world/bin/python submission/experiments/EXP-001_v2b_124/run_rollout.py --config submission/experiments/EXP-001_v2b_124/config.yaml --stage extend124 --path AA --gpu <idle_gpu>
```

DD同理；旧runner每次只生成一块，extend124按checkpoint续接调用两次。若改用版本化文件名，在report记录实际命令。不要并发运行同一路径。

## 5. Resource Budget

| 项目 | 当前授权上限 |
| --- | --- |
| 已用 | 四条第三块120sampling +12diagnostic=132次；约0.3492 GPU-hours、12次VAE decode |
| 剩余V2b | AA/DD各3块×30，共≤180次sampling；不重跑旧块 |
| Original | 优先复用；必要时最多2条×30=60次 |
| 新GPU诊断/训练 | 0；已有12次诊断预算已用完；0 optimizer updates |
| 本修订预期总调用 | ≤372次（132+180+60），仍受原612次任务硬上限约束；额外余量不自动授权新实验 |
| VAE | 后续V2b≤6次、必要Original≤2次；总≤20次，0新RGB re-encode |
| GPU并发 | 项目合计2026-10-10 09:00 HKT前≤8张，截止后≤3张；本轮两条持续路径正常只需2张 |
| 显存 | 每卡allocated≤44GiB；实时确认空闲，保留既有offload与精度 |
| 时间 | 整个EXP-001累计≤8 GPU-hours且从首个GPU进程起≤8小时；包括旧运行、加载、VAE、失败和活跃进程，不能重置 |

08:30后不启动使占卡超过3的新chunk；09:00前安全保存并释放多余GPU，暂停但仍占显存不算释放。调度器/守护只操作本任务登记且PID与启动标识匹配的进程，不影响其他用户。09:00后不自动恢复夜间8卡，等待用户新授权。当前硬件有8卡，不要求占满。

## 6. Evaluation / Acceptance

### Action Fidelity

固定原flow计算方法，逐块及全片报告；看完整序列中的动作语义及与Original的差别。flow只是辅助证据，轻微反号或幅度变化不单独决定失败。明显持续沿错误方向、动作长期无区分或与指令冲突才构成控制失败；不确定时标PARTIAL并说明，不追加大量诊断来追求完美结论。

本轮正式范围为持续A/D。切换路径已有疑点，不能宣称切换能力、所有四路径或泛化都通过。不同自身history的两视频差分不描述为同状态counterfactual。

### Visual Stability

完整视频帧序列和必要原分辨率检查；关注单体人物、可辨肢体、场景连续、边界和累计漂移。严重ghosting、透明分解、人物消失/换场属于关键失败；普通模糊或短暂细节形变记录程度，不要求像素完美。已发布RGB hash必须保持不变，报告过去末5帧重解码差异。

完整解码不等于画质通过；静态逐帧检查与实际播放分开记录。证据足够后结束检查，不重复制作大量画廊或重复跑测试。

### Efficiency

记录每块sampling、forward、VAE/搬运/编码、GPU峰值和CPU offload。CPU hidden KV=0。复用前73帧后的时间为增量成本；完整180次forward与Original30次不同，不宣称公平加速。测量完整性与模型加速能力分开，后者本轮没有PASS目标。

### Judge验收尺度

- 若完整A/D124帧具有可辨动作差别和可用人物/场景结构、对照可追溯：可接受**单场景/seed、持续动作的V2b多窗口结果**，同时保留切换/效率限制。
- 若只一条可用或动作语义仍不明确：接受有效任务执行，模型能力记PARTIAL，明确可用范围。
- 若严重退化：接受有证据的负结果、模型能力FAIL；不因没有成功长片让Worker无期限返工。
- 协议/输入/源码错误则INVALID或needs_revision；这种基本正确性不能用ROI理由放宽。

## 7. Stop Conditions / Out of Scope

协议错误、未来信息泄漏、原始证据被覆盖、历史帧回改、NaN/OOM、预算超限或明确严重画质/控制失败时停止相关任务并报告；一条失败不自动取消另一条仍有信息价值的持续动作对照。结果已足以改变下一步决策时结束。

不做训练、额外seed/场景、超参数搜索、AD/DA继续生成、专项DA诊断、KV迁移、AnyFlow、DMD、10/20秒视频或性能重复排名。

## 8. Deliverables / Judge交接

交付原实验目录的版本化代码/config、metrics、MANIFEST、日志、AA/DD完整或失败视频、Original并排对照、每块latent/发布RGB hash与[report.md](report.md)。保留四路径第三块所有原始结果与旧门槛结论。

Judge收到completed/blocked/failed报告后，核对协议与少量关键原始证据、完整相关帧序列和主要限制；足够判断即收敛。验收写入任务书，完整任务书/报告/评价入archive，progress更新后才发布下一任务。用户已授权该交接，不再重复询问一般性许可。

当前Judge验收pending。后续优先根据完整对照决定“值得延长”还是“应改模型/历史协议”；不预排一串局部消融，也不为了用满GPU增加新分支。


### 用户最新验收解释：可行性阶段

2026-10-10用户进一步明确：模型未经大量训练，效果差一些可以接受；方向实在不行则停止。当前v3的运行协议、路径与预算保持原授权；验收以是否形成有研究价值的持续动作多窗口证据为准，允许画质、自然度与边界方面的原型限制。有效负结果同样可以结束任务，不要求追加实验或训练去挽救每条路线。V3后续也以核心机制和可用能力可行性为目标，不能用成熟产品画质要求无期限卡住推进。详见guideline第17节。


## 9. Judge最终验收（2026-10-10）

accepted。接受有效执行及单停车场seed13持续A/D124帧的动作/视觉基本可行性；动作切换PARTIAL，原四路径DA门槛FAIL保留。312次前向、18次VAE、1.029126 GPU-hours，零训练。AA边界跳变、DD构图漂移和节奏/细节限制保留；无persistent KV、无公平speedup，V3尚未完成。详见[正式评价](submission/experiments/EXP-001_v2b_124/judge/FINAL_REVIEW.md)。本任务关闭，不追加V2b消融或长片。


## 完整Worker报告

来源：submission/experiments/EXP-001_v2b_124/worker_report.md；SHA-256 `aa1b8fb23a98cddb4ad5a5b19c9e0944ec83c24bca3312a57c139e9800039870`。

# EXP-001 Worker 执行报告：V2b 持续 A/D 124 帧

报告日期：2026-10-10，Asia/Hong_Kong。Task ID EXP-001，当前 plan_version 3。**Worker Status: completed；Judge Acceptance: pending。** 这表示限定实验和交付物已完成，不表示 V2b 升级为正式高效因果模型。研究负责人的正式验收仍以 [next_plan.md](next_plan.md) 为准。

## 结论与验收范围

在停车场单场景、seed 13、Original H3 + released action LoRA、零新训练的 Same-σ 自生成历史协议下，持续 A 与持续 D 均从已有 56 帧续写至完整 **124 RGB 帧 / 24 fps / 5.17 s**。全部新增块中，A 的水平光流代理保持正向，D 保持负向；人物和场景一直可辨，没有早期方案那种严重人物分解或瞬间换场。对照 Original 的两条真实并排 MP4 已生成并通过完整解码。由此可以支持“此协议在这一个场景/seed 下具备持续 A/D 多窗口生成的可行性”，不能扩展成动作切换、长期泛化或高效严格因果化已经通过。

原 v1/v2 四路径门槛失败仍是事实：DA 在第三块 RGB56–72 输入 A 的水平 flow 为 −0.175，前后半段均负，人物更多向镜头运动；AD/DA 按原停止规则止于 73 帧。v3 按用户 ROI 指示仅继续 AA/DD，既未重标原四路径 PASS，也未追加训练、DA 专项诊断、AnyFlow 或 DMD。

## 历史证据与本轮实际执行

历史 V2b 正控仅到 56 帧第二块，见 [历史实验](submission/experiments/11_causal_12_then5_selfhistory/README.md)。本任务复用其 first12 和 second5 生成 endpoint。v1 在四条路径各生成 [17,22) 第三块，得到 AA/DD/AD/DA 四条 73 帧；v2 因 DA 触发预登记 flow 符号门槛而停止。两版任务书与在途报告已由 Judge 归档。v3 没有改变模型协议，仅将后续能力问题缩到持续 AA/DD：各续 [22,27)、[27,32)、[32,37)，累计 RGB 90、107、124。

模型协议：Original H3 底座 + released action LoRA，Single I0、原生 action/time 与全局 RoPE、原始 directed action routing；首块 12 latent，后续每块 5 latent；每块 30-step native schedule、flow shift 2.22。每个 sigma 将自己已生成的完整可见历史临时加到同一噪声水平，与当前块联合重算，只更新当前 chunk。没有未来动作/视频、GT reset、额外 anchor、新 adapter、persistent hidden KV 或输出平滑。保存的历史 latent 与已发布 RGB 均不回改。该 T2 局部联合重算 **不是** strict chunk-causal masked attention + persistent-KV V3。

运行前核对六份 endpoint、输入、released LoRA 和 13 份 base transformer shard SHA。v1 第三块 first12 与 second5 旧 API 回放 relative RMS 均为 0。v3 对 active path、绝对 09:00 HKT 资源截止、活跃进程 GPU-hours 与恢复不重置预算的 CPU 检查为 3 passed。v1 原代码/配置保留；v3 使用独立 [run_rollout_v3.py](submission/experiments/EXP-001_v2b_124/run_rollout_v3.py)、[config_v3.yaml](submission/experiments/EXP-001_v2b_124/config_v3.yaml)和[来源 manifest](submission/experiments/EXP-001_v2b_124/source_manifest_v3.json)。v1 旧 6/4 卡字段只是运行快照，当前 v3 授权为 09:00 前最多 8 卡、以后最多 3 卡，本轮 v3 实际仅使用 GPU 0/1 两张。

## 结果：动作、画面与边界

下表为每段新 17 RGB 帧固定中心裁剪 Farneback 水平光流均值，单位 px/相邻帧；正负是运动代理而非动作正确率。

| 新增 RGB | AA：持续 A | DD：持续 D | AA/DD 边界灰度 MAD |
| --- | ---: | ---: | ---: |
| 56–72 | +0.347 | −0.879 | 4.51 / 9.41 |
| 73–89 | +1.477 | −0.870 | 16.25 / 7.82 |
| 90–106 | +0.685 | −0.611 | 3.22 / 2.30 |
| 107–123 | +0.987 | −1.260 | 3.07 / 2.41 |

全片 124 帧 Original A/D flow 为 +1.077/−1.602，V2b 为 +1.035/−1.050。完整帧序列、所有新增块接触图与必要原尺寸边界检查中，人物保持单体可辨且停车场结构延续。AA 在 RGB72→73 出现明显转身/位置跳变，不能宣称块边界完美；DD 末段人物贴近左下边缘，有构图漂移。静态逐帧查看由 Worker 完成；Judge 已独立审阅相关完整序列和原尺寸细节。当前视觉判断是可用于可行性展示，达不到与 Original 等质或长期稳定的结论。AD/DA 第三块视频仍保留为动作切换限制的证据。

完整性与可复现性：[CPU 视频审计](submission/experiments/EXP-001_v2b_124/artifacts/video_integrity_audit_v3.json)解码所有 124/73 帧 rollout、每块 17 帧和 2 帧边界片，验证 H.264、24 fps、PTS、帧数、endpoint 文件及逐块已发布 RGB prefix SHA；全部通过。旧 73 帧及每个续写前缀均没有被改写。Original A/D 输入审计核对 initial video/audio noise、anchor、prompt 和全局位置等可比条件；Original 为全片双向、原生联合音视频去噪，V2b 为固定 audio noise 与自身历史协议，因此对照是能力比较，不能把变化归于单一因素。

## 效率与实际预算

Original 两条 124 帧各为全片 30 denoiser forward，已保存运行的完整 wall including shared setup 分别为 A **478.4 s**、D **454.6 s**。V2b 六块需 180 sampling forward/路径并每步重算历史；本轮测得从已保存 73 帧继续到 124 帧的三次独立进程增量 wall 合计 A **1216.8 s**、D **1194.9 s**，不包括旧 73 帧生成时间。两种时间范围不同，不能计算公平速度比，也没有 persistent-KV 加速证据。

EXP-001 全任务实际消耗 **300 sampling + 12 diagnostic = 312 denoiser forward**、18 次 VAE decode、0 optimizer update、3704.85 GPU-seconds（1.0291 GPU-hours）。最高单卡 torch.cuda.max_memory_allocated 为 26392.86 MiB，CPU hidden KV 为 0。首个 GPU 进程 03:10:43、最后退出 03:47:40 HKT；v3 两进程最多同时占 2 卡，均在 09:00 前退出。预算和逐路径 latency/显存原始值见[实验指标](submission/experiments/EXP-001_v2b_124/metrics.json)与[账本快照](submission/experiments/EXP-001_v2b_124/artifacts/raw_json/budget.json)。CPU 模型 offload 按旧协议保留，但未单独精确量化其主机内存峰值，不以此声称内存优势。

## 视频与交付物

- [Original A vs V2b AA，124 帧](submission/experiments/EXP-001_v2b_124/artifacts/videos/Original_vs_V2b_A_124.mp4)
- [Original D vs V2b DD，124 帧](submission/experiments/EXP-001_v2b_124/artifacts/videos/Original_vs_V2b_D_124.mp4)
- [A/D 依次播放的并排总览，248 帧](submission/experiments/EXP-001_v2b_124/artifacts/videos/Original_vs_V2b_AD_overview_248.mp4)
- [完整 AA 原片](submission/experiments/EXP-001_v2b_124/artifacts/videos/AA_rollout_124.mp4)、[完整 DD 原片](submission/experiments/EXP-001_v2b_124/artifacts/videos/DD_rollout_124.mp4)、[AD 73 帧](submission/experiments/EXP-001_v2b_124/artifacts/videos/AD_rollout_73.mp4)、[DA 73 帧](submission/experiments/EXP-001_v2b_124/artifacts/videos/DA_rollout_73.mp4)
- [视频和计时来源 manifest](submission/experiments/EXP-001_v2b_124/artifacts/comparison_manifest.json)、[实验 README](submission/experiments/EXP-001_v2b_124/README.md)、[权重/代码/输入 MANIFEST](submission/experiments/EXP-001_v2b_124/MANIFEST.md)、[旧 v2 失败指标快照](submission/experiments/EXP-001_v2b_124/metrics_v2_interim.json)

以上两条并排片左 Original、右 V2b，标明同首图/seed/noise、步数和不同计时口径；三条新并排片均通过全片解码、SHA、FPS 与 PTS 检查。原始 large latent、未压缩 RGB 和模型权重保留原位；Git 交付目录复制了主要真实 MP4 与关键 JSON，没有覆盖原始文件。

## 可支持的判断、限制与下一步

本任务证明 V2b Same-σ 局部联合重算可在单场景/seed 保持持续 A/D 响应及可辨人物到 124 帧，同时暴露边界和构图漂移。它没有证明动作切换可靠，更没有证明 strict causal + KV 的高效 V3 已实现。性能上还慢于 Original 的量级，少步 AnyFlow 与 DMD 目前仍是旁支探索，不能把本片归因于它们。后续方向应由 Judge 根据这份完整对照决定；Worker 不自行启动新训练或诊断。

**提交验收：** v3 指定两条路径、完整原片/对比片、指标、源码快照和预算均已交付；Judge Acceptance 仍为 pending。此前 v2 四路径 stop/DA FAIL 独立保留，最终主线版本及 Git 提交由 Judge 维护。


## Judge正式验收

来源：submission/experiments/EXP-001_v2b_124/judge/FINAL_REVIEW.md；SHA-256 `3b9437737254a1bde4ed4e69e3b440973449619d527457d00c9735b1356477ee`。

# EXP-001 / v3：Judge正式验收

日期：2026-10-10，Asia/Hong_Kong。**Judge Acceptance: accepted；Decision: accept / close。**

## 结论与范围

接受本任务执行，并接受**单停车场、seed13、持续A/D、124 RGB帧（24fps，5.17秒）的V2b多窗口可行性**。两条路径人物保持单体可辨、停车场结构连续，呈现不同方向的持续运动；辅助flow与这一观察一致。当前是未经额外大量训练的原型验证，不要求成熟产品画质。

| 维度 | 判定 | 边界 |
| --- | --- | --- |
| 执行与证据 | PASS | v1四路第三块、v3仅AA/DD三块续写，零训练，输入/源码/历史可追溯 |
| 持续A/D动作可行性 | PASS，限定本场景/seed/124帧 | 非动作准确率、非同状态反事实、非跨场景泛化 |
| 多窗口视觉基本结构 | PASS，可行性范围 | AA RGB72→73明显姿态跳变；动作节奏不均/细节软化；DD末段靠近画面下边缘 |
| 动作切换 | PARTIAL / 未通过联合验收 | AD/DA停于73帧；DA第三块A的flow −0.175，原四路径符号门槛FAIL保留 |
| 效率测量 | 已记录；公平speedup NOT_TESTED | 每步重算全部可见历史，persistent hidden KV=0 |
| V3 | NOT_COMPLETED | 本配置缺少strict chunk-causal历史表示与真实persistent KV复用 |

不为普通画质缺陷追加实验；也不把本轮成功范围扩展到切换、10/20秒、泛化或V3。

## 独立审核与可复现性

Judge已读取最终Worker报告、任务v3与用户ROI/可行性补充、runner/config/来源manifest、原始逐块JSON、预算和对比来源。四路前73帧的完整静态帧序列及第三块全部人物原尺寸裁剪已审阅；随后AA/DD每段新增17帧静态序列全部查看，另检查关键边界和末帧原尺寸画面，覆盖两条完整124帧。没有声称正常速度实时播放。Original匹配参考完整解码，抽看12帧/路径；对比片查看排版/标签并完整解码。

独立核对：v1/v3冻结worker文件hash均匹配；两条124帧、24fps、单调PTS；每个已发布RGB前缀、生成endpoint hash均匹配；三条并排片的hash、124/124/248帧、1664×570与PTS均通过。见[最终技术检查](final_technical_checks.json)、[对比片检查](comparison_checks.json)及[逐阶段审核](REVIEW.md)。保留旧源码/config/旧四路径门槛，未追溯改写旧FAIL。

Original为2026-10-01-21的已存A/D参考，输入审计确认初始video/audio noise、anchor、prompt、位置等相符。Original联合音视频整段去噪，V2b固定audio条件且逐块重算自身历史，属于协议能力比较，不能作单变量归因。

## 成本与限制

全任务300 sampling +12 diagnostic =312 denoiser forwards，18 VAE decodes，0训练；3704.853 GPU-seconds（1.029126 GPU-hours）；allocated峰值26392.86MiB。10个登记GPU运行均已结束，v3只用两卡，未接近09:00截止。CPU截止/预算逻辑有检查，不能宣称实际经历跨09:00切换。历史资源字段与重复配置字段按各自版本解释，实际执行满足最新8/3授权和v3更紧预算。

V2b完整六块协议每条180 sampling forwards；本次前两块复用。73→124三块增量wall为AA1216.825s/DD1194.907s，Original全片E2E为478.447s/454.570s，口径不同，无公平加速比。CPU offload主机内存峰值未单独量化，不据此宣称内存优势。全片flow为V2b A +1.034788/D −1.049849，Original A +1.076725/D −1.601923，仅作辅助。

历史CPU测试日志有8项/当前收集4项的数量差异，Worker已披露；不以此阻止已有代码/输出证据充分的任务。冻结v3测试首项依赖启动时state=22，仅是开跑前检查；最终状态=37时不应把它作为可重复运行的通用测试命令。文档改为两项纯预算检查及最终产物审计，冻结源码不变。

## 研究决定

V2b完成所需的持续动作多窗口对比，停止在本方向追求边界或小指标收益。下一项采用native Single I0/12→5的既有current-prefix严格缓存候选，以有限短视频判断能否同时保留基本动作/结构与真实KV。候选失败时停止无训练拓扑微调，只有存在明确可修复信号才考虑有限适配，否则换路线。下一任务须单独发布，不由本验收自动授权GPU。



## EXP-001结案补注：Worker收尾测试版本核对

Worker结案前保留开跑时测试/manifest并增加完成态测试。此前归档评价的测试说明由下列完整修订版替代，能力判定与运行结果不变。SHA-256 `9db335be85b3faab1ed11ba6bae39b8a6792b07e870fd4bfbf135f43af05d620`。

# EXP-001 / v3：Judge正式验收

日期：2026-10-10，Asia/Hong_Kong。**Judge Acceptance: accepted；Decision: accept / close。**

## 结论与范围

接受本任务执行，并接受**单停车场、seed13、持续A/D、124 RGB帧（24fps，5.17秒）的V2b多窗口可行性**。两条路径人物保持单体可辨、停车场结构连续，呈现不同方向的持续运动；辅助flow与这一观察一致。当前是未经额外大量训练的原型验证，不要求成熟产品画质。

| 维度 | 判定 | 边界 |
| --- | --- | --- |
| 执行与证据 | PASS | v1四路第三块、v3仅AA/DD三块续写，零训练，输入/源码/历史可追溯 |
| 持续A/D动作可行性 | PASS，限定本场景/seed/124帧 | 非动作准确率、非同状态反事实、非跨场景泛化 |
| 多窗口视觉基本结构 | PASS，可行性范围 | AA RGB72→73明显姿态跳变；动作节奏不均/细节软化；DD末段靠近画面下边缘 |
| 动作切换 | PARTIAL / 未通过联合验收 | AD/DA停于73帧；DA第三块A的flow −0.175，原四路径符号门槛FAIL保留 |
| 效率测量 | 已记录；公平speedup NOT_TESTED | 每步重算全部可见历史，persistent hidden KV=0 |
| V3 | NOT_COMPLETED | 本配置缺少strict chunk-causal历史表示与真实persistent KV复用 |

不为普通画质缺陷追加实验；也不把本轮成功范围扩展到切换、10/20秒、泛化或V3。

## 独立审核与可复现性

Judge已读取最终Worker报告、任务v3与用户ROI/可行性补充、runner/config/来源manifest、原始逐块JSON、预算和对比来源。四路前73帧的完整静态帧序列及第三块全部人物原尺寸裁剪已审阅；随后AA/DD每段新增17帧静态序列全部查看，另检查关键边界和末帧原尺寸画面，覆盖两条完整124帧。没有声称正常速度实时播放。Original匹配参考完整解码，抽看12帧/路径；对比片查看排版/标签并完整解码。

独立核对：v1/v3冻结worker文件hash均匹配；两条124帧、24fps、单调PTS；每个已发布RGB前缀、生成endpoint hash均匹配；三条并排片的hash、124/124/248帧、1664×570与PTS均通过。见[最终技术检查](final_technical_checks.json)、[对比片检查](comparison_checks.json)及[逐阶段审核](REVIEW.md)。保留旧源码/config/旧四路径门槛，未追溯改写旧FAIL。

Original为2026-10-01-21的已存A/D参考，输入审计确认初始video/audio noise、anchor、prompt、位置等相符。Original联合音视频整段去噪，V2b固定audio条件且逐块重算自身历史，属于协议能力比较，不能作单变量归因。

## 成本与限制

全任务300 sampling +12 diagnostic =312 denoiser forwards，18 VAE decodes，0训练；3704.853 GPU-seconds（1.029126 GPU-hours）；allocated峰值26392.86MiB。10个登记GPU运行均已结束，v3只用两卡，未接近09:00截止。CPU截止/预算逻辑有检查，不能宣称实际经历跨09:00切换。历史资源字段与重复配置字段按各自版本解释，实际执行满足最新8/3授权和v3更紧预算。

V2b完整六块协议每条180 sampling forwards；本次前两块复用。73→124三块增量wall为AA1216.825s/DD1194.907s，Original全片E2E为478.447s/454.570s，口径不同，无公平加速比。CPU offload主机内存峰值未单独量化，不据此宣称内存优势。全片flow为V2b A +1.034788/D −1.049849，Original A +1.076725/D −1.601923，仅作辅助。

历史CPU测试日志有8项/当前收集4项的数量差异，Worker已披露；不以此阻止已有代码/输出证据充分的任务。开跑时v3测试首项依赖state=22；Worker在结案前将原文件及manifest保留为test_v3_contract_at_run.py/source_manifest_v3_at_run.json，再将当前test_v3_contract.py适配到完成态37，manifest记录了两份hash。Judge已检查该改动仅覆盖完成态断言，并运行当前三项CPU检查，3 passed；生成runner/config未变。

## 研究决定

V2b完成所需的持续动作多窗口对比，停止在本方向追求边界或小指标收益。下一项采用native Single I0/12→5的既有current-prefix严格缓存候选，以有限短视频判断能否同时保留基本动作/结构与真实KV。候选失败时停止无训练拓扑微调，只有存在明确可修复信号才考虑有限适配，否则换路线。下一任务须单独发布，不由本验收自动授权GPU。


## EXP-001 GitHub交付收据与下一任务

正式提交`679258c76e0ceaa2ec54912cc62f1df0e5304e05`已正常推送origin/main；ls-remote核对远端与本地一致。推送前fetch无分叉，不需合并；未force push。160文件，保留原始日志空白，代码/文档diff检查及入口链接通过，无权重/大型tensor。

EXP-002/v1随后发布：共同first12_A历史分支A/D、native current-prefix严格缓存短视频，≤128次前向、零训练。属于独立新任务，不改写EXP-001结果。


EXP-002/v1任务快照已提交并推送`021f1f064d3fb5d7690707475e00649e8909268b`；ls-remote与本地SHA一致，工作区在交接时干净。已通知既有Worker执行，未创建新Worker或新增额外实验。


# EXP-002 / v1 完整归档：73帧严格缓存候选 accepted


## 执行任务书

# EXP-002：Native条件下的严格缓存视频候选

发布：2026-10-10（Asia/Hong_Kong），Judge。EXP-001/v3已验收并推送GitHub：679258c76e0ceaa2ec54912cc62f1df0e5304e05。**本文件是当前唯一GPU任务授权。**

## 1. Task ID / Version / Status

- Task ID / Plan Version：EXP-002 / 1。
- Research Track：Mainline / V3 candidate；不是正式V3版本。
- Parent Version：Original H3 + released action LoRA；继承V2b的Single I0/native条件与12→5分块，复用既有current-prefix候选工程。
- Worker Status：running（AA/AD第二块56帧已完成，续第三块）；Judge Acceptance：pending。

## 2. Research Question / Hypothesis / Baseline

唯一问题：严格chunk因果、private current prefix反馈与真实persistent video KV，在native条件下能否生成具有可用人物结构和持续A/D响应的自身历史续写？

假设：保留原生初图/时间/位置和首窗长度后，既有current-prefix候选可能形成可用的无训练缓存起点。该假设未证实；若视频失败，停止无训练拓扑微调；依据失败是否有可修复信号决定有限适配或直接换路线，不默认追加训练。

基线复用V2b AA与AD的56/73帧已有视频、同first12_A历史与噪声。AA有EXP-001124帧证据；AD仅有73帧，明确其范围，不能把DD当成AD匹配基线。Original持续A片可作AA参考，Original A→D匹配片缺失如实标记，本轮不新生成基线。

历史证据：`submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate/`已经实现本候选连接。旧RGB-dual/5latent的18固定状态诊断只恢复首块Original身份，后续动作delta cosine未恢复，且没有候选生成视频。本轮新内容是native Single I0/12→5条件下的实际cached rollout；不得将旧方案当作首次提出或正结果，不重复18点诊断。

## 3. Controlled / Changed Variables / Inputs

保持Original权重与released action LoRA、停车场832×480、seed13、full37初始噪声、Single I0、native时间、全局RoPE、固定audio条件、30步/chunk、shift2.22、既有实际精度/backend与offload。

联合变化：Same-σ全历史双向重算改为严格chunk因果、sigma0/native条件clean-commit的历史hidden KV；current action own routing/feedback；公共非动作prefix只读取当前chunk video；每个chunk拥有独立prefix，不追溯更新祖先KV。不能声称这是单因素消融。

输入：EXP-001冻结manifest、V2b旧first12_A endpoint、匹配full37 noise/conditioning；旧`current_prefix.py`与测试；`H3-World/code/causal/h3_cached.py`。首12复用仅在空cache时候选图/条件与旧局部Original一致后成立；第二块以后必须本候选自己的输出，不得接V2b后续块或teacher重置。

## 4. Execution Plan

### S0：最小实现与正确性

在独立`submission/experiments/EXP-002_native_cached/`实现cached interval入口。全局(start,stop)与cache chunk index分离，支持[0,12)、[12,17)、[17,22)；不能沿用`index*chunk_frames`作为不等长分块的全局位置。Single I0、`fixed_prefix_timesteps=False`，未来video/action在进入模型前物理移除。

复用既有current-prefix实现。公共prefix不读历史video KV；current action只反馈自己控制的current video；current video读取历史video KV与自己的action。历史缓存只读，每个祖先用其生成时独立prefix/native条件提交一次。commit后仅追加新chunk，禁止每步重算历史Transformer冒充cache。

CPU检查覆盖本轮改动的全局位置/action对齐、未来隔离、cache不变、首12空cache身份、每块一次commit与逐祖先重放参考。已有未改动测试优先复用；不开展后端bitwise一致工程。最多4次真实模型回放用于首窗/缓存复用必要核对，记录实际误差和判定依据；明显图或条件不一致必须修复或报告，不能以ROI豁免。

### S1：共同A-history分支AA/AD，各续第二块到56帧

两条都复用同一个first12_A、自身生成首39RGB和相同初始noise。各自按完全相同的sigma0/native条件prefill，核对两条初始历史KV摘要一致；唯一分支变化是当前[12,17)动作A或D（AA/AD），30步采样只读同语义cache。这样直接检验固定历史下的当前动作响应，避免两个不同历史的差异冒充控制效果。冻结已有首39RGB，只追加新17帧。保存cache摘要/字节数、输入hash、采样/搬运/VAE时间及原片，与V2b相同范围对比。

若严重结构崩坏或明显持续控制失败，停止受影响路径并提交有限负结果；不要追加mask、gain、history-sigma消融。普通边界跳变、模糊或辅助flow轻微反号允许记录限制后继续。

### S2：有信息价值的路径再续第三块到73帧

同一候选协议将自己的第二块sigma0 commit一次，继续[17,22)30步。不得重新接V2b历史、清空cache或重跑首窗。每条最多两个新chunk，总4个；其中一条明确失败不要求另一条也停止。

### S3：报告并等待Judge

提交完整候选片、左V2b右candidate的匹配范围并排片，保留真实边界和失败长度；明确V2b自身历史与候选自身历史后续不同，不描述为同状态反事实。静态帧序列与实际播放检查分开记录。

建议入口（由Worker实现后冻结实际命令）：

```bash
.venvs/h3world/bin/python submission/experiments/EXP-002_native_cached/run_rollout.py --config submission/experiments/EXP-002_native_cached/config.yaml --path AA --gpu <idle_gpu>
```

AD同理（共同A首窗，当前及后续D）；runner应先产出第二块供检查，再显式恢复第三块，不自动排队新实验。最终报告提交后停止GPU扩展。

## 5. Resource Budget

- 0 optimizer updates；两条当前动作分支AA/AD（首窗共同A，其后持续A或D）、一个scene/seed。
- 新sampling≤120（4×30）；first12 prefill与second-chunk commit合计≤4；必要模型回放≤4；完整denoiser前向总≤128，失败/重试也计入。
- 新VAE decode≤4，0 RGB re-encode；复用首39已发布RGB。
- 全任务≤3 GPU-hours且首个GPU进程起≤4h elapsed，计入加载、失败、commit与活跃进程；每卡allocated≤44GiB。
- 项目总GPU：2026-10-10 09:00 HKT前≤8，之后≤3；08:30起不新增超过3卡的占用，09:00前释放多余卡，之后不自动恢复8卡。本任务通常2卡即可。

## 6. Evaluation / Acceptance

分别报告：执行有效性、strict causal/persistent KV实现、动作响应、人物/场景结构、效率测量。辅助flow/cosine只作解释，实际视频与主要结构决定能力范围。

只在同一个候选中确认历史未来隔离、cache真实复用、动作与视觉共同可用，才接受为值得延伸的V3候选。首12身份不算新的能力收益；实现正确不代表视频通过；本轮73帧不覆盖124帧或跨场景。

计时包含prefill/commit、sampling、CPU-KV传输、VAE、编码；报告每块cache tokens/bytes及GPU峰值。匹配硬件/offload与计时范围后讨论观察成本，不将理论KV直接写成实测加速。

执行正确且结果为负也可accepted，模型能力按PASS/PARTIAL/FAIL真实记录。有限片段已有充分证据决定下一步时收敛，不要求补完所有内部机制。当前为可行性验证，接受可辨动作与基本结构伴随模糊、节奏不均及普通边界缺陷；不以高保真或动作自然度作为暗加门槛。

## 7. Stop Conditions / Out of Scope

协议/输入错误、未来泄漏、历史回改、NaN/OOM、预算超限、严重结构崩坏或持续控制失败时停止相应运行；向Judge交付事实。不得自行提高预算。

不做训练、额外场景/seed、124帧扩展、额外切换序列（已授权AA/AD共同首窗分支除外）、超参数搜索、AnyFlow/DMD、重复18点诊断或新增基线生成。

## 8. Deliverables / Next Decision

独立实验目录README/config/源码manifest/metrics/MANIFEST、实际命令日志、视频、cache正确性与资源证据；根report.md标明实际状态并通知Judge。Judge审阅、归档、更新submission并同步GitHub后发布下一任务。

- 若同协议动作/结构可用：下一任务做124帧自身历史和效率验证，再判断V3正式验收。
- 若失败：停止此无训练方向；只有存在明确可修复信号时才设计一次有限适配，否则归档并换更有依据的路线。不默认训练挽救，也不继续零散mask消融。

本任务授权限于上述实现与≤128次前向。完成或停止后提交report，由Judge验收；不得自动延长。


## Worker原始报告

# EXP-002 / v1 Worker 执行报告：Native 严格缓存候选 73 帧

更新：2026-10-10 04:20 Asia/Hong_Kong。**Worker Status: completed；Judge Acceptance: pending。** 本报告仅提交执行事实与 Worker 判断，不宣布正式 V3 已完成。任务依据 [next_plan.md](next_plan.md)，详细材料位于 [EXP-002 实验目录](submission/experiments/EXP-002_native_cached/README.md)。

## 研究问题与结果

本轮验证 Original H3 + released action LoRA 在 Single I0、native 时间、12→5→5 latent 分块下，用 current-prefix 严格分块因果路由和真实 persistent raw video KV，是否能在自身生成历史里保留人物结构和 A/D 动作响应。AA 与 AD 共享同一个已保存 `first12_A` 生成首窗、同一首窗 clean-commit KV、首 39 RGB、初始 noise、anchor、audio、seed 13、全局 RoPE 和 30-step/块；第二块唯一动作差别是 A 对 D。第三块接各自的第二块生成 latent 与 committed KV，没有 GT 或 V2b 后续块重置。本轮没有训练、AnyFlow、DMD 或新基线推理。

**有限正结果：** 两条路径均实际生成并逐帧解码至 **73 RGB 帧**。第二块（RGB39–55）AA/AD 水平 Farneback flow 为 **+1.347/−1.458 px/帧**；第三块（RGB56–72）为 **+0.474/−0.930**。方向在两块持续区分，人物保持单体可辨、停车场结构延续，没有早期路线的透明分解或瞬间换场。AA 第三块人物转身，边界灰度 MAD **15.01**，明显高于本块帧内 **3.83**；视觉质量不能写成无瑕或长期稳定。AD 第三块边界 MAD **6.32**。flow 是辅助运动代理，不是严格动作正确率。第二块是同历史反事实；第三块的两条历史已分化，只证明各自持续动作在自身历史中仍有响应。

| 新增块 | AA flow | AD flow | AA/AD 边界灰度 MAD | 两条片段状态 |
| --- | ---: | ---: | ---: | --- |
| RGB39–55 | +1.347 | −1.458 | 3.48 / 3.57 | 人物和场景可辨，动作方向分化 |
| RGB56–72 | +0.474 | −0.930 | 15.01 / 6.32 | 人物仍可辨，AA 有明显边界/姿态变化 |

参考 V2b Same-σ 局部双向重算在相应块的 flow 为 AA +2.146/+0.347、AD −1.752/−0.972；本候选不是从 V2b checkpoint 顺承，也不是单变量替换，故这些数值只能说明能力范围，不能推断严格缓存单独造成改善或退化。Original A→D 匹配片仍缺失，本轮按任务书没有新增。

## 协议与缓存证据

新 [interval_cached.py](submission/experiments/EXP-002_native_cached/interval_cached.py) 将全局 latent start/stop 与 cache index 分离，用原生 `fixed_prefix_timesteps=False`，输入进入 refiner 前由 `visible_inputs` 物理删除未来 action/video；current-prefix 路由来自已有候选，没有改 H3 权重。CPU 冻结真实输入检查 **9 passed**，旧 current-prefix tiny-H3 测试 **10 passed**。首次 CPU 测试输出只在工具记录中，未落盘；[04:20 HKT 的事后复核日志](submission/experiments/EXP-002_native_cached/artifacts/cpu_tests_recheck_20261010.log)记录了同两组测试再次 **9+10 passed**，不冒充首次日志。真实 H3 首12固定 noisy state 的候选 velocity 与保存 Original 参考 relative RMS/max abs 都是 **0**，说明首窗身份没有引入数值差异。

首窗 raw KV 为 50 层、每层 4,680 video tokens，**6,799,104,000 bytes**；第二块各自 clean-commit 后每层累计 6,630 tokens，**9,632,064,000 bytes**。共同首窗 KV 文件 SHA 两路径一致，第二块后各自缓存 SHA 不同。AD 第二块和两条第三块的采样前后，内存 entry 身份、storage 指针、tensor version、commit 数都保持不变；历史没有在每个 sigma 重算。已发布 RGB 前缀也经 `.npy` 逐像素核对：两路径首39完全一致，每条73片的前56与原已发布56帧完全一致。AA 第二块在增强 instrumentation 前执行，仅记录磁盘首窗 KV 文件 hash 不变，没有进程内 identity 检查；原运行源码已冻结为 [AA at-run 快照](submission/experiments/EXP-002_native_cached/run_rollout_aa_at_run.py)，没有为补这一个测量重跑 GPU。

## 视频、完整性与预算

- [AA 73 帧原片](submission/experiments/EXP-002_native_cached/artifacts/videos/AA_rollout_73.mp4)、[AD 73 帧原片](submission/experiments/EXP-002_native_cached/artifacts/videos/AD_rollout_73.mp4)。56 帧原片也在实验 README 中。
- [左 V2b、右 strict KV：AA 73 帧](submission/experiments/EXP-002_native_cached/artifacts/videos/V2b_vs_cached_AA_73.mp4)、[AD 73 帧](submission/experiments/EXP-002_native_cached/artifacts/videos/V2b_vs_cached_AD_73.mp4)；56 帧并排片同样已交付。两侧帧数、24fps、可视时间一致，顶部标明协议，底部标动作与块边界。对比是跨协议能力比较：V2b 同σ历史联合重算，候选为 sigma0 hidden KV 与严格分块因果。新增块之后两侧历史不再相同。
- 8 个 MP4 逐帧解码、H.264、24 fps、尺寸、PTS 递增与帧数已通过 [完整性审计](submission/experiments/EXP-002_native_cached/artifacts/video_integrity.json)。原视频没有被覆盖；小 MP4、JSON 已复制到 submission，大 tensor/KV 保留在外部输出目录并由 [artifact manifest](submission/experiments/EXP-002_native_cached/artifact_manifest.json) 关联。

预算实际消耗：**120 sampling + 3 prefill/commit + 1 identity diagnostic = 124 次 denoiser forward**，4 次 VAE decode，0 optimizer update；GPU 0 顺序执行四个进程，合计 **825.24 GPU-seconds / 0.2292 GPU-hours**，首至末约 15 分钟，峰值 `torch.cuda.max_memory_allocated` **26,686 MiB**，均未触及任务上限。第一条 AA 运行在 prefill 后重置了 peak，其单条 25,498 MiB 未包含 prefill 峰值；后续三条完整记录其各自 sampling、commit、decode 的进程内峰值。四次运行单块 wall 分别为 AA 219.2/200.8 s、AD 188.1/191.2 s，包含每次模型加载和当前块处理，但不等于从零 73 帧端到端时间，不能与 Original 30-step 全片时间直接相除得加速比。逐块原始指标和预算见 [metrics.json](submission/experiments/EXP-002_native_cached/metrics.json) 与 [budget.json](submission/experiments/EXP-002_native_cached/artifacts/budget.json)。

## Worker 判断与边界

这是一条**有信息价值的 V3 可行性候选**：首窗 identity、严格未来隔离、真实 raw KV 复用、同历史 A/D 第二块分化以及两条 73 帧基本视觉结构同时得到证据。它仍不是正式高效 V3 的完整证明：只测了单场景/seed、持续 A 或先 A 后持续 D 到 73 帧；没有 124 帧、其它动作、多场景或公平端到端加速验证。第三块 AA 的边界跳变也需保留为限制。按任务书本轮停在 73 帧，后续扩展或正式版本判定交 Judge 决定；Worker 不自动启动新实验。


## Judge正式验收

# EXP-002 / v1 — Judge final review

2026-10-10，Asia/Hong_Kong。**Decision: accept；任务关闭。接受73帧严格缓存候选的可行性证据，正式124帧V3尚未完成。**

## 结论与范围

| 维度 | 判定 | 证据与边界 |
| --- | --- | --- |
| 执行与协议 | PASS | Original H3 + released action LoRA，native Single I0，12→5→5，30步/chunk；0训练、无GT重置、未来video/action物理裁剪 |
| Strict chunk causal / persistent KV | PASS（本协议） | 源码与CPU检查、首12模型身份relative RMS/maxabs=0、真实50层raw video KV持久化与自己的第二块commit；后续采样不重算历史 |
| 动作可行性 | PASS（单scene/seed、73帧） | 同first12_A下第二块AA/AD flow +1.347/−1.458且视觉响应可辨；第三块+0.474/−0.930为各自历史续写，不能当同状态反事实 |
| 人物/场景可行性 | PASS，有质量限制 | 新增RGB39–72全部静态序列与原尺寸端点已看；单体人物、停车场可辨；AA55→56明显姿态位置跳变、节奏与细节软化保留 |
| 效率记录 | PASS（增量成本）；完整E2E/公平speedup NOT_TESTED | 124 forwards、4 VAE、825.238 GPU秒；缓存真实复用不等于Original公平加速 |
| 124帧与跨场景 | NOT_TESTED | 下一步只延伸同一候选，不追加短片消融或训练 |

共享首12 latent是既有模型生成历史，首39 RGB复用既有生成片，非GT。候选空cache图的真实模型身份诊断支持首窗复用；73帧不能描述为本轮全部从零采样。源模型、输入与代码凭据见[MANIFEST](../MANIFEST.md)、[Judge源码摘要](source_manifest.json)和逐块原始JSON。

## 审阅内容与可追溯证据

Judge独立读取interval入口、两版runner、路由/缓存代码与测试。9项本轮CPU输入/位置检查和10项既有current-prefix检查通过；不重复18状态模型诊断。独立检查两条第二块输入配对、第三块恢复cache链、endpoint及published RGB摘要；每条已显示56帧保持冻结。完整解码56/73原片；新增全部帧的静态图与原尺寸端点见本目录，不声称原速播放。最终20项交付manifest均匹配，两条73帧并排片完整解码、24fps/73帧/PTS通过，布局抽看通过。

- [阶段审阅](S0_REVIEW.md)、[同history配对](second_pair_checks.json)、[AA第三块](AA_third_checks.json)、[AD第三块](AD_third_checks.json)、[交付检查](delivery_checks.json)。
- [冻结Worker原始报告](../worker_report.md)、[原始metrics快照](../worker_metrics_snapshot.json)、[执行任务书](../accepted_taskbook.md)。原始报告中的相对链接按根report.md位置写，实验入口提供当前有效导航。

## 成本与限制

120 sampling +3 prefill/commit +1首窗身份诊断 =124 forwards；4 VAE、0 optimizer、单GPU0依次执行，0.22923284 GPU-hours，首至末914.45秒。最大已记录allocated26,686.14MiB。首12 cache6,799,104,000 bytes；through17为9,632,064,000 bytes，50层各6630历史video tokens。

首条AA在instrumentation增强前执行：仅磁盘cache hash不变，没有进程内identity/version记录；peak在prefill后reset，缺该阶段峰值。冻结原代码并披露，不重跑。后三次进程有内存只读与各阶段峰值；这些证据结合源码足以支持本轮可行性判断。原JSON的commit_calls_per_layer实际为累计层级提交数，不能当模型forward次数。

每次增量进程包含重新加载/缓存序列化，不能当完整首屏或73帧从零E2E；V2b同时改变拓扑、history sigma及缓存，不可单因素归因。普通画质缺陷按用户可行性尺度接受；单停车场seed13、仅AA/AD、73帧不覆盖泛化和长期质量。

## 下一步决策

方向已有有效信号，下一项有价值的验证是同配置自己的历史续写73→124，记录逐块缓存与实际成本。不给新场景、seed、训练或额外诊断预算。若出现严重持续失效则记录边界、停止相应方向，不为挽救结果无限追加实验。EXP-003须在本轮归档与Git交付后正式发布。


EXP-002 Git交付：bdcb43a5bf7568b6182b8563a9b08a8391fa53fb，正常push并核对远端；发布前fetch无分叉。CPU日志artifacts/cpu_tests_recheck_20261010.log为04:20 CPU重查9+10通过，未增加GPU开销。


# EXP-003 / v1 完整归档：V3可行性版本 accepted


## 任务书

# EXP-003：同一严格缓存候选的124帧与效率可行性

发布：2026-10-10 04:23 HKT，Judge。EXP-002已验收归档并推送GitHub：bdcb43a5bf7568b6182b8563a9b08a8391fa53fb。本文件是当前唯一GPU任务授权。

## Task / Track / Parent

EXP-003 / v1；Worker Status: running（单GPU0，AA先行）；Judge Acceptance: pending；Mainline / V3 feasibility。Parent为EXP-002 native Single I0/current-prefix/clean-commit persistent-KV候选，Original H3 + released action LoRA，零新增训练。

## 核心问题与假设

同一候选在自己的历史KV下延伸至124帧，是否仍具备可辨动作、基本人物/场景结构，并体现真实历史复用与可解释的推理成本？假设是EXP-002短窗口能力可延续，不以正式V3通过为预设结果。

已有回答与本轮价值：EXP-002检查同一个A首窗历史下当前A/D（AA/AD）并续至73帧；本轮只增加后续三块，验证增长后的自身历史及成本，不重复短窗口和18点诊断。正结果可用于同配置V3可行性验收；严重持续失效则记长度边界，停止该无训练方向，不自动加训练挽救。

## 保持条件 / 唯一变化 / 输入

保持Original/released LoRA、停车场832×480、seed13、full37相同noise、Single I0、native时间与全局RoPE、固定audio条件、30steps/chunk、shift2.22、own action/current-prefix连接、sigma0/native clean commit、原精度/backend/offload与CPU raw KV。AA全A，AD首12为A、其后均D。

从各自EXP-002完成的first12_A + second5 + third5 endpoints与cache_through17续接。先用自己的third5按原图/条件提交一次得到cache_through22，不能重建于当前新prefix下。首73已发RGB保持原样。继续latent[22,27)、[27,32)、[32,37)，累计90/107/124RGB，完整partition[12,5,5,5,5,5]。

本轮只扩大生成范围；不改变mask、anchor、history sigma、窗口淘汰、精度、权重或动作协议。六块范围内保留全部历史；cache index与全局start分开，历史K/V是祖先提交的原表示，不能每步重算历史或用V2b/teacher reset。

## 执行步骤

1. 冻结EXP-002源码/输入/缓存/endpoint/RGB hash；隔离EXP-003入口，扩展显式区间到37latent。CPU验证新增索引/位置、缓存覆盖及预算；复用已有动作/未来隔离证明，不重跑模型诊断。
2. AA/AD各自续写三个chunk至124帧，保留中间checkpoint和真实帧；一个有限任务里可连续执行，无需每块提交正式验收。发现明确严重结构或持续控制失败时停止受影响路径，保留结果；普通细节、节奏或边界问题不取消所有路径。
3. 每块只读历史KV采样30步；需要进入下一块时当前endpoint sigma0 commit一次。最后124帧无须多余commit。生成帧只追加，不对过去末5帧进行回改或平滑。
4. 交付候选AA/AD完整124帧、Original持续A vs candidate AA124、V2b AA vs candidate AA124，以及V2b AD/candidate AD共有73帧。Original A→D匹配视频缺失则标记；不使用Original持续D冒充，也不新生成基线。各片标明动作、范围、协议与计时口径。
5. 提交report后停止GPU扩展，Judge判断是否可以发布正式V3可行性版本并同步GitHub。

## 预算与资源

0训练；两路径、单scene/seed。6个新chunk×30=180 sampling；续接third5提交2次、90/107帧后提交各2次，总commit≤6；诊断0；完整denoiser总≤186。VAE≤6、RGB re-encode0，失败/重试计入。

全任务≤2 GPU-hours，首个GPU进程起≤3h elapsed；allocated每卡≤44GiB。项目在2026-10-10 09:00 HKT前最多8卡，之后最多3卡；本任务最多2卡且可顺序运行，08:30后不增加超过3卡占用，09:00前释放多余卡，不自动恢复8卡。

建议一个进程连续处理同一路径剩余chunk，保留必要offload；若逐块重启，明确加载/保存cache成本，不将其混入纯sampling比较。所有活跃进程、加载、提交、VAE、失败都计入GPU-hours。

## 效率与验收

记录每块sampling、clean commit、cache读取/传输/保存、VAE和wall时间、实际前向次数、cache层/token/bytes、GPU allocated峰值；核对只对新chunk做Transformer计算和实际历史cache未变。首73复用时，时间明确为增量；首屏/完整E2E未测则写未测，不能补造。

对照优先复用EXP-001 AA对应三块实测：310.26/377.53/447.19秒sampling；记录硬件/offload/步数及运行时间差异。可描述已记录条件下的观察成本，不能当严格单变量KV收益或统一重复均值speedup。Original整片30步和本候选每块30步不直接求加速比。

同一个明确配置同时满足以下条件时，Judge可验收V3的可行性版本：strict chunk-causal、真实persistent KV复用；EXP-002同history A/D有可辨条件响应；本轮124帧自身历史里人物/场景基本可用且动作无明显持续失效；代码/输入/视频可追溯，效率测量足以说明历史复用与实际成本。保留单场景/seed、切换范围、质量、无整体Original speedup和未训练等限制。不要求成熟产品画质、10/20秒、多场景泛化或AnyFlow/DMD已经成功。

若只有一条可用则能力PARTIAL；若关键机制错误则INVALID；有效负结果可接受任务执行但不能宣布V3。普通模糊、肢体细节、短暂边界和小幅proxy反号不独立判失败。

## 输出与停止

独立EXP-003目录代码/config/manifest/metrics/日志/真实视频/效率表、根report.md；最终说明同一配置解决什么和还缺什么。协议泄漏、历史回改、NaN/OOM、预算超限、严重质量/控制失败时停止对应运行并报告。无训练、额外seed/场景、参数搜索、额外诊断、AnyFlow/DMD或更长视频授权。

本任务授权限于上述186次前向与资源范围；完成或提前停止后提交report，不自动扩展。


## Worker原始报告

# EXP-003 / v1 Worker 执行报告：严格缓存候选续写至124帧

更新：2026-10-10 04:58 HKT。**Worker Status: completed；Judge Acceptance: pending。** 本报告提交本轮实际执行与证据；正式 V3 可行性版本是否验收由 Judge 决定。任务依据 [next_plan.md](next_plan.md)，完整目录见 [EXP-003 README](submission/experiments/EXP-003_native_cached_124/README.md)。本轮结束后 GPU 进程均已退出。

## 结论与协议

在 EXP-002 已验收的 Original H3 + released action LoRA、native Single I0、current-prefix own-action routing、严格分块因果、CPU persistent raw KV、30 native FM steps/chunk、seed13、停车场单场景的**同一配置**下，AA 和 AD 均从自己已生成的73帧 endpoint/cache 续写至 **124 RGB帧 / 24fps / 5.17秒**。先用各自保存的第三块 clean endpoint 在原始条件下提交一次 KV，然后各生成三个5-latent chunk；每块采样只读祖先KV，进入下一块时只提交当前 clean endpoint，末块不做无用commit。未来action/video在进入refiner前物理删除，保留全局RoPE/原生时间和同一I0，没有teacher/GT reset、RGB anchor、训练、AnyFlow或DMD。

两个124帧原片中人物与停车场到末尾仍可辨，A与D在新增三个chunk的水平光流代理符号持续相反。**但 AA 在约RGB79–86出现明显人体形变和游离肢体残影**，约RGB108–110另有短暂背景残影；AD约RGB91有短暂肢体残影。AA第五、第六块的人物恢复为单体，缺陷没有持续累积。故这支持“单场景/seed下有明确质量限制的长窗口严格缓存可行性”，不支持全帧高质量、10/20秒稳定、多场景泛化或成熟交互体验的声明。flow只是运动代理，不等于动作语义判分。EXP-002第二块的AA/AD同历史反事实仍是动作控制的更强证据；本轮73帧后两路径历史已经不同，不能再称同状态反事实。

| 新增RGB范围 | AA水平flow px/帧 | AD水平flow px/帧 | AA/AD边界灰度MAD | 实际观察 |
| --- | ---: | ---: | ---: | --- |
| 73–89 | +1.463 | −1.304 | 13.82 / 7.18 | AA约79–86明显人体形变/残影，AD基本完整 |
| 90–106 | +0.953 | −0.648 | 9.69 / 3.34 | AA恢复单体；AD约91帧短暂残影 |
| 107–123 | +1.335 | −1.240 | 2.91 / 2.50 | 两条基本结构可用；AA短暂背景残影 |

## 原始视频与对照

- [候选 AA 124帧](submission/experiments/EXP-003_native_cached_124/artifacts/videos/AA_rollout_124.mp4)和[候选 AD 124帧](submission/experiments/EXP-003_native_cached_124/artifacts/videos/AD_rollout_124.mp4)；90/107帧中间片也在 [实验 README](submission/experiments/EXP-003_native_cached_124/README.md) 中。
- [左 Original H3 持续A、右候选 AA 124](submission/experiments/EXP-003_native_cached_124/artifacts/videos/Original_vs_candidate_AA_124.mp4)；[左 V2b AA、右候选 AA 124](submission/experiments/EXP-003_native_cached_124/artifacts/videos/V2b_vs_candidate_AA_124.mp4)。两片来自真实保存的MP4，显示动作、协议与不同计时口径。
- [左 V2b AD、右候选 AD 共73帧](submission/experiments/EXP-003_native_cached_124/artifacts/videos/V2b_vs_candidate_AD_73.mp4)复用EXP-002已验收对照。Original A→D匹配片缺失，未用Original持续D冒充，也未增加基线推理。

Original整片30-step与候选每块30-step不是相同denoiser调用数；对照是能力比较而非单因素消融。Original A整片E2E为478.4s；候选AA的73→124帧进程增量wall为798.9s，含加载、cache IO、采样、提交、VAE、编码及人工检查等待，**不含旧73帧生成**，不能相除得端到端speedup。V2b AA已记录三个续写块sampling为310.26/377.53/447.19s；候选为148.45/176.43/200.42s。该观察包含拓扑、历史协议、KV策略和计时instrumentation差异，不能单独归功于KV。细节见 [效率表](submission/experiments/EXP-003_native_cached_124/efficiency_table.md)及[原始指标](submission/experiments/EXP-003_native_cached_124/metrics.json)。

## 实现正确性、预算及限制

[独立入口](submission/experiments/EXP-003_native_cached_124/interval_cached.py)显式使用全局区间 `[17,22)→[22,27)→[27,32)→[32,37)` 和独立cache index；前者只用于已保存third5提交，后面三段用于新采样。[计时路由](submission/experiments/EXP-003_native_cached_124/metered_prefix.py)只测已有current-prefix路由的历史KV读取/搬运，CPU测试与原路由逐元素输出相同。冻结的18份来源文件hash和两条EXP-002缓存的50层/token覆盖经 [CPU预检](submission/experiments/EXP-003_native_cached_124/preflight_cpu.log)核对；新增 [11项CPU测试](submission/experiments/EXP-003_native_cached_124/artifacts/cpu_tests_postrun.log)通过。

六段采样前后历史cache的entry身份、storage地址、tensor version与提交计数未变；每次commit后只追加当前chunk，最后块无commit。首次已发布73帧、90/107帧前缀在`.npy`层逐像素保持，endpoint/cache文件SHA与 [15份MP4及片段完整性审计](submission/experiments/EXP-003_native_cached_124/artifacts/video_integrity.json)均通过：H.264、24fps、帧数、尺寸、PTS递增。大latent、原始raw KV与未压缩RGB保留在外部输出目录，提交包用 [MANIFEST](submission/experiments/EXP-003_native_cached_124/MANIFEST.md) 与 [artifact manifest](submission/experiments/EXP-003_native_cached_124/artifact_manifest.json)关联，未复制33B权重或数十GB缓存入Git。

任务账本实际消耗 **180 sampling + 6 clean commit = 186完整denoiser forwards**，6次VAE，0 optimizer update，GPU 0顺序执行AA/AD，总 **1572.27 GPU-seconds / 0.43674 GPU-hours**，低于2 GPU-hours与3小时elapsed上限。记录到的新增采样块/commit/VAE最大`torch.cuda.max_memory_allocated`为 **26,876.70 MiB**；开头third5 clean commit只执行44GiB上限断言、未保存精确峰值，因此不能声称这是完整进程峰值。CPU KV从第三块提交后的12.465GB增长到末块采样前18.131GB。KV GPU transfer时间是sampling的组成部分，不可重复相加。没有从零完整124帧E2E、首块延迟或多次warmup均值。

Worker判断：任务协议有效，双路径124帧有动作与基本结构的有限正证据，且效率记录证实真实历史KV复用；AA中段明显结构缺陷必须随结果公开。是否将其验收为正式V3**可行性版本**，以及下一步是质量适配还是其它路线，交Judge根据这组完整证据决定。本轮不再启动GPU诊断或修复。


## Judge最终验收

# EXP-003 / v1 — Judge final review / V3 feasibility release

2026-10-10，Asia/Hong_Kong。**Decision: accept；V3 Efficient Causal H3-World 可行性版本成立。** 范围为单停车场、seed13、AA/AD自身历史六块124 RGB帧（24fps，5.17秒），Original H3 + released action LoRA，零新增训练。它有明确画质缺陷，不代表成熟模型、长期泛化或公平Original端到端加速。

## 为什么现在收口

EXP-002已经在同一个A首窗历史下证明当前A/D响应，并验证native条件的严格分块因果和真实persistent raw video KV；EXP-003保持相同图、权重和条件，只延伸自己的历史至124帧，六个新增块的方向信号持续，后段人物/场景仍基本可用。实际缓存、资源和成本均有证据，达到本阶段判断路线可行所需范围。

AA第四块有明显瞬态人体形变/游离肢体残影，约RGB79–86，不能归类成普通模糊，也不能删去；第五、六块恢复单体人物，没有持续恶化。AD第五块开头有短暂姿态/肢体异常，后续恢复。按用户明确的可行性与ROI要求，这些缺陷降低画质和连续性评级，但不否定同协议下的核心能力验证。没有追加修复训练、种子筛选、mask消融或模型诊断。

## 分维度判定

| 维度 | 判定 | 证据 / 限制 |
| --- | --- | --- |
| 任务执行 | PASS | 186次前向、6VAE、零训练/诊断/重试；AA/AD124帧，预算内结束 |
| Strict chunk-causal | PASS（本协议） | 未来video/action物理裁剪；全局start与cache index分开；private current prefix，不回改祖先隐藏状态 |
| Persistent KV | PASS | 每层祖先raw K/V+RoPE持久保存；采样内存身份/指针/version不变；只提交新的clean endpoint；最后块无额外commit |
| 动作可行性 | PASS（本scope） | EXP-002同history第二块A/D响应；本轮两条三新块有可辨持续方向，flow为辅助证据 |
| 多窗口基本视觉可用性 | PASS，有明显质量限制 | 两条到124帧人物/停车场仍可辨，AA瞬态严重形变后恢复；非全帧稳定或高保真 |
| 严格连续性 / 画质成熟度 | PARTIAL | AA边界姿态跳变、RGB约79–86结构异常；动作僵硬、细节软化；不能写成画质追平V2b |
| 历史复用与实测成本 | PASS（增量口径） | 有逐块sampling/commit/CPU cache搬运/保存/VAE/wall；历史不每步重算 |
| 完整从零E2E、公平speedup、跨场景/长时泛化 | NOT_TESTED | 首73复用、单场景seed；Original全片30步与每块30步不能直接相除 |

## 结果与成本

| 路径 / 总RGB | 新块flow px/帧 | sampling秒 | 边界灰度MAD |
| --- | ---: | ---: | ---: |
| AA90 | +1.4629 | 148.45 | 13.82 |
| AA107 | +0.9529 | 176.43 | 9.69 |
| AA124 | +1.3354 | 200.42 | 2.91 |
| AD90 | −1.3040 | 148.94 | 7.18 |
| AD107 | −0.6482 | 176.66 | 3.34 |
| AD124 | −1.2402 | 199.43 | 2.50 |

实际180sampling+6commit=186完整前向、6VAE、0optimizer、0额外诊断。GPU0/NVIDIA L40顺序执行AA与AD，798.946+773.320=1572.266 GPU秒，即**0.436741 GPU-hours**，含占卡检查等待；首至末约26.8分钟。EXP-002与EXP-003合计310实际前向、10VAE、约0.66597 GPU-hours；共享首12的原始采样开销不在这两轮内，不能把此数当完整从零端到端速度。

每块采样前历史KV22/27/32latent，50层各8580/10530/12480 video tokens，CPU cache12,465,024,000 /15,297,984,000 /18,130,944,000 bytes。每个sigma搬运历史cache到GPU，未使用GPU驻留缓存；AA三块搬运GPU事件累计19.47/25.47/34.97秒，已包含于148.45/176.43/200.42秒sampling，不能重复加到总耗时。

已记录新块sampling/commit/VAE allocated峰值26,876.70MiB（约26.25GiB）。初始third5提交执行≤44GiB上限断言，但其third_peak局部值未写盘；不能称这是真正完整进程峰值。此缺项披露，不补跑GPU。EXP-002首条AA另有已披露测量缺项，旧结果保持原样。

已有V2b AA对应sampling310.26/377.53/447.19秒，本候选单次为其47.8%/46.7%/44.8%。二者均为同机L40、每块30步、记录的native精度/offload，但拓扑、history sigma与缓存共同变化，运行时刻和计时instrumentation不同；可称协议整体的观察成本降低，不是纯KV单因素收益、重复均值benchmark或相对Original完整E2E speedup。

## 独立审核与来源

Judge阅读interval、metered_prefix、runner与测试，计时改动保持attention数学。新增11项CPU检查由Worker执行通过，旧19项检查来自EXP-002；未追加真实模型诊断。读取来源18项hash、逐块实际源码/输入摘要与GPU账本。

两条共102个新RGB帧全部静态查看，包含每块边界与native末帧，并额外看AA82/85原尺寸细节；完整解码90/107/124，核对24fps/帧数/单调PTS、旧RGB逐像素冻结、端点摘要、内存cache不变、参数版本。首73继承EXP-002已审证据，历史源为生成数据，不是GT。不声称进行了正常速度实际播放。

- [逐块观察](S0_REVIEW.md)、本目录六份`*_checks.json`和完整contact sheets。
- [成本独立汇总](costs_checked.json)、[最终交付检查](delivery_checks.json)。
- [冻结Worker报告](../worker_report.md)、[原始metrics](../worker_metrics_snapshot.json)、[执行任务书](../accepted_taskbook.md)、[来源和命令](../MANIFEST.md)。

完整V2b和Original AA124对比只作跨协议能力比较；AD匹配V2b只覆盖共有73帧，Original A→D匹配片缺失，未拿持续D冒充。原片保留所有坏帧和真实边界。

## 版本与下一步

发布V3可行性版本的依据是EXP-002/003同一权重与推理协议；没有将V2b重命名，也没有把V2a与V2b不同能力相加。正式版本定义、代表片、配置与代码摘要一起冻结。AnyFlow/DMD尚未完成；30steps/chunk也并非实时生成。

当前任务关闭，不再扩长片、补小消融或修复几帧。下一阶段建议以该V3为冻结基线，优先做有明确预算的少步生成可行性；是否进入新训练须另立任务书。持续无效方向及时归档，普通质量缺陷不自动触发追加训练。


EXP-003/V3 Git交付：`d039352941708d4c7c757d696009b67c3e0e2468`；fetch确认无分叉，正常commit/push，无force push，远端SHA一致，工作树干净。73项本轮提交约22.82MB，不含权重/latent/KV/bytecode。


# EXP-003关闭后的任务状态快照（发布EXP-004前）

# EXP-003：同一严格缓存候选的124帧与效率可行性

发布：2026-10-10 04:23 HKT，Judge。EXP-002已验收归档并推送GitHub：bdcb43a5bf7568b6182b8563a9b08a8391fa53fb。本任务已关闭，无继续GPU授权。

## Task / Track / Parent

EXP-003 / v1；Worker Status: completed；Judge Acceptance: accepted（V3 feasibility）；Mainline / V3 feasibility。Parent为EXP-002 native Single I0/current-prefix/clean-commit persistent-KV候选，Original H3 + released action LoRA，零新增训练。

## 核心问题与假设

同一候选在自己的历史KV下延伸至124帧，是否仍具备可辨动作、基本人物/场景结构，并体现真实历史复用与可解释的推理成本？假设是EXP-002短窗口能力可延续，不以正式V3通过为预设结果。

已有回答与本轮价值：EXP-002检查同一个A首窗历史下当前A/D（AA/AD）并续至73帧；本轮只增加后续三块，验证增长后的自身历史及成本，不重复短窗口和18点诊断。正结果可用于同配置V3可行性验收；严重持续失效则记长度边界，停止该无训练方向，不自动加训练挽救。

## 保持条件 / 唯一变化 / 输入

保持Original/released LoRA、停车场832×480、seed13、full37相同noise、Single I0、native时间与全局RoPE、固定audio条件、30steps/chunk、shift2.22、own action/current-prefix连接、sigma0/native clean commit、原精度/backend/offload与CPU raw KV。AA全A，AD首12为A、其后均D。

从各自EXP-002完成的first12_A + second5 + third5 endpoints与cache_through17续接。先用自己的third5按原图/条件提交一次得到cache_through22，不能重建于当前新prefix下。首73已发RGB保持原样。继续latent[22,27)、[27,32)、[32,37)，累计90/107/124RGB，完整partition[12,5,5,5,5,5]。

本轮只扩大生成范围；不改变mask、anchor、history sigma、窗口淘汰、精度、权重或动作协议。六块范围内保留全部历史；cache index与全局start分开，历史K/V是祖先提交的原表示，不能每步重算历史或用V2b/teacher reset。

## 执行步骤

1. 冻结EXP-002源码/输入/缓存/endpoint/RGB hash；隔离EXP-003入口，扩展显式区间到37latent。CPU验证新增索引/位置、缓存覆盖及预算；复用已有动作/未来隔离证明，不重跑模型诊断。
2. AA/AD各自续写三个chunk至124帧，保留中间checkpoint和真实帧；一个有限任务里可连续执行，无需每块提交正式验收。发现明确严重结构或持续控制失败时停止受影响路径，保留结果；普通细节、节奏或边界问题不取消所有路径。
3. 每块只读历史KV采样30步；需要进入下一块时当前endpoint sigma0 commit一次。最后124帧无须多余commit。生成帧只追加，不对过去末5帧进行回改或平滑。
4. 交付候选AA/AD完整124帧、Original持续A vs candidate AA124、V2b AA vs candidate AA124，以及V2b AD/candidate AD共有73帧。Original A→D匹配视频缺失则标记；不使用Original持续D冒充，也不新生成基线。各片标明动作、范围、协议与计时口径。
5. 提交report后停止GPU扩展，Judge判断是否可以发布正式V3可行性版本并同步GitHub。

## 预算与资源

0训练；两路径、单scene/seed。6个新chunk×30=180 sampling；续接third5提交2次、90/107帧后提交各2次，总commit≤6；诊断0；完整denoiser总≤186。VAE≤6、RGB re-encode0，失败/重试计入。

全任务≤2 GPU-hours，首个GPU进程起≤3h elapsed；allocated每卡≤44GiB。项目在2026-10-10 09:00 HKT前最多8卡，之后最多3卡；本任务最多2卡且可顺序运行，08:30后不增加超过3卡占用，09:00前释放多余卡，不自动恢复8卡。

建议一个进程连续处理同一路径剩余chunk，保留必要offload；若逐块重启，明确加载/保存cache成本，不将其混入纯sampling比较。所有活跃进程、加载、提交、VAE、失败都计入GPU-hours。

## 效率与验收

记录每块sampling、clean commit、cache读取/传输/保存、VAE和wall时间、实际前向次数、cache层/token/bytes、GPU allocated峰值；核对只对新chunk做Transformer计算和实际历史cache未变。首73复用时，时间明确为增量；首屏/完整E2E未测则写未测，不能补造。

对照优先复用EXP-001 AA对应三块实测：310.26/377.53/447.19秒sampling；记录硬件/offload/步数及运行时间差异。可描述已记录条件下的观察成本，不能当严格单变量KV收益或统一重复均值speedup。Original整片30步和本候选每块30步不直接求加速比。

同一个明确配置同时满足以下条件时，Judge可验收V3的可行性版本：strict chunk-causal、真实persistent KV复用；EXP-002同history A/D有可辨条件响应；本轮124帧自身历史里人物/场景基本可用且动作无明显持续失效；代码/输入/视频可追溯，效率测量足以说明历史复用与实际成本。保留单场景/seed、切换范围、质量、无整体Original speedup和未训练等限制。不要求成熟产品画质、10/20秒、多场景泛化或AnyFlow/DMD已经成功。

若只有一条可用则能力PARTIAL；若关键机制错误则INVALID；有效负结果可接受任务执行但不能宣布V3。普通模糊、肢体细节、短暂边界和小幅proxy反号不独立判失败。

## 输出与停止

独立EXP-003目录代码/config/manifest/metrics/日志/真实视频/效率表、根report.md；最终说明同一配置解决什么和还缺什么。协议泄漏、历史回改、NaN/OOM、预算超限、严重质量/控制失败时停止对应运行并报告。无训练、额外seed/场景、参数搜索、额外诊断、AnyFlow/DMD或更长视频授权。

本任务186次前向已执行完毕并验收；GPU已释放，无新增实验授权。


## Judge最终决策 / Next plan

EXP-003已验收，结合EXP-002同配置证据，正式发布V3可行性版本。AA/AD124帧，strict chunk-causal、真实persistent KV和可辨动作成立；AA约79–86帧明显瞬态形变后恢复，画质/严格连续性PARTIAL。详见submission/experiments/EXP-003_native_cached_124/judge/FINAL_REVIEW.md。

本阶段结束，不追加长视频、场景/seed、微小调参或修复训练。下一阶段建议：冻结V3，另立有限预算的少步生成可行性任务，再决定是否扩大AnyFlow训练；DMD尚未进入。该建议不是GPU授权，当前没有EXP-004在运行。

继续执行项目总GPU限制：2026-10-10 09:00 HKT之后最多3卡，不自动恢复8卡。当前GPU已释放。

正式交付Git：`d039352941708d4c7c757d696009b67c3e0e2468`，origin/main已核对一致。V3阶段完成，当前无运行任务。



## 2026-10-10：EXP-004 / v1 完成验收

Judge accepted：原权重8步续写AA/AD73帧有限可行性，画质PARTIAL。34forward/4VAE/0.12944943 GPU小时，零训练。首39帧借用30步结果，AA明显肢体拖影如实保留；停止本轮，下一步建议全程8步初始化验证。

### 冻结原任务书

~~~~markdown
# EXP-004：V3原权重8步续写的有限可行性

发布：2026-10-10，Judge。用户要求继续下一阶段；V3正式基线已冻结于submission commit `d039352941708d4c7c757d696009b67c3e0e2468`。本文件为当前唯一新GPU任务授权。

## 1. Task / Version / Track / Parent

- Task ID / Plan Version：EXP-004 / v1。
- Worker Status：authorized；Judge Acceptance：pending。
- Research Track：Mainline efficiency / 少步基线；不是AnyFlow训练。
- Parent：已验收V3（Original H3 + released action LoRA，native Single I0/current-prefix/严格chunk causal/CPU persistent raw video KV）。

## 2. 唯一问题 / 假设 / 决策价值

**仅把新生成chunk的native FM采样从30步减为8步，不训练、不改其它协议，能否保留同历史A/D响应，并在自己的8步历史下续到73帧的基本人物/场景结构？**

假设是已有权重可能有足够的减步余量，未预设通过。本轮不是AnyFlow，也不把普通Euler减步结果用于证明AnyFlow成败。

已有8步V1/V2a与旧AnyFlow试验使用不同图/anchor/权重，不能回答本轮V3同协议问题；EXP-002/003只验证30步。先做本轮比新开蒸馏训练更便宜。

- 正结果：后续优先验证8步自己的历史能否延伸；再决定是否需要训练，而非直接开大AnyFlow任务。
- 负结果：停止原权重直接8步路线，不扫4/12/16步、shift或gain；保存30步V3与8步负对照。再依据可修复信号决定独立的有限少步适配任务，绝不自动扩训。

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

在`submission/experiments/EXP-004_v3_8step/`建独立config/runner；优先复用EXP-002 interval及既有路由，不重写注意力实现。新改动仅8步schedule、复用首窗与账本/输出路径。CPU检查覆盖实际8步schedule到终点、预算、输入/首RGB/缓存摘要；继承已通过因果和cache测试，不为未改部分重复GPU诊断。

冻结实际源码/config/命令。每次forward和VAE尝试调用前记账，包括失败；禁止自动重试。项目其它用户GPU不得干预。

### S1：同A历史下AA/AD分别8步续第二块到56帧

两个分支读取同一首12及raw KV，噪声/anchor/audio/全局位置/RGB前缀相同，仅当前动作A或D不同。interval[12,17)、index1；8步sampling只读历史，记录内存cache identity/storage/version/commit计数不变，冻结39RGB并追加17。

先看两条新增帧。如果严重结构崩坏或动作条件明显失效，则停受影响路径并保留真实负结果，不再自动续该路径；普通细节/边界/局部proxy异常按可行性尺度判断。观察足够改变决策即可收口。

### S2：值得继续的路径续自身第三块到73帧

只将自己的8步第二块clean endpoint按原sigma0/native/原stop17动作条件提交一次，index1；不得换回30步第二块。然后[17,22)、index2再8步sampling，只读已提交cache；冻结56RGB，追加17。最后第三块不做无用commit。

每条最多两新chunk，最多四条8步采样。Worker可在授权内逐块快速视觉检查后继续，不必等待Judge逐块许可；明显失败按停止条件报告。

### S3：交付并停止

给出8步AA/AD完整56/73原片、左V3 30步右8步的共有长度并排片。清楚标明首窗仍是复用30步历史，新块是8步；若路径提前停则按真实长度对比，不补片或循环。第二块可同history比较；第三块两种steps/动作路径的历史已不同，要区分。

提交完整report、代码/配置、来源/产物manifest、实际命令/日志、逐块成本与账本。提交后停GPU，由Judge验收、归档和Git发布，不自动开展下一实验。

## 6. Budget / GPU

- 0 optimizer update、0额外真实模型diagnostic、0首窗prefill、0新增基线生成。
- 新sampling≤32（最多4×8），第二块clean commit≤2；完整denoiser总≤34。最后块不commit，失败/重试计入上限。
- VAE decode≤4；RGB re-encode0；单scene/seed，仅AA/AD。
- 全任务≤0.5 GPU-hours，首个GPU进程起≤90分钟elapsed；每卡allocated≤44GiB。加载/缓存搬运/commit/VAE/占卡检查等待/失败都计GPU-hours。
- 现在已过2026-10-10 09:00 HKT，**项目总GPU≤3，本任务最多1卡**。启动前重新确认空闲卡；当前优先空闲GPU0，但不能仅凭编号猜测可用。不得终止或迁移他人进程。

## 7. Evaluation / Acceptance

分别报告执行有效性、动作、基本视觉、缓存与效率。静态全部新增帧和必要native细节必须查看；不能仅看末帧。普通质量缺陷允许，但明显持续的人物分解/控制消失须明确失败，不能为了减步利益隐藏。

主要对照为同V3协议30 vs8。第二块匹配历史/初始noise/条件，主动变量只有schedule步数；第三块接各自生成历史，属于闭环协议结果。记录实际sigma列表、8次sampling、commit数、cache未变、RGB冻结与checkpoint/hash。

记录每块sampling、commit、VAE、wall、cache tokens/bytes和GPU峰值。sampling含缓存搬运；若未另测搬运子项就标未拆分，不为补计时复制路由或重复运行。峰值应覆盖本进程真实sampling/commit/VAE；缺项披露。比较8步与既有30步的单次观察成本，注明不同时刻共享硬件、首窗复用；不要直接宣称端到端3.75倍加速。

结果是负也可以接受任务执行；8步能力按PASS/PARTIAL/FAIL区分。只因8步推理完成或flow符号符合不宣布能力通过。8步成功也不改名为AnyFlow或宣布训练完成。

## 8. Stop / Out of Scope / Deliverables

协议错误/未来泄漏/历史回改/NaN/OOM/预算超限立即停相关进程；严重持续质量或动作失败停相关路径，充分证据后停止扩展。无4步、其它步数、noise/shift/gain/mask扫描、长至124帧、新scene/seed、训练、AnyFlow/DMD或新Original基线授权。

独立目录：README、config、源码与hash、metrics、MANIFEST、原片/并排片、日志和budget；根report.md真实状态。Judge审阅后更新progress/next_plan/archive/submission，正常Git合并与推送。本任务只授权上述≤34前向。

~~~~
### 完整Worker原报告

~~~~markdown
# EXP-004 / v1 Worker 执行报告：V3 原权重 8-step native FM 续写

更新：2026-10-10 HKT。**Worker Status: completed；Judge Acceptance: pending。** 本轮按 [EXP-004 任务书](next_plan.md)执行，完整证据在 [独立实验目录](submission/experiments/EXP-004_v3_8step/README.md)。GPU 进程已全部退出；没有启动训练、AnyFlow 或 DMD，也没有超出预算的推理。

## 研究问题与结论

只将 V3 新增 chunk 的 native FM/Euler sampling 从30步改为8步，是否仍保留同历史 A/D 条件响应，并能接自己的8步历史续至73帧？**本轮有限单场景/seed实验得到正的可行性信号，但画质与动作幅度不完全保真。** AA/AD第二块共用 EXP-002 保存的首12 latent、首39 RGB、首窗 CPU raw KV、noise、anchor、audio、全局位置、权重与推理协议，只改变当前动作；8步水平光流分别为 **+0.781 / −1.510 px/帧**，方向相反。两条路径各自接自己的8步第二块后，第三块为 **+0.978 / −0.732 px/帧**，人物和停车场到73帧仍可辨。第三块两侧的历史已不同，不能当作同状态反事实。

首块是**复用的30-step**结果，只有39–55和56–72两个新增RGB区间对应8-step采样。因此本轮不能声称完整73帧从零8-step、8-step首屏延迟或完整E2E加速。直接减步用的是原生FM，不是AnyFlow目标或蒸馏。

| 路径 / 新增RGB | 本轮8步flow | 已存V3 30步flow | 本轮sampling / 30步sampling | 本轮边界灰度MAD | 视觉判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| AA 39–55 | +0.781 | +1.347 | 70.87 / 142.91 s | 3.54 | 人物单体、场景可辨；动作幅度较小、细节更软 |
| AD 39–55 | −1.510 | −1.458 | 40.52 / 135.69 s | 2.81 | 人物单体、方向可辨，普通腿部细节缺陷 |
| AA 56–72 | +0.978 | +0.474 | 37.76 / 155.14 s | 4.89 | 仍可辨，但腿部拖影、局部透明残影较明显 |
| AD 56–72 | −0.732 | −0.930 | 42.51 / 146.39 s | 7.95 | 仍可辨，块边界变化与局部腿纹理残影 |

30步实验在清晨、本轮在下午共享硬件有其他作业；上表是**各自一次运行的实际sampling观测**，不能直接计算公平的速度倍数。flow是运动代理，不能单独代替动作语义或画质判断。AA第三块与30步出现不同的人物朝向和运动轨迹；不能把两侧不同历史的第三块差异归因于单步velocity变化。

## 视频、协议与核验

- [AA 56帧：左V3 30步、右8步](submission/experiments/EXP-004_v3_8step/artifacts/videos/V3_30_vs_8_AA_56.mp4)；[AD 56帧同历史对照](submission/experiments/EXP-004_v3_8step/artifacts/videos/V3_30_vs_8_AD_56.mp4)。
- [AA 73帧闭环对照](submission/experiments/EXP-004_v3_8step/artifacts/videos/V3_30_vs_8_AA_73.mp4)；[AD 73帧闭环对照](submission/experiments/EXP-004_v3_8step/artifacts/videos/V3_30_vs_8_AD_73.mp4)。四条8步原片也保存在同一 [视频目录](submission/experiments/EXP-004_v3_8step/artifacts/videos/)。全部新帧与必要原分辨率细节已查看，坏帧未裁除。
- 原生8步sigma为 `1, 0.939540, 0.869452, 0.787234, 0.689441, 0.571184, 0.425287, 0.240781, 0`；没有截断30步sigma。Second chunk采样只读原始首窗KV；进入third前只将本轮自己的8步second clean endpoint在sigma=0提交一次；third无多余commit。future action/video在模型前物理裁剪，保留accepted current-prefix own-action路由、Single I0与全局RoPE。
- 四段源代码和条件摘要一致；第二块AA/AD的首窗latent、首窗KV、初始noise、anchor、audio、位置、sigma和旧RGB摘要相同，仅prompt动作摘要不同。采样前后历史cache的entry identity、storage address、tensor version与commit计数未变；RGB前39及随后前56帧在`.npy`层逐像素冻结。四原片与四并排片均通过H.264、24fps、帧数、尺寸、单调PTS和完整解码检查。详见 [来源与命令](submission/experiments/EXP-004_v3_8step/MANIFEST.md)、[视频审计](submission/experiments/EXP-004_v3_8step/artifacts/video_integrity.json)、[机器指标](submission/experiments/EXP-004_v3_8step/metrics.json)。

## 预算、成本与边界

实际 **32 sampling + 2 clean commit = 34次完整denoiser forward、4次VAE decode、0 optimizer update**。GPU 0顺序执行四个进程，共 **466.02 GPU-seconds = 0.12945 GPU-hours**，低于0.5 GPU-hours与90分钟elapsed上限；没有失败/自动重试。记录到的单进程最大 `torch.cuda.max_memory_allocated` **25,682.07 MiB**，第二/第三块采样前的CPU raw KV分别约 **6.799 / 9.632 GB**。commit约12.02/12.19秒；这些测量包含真实cache使用，sampling内的搬运未单独拆分。进程增量wall和采样时间见 [逐块账本](submission/experiments/EXP-004_v3_8step/artifacts/metrics/)；不包含首39帧旧生成成本。

Worker判断：执行协议有效，同历史第二块A/D动作差异和自身历史第三块结构有明确有限正结果；本轮8步画质及AA动作幅度较30步下降，AA后段残影与AD边界变化需保留。因此建议将**原权重直接8步的局部能力**标为 `PARTIAL`，由Judge独立决定是否接受为后续少步研究基线。尚未验证124帧、其它场景/seed、动作切换或完整端到端效率；不自动扩展至这些范围，也不把这项结果等同AnyFlow/Stage2。

~~~~
### Judge完整审核

~~~~markdown
# EXP-004 / v1：Judge 最终验收

2026-10-10，Asia/Hong_Kong。Decision: **accept**；Worker execution: **completed**；Judge acceptance: **accepted**。按当前可行性阶段与用户ROI要求，证据足以回答本轮问题，实验收口。

## 结论

V3原权重普通native FM/Euler的新块采样从30步减为8步后，AA/AD在共同30步首窗上保留可辨的动作响应，且各自接入自己的8步第二块后均续到73 RGB帧（24fps，3.04秒），人物/停车场基本可用。协议有效、严格因果与真实KV复用维持，零新增训练。

这支持**8步续写有限可行性**。首12 latent/39 RGB来自既有30步流程；全程8步初始化、8步124帧、首屏/完整E2E与跨scene/seed未测。普通FM减步证据不能称AnyFlow训练或DMD成功。正式30步V3的124帧定义不变。

| 能力/执行项 | 判定 | 范围 |
| --- | --- | --- |
| 执行、来源与预算 | PASS | 4条8步新块，2次own-history commit，实际资源闭合 |
| strict causal / persistent KV | PASS | 继承已验收图与路由，采样不改历史cache，无额外诊断 |
| 动作响应 | PASS（有限） | 第二块同history AA/AD对照；第三块仅闭环续写 |
| 基本人/场景结构 | PASS（有限） | 全部新增68帧检查，2条至73帧 |
| 画质、严格连续性 | PARTIAL | AA后段肢体透明拖影明显；软化、纹理重叠、姿态跳变 |
| 新块采样减步与观察成本 | PASS | 实际8步；旧30步计时作观察参考 |
| 全程8步、长片与公平E2E | NOT_TESTED | 不将借用30步首窗和不同时刻计时外推 |

## 数值与视觉

| 路径/新RGB | 8步flow | 旧30步flow | 8步sampling秒 | 旧30步sampling秒 | 边界灰度MAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| AA/39–55 | +0.780621 | +1.347308 | 70.869 | 142.913 | 3.539 |
| AD/39–55 | −1.509824 | −1.457716 | 40.524 | 135.687 | 2.814 |
| AA/56–72 | +0.978349 | +0.473785 | 37.764 | 155.140 | 4.895 |
| AD/56–72 | −0.732276 | −0.929845 | 42.510 | 146.390 | 7.953 |

Judge看过四块的完整新增帧静态序列及原分辨率末帧。第二块AA/AD人物场景基本可辨，有软化/背景纹理重叠；AA第三块后段透明肢体、腿部拖影/重复更明显，AD第三块局部腿部纹理与残影。未见持续整体人体崩坏，不能描述为无缺陷。flow只是运动代理，不单独作为动作语义评分。第三块历史已分化，方向差异不能称同状态反事实。

## 核验与成本

- Runner复用EXP-002 interval_cached与current-prefix；native 8步schedule含9个sigma端点、shift2.22不变。第二块不commit；第三块前只提交自己的8步第二块一次，末第三块不commit。
- 独立比对第二块AA/AD共同history/noise/cache/anchor/audio/global位置/旧RGB，当前动作条件不同；每条与旧30步输入条件匹配。第三块endpoint来源确属自己8步输出。
- 旧RGB前缀逐像素冻结、endpoint/RGB摘要、采样前后内存cache identity/storage/version与文件摘要、权重版本检查通过。Runner/config/interval/router当前hash与4个运行记录全部一致，详见[成本/来源核对](costs_checked.json)。
- 四个原片完整解码、24fps、帧数/PTS通过；四个30vs8并排片来源/输出hash、尺寸1664×560、帧数、24fps/PTS通过。检查了实际并排帧及文字：左30右8，RGB56起明确各自历史。见[comparison_checks.json](comparison_checks.json)。
- 32sampling+2commit=**34 denoiser forwards**，**4VAE**，0训练/额外模型诊断/首窗prefill/reencode。GPU0四个顺序进程均完成，**466.01794 GPU秒=0.12944943 GPU小时**；首进程至末进程结束573.55258秒，**峰值allocated25,682.066MiB**，低于34forward/4VAE/0.5GPU小时/90分钟/44GiB预算。
- 历史raw KV由首12的6,799,104,000 bytes增至through17的9,632,064,000 bytes，最后第三块不提交。sampling含cache读入/搬运，子项未拆分。旧30步为不同时刻共享硬件单次运行，不声称精确或公平E2E速度比；首窗原始成本未包含。

Worker建议将整体局部能力标为PARTIAL；Judge保留其原文，独立区分基本可行性PASS与画质/连续性PARTIAL，未更改Worker原判断。最终报告、manifest与CPU核验日志已审阅，13份小型source文件及29份产物hash匹配；大型LoRA/KV复用实际运行摘要与先前验证，不增加模型诊断。

## 决策与交付

接受本轮有限正证据，停止GPU与额外调参。保留Worker原报告/指标快照和原始日志，独立Judge结论不回写伪造旧报告。8步结果作为V3增量效率证据进入submission/report，30步V3正式基线继续保留。

下一项最有价值的问题：让首窗也使用8步，检查全程少步能否在自己的历史下成立，再决定是否有必要训练。建议短窗可用再有限延伸；不扫描步数/shift/gain，不为局部画质补大量实验。EXP-004预算已关闭，建议不构成EXP-005执行授权。

[实验说明](../README.md) · [原Worker报告](../worker_report.md) · [原Worker指标](../worker_metrics_snapshot.json) · [任务书](../taskbook_v1.md) · [在途逐块审核](S0_REVIEW.md)

~~~~

EXP-004正式结果提交`4a90b78623a728502347287d9dc0fff290bbb998`已正常推送origin/main，远端SHA核对一致，submission工作树干净；提交前fetch无分叉。提交包含代码/视频/真实日志/冻结报告和Judge审核，未包含大型权重或缓存。


## 2026-10-10：用户调整研究分类，原V3归入V2c

V2a RGB Anchor、V2b Same-σ、V2c Strict Causal + Persistent KV均归类为修复V1视觉与动作退化的并行路线，三者没有顺次权重继承关系。原V3 Efficient Causal的结果现称V2c；未来V3尚未定义。

report与mainline增加v2家族层级，V2c内收纳73帧证据和8步续写。EXP-002/003中性目录不变、当前标题标为V2c；EXP-004目录改为EXP-004_v2c_8step，旧路径为兼容符号链接。历史任务书、Worker报告、Judge审核、日志、原始指标、代码与实验视频保持原文/字节，现行摘要使用新分类。之前出现的旧V3须按历史日期区分：早期曾指现V2b，近期Efficient Causal指现V2c。

迁移前现行文档快照、新旧路径与SHA记录见[分类迁移档案](submission/archive/taxonomy_v2c_20261010/README.md)。本次零模型推理/训练，仅CPU更新两段展示片的标题；原视频仍保留。根目录仍为六份Markdown。

V2c分类与目录迁移提交`575530a3b5d660f8ac3559502a848d08c3e2a1b1`已推送origin/main，远端SHA一致，submission工作树干净。471个现行本地链接通过；132个迁移文件中115个逐字节保持，冻结证据与正式124帧视频摘要通过；仅两段8步展示片在CPU侧更新标题，无模型推理/训练。


## 2026-10-10：恢复V3 Baseline，EXP-005阶段一CPU收口

用户最新决定恢复V3 Original Feasibility Baseline，独立新增V3-SW-G/SW-L。EXP-002/003已验收结果与冻结源码/视频保持不变；report/mainline改为v3家族三级入口，V2只保留a/b，兼容旧实验路径。迁移清单见[档案](submission/archive/v3_baseline_restore_20261010/README.md)。

EXP-005/v1只授权CPU。Worker完成显式12→5分块、精确五祖先cache校验、Global/Local入口与14项测试；Judge独立复跑14 passed in 3.74s，接受已测CPU范围。没有GPU forward/VAE/训练/生成视频，两个SW候选生成能力NOT_TESTED。原生>37条件fixture与GPU runner仍未认证，入口继续关闭；阶段二提案未授权。

[冻结任务书](submission/experiments/EXP-005_v3_sliding_window/taskbook_v1.md) · [Worker完整报告](submission/experiments/EXP-005_v3_sliding_window/worker_report.md) · [Judge审核](submission/experiments/EXP-005_v3_sliding_window/judge/FINAL_REVIEW.md) · [GPU预算与停止条件](submission/experiments/EXP-005_v3_sliding_window/GPU_PLAN.md) · [独立FM8/AF计划](submission/experiments/EXP-005_v3_sliding_window/FUTURE_ANYFLOW.md)。

核心提案G0→G1→L1共281forward、9VAE、1.70GPU小时，分段批准；C9默认关闭。当前预算仍0，项目总卡数≤3。按ROI与可行性尺度审核，不为普通画质瑕疵增加无限消融。

本轮Git交付：`bd5711d18b4d35ec8711d52c39c205bd2a96264e`已正常推送origin/main并核对远端SHA一致；提交前fetch无分叉，提交后submission工作树干净。未提交权重、原始KV/latent或字节码。
