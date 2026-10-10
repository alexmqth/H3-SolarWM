# 当前项目进展

更新：2026-10-11 05:00 HKT，Judge。可行性与ROI优先，普通画质缺陷如实记录。

**当前任务：EXP-010/v1，全程普通FM8 + SW-G 124/158帧联合验证，当前仅CPU准备、GPU须分阶段放行。EXP-009已完成：FM4 AA/AD73续写有限可行PASS/quality PARTIAL；AF4 AD C3持续人物分解FAIL，当前配置停止。EXP-008 DMD配置已按生成FAIL归档。** 详见[next_plan](next_plan.md)，夜间连续监督至09:00。

## 正式结论

**V3 Original Feasibility Baseline继续作为已验收正式参考。** EXP-002/003：Original H3 + released Action LoRA，Single I0/native timestep/own-action/action feedback/current-prefix feedback，strict causal + persistent raw KV，30步Global RoPE，AA/AD六块124帧，零新训练。瞬态人体形变、连续性PARTIAL，单停车场seed13；冻结证据未覆盖。

**EXP-005/v2已完成并审核：SW-G通过158帧有限可行性；SW-L工程通过、生成PARTIAL，无已证实收益，当前无训练Local方向归档。**

| 版本 | 已有结果 | 当前定位/限制 |
| --- | --- | --- |
| V0 Original H3 | 原始双向动作/视觉参考 | 非strict causal |
| V1 Native causal | 分块/KV工程可行 | 动作与视觉未同时通过 |
| V2a RGB Anchor | 124帧基本结构 | 动作失败，20秒后段崩坏 |
| V2b Same-σ | 持续AA/DD124帧动作/基本结构可行 | 历史双向重算，无persistent KV，切换PARTIAL |
| V3 Baseline | AA/AD124帧strict causal/KV/动作/结构可行 | 正式参考，冻结 |
| V3-SW-G | A继续/D切换各158帧，真实最近5祖先淘汰 | 优先滑窗候选；边界跳变、D-C8持续拖影，quality PARTIAL |
| V3-SW-L | 同窗口158帧，动作方向保留、cache正确 | 场景几何/亮度跳变更明显，无联合收益，归档 |
| V3-FM8 | 新8步首39+AA/AD73有限可行性 | 零训练，quality PARTIAL，普通少步对照 |
| V3-FM4 / AF4 | FM4共同FM8首窗后AA/AD73有限可用；AF4 AD C3失败 | EXP-009收口，FM4 quality PARTIAL；不扫更低步数 |
| V3-DMD | cycle8 AA C2全17新增帧彩噪，生成FAIL | 当前配置归档；AD中断未评估，C3未执行 |
| V3-AF | 32updates后8NFE AA/AD73续写有限可行性 | 共同FM8首窗，quality PARTIAL，尚无优于FM8整体证据 |

[V3家族](submission/report/v3/README.md) · [V2家族](submission/report/v2/README.md) · [浏览器视频入口](submission/report/index.html)

## EXP-005本轮证据

- 阶段一14项CPU测试已独立复核。显式长分块/精确indices/未来行隔离/Local位置与prefix约束通过。
- G0真实模型两状态旧/新入口差异0；30-step C6 endpoint与全部124RGB逐值复现冻结V3。
- G1首次clean commit C6淘汰C1；C7祖先C2–C6，C8为C3–C7，全部50层检查。历史video KV恒定14,164,800,000 bytes（13.19GiB），旧RGB均不变。
- Global A C7/C8 flow +0.819/+0.768；D −1.575/−1.185。方向和首新增块切换响应可辨。Local相应+1.120/+0.343、−0.598/−1.269；主体仍可辨，但重复几何、亮度闪变更多。
- 同G历史/C8 noisy state的Global/Local velocity relative RMS差异18.25%；历史不变。Local不是无损重定位，也不等价重算历史。

[Judge最终审核](submission/experiments/EXP-005_v3_sliding_window/judge/STAGE2_FINAL_REVIEW.md) · [实际Worker报告](report.md) · [三个候选协议](submission/experiments/EXP-005_v3_sliding_window/judge/STAGE2_PROTOCOL.md) · [完整证据目录](submission/experiments/EXP-005_v3_sliding_window/README.md)

## 成本与边界

本轮单GPU0顺序运行，实际**281forward、9VAE、0训练、0.662707GPU小时**，低于1.70GPU小时；GPU allocated峰值26.621GiB，核心elapsed48.49分钟。EXP-005作业均已结束，无C9/重试。当前用户夜间新授权见下，至10月11日09:00不设人为项目卡数上限，之后最多3卡。

仅验证158帧/24fps约6.58秒与两次真实淘汰；不能声称无限长、跨scene/seed稳定。只历史video KV有界，prefix、latent/RGB保存、VAE全前缀解码仍可增长。长fixture显式保持旧37条件，并非默认H3长packed重建。阶段增量时间不是从零生成全片E2E，也不支持公平speedup排名。

## 已完成：EXP-007 V3-AF

EXP-006已经完成：**新8步首39+AA/AD73有限可行性通过，quality PARTIAL**。方向C2为+0.846/−0.321、C3为+0.505/−0.826；人物/场景可用，边界跳变与残影保留。43forward/5VAE/0训练/0.136905GPU小时，allocated peak26.061GiB。[Judge验收](submission/experiments/EXP-006_v3_fm8_full/judge/FINAL_REVIEW.md) · [V3-FM8视频](submission/report/v3/v3_fm8/README.md)。不继续扫步数或强行扩长。

已完成的EXP-007/v3：AF1真实模型单步训练工程通过，初始化diagonal完全一致、target/QKV均有效更新，基础权重不变，更新后KV须自建。Judge独立核对配对checkpoint/RNG/optimizer/hash全部通过。两attempt累计22forward/4backward/1update/0VAE/186.58153秒，峰值26.63145GiB。[AF1审核](submission/experiments/EXP-007_v3_anyflow/judge/AF1_REVIEW.md)。AF2已完成新增31updates并通过工程审核；AF3匹配FM8历史与噪声的8NFE视频已完成，有限续写可行性通过、quality PARTIAL，旧训练产物不作新V3-AF证据。

用户授权Judge持续到2026-10-11 09:00，阶段完成后自主审核/布置下一项任务，不重复请求用户批准。期间按已批预算使用实际空闲卡，不人为限制项目总卡数，不抢占他人进程；09:00后最多3卡。V3-AF真实warmup通过后再批有限训练与匹配8NFE视频；DMD在基础student可用后独立推进，保留[协议预审](submission/experiments/EXP-007_v3_anyflow/FUTURE_DMD.md)。

## 历史与交付

EXP-001 V2b124、EXP-002/003 V3 Baseline124、EXP-004普通FM8续写73（首窗30步）、EXP-005阶段一CPU均已归档。历史提交与结果见[archive.md](archive.md)。EXP-005提交`7639bea7cc0fb9e3b27295ce87422cf2d06c076e`已正常推送origin/main，本地与远端SHA一致，当时submission工作树干净。505个本地链接、79项源码/基线摘要、45项产物摘要通过；未提交权重、大KV/latent/npy或字节码。

EXP-006正式结果与EXP-007 AF0准备已发布：`e2ac8d091d3f3c647de0aac2b1cac4df84813daa`，正常推送origin/main并独立核对远端SHA一致。73文件约9.28MB；394现行链接、17小型源码摘要通过。EXP-007后续运行文件单独归档。

EXP-007/v2 AF1结果、Judge审核及v3任务书已发布：`e9a65945d9e8650c7bbd0ff2f9c6c55771e29df1`，正常push并独立核对origin/main SHA一致。22文件；11归档/外部checkpoint摘要、21新增导航链接通过，原始log空白保持原样，其他diff检查通过。

AF2于约02:09 HKT在GPU1启动并已完成退出，原PID `4087679`。输出`H3-World/outputs/EXP-007_v3_anyflow_af2/`，日志`H3-World/outputs/EXP-007_af2_run.log`；step1配套checkpoint/optimizer及CPU/CUDA/logical RNG恢复、step32最终产物均已审核。

后续DMD CPU准备已完成：[独立tiny-H3核查](submission/experiments/EXP-007_v3_anyflow/dmd_cpu/README.md)。实际8-map连续梯度、teacher/fake/student角色隔离、同renoise评分符号、各权重独立KV和fake更新后重建通过，2.613秒/0GPU；不代表33B DMD能力，也没有新增DMD GPU授权。

02:26 HKT：AF2累计step8 checkpoint已保存并经Judge独立CPU审核：target/QKV/optimizer配对SHA、20项optimizer step8、RNG、AA/AD顺序、每步17forward/4backward和cache检查通过，运行源码摘要未变化。前7次续训平均137.37秒，peak26.76744GiB。训练继续至原上限，尚无AF2视频结论。[中间checkpoint审核](submission/experiments/EXP-007_v3_anyflow/judge/af2_step8_audit.json)。

## AF2完成，AF3视频放行（03:21 HKT）

AF2最终step32工程审计通过：新增31updates、527forward/124backward/0VAE，1.191072GPU小时，peak26.76744GiB；step8/32配对权重/optimizer/RNG与所有账本检查通过。[Judge审核](submission/experiments/EXP-007_v3_anyflow/judge/AF2_REVIEW.md)。已批准最终step32匹配FM8的8NFE AA/AD73续写，35forward/4VAE/0.35GPUh；当时尚无新AF视频能力结论；后续03:36 AF3审核已完成。

## 2026-10-11 03:36 HKT：EXP-007 AF3收口

step32新AnyFlow在共同FM8首窗后的AA/AD73帧8NFE续写有限可行性通过，quality PARTIAL，尚无整体优于FM8的证据。独立来源/账本/缓存/匹配输入/历史RGB/全片解码审计通过，Judge已看全部68张新增帧与边界。35forward/4VAE/0update/473.000176秒（0.131389GPUh），peak26.33646GiB，含首次导入失败后明确授权重试。D分支C3幅度较弱、边界跳变与残影保留，不追加扫参。[正式审核](submission/experiments/EXP-007_v3_anyflow/judge/AF3_REVIEW.md) · [报告与视频](submission/report/v3/v3_anyflow/README.md)。下一项EXP-008独立DMD单cycle，当前CPU迁移准备，GPU待代码/预算冻结。

## 当前执行：EXP-008/v1 DMD单cycle（03:40 HKT）

AF3已验收后，Judge独立CPU预检及迁移审查通过，正式批准三卡0/2/5单cycle：17forward/3backward/3update/0VAE，≤30min wall/≤1.5GPUh，每卡44GiB。完整8map链保持梯度，各角色独立KV，不自动扩展。授权及冻结来源见[GPU放行](submission/experiments/EXP-008_v3_dmd/GPU_RELEASE.md)与[next_plan](next_plan.md)。

EXP-007 AF2/AF3完整结果与EXP-008冻结pilot任务已发布：`aff206b9f14d18c7a91c0b4eb553d36166afad8b`。正常push后独立核对origin/main相同，83文件；14归档摘要、95导航链接和两条对比视频完整解码通过，不含基础权重或大KV。发布时submission工作树干净，DMD运行输出仍在外部outputs。

## EXP-008 pilot完成，准备有限延续（03:46 HKT）

真实33B三角色单cycle工程通过，Judge独立审计全PASS，进程849858已退出。17forward/3backward/3update/0VAE，254.898624秒，三卡保守0.21241552GPUh；teacher/fake/student峰25.070/25.879/27.965GiB。8map梯度与target/QKV有效更新、权重/optimizer/RNG配对验证通过，尚无DMD视频能力结论。[审核](submission/experiments/EXP-008_v3_dmd/judge/PILOT_REVIEW.md)。

v2拟新增7cycles到累计8、99forward/14backward/14update/≤2.25GPUh，随后匹配视频另放行。当前CPU实现/恢复预检，训练须独立marker；不凭pilot授权自动扩展。

## EXP-008/v2训练放行（03:52 HKT）

Judge独立CPU预检及代码审查通过，29项来源冻结，新增7cycles至累计8已批准：GPU0/2/5、99forward/14backward/14update/0VAE、45min wall/保守2.25GPUh。独立marker绑定manifest；视频待训练审核另放行。[训练放行](submission/experiments/EXP-008_v3_dmd/TRAIN_RELEASE_V2.md)。

DMD v2已于约03:53在GPU0/2/5实际启动，PID1017778，输出`H3-World/outputs/EXP-008_v3_dmd_train_v2/`；继续原99forward/14backward/14update及2.25GPUh上限。

DMD pilot完整结果、Judge审核与v2冻结延续任务已发布：`8450c423e42ba42d4851ae18733cadecb6fd4276`，正常push后独立核对origin/main一致；20文件，原始日志/JSON副本字节一致、25条pilot导航链接通过。无权重或大型KV进入Git。发布时工作树干净，训练进程1017778继续运行。

04:05 HKT：DMD v2累计cycle4 checkpoint已保存，Judge独立CPU核对四文件SHA、元数据、student/fake optimizer step4/5、cycle2–4训练噪声精确重放及两generator状态，全部通过。[中间审核](submission/experiments/EXP-008_v3_dmd/judge/cycle4_audit.json)。已完成追加cycle耗时211.64/234.72/230.50秒，student peak约28.10GiB；训练进入cycle5，仍只评估最终checkpoint，无中途视频扫选。

## DMD v2训练完成、视频放行（04:21 HKT）

累计student cycle8已完成并经Judge全量独立审计通过：新增99forward/14backward/14update/0VAE，1673.640974秒，三卡1.394700812GPUh，student峰28.10182GiB。全7轮噪声可从pilot RNG精确重放，cycle4/8配套权重/optimizer/RNG通过。Fake loss曾出现31.53尖峰，后约2.5；不把surrogate下降当画质收益。[审核](submission/experiments/EXP-008_v3_dmd/judge/TRAIN_V2_REVIEW.md)。

最终cycle8评估CPU preflight通过，现批准最多35forward/4VAE/.35GPUh，先AA/AD C2；若持续主体/场景崩溃则停止C3并收口。普通缺陷按PARTIAL。生成能力尚待视频。[放行](submission/experiments/EXP-008_v3_dmd/EVAL_RELEASE.md)。

DMD v2完整训练结果与最终cycle8评估授权已发布：`9cfde9482bcf87862f44ede350f884a3fbcec04d`，正常push后独立核对origin/main一致。23文件；训练原始log/JSON副本逐字节一致、32条导航链接通过，未提交checkpoint或大KV。评估首窗prefill已成功，AA C2正在运行。

## 04:35 HKT：DMD生成失败收口，准备4NFE方法对比

最终cycle8 AA C2新增17帧全为彩色噪声，停止当前配置，未追加训练或筛checkpoint。AD第6个sampling forward中断，5步完成，无完整端点/视频；C3未启动。来源/执行审计通过但生成FAIL。[Judge正式结论](submission/experiments/EXP-008_v3_dmd/judge/FINAL_REVIEW.md) · [三列负结果视频](submission/experiments/EXP-008_v3_dmd/dmd_eval/artifacts/failure_comparison/AA_FM8_AF8_DMD8_cycle8_collapse_56.mp4)。评估16已启动/预记账forward（1中断）、1VAE、0.05697321GPUh；整个DMD任务约1.66409GPUh。仅否定本有限训练配置，不推广为所有DMD不可行。

下一EXP-009固定原AF2 step32、普通FM4、共同FM8 C1/seed13，AA/AD最多73帧，零新训练。候选总预算38forward/8VAE/.6GPUh；目前仅CPU准备，待Judge审查与marker。直接检验4NFE收益，不继续修补DMD失败配置。

DMD负结果与EXP-009 CPU任务书已发布：`6811a2eb754aa72bcd2f7988c5e34147ced62b53`，正常push并独立核对远端main一致。33文件，15归档SHA、19导航链接、56帧24fps三列视频完整解码/PTS核对通过，无权重或大型KV。

EXP-009独立CPU预检通过，28项来源已冻结；GPU0 C2阶段22forward/4VAE已放行，总任务上限38forward/8VAE/.6GPUh不变。[放行](submission/experiments/EXP-009_v3_fm4_af4/GPU_RELEASE.md)。

04:51 HKT：EXP-009 C2四组完成且独立执行审计PASS，22forward/4VAE/.1242915GPUh。FM4 AA/AD有限可用；AF4 AD方向弱/含混、纹理扭曲加重，尚无AF4整体优势。仅放行原计划C3，最多新增16forward/4VAE，总.6GPUh不变。[C2审核](submission/experiments/EXP-009_v3_fm4_af4/judge/C2_REVIEW.md)。

## 05:00 HKT：EXP-009正式收口，EXP-010 CPU准备

普通FM4在共同FM8首39帧后的AA/AD73续写有限可行PASS，quality PARTIAL。AF4 AA保留动作但持续透明，AD C3人物分解成大片透明碎片且响应近消失，生成FAIL，不接受为联合可行或改进版本。28项来源、各模型自有KV、匹配noise/action、历史RGB不变与所有视频完整解码审计通过，38forward/8VAE/0update/782.290028秒=.217302786GPUh、峰26.33646GiB。停止该AF4配置，不追加2NFE或训练。[最终审核](submission/experiments/EXP-009_v3_fm4_af4/judge/FINAL_REVIEW.md)。

EXP-010准备复用EXP-006 AA73、全程普通FM8继续AA124后晚分叉A继续/D切换至158，测试真实KV淘汰和少步推理的组合。最多62forward/7VAE/.75GPUh，分124与158两阶段GPU审查，目前仅CPU授权。现有30步SW-G原片复用，基线冻结。

EXP-009完整结果与EXP-010 CPU任务已发布：`ddf7598af18f46c30e85f2fd9bf32c67706e8682`。正常push后独立核对远端main一致；89文件，35归档SHA、42导航链接、两条73帧24fps四宫格完整解码与PTS通过，最终账本与C3原日志副本逐字节一致。未提交权重、大KV或npy。
