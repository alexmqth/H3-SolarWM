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
