# 当前项目进展

更新：2026-10-11 05:42 HKT，Judge。当前为可行性验证阶段，ROI优先；接受明确边界下的PARTIAL，持续结构失败的配置停止归档。

**当前唯一执行任务：EXP-011/v1，两个固定 ABot 初图的 V3 FM30 / FM8 短程迁移，P0独立CPU审核通过，P1输入编码已放行；G1/G2未放行。EXP-010 已验收：全 FM8 + SW-G 的 A继续/D晚切换各158帧有限可行PASS，画质PARTIAL、首切换块响应PARTIAL。** 任务书见[next_plan.md](next_plan.md)，Worker执行见[report.md](report.md)。用户授权持续监督到2026-10-11 09:00 HKT，尚未到期。

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
| V3-FM8 / EXP-006、EXP-010 | 从首窗8步AA/AD73；共享AA124后A继续/D晚切换各158，真实淘汰 | 有限可行PASS，quality PARTIAL；首晚切换块弱，第二块反向响应可辨 |
| V3-AF8 / EXP-007 | 新target-time student累计32updates，8NFE AA/AD73续写 | 共同FM8首窗，有限可行PASS/quality PARTIAL，尚无整体优于FM8证据 |
| V3-DMD8 / EXP-008 | 累计8 student cycle训练工程PASS；AA C2全部17新帧彩噪 | 生成FAIL，当前配置归档；AD中断未评估、C3未执行 |
| V3-FM4 / EXP-009 | 共同FM8首窗后AA/AD73续写可用 | 有限可行PASS，quality PARTIAL；4步首窗和长时未测 |
| V3-AF4 / EXP-009 | AA透明化加重，AD C3持续人物分解、响应近消失 | AD C3生成FAIL，无联合收益，当前4NFE配置归档 |

V2a/V2b为并行视觉/动作修复路线，没有权重继承关系。用户最终保留V3 Baseline正式命名；短暂V2c命名只存历史记录。普通FM减步、AnyFlow finite-map训练、DMD分别评价，旧协议AnyFlow产物不是新V3-AF证据。

[V3家族](submission/report/v3/README.md) · [V2家族](submission/report/v2/README.md) · [浏览器视频入口](submission/report/index.html)

## 关键已验收证据

### SW-G / SW-L

EXP-005：14项CPU正确性、实际模型G0两状态零差异、C6 endpoint与124RGB逐值复现冻结V3通过。C6 commit淘汰C1，C7读取C2–C6，C8读取C3–C7；所有50层核查，历史video KV恒定14,164,800,000 bytes（13.19GiB），旧RGB不变。

Global A C7/C8 flow +.819/+.768，D −1.575/−1.185，动作与切换可辨；保留边界跳变和D C8拖影。Local动作仍存在，但重复场景/亮度变化更多。同Global历史/C8 noisy state下Local velocity相对RMS差异18.25%，不是无损重定位或历史重算等价。

实际281forward/9VAE/0训练/.662707GPUh，allocated峰26.621GiB。只证明158帧约6.58秒和两次淘汰；仅历史video KV有界，prefix、latent/RGB与全前缀VAE成本仍可增长。冻结long47 fixture不等价默认长packed重建。[最终审核](submission/experiments/EXP-005_v3_sliding_window/judge/STAGE2_FINAL_REVIEW.md)。

### 普通FM8

EXP-004曾仅测30步首窗后的8步续写；EXP-006补上首12 latent也新8步生成，AA/AD自己的历史续写至73帧。C2 flow +.846/−.321，C3 +.505/−.826；人物/场景可用，边界与肢体残影保留。43forward/5VAE/0训练/.136905GPUh，峰26.061GiB。[正式审核](submission/experiments/EXP-006_v3_fm8_full/judge/FINAL_REVIEW.md) · [视频报告](submission/report/v3/v3_fm8/README.md)。

### AnyFlow8与DMD

EXP-007 AF1单更新真实模型工程通过；AF2恢复到累计step32，新增31updates、527forward/124backward/1.191072GPUh。AF3固定step32、匹配8NFE继续AA/AD73，有限可行PASS、quality PARTIAL，但没有优于FM8整体证据；35forward/4VAE/.131389GPUh（含首次导入失败），峰26.336GiB。[AF2审核](submission/experiments/EXP-007_v3_anyflow/judge/AF2_REVIEW.md) · [AF3审核](submission/experiments/EXP-007_v3_anyflow/judge/AF3_REVIEW.md)。

EXP-008从AF2 step32启动真实33B三角色DMD，完整8-map反向、角色隔离、自身KV与checkpoint/optimizer/RNG工程通过。最终cycle8 AA C2新17帧全彩噪，生成FAIL；AD完成5 sampling、第6forward中断，无端点/视频，为NOT_EVALUATED；C3未执行。总约1.66409GPUh，其中最终评估16已启动/预记账forward含1中断、1VAE/.056973GPUh。只否定当前有限配置，不推广为所有DMD方法失败，不继续加训练或筛checkpoint。[最终审核](submission/experiments/EXP-008_v3_dmd/judge/FINAL_REVIEW.md) · [负结果视频](submission/report/v3/v3_dmd/README.md)。

### 4NFE对比

EXP-009固定普通FM4 vs AF2 step32 AF4，共同FM8首39，各自KV和C3历史。FM4 C2 AA/AD flow +.706/−.190、C3 +.336/−.618，有限可用/quality PARTIAL。AF4 C2 +.489/+.162、C3 +.268/−.0075；AD C3持续人物分解成大块透明碎片，生成FAIL。Judge看过8段全部136张新增帧和必要原分辨率细节，不仅依赖proxy。当前AF4配置停止，无2NFE或新增训练；AF8有限可行结论保持。

28项来源、各自KV、匹配输入、旧RGB不变与完整解码审计通过。38forward/8VAE/0update/782.290028秒=.217302786GPUh，峰26.33646GiB。[最终审核](submission/experiments/EXP-009_v3_fm4_af4/judge/FINAL_REVIEW.md) · [完整视频报告](submission/report/v3/v3_fm4_af4/README.md)。

## 最新FM8滑窗与当前任务

EXP-010完整验收：自有全FM8 AA73续到AA124，再C7/C8 A继续/D晚切换到158；C7 flow A+.439/D+.034，C8 A+.587/D−.858。人物/车库可用，透明腿残影与场景边界变化；D首切换块弱，次块响应更清楚。真实50层KV indices1–5再2–6，恒定14,164,800,000 bytes，旧RGB不改写。新增62forward/7VAE/0训练/1137.108658秒=.315863516GPUh，峰26.59566GiB。仅停车场seed13、158帧6.58秒、冻结long47位置，不宣称无限长度或公平E2E速度。[正式审核](submission/experiments/EXP-010_v3_fm8_sw158/judge/FINAL_REVIEW.md)。

EXP-011：P0已审核，P1仅6text/2image encode已放行；G1/G2待审。任务为：固定现有validation中的户外工业与中世纪村落两张初图，重新构建native Single I0，比较V3 FM30/FM8首39及各自AA/AD到56；不复用旧Dual Anchor编码、不称绝对未见或现实世界数据。预处理、首窗、续写各需Judge放行。总上限232forward/12decode/0训练，编码≤.25GPUh、推理≤1.25GPUh，09:00截止；失败配置停止续写，不挑图/调参/扫seed。详见[next_plan.md](next_plan.md)。

## 资源与协作规则

用户授权Judge自主审核/布置下一阶段并持续监督到2026-10-11 09:00 HKT；期间使用实际空闲卡，不人为限制总卡数、不抢占他人进程。GPU3/4目前有他人VLLM进程，不操作。09:00后项目最多3卡；新GPU任务必须有编号、预算、冻结输入与Judge分阶段授权，无需用户重复批准已授权范围。

Judge管理next_plan/progress/archive与Git发布，现有Exp Worker实现/执行/提交报告；不创建新agent。普通画质缺陷不机械卡死，协议真实性必须成立。负结果归档，不为占满GPU追加实验。正式结果均冻结，根目录保持六个管理Markdown文件。

## 最新发布与历史

最新已发布EXP-010完整158证据与EXP-011 CPU任务：`3db745dec1c1d0a4ebd136ed9237012b45a15614`，44文件，正常push且独立核对远端main一致。22项branch归档SHA、72导航链接、原始日志/账本副本与两条完整对比视频核查通过；未提交大型tensor或权重。

此前EXP-009完整结果为`ddf7598af18f46c30e85f2fd9bf32c67706e8682`，DMD负结果为`6811a2eb754aa72bcd2f7988c5e34147ced62b53`。完整历史见[archive.md](archive.md)；早期详尽进展见[历史快照](submission/experiments/EXP-010_v3_fm8_sw158/management_snapshot/progress_before_20261011_0505.md)。
