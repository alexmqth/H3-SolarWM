# 当前项目进展

更新：2026-10-11 07:49 HKT，Judge。当前为可行性验证阶段，ROI优先；接受明确边界下的PARTIAL，持续结构失败的配置停止归档。

**当前任务EXP-014：四train初图的native V3 FM30教师目标。P0/P1与首图T1均验收，余三图T2在GPU0/1/2并行生成，逐图审核，新训练0。** EXP-013数据审计已验收；EXP-012当前AF8 step32在两场景无联合迁移收益，停止扩展。普通FM8保留低成本对照。持续监督至2026-10-11 09:00 HKT。[当前任务书](next_plan.md)。

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
| V3-AF8 / EXP-007、EXP-012 | 停车场8NFE AA/AD73有限可行；两场景匹配历史迁移无联合收益 | 工业D切换PARTIAL，村落AA结构FAIL；当前step32停止扩展，普通FM8优先 |
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

EXP-011最终：新native Single I0的工业/村落FM30与FM8四配置首39及各自AA/AD56均有限可行通过、quality PARTIAL。Judge检查156首窗帧+136新增帧及原分辨率细节；四套实际50层C1 KV、原39RGB不变、动作/噪声/native网格与完整视频审计PASS。村落FM8颗粒/植被和人物边缘重影明显更多；动作分叉仍可辨。232forward/12decode，加6text/2image encode，推理1781.710秒、P1含失败70.565秒，总0.514521GPUh，零训练。两固定初图/seed13/56帧，G2各方法自身历史，非广泛泛化或同KV步数消融。无C3。[最终验收](submission/experiments/EXP-011_v3_scene_transfer/judge/FINAL_REVIEW.md) · [视频](submission/report/v3/v3_scene_transfer/README.md)。

## 资源与协作规则

用户授权Judge自主审核/布置下一阶段并持续监督到2026-10-11 09:00 HKT；期间使用实际空闲卡，不人为限制总卡数、不抢占他人进程。GPU3/4目前有他人VLLM进程，不操作。09:00后项目最多3卡；新GPU任务必须有编号、预算、冻结输入与Judge分阶段授权，无需用户重复批准已授权范围。

Judge管理next_plan/progress/archive与Git发布，现有Exp Worker实现/执行/提交报告；不创建新agent。普通画质缺陷不机械卡死，协议真实性必须成立。负结果归档，不为占满GPU追加实验。正式结果均冻结，根目录保持六个管理Markdown文件。

## 最新发布与历史

EXP-012完整负结果已发布`efff5c535912b52b53a507da9d0b9ba4ab2ceb53`，63文件，normal push及独立远端main核对一致。39项归档、28个原始副本、33链接、4比较片解码及内容对位通过。P0代码/工业放行发布`f9bf33406de252e15dbf88a76134ff12c7495b40`。

EXP-011完整迁移验收与8比较片发布`43c56e85bb4e42b0ca38a9ecfb547be09e5fcf43`；EXP-010完整FM8+SW-G158证据`3db745dec1c1d0a4ebd136ed9237012b45a15614`。完整历史见archive与各实验Judge文件。

## 最新AF8迁移结论

EXP-012两固定其他场景、共同FM8 C1、native8NFE：工业AA/AD flow+28.14/+11.07，AD反向响应未证实；村落+70.33/−68.63，AD可用但AA后半人物/前景大片分叉透明残影、结构FAIL。两套实际50层cache与匹配条件/旧RGB/视频审计PASS。34forward/4decode/354.403秒=.098445GPUh，0训练；不把工程PASS当能力PASS，不追加C3/训练/调参。[最终审核](submission/experiments/EXP-012_v3_af8_scene_transfer/judge/FINAL_REVIEW.md)。

## EXP-013数据准备验收

24个39帧clip、936帧完整解码/原源二值动作独立核查PASS；4train episodes/16clip，2validation episodes/8clip，源和episode隔离。四初图按最早A固定，不按质量挑选。真实录屏有A+S/L/J组合，未来纯A/D teacher是反事实伪标签；旧Dual Anchor编码不能复用，现有clip没有C2 GT。独立AnyFlow训练仍为草案，新评估优先复用FM8参考，将新增预算控制为126forward/15decode。0GPU/0训练。[正式验收](submission/experiments/EXP-013_v3_multiscene_data_plan/judge/FINAL_REVIEW.md)。

EXP-013数据审计与EXP-014任务书已发布`a216ca863bdb4de62ab7a9dc3bf467f46fe63449`，24文件；正常push及独立ls-remote确认一致。26个报告链接检查无缺失，归档数据与已独立审计摘要相同。EXP-014仅P0 CPU准备获批，尚无GPU marker。

## EXP-014 P1编码运行（07:34 HKT）

P0独立预检与真实runtime摘要PASS；GPU0 P1编码四train图已启动，12text/4image预算，600GPU秒，0denoiser/0训练。T1/T2未授权。代码清单SHA `5afe075f4db7e5b6b107ffabbfcd1feb408dbccd173649b6548c74f876173331`。[P0验收](submission/experiments/EXP-014_v3_multiscene_teacher/judge/P0_REVIEW_AND_P1_RELEASE.md)。

## EXP-014 P1验收/T1已启动（07:37 HKT）

四native fixture独立CPU验收PASS，12text/4image encode实际110.213939GPU秒，峰40.577766GiB，无重试。首图s0_43866101的FM30 C1+AA/AD C2已在GPU0运行，91forward/3decode/1350秒上限；余三图T2未放行，新训练0。[P1审核](submission/experiments/EXP-014_v3_multiscene_teacher/judge/P1_REVIEW_AND_T1_RELEASE.md)。P0代码与编码授权已发布`2244dbcfb84feba79dd8c18846febf0e6c4f8272`，远端核对一致。

EXP-014 P1归档与首图授权已发布`928286da635119ef61e7411af1ddb4c82be70321`，12文件、远端main摘要核对一致；5份P1原始副本逐字节相等，6个报告链接有效。

## EXP-014首图T1验收/余三图运行（07:49 HKT）

首图39+17+17全帧及原分辨率检查通过，AA/AD flow辅助+42.147/−27.459，人物/场景可辨，teacher有限可用/quality PARTIAL。实际50层index0、同C1/noise/Global位置、旧39RGB不变与视频核查PASS。91forward/3decode/532.260704GPU秒，峰38.181200GiB。余三固定图T2已获marker并在GPU0/1/2独立并行，逐图验收；0训练。[T1审核](submission/experiments/EXP-014_v3_multiscene_teacher/judge/T1_REVIEW_AND_T2_RELEASE.md)。
