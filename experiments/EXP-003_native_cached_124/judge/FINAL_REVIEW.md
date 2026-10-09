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
