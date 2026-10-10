# Project Progress

更新：2026-10-10（Asia/Hong_Kong），Judge。当前为可行性阶段，按用户ROI要求收口。

## 1. 当前结论

**V2c Strict Causal KV H3-World可行性版本已完成验收。** EXP-002/003使用同一个Original H3 + released action LoRA/native Single I0/current-prefix协议，结合strict chunk-causal、真实persistent raw video KV、同history A/D响应与AA/AD自身历史六块124帧（5.17秒）。没有新增训练。

AA约RGB79–86有明显人体形变/游离肢体残影，第五/六块恢复；AD有短暂肢体异常。基本多窗口结构可用，严格连续性/画质成熟度PARTIAL。没有成熟质量、跨scene/seed、10/20秒或公平Original从零E2E加速结论。

## 2. 主线状态

| 版本 | 现有证据 | 主要限制 |
| --- | --- | --- |
| V0 Original | 原始双向H3 + released LoRA，动作/视觉参考 | 非严格因果AR，非GT |
| V1 Native causal | strict chunk-causal / persistent KV工程可行 | 动作与视觉未同时通过 |
| V2a RGB anchor | 124帧视觉结构改善，已有适配 | A/D失败，20秒后段崩坏 |
| V2b Same-σ local bidir | EXP-001持续AA/DD124帧动作/基本结构可行 | 无persistent hidden KV；切换PARTIAL，历史每步双向重算 |
| V2c Strict Causal KV | EXP-002/003同协议AA/AD124帧严格因果/KV/动作/基本结构可行性accepted | AA瞬态明显形变、边界与节奏问题；单scene/seed，完整E2E与泛化未测 |

V2a/V2b/V2c是解决V1退化的三条并行路线。V2c实际权重仍从Original出发，无新checkpoint；原展示名称V3 Efficient Causal按用户要求重分类为V2c，三支间无顺次权重继承。

## 3. 最新交付

- [V2c正式定义与全部视频](submission/mainline/v2/v2c_strict_causal_kv/README.md) · [浏览器展示](submission/report/index.html)。
- [Original vs V2c AA124](submission/report/v2/v2c_strict_causal_kv/videos/Original_vs_V2c_AA_124.mp4) · [V2b vs V2c AA124](submission/report/v2/v2c_strict_causal_kv/videos/V2b_vs_V2c_AA_124.mp4) · [V2c AD124原片](submission/report/v2/v2c_strict_causal_kv/videos/V2c_AD_124.mp4)。
- [Judge最终验收](submission/experiments/EXP-003_native_cached_124/judge/FINAL_REVIEW.md) · [独立成本核对](submission/experiments/EXP-003_native_cached_124/judge/costs_checked.json)。
- [Original vs V2b持续A/D124帧总览](submission/report/v2/v2b_same_sigma_local_bidir/videos/Original_vs_V2b_AD_overview_248.mp4)保留。

## 4. 资源与成本

EXP-003：186forward（180sampling+6commit）、6VAE、零训练/额外诊断，GPU0顺序两进程合计0.436741GPU-hours。EXP-002/003合计310实际前向、10VAE、约0.66597GPU-hours，首窗采样复用且不在该成本内。

AA三个新块sampling148.45/176.43/200.42秒，已有V2b相应310.26/377.53/447.19秒。仅为协议整体观察成本，不单独归因KV，也不是完整Original E2E speedup。CPU历史缓存最终18.131GB；新块记录peak26,876.70MiB，初始commit具体peak缺项已披露。

## 5. 当前任务与下一步

**EXP-004 / v1 completed & accepted：V2c原权重8步续写AA/AD到73帧，零训练。** 同历史第二块动作响应可辨（flow +0.780621/−1.509824），各自8步历史第三块+0.978349/−0.732276；基本人物场景可用。AA后段肢体透明拖影较明显，画质/连续性PARTIAL。首39帧仍复用30步结果，尚无全程8步、8步124帧或完整E2E结论。

实际34forward（32sampling+2commit）、4VAE、0.129449 GPU小时，GPU0已释放，peak25,682.07MiB。每新块sampling37.76–70.87秒，旧30步135.69–155.14秒；不同时刻共享硬件的单次观察，不作为公平E2E速度比。原报告、对比和成本见[8步证据](submission/report/v2/v2c_strict_causal_kv/8step_continuation/README.md)。

下一项高ROI建议：从首窗开始也使用8步，再续自己的历史，检验直接减步能否独立成立，再决定是否需要训练。EXP-004已关闭，尚无EXP-005授权；不追加局部调参或无预算扩展。AnyFlow/DMD仍未完成。任务收口和建议见[next_plan.md](next_plan.md)。

GPU项目总限制：2026-10-10 09:00 HKT前≤8，之后≤3，不自动恢复8卡；本轮只用了1卡。

## 6. 完成任务与Git交付

| 任务 | 正式结论 | 入口 |
| --- | --- | --- |
| EXP-001/v3 | V2b持续AA/DD124可行性，切换PARTIAL | [审核](submission/experiments/EXP-001_v2b_124/judge/FINAL_REVIEW.md) |
| EXP-002/v1 | 73帧严格缓存候选，同history A/D响应 | [审核](submission/experiments/EXP-002_native_cached/judge/FINAL_REVIEW.md) |
| EXP-003/v1 | 同协议AA/AD124可行性，发布V2c | [审核](submission/experiments/EXP-003_native_cached_124/judge/FINAL_REVIEW.md) |
| EXP-004/v1 | 原权重8步续写AA/AD73可行性，画质PARTIAL | [审核](submission/experiments/EXP-004_v2c_8step/judge/FINAL_REVIEW.md) |

EXP-001结果679258c已推送；EXP-002结果bdcb43a已推送；EXP-003任务d536f25已推送。EXP-003正式结果与V2c版本提交`d039352941708d4c7c757d696009b67c3e0e2468`已正常推送origin/main，远端SHA一致；提交前fetch无分叉，提交后工作树干净。全部原始任务/Worker报告/Judge结论见[archive.md](archive.md)，历史负结果保留。

EXP-004正式结果提交`4a90b78623a728502347287d9dc0fff290bbb998`已正常推送origin/main，远端SHA核对一致，submission工作树干净；提交前fetch无分叉。提交包含代码/视频/真实日志/冻结报告和Judge审核，未包含大型权重或缓存。

## 7. 分类更新

2026-10-10：原V3归入V2c Strict Causal + Persistent KV；report/mainline按v2家族分组。EXP-004目录改为EXP-004_v2c_8step，EXP-002/003保留稳定中性目录并更新V2c标题。历史快照保留旧称。V3尚无新定义或实验授权；下一研究建议仍是V2c从首窗开始的8步验证。

V2c分类与目录迁移提交`575530a3b5d660f8ac3559502a848d08c3e2a1b1`已推送origin/main，远端SHA一致，submission工作树干净。471个现行本地链接通过；132个迁移文件中115个逐字节保持，冻结证据与正式124帧视频摘要通过；仅两段8步展示片在CPU侧更新标题，无模型推理/训练。
