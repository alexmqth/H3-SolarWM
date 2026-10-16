# 当前研究任务：CVPR动机、贡献与机制诊断准备

2026-10-16 HKT，用户明确主要目标为CVPR。Judge调整主线：以可重复机制问题和可验证方法贡献组织实验，不继续追逐EXP-027约1%代理收益。仍聚焦ABot/H3-World语言动作，SolarWM官方参考停止。

[研究计划](submission/research/cvpr/RESEARCH_PLAN.md) · [CPU授权](submission/research/cvpr/AUTHORIZATION.json)。原Exp Worker立即在20分钟CPU预算内整理已有证据与一个最小机制诊断方案，不执行新GPU任务。

## 优先研究问题

如何在动作条件视频模型的因果化及历史KV复用中，保留当前动作对未来的可控影响，同时维持已生成世界的连续性？先区分合法条件通路被削弱与历史表征/新动作冲突两种待证伪假设。现有V1/V3版本差异混杂，不能直接证明机制；若只是修复遗漏attention边，则不自动构成论文方法。

## 当前交付与停止条件

Worker输出submission/research/cvpr/EVIDENCE_MAP.md和DIAGNOSTIC_PROPOSAL.md：已有证据/混杂/未知、最接近本地可核验工作与待文献核实项、一个能否定假设的配对诊断、精确输入/权重/作用变量/预算草案。CPU≤20分钟、4workers/RSS32GiB，0下载/0GPU/0新视频。Judge审后再决定GPUpilot，不扩大评测或预设SGF+/Memory方法。

EXP-027已accepted收口，64updates/四视频、无清楚新增能力；正式V3保持冻结。后续基线训练用于公平对照和排除欠训练混杂，不能以未经充分训练的负结果证明架构缺陷。论文novelty及目标CVPR届次/截止时间尚待核实，不作先验保证。
