# 当前执行任务：EXP-027/v1 ABot 基线诊断准备

2026-10-16 HKT。用户决定当前就在ABot上研究。原Exp Worker立即复用EXP-023/025数据和训练证据，20分钟内提交一个优先、具体可执行的GPU实验方案。当前仅CPU，无新GPU任务。

[任务书](submission/experiments/EXP-027_abot_baseline_diagnosis/taskbook_v1.md) · [授权](submission/experiments/EXP-027_abot_baseline_diagnosis/AUTHORIZATION.json)。

## 主线与优先事项

保留Original H3-World + released Action LoRA作为语言动作参考，V3 strict causal/persistent KV作为研究对象。盘点32train+2dev的动作/切换/episode覆盖，从既有学习记录判断先验证小样本拟合，还是直接做真实/自生成历史与动作切换诊断；优先一项，不并行搜索参数。草案明确样本、初始化、updates与推理NFE、时间/显存预算和停止条件，Judge审后直接安排Worker执行。

CPU≤20分钟/4workers/RSS32GiB/新增盘1GiB，0下载/0GPU；不重编码、不重复全量视频审核、不建新框架。已有小预算负结果不代表充分训练失败，1%左右代理收益不作为继续投入的充分理由。

## SolarWM参考支线

用户决定暂时不做，停止数据访问、下载和环境工作，不作为ABot实验前置。EXP-026保留四卡真实加载PASS和输入401/403证据；公开状态不作超出证据的推断。除非用户后续重新决定研究该参考，不自动重开。
