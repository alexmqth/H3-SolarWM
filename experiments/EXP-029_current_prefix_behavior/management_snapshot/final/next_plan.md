# 当前任务状态：EXP-029 已收口，暂无新GPU任务

2026-10-17，Judge。EXP-029的两场景×ON/OFF×A/D八条续写现已完整交付。夜间7条完成后达到wall上限；用户白天要求GPU1/6尝试，GPU1只补缺失s3 OFF_D成功，其他七条保持不变。恢复进程已退出。

## 验收与研究决定

[Judge最终审核](submission/experiments/EXP-029_current_prefix_behavior/judge/FINAL_REVIEW.md) · [八条视频与两个对比](submission/report/v3/fm2anyflow/current_prefix_behavior/README.md) · [夜间补交报告](submission/research/overnight_20261017/FINAL_REPORT.md)。

接受实现、协议和有限行为证据，不升级V3。山地场景ON/OFF都有A/D分叉；暗草地两种设置分叉均弱。该合法prefix边会改变画面/姿态，但没有建立它是控制退化主因的证据；不将RGB差异当动作准确率。

## 成本

原始3597.816秒+恢复321.700秒，共3919.516单卡进程秒（65.325分钟），266DiT尝试=3commit+263sampling，8decode，0训练。累计包含原缺失分支的23次中断工作。每条完整输出39旧帧+17新帧、30NFE。共享GPU/offload条件影响耗时，不能作为ON/OFF纯速度比较。

## 下一步边界

停止扩展当前prefix消融和角色梯度pilot；当前无额外GPU、双Adapter、AnyFlow或DMD授权。下一项若继续，应先从现有证据区分动作响应延迟与历史运动倾向等不同假设，提出能改变决策的最小匹配对照，再单独下发编号、预算和停止条件。不要只为已有模块故事重复消融，也不继续追逐约1%MSE改善。

用户要求的夜间09:00截止已执行，Judge报告/最终发布延迟至白天补交，未改写为准时完成。白天恢复由用户新指令单独授权，其20分钟预算与13:00截止现因成功退出而收口。
