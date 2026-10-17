# 当前任务：EXP-030/v1 Original-visible 与 V3 匹配动作对照

2026-10-17，用户明确“开始吧”，Judge授权原Exp Worker先完成有限CPU准备，通过后自动执行四条新续写。此前EXP-029收口/无新GPU状态由本任务取代；EXP-028/029原始证据和验收不变。

## 唯一有效任务与预算

[完整任务书](submission/experiments/EXP-030_original_visible_behavior/taskbook_v1.md) · [机器授权](submission/experiments/EXP-030_original_visible_behavior/AUTHORIZATION.json)。

- 核心问题：同一自生成历史/噪声下，Original-visible是否保留V3缺少的A/D行为响应？
- s0山地、s3暗草地各新增Original-visible A/D两条，复用EXP-029四条ON V3视频，不重跑。
- 原版权重及released Action LoRA、30NFE，39旧帧+17新增帧；不训练。Original-visible是固定clean历史的双向重算参考，不等于原版完整联合生成；结果只先归因于协议整体差异。
- CPU准备≤45分钟；单GPU1或6，共≤5400进程秒、120DiT采样、4decode、0backward/updates/encode/download。显式reserve18，实际free≥32GiB；共享GPU允许，无需完全空卡。
- 最晚14:25 HKT启动，16:00硬停止；所有失败成本累计，单次GPU尝试，无自动GPU重试。实际PID/GPU/首forward须通知Judge。
- 完整采样端点、history clamp、动作/位置/prefix协议及保存路径通过CPU检查后自动启动；协议/资源异常立即停止。
- 交付四条新片、两组2×2对比、全部新增帧/边界和人物局部检查、配置日志与耗时/内存报告。MAD与光流不作动作准确率，0.71秒短窗口可判不确定。

## 停止边界

不追加GT/Self矩阵、更多scene/seed、长视频、prefix/梯度消融、双Adapter、AnyFlow或DMD。Worker完成后停进程并通知Judge验收；Judge负责GitHub提交与push。具体停止和决策条件见任务书。
