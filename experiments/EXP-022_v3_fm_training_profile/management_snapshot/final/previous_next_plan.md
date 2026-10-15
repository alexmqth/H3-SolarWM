# 当前可执行任务：EXP-022/v1

2026-10-15 HKT，Judge已授权原Exp Worker立即执行CPU实现与有限GPU profile。**CPU前置检查通过后直接跑GPU，无需再次等常规放行。**

[完整任务书](submission/experiments/EXP-022_v3_fm_training_profile/taskbook_v1.md) · [授权/预算](submission/experiments/EXP-022_v3_fm_training_profile/AUTHORIZATION.json)

- 补齐36条候选RGB视频；复用既有b784d995编码，不让新数据准备阻塞profile。
- 实现独立可微V3 wrapper，checkpoint0对齐冻结V3前向/KV。
- 最多2次真实FM更新，保存重载，测显存、host RAM与逐阶段耗时。
- 单空闲GPU，≤3600GPU秒/20顶层DiT forward/2backward/2updates；0VAE/text调用，0生成采样。CPU≤8线程/2小时，host RAM≤128GiB，新增磁盘≤12GiB。
- 协议差异、OOM、非有限梯度、预算耗尽停止受影响GPU，不自动重试或扩量。任务书包含完整控制变量与验收范围。

## Judge已完成验收

EXP-019/020已accepted，质量/动作PARTIAL。EXP-021 CPU准备已accepted：32train+8dev候选；仅4train已物化；微型4项CPU单测通过，真实33B训练尚未验证。[EXP-021审核](submission/experiments/EXP-021_v3_fm_baseline_preparation/judge/FINAL_REVIEW.md)。

用户要求任务下发即消息通知固定Exp Worker。Judge负责授权、审核和发布，Worker负责执行和报告。正式V3冻结；新adapter仅测训练通路，不以两次更新评估训练容量或画质。

2026-10-15交接确认：已向固定Worker会话发送EXP-022立即执行指令；Worker已回复“EXP-022/v1已开工”，CPU准备后按授权直接启动GPU profile。EXP-021证据与EXP-022授权已push main `ea5a0c86bc0f6558f024886b4a66bdf95e80824c`，独立远端SHA一致。
