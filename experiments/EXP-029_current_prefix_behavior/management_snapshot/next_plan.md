# 当前任务：EXP-029 同状态current-prefix动作行为对照

2026-10-17 HKT，Judge。用户授权今晚继续主线，08:59:45开始停止、09:00硬截止并汇报/push。原Worker完成CPU检查后直接执行已批准GPU任务，无需再次用户确认。截止守护已运行，登记新进程。

[任务书](submission/experiments/EXP-029_current_prefix_behavior/taskbook_v1.md) · [授权](submission/experiments/EXP-029_current_prefix_behavior/AUTHORIZATION.json) · [夜间约束](submission/research/overnight_20261017/README.md)。

## 为什么做

EXP-028已验收，完整8条诊断未建立V3特有的明显角色梯度冲突；不扩梯度层、不直接训练双Adapter。接下来用真实视频检验当前非动作prefix读取当前video这条合法路径的行为作用，而非继续优化cosine。

## 固定对照

复用EXP-014 s0_43866101与s3_b784d995保存的自生成C1和A/D fixture；一次正确V3 clean commit后的同一KV，固定初始噪声、原始H3+released Action LoRA、native time/Global RoPE/own-action feedback。每图ON_A/ON_D/OFF_A/OFF_D，只开关current非动作prefix反馈。完整30NFE，39历史帧不改，新增17帧；不冒充新生成5秒或长视频。

## 预算

CPU实现≤45分钟/4workers/RSS32GiB。GPU1共享允许但启动free≥40GiB，单次≤3600进程秒且必须在08:59:45前结束；242DiT（2commit+240sampling），8VAE decode，0backward/updates/编码/下载，GPUallocated<40GiB，RSS<96GiB。8原片+2个2×2对比及全部新增帧联系表。每个新进程登记deadline guard。

## 审核与停止

CPU明确唯一mask改动、A/D固定布局、历史KV不变、未来裁剪、真实入口和计数。GPU异常/超限停止不重试扩展。人物/相机分开观察，光流辅助；动作更静止不算收益。Worker第一场景完成和最终完成时消息Judge；Judge逐帧审核后决定方法价值。正式V3不覆盖，SGF+/Memory/双Adapter尚非已成立方法。
