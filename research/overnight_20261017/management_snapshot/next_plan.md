# 当前执行补充：EXP-029/v2 有限恢复（2026-10-17 11:49 HKT）

用户现在要求GPU1/6尝试，授权原Worker只补夜间超时缺失的s3 OFF_D。单GPU1优先/6后备，≤1200秒、31DiT/1decode、0训练；32GiB free与reserve18不变，最晚12:35启动、13:00硬停止。其余七条不重跑。CPU通过后直接运行并消息Judge。[恢复任务书](submission/experiments/EXP-029_current_prefix_behavior/ADDENDUM_03_FINITE_RECOVERY.md)。

夜间原任务已在04:10触及一小时预算停止（235DiT/7decode），不是仍在候卡。9:00守护已执行，所有登记旧进程已核查退出。以下为冻结原任务背景；其过期准入不适用于恢复任务。

---

# 当前任务：EXP-029 同状态current-prefix动作行为对照

2026-10-17 HKT，Judge。用户授权今晚继续主线，08:59:45开始停止、09:00硬截止并汇报/push。原Worker完成CPU检查后直接执行已批准GPU任务，无需再次用户确认。截止守护已运行，登记新进程。

[任务书](submission/experiments/EXP-029_current_prefix_behavior/taskbook_v1.md) · [授权](submission/experiments/EXP-029_current_prefix_behavior/AUTHORIZATION.json) · [夜间约束](submission/research/overnight_20261017/README.md)。

## 为什么做

EXP-028已验收，完整8条诊断未建立V3特有的明显角色梯度冲突；不扩梯度层、不直接训练双Adapter。接下来用真实视频检验当前非动作prefix读取当前video这条合法路径的行为作用，而非继续优化cosine。

## 固定对照

复用EXP-014 s0_43866101与s3_b784d995保存的自生成C1和A/D fixture；一次正确V3 clean commit后的同一KV，固定初始噪声、原始H3+released Action LoRA、native time/Global RoPE/own-action feedback。每图ON_A/ON_D/OFF_A/OFF_D，只开关current非动作prefix反馈。完整30NFE，39历史帧不改，新增17帧；不冒充新生成5秒或长视频。

## 预算

CPU实现≤45分钟/4workers/RSS32GiB。GPU1优先，可选0/6/7中的一张，共享允许但启动free≥32GiB（补充02显式CPU卸载reserve18），单次≤3600进程秒且必须在08:59:45前结束；242DiT（2commit+240sampling），8VAE decode，0backward/updates/编码/下载，GPUallocated<40GiB，RSS<96GiB。8原片+2个2×2对比及全部新增帧联系表。每个新进程登记deadline guard。

## 审核与停止

CPU明确唯一mask改动、A/D固定布局、历史KV不变、未来裁剪、真实入口和计数。GPU异常/超限停止不重试扩展。人物/相机分开观察，光流辅助；动作更静止不算收益。Worker第一场景完成和最终完成时消息Judge；Judge逐帧审核后决定方法价值。正式V3不覆盖，SGF+/Memory/双Adapter尚非已成立方法。

**准入补充01：** GPU1被其他新任务占用后free不足，允许单卡候选[1,0,6,7]择一，不增加GPU数量或预算；实际空余门槛后由补充02修订为32GiB，不抢占。

**准入补充02：** 默认reserve5卸载策略曾达38.181GiB，不能直接按旧策略降低门槛。显式采用EXP-019已验证的`ABOT_VRAM_RESERVE_GIB=18`（同56帧任务reserved峰27.197GiB），准入32GiB free；四分支统一，权重/精度/NFE不变，搬运耗时计入原3600秒。Worker仅在attempt0时冻结并更换候卡控制器、完成CPU检查后继续。[补充02](submission/experiments/EXP-029_current_prefix_behavior/ADDENDUM_02_MEMORY_ADMISSION.md)。
