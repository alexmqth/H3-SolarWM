# 当前任务：EXP-026 plan v6 单样本官方参考准备

2026-10-16 HKT，用户批准继续。原Exp Worker立即执行一次有预算的官方输入检查和CPU reader/推理方案准备，收到就绪证据后Judge签发单样本GPU阶段，不再重复询问用户开始许可。

[任务书](submission/experiments/EXP-026_solar_stage2_reference/taskbook_v6.md) · [授权](submission/experiments/EXP-026_solar_stage2_reference/AUTHORIZATION_v6.json)。

## 当前执行范围

CPU≤30分钟、8workers/RSS96GiB；下载≤20GiB/新增磁盘≤30GiB。每提供方最多一次小文件访问；仍401/403且无新访问条件则收口，不反复试地址。检查真实官方样本与支持文件，复用v5已成功环境与加载补丁，0GPU/训练。输入齐全后提交单条158帧、4NFE、四卡的推理方案，GPU阶段建议上限9600GPU秒但尚未签发。

## 价值与停止条件

只做一次官方原生相机控制参考，不作为V3研究的强制前置，不将其当作语言人物动作的公平对照。不能从一个样本推断整体能力，也不能隔离训练量、SGF、DMD或数据的贡献。不训练SolarWM、不扩大评测集、不重新编译环境。若数据仍不可访问，暂停此支线，主线不因其阻塞。

v5已通过环境与四卡33B/EMA加载：157.989秒/631.956GPU秒，含历史累计737.221GPU秒；每卡峰allocated22.443GiB，0forward/训练/视频。[Judge审核](submission/experiments/EXP-026_solar_stage2_reference/judge/V5_REVIEW.md)。正式V3保持冻结，EXP-025小预算训练尚无明确新增视频收益。
