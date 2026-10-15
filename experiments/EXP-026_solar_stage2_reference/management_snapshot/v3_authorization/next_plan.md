# 当前可执行任务：EXP-026/v3 四卡真实加载

2026-10-16 HKT，用户已批准。原Worker立即做四卡SP4/DP1最小调整，CPU/资源门控通过后自动执行一次真实33B+EMA load-only，不再等待额外确认。

[任务书](submission/experiments/EXP-026_solar_stage2_reference/taskbook_v3.md) · [授权](submission/experiments/EXP-026_solar_stage2_reference/AUTHORIZATION_v3.json)。

- 优先物理GPU0/1/6/7，启动前重新查空闲；不动2–5。
- 单次4卡×600秒≤2400GPU秒，0forward/codec/训练，失败不自动重试。
- 初始MemAvailable≥400GiB，运行时至少128GiB；本任务RSS≤400GiB，每rank allocated≤40GiB；watchdog及09:00前退出。
- 真实数据缺失不阻塞纯加载，不伪造输入。加载成功不代表官方生成能力通过。
- Worker只改独立memory_load_v3/副本，主动消息开工/启动/结果；Judge审核并push。

v2 CPU补丁已accepted；旧八卡限制已更正。正式V3及既有实验保持冻结。
