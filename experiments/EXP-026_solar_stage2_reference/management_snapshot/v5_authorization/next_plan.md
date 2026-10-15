# 当前执行任务：EXP-026 plan v5 用户目录工具链修复与四卡加载

2026-10-16 HKT，用户已批准开始。原Worker立即在用户目录准备CUDA12.4开发工具链，固定官方FlashAttention源码构建兼容扩展；CPU import/backend门控通过后自动执行一次4卡SP4/DP1/shard4真实33B+EMA load-only，无需额外确认。

[完整任务书](submission/experiments/EXP-026_solar_stage2_reference/taskbook_v5.md) · [授权](submission/experiments/EXP-026_solar_stage2_reference/AUTHORIZATION_v5.json)。

CPU总90分钟/编译累计60分钟，MAX_JOBS4、RSS≤48GiB、下载≤8GiB/磁盘≤20GiB；不改系统CUDA/driver、共享venv、torch或官方attention语义。

GPU优先0/1/6/7，门控通过后自动一次4×600秒≤2400GPU秒，0forward/codec/训练，失败不自动重试；资源/09:00前退出规则见任务书。真实数据缺失不阻塞纯加载，视频未授权。

v3失败105.265GPU秒与v4 CPU阻塞证据冻结。任务发布后Judge消息原Worker并确认实际开工；Worker阶段回报，Judge审核/push。
