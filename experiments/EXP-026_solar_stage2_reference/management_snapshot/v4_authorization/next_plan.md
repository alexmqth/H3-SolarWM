# 当前任务：EXP-026/v4 隔离FlashAttention修复与四卡加载

v3一次GPU尝试因FlashAttention/PyTorch ABI不兼容失败，26.316秒/105.265GPU秒；四rank NCCL ready，但FSDP/EMA未到，显存可行性仍未测。四卡已释放。[v3审核](submission/experiments/EXP-026_solar_stage2_reference/judge/V3_REVIEW.md)。

现授权原Worker立即修复独立overlay中的FlashAttention，保持torch与官方backend不变；CPU真实import/需求检查通过后直接执行一次同协议四卡load-only，不等额外确认。[v4任务书](submission/experiments/EXP-026_solar_stage2_reference/taskbook_v4.md) · [授权](submission/experiments/EXP-026_solar_stage2_reference/AUTHORIZATION_v4.json)。

CPU45分钟、下载≤2GiB/新磁盘≤8GiB；一次4×600秒≤2400GPU秒，优先GPU0/1/6/7，0forward/codec/训练；资源/09:00限制沿用v3。失败不自动重试，不覆盖旧证据，不改共享环境。

官方数据访问阻塞不影响本轮纯加载；视频仍未授权。Worker消息回报各阶段，Judge审核和push。
