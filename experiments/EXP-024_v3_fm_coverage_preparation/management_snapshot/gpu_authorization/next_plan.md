# 当前可执行任务：EXP-024/v2 有限 GPU profile

CPU准备已由Judge accepted，原Worker立即完成来源清单冻结后执行单卡真实C1/C2 profile。

- 全50层QKV rank8，10,752,000参数，新零增量初始化；冻结H3+released Action LoRA。
- C1/C2各1次更新，最多3顶层DiT尝试、2backward、2updates、1800GPU秒；0编码/新视频，不自动扩训或重试。
- 单张启动时空闲L40，优先GPU0；外层timeout1770秒、kill-after10秒。
- optimizer前检查50层路由/冻结梯度/adapter有限梯度；历史完整payload摘要不变。C1失败即停。

[GPU任务书](submission/experiments/EXP-024_v3_fm_coverage_preparation/taskbook_v2_gpu_profile.md) · [GPU授权](submission/experiments/EXP-024_v3_fm_coverage_preparation/GPU_AUTHORIZATION.json) · [CPU Judge审核](submission/experiments/EXP-024_v3_fm_coverage_preparation/judge/CPU_FINAL_REVIEW.md)。Worker完成后立即更新report并消息回报，由Judge决定下一训练预算。

EXP-023已accepted收口，视频增益未证实，正式V3不升级。最终证据和024初始任务已push a1ec25cb29fb606b9ee3f594a0d881c399b52e06，GPU新授权另行发布。
