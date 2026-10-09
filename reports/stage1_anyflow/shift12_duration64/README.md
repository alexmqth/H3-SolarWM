最新完整64次结果：[FINAL_RESULTS.md](FINAL_RESULTS.md)。下文保留当时快照。

# Shift12有限训练量对照

step32训练及A/D39f的4/8-step评测全部完成并失败：4-step分离度−0.019565，8-step0.172691，A均错误。4-step后段仍雾化，8-step无明确改善。[完整结果与视频](STEP32_RESULTS.md)、[训练曲线](GPU_RESULTS.md)。GPU1已按既定规则继续到总64。

[计划](PLAN.md)、[预检](preflight.json)、[恢复审计](gpu_resume_audit.json)、[队列快照](run.json)。恢复各自权重/Adam/RNG，只改变更新总数，不交换两条训练分支的state。数字gate未过时原控制器继续到最多64。

控制器依赖源机器冻结runtime；跨机器使用REPRODUCE.md的通用入口。日志训练shift字段的误标参见../metadata_shift_correction.json，不改写原始产物。

32→64真实GPU回载的权重/Adam/RNG审计：[gpu_resume32_audit.json](gpu_resume32_audit.json)。
