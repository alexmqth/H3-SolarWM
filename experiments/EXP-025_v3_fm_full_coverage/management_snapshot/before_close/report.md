# EXP-025/v1 Worker 交付：全层 C1/C2 普通 FM 小数据拟合

2026-10-16 HKT。状态 `completed`，**待 Judge 验收**。按冻结任务书用单张 GPU0 完成全50层 QKV rank8 新零增量 adapter、32条训练clip、C1/C2各128次普通FM更新，并完成checkpoint0/256各6条56帧固定评测。没有运行AnyFlow、DMD或额外GPU分支；没有修改冻结EXP-023/024源码。

工程链路通过：零adapter的C1输出、clean commit完整KV、C2输出逐值一致；首个C1/C2真实梯度门控及全部后续KV摘要检查通过；checkpoint0/64/128/256落盘并重载核对。总计570顶层DiT、256 backward、256 optimizer、12 VAE decode、0 encode，GPU进程11,566.6秒；峰allocated24.45GiB、reserved28.14GiB，CPU raw KV每块6,799,104,000 B，均在预算内。

固定C2 FM probe MSE在train/dev分别下降1.42%/1.25%，C1分别上升0.21%/0.26%。六组三列GT/zero/256对照视频及完整解码检查已完成，但**未看到跨角色一致、明确的画质或动作收益**；self-history的boundary MAD下降只是一项运动敏感proxy，不能单独判定质量改善。本轮没有同状态A/D方向测试，也没有长时生成证据。因此建议保留EXP-025为已执行的受控联合基线，不将其升为正式V3能力checkpoint，不自动追加同类训练预算。最终结论待Judge独立审核。

[完整Worker报告](submission/experiments/EXP-025_v3_fm_full_coverage/WORKER_REPORT.md) · [六组视频导航](submission/experiments/EXP-025_v3_fm_full_coverage/README.md) · [固定指标](submission/experiments/EXP-025_v3_fm_full_coverage/METRICS_SUMMARY.json) · [GPU账本](submission/experiments/EXP-025_v3_fm_full_coverage/GPU_LEDGER.json) · [独立视频审计](submission/experiments/EXP-025_v3_fm_full_coverage/INDEPENDENT_VIDEO_AUDIT.json)。Worker未操作Git。
