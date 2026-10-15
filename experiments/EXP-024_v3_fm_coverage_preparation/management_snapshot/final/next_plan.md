# 当前可执行任务：EXP-025/v1

EXP-024/v2已accepted：真实全层C1/C2 profile完成，3DiT/2backward/2updates，76.367GPU秒，峰allocated24.163GiB、reserved27.508GiB；质量未测。

现授权原Worker立即实现并执行全层rank8/C1+C2普通FM小数据拟合：32train×每block4遍=256updates，固定前后各6视频及C1/C2同状态probes。CPU自检、零adapter回归和真实梯度门控通过后自动连续执行。

单空闲L40、总13800GPU秒/576顶层DiT/256backward/256updates/12decode；0新编码；失败不自动重试，不扩展新分支。[完整任务书](submission/experiments/EXP-025_v3_fm_full_coverage/taskbook_v1.md) · [授权](submission/experiments/EXP-025_v3_fm_full_coverage/AUTHORIZATION.json)。Worker按阶段消息回报，Judge审核/push。

正式V3保持冻结。EXP-023视频收益未证实，不再微调旧tail8配置；本轮容量与训练块覆盖共同改变，不能作单因素归因。
