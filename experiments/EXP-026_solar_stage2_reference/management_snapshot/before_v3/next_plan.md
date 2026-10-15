# 当前任务：EXP-026/v2 CPU准备已收口，官方复现受阻

2026-10-16 HKT。Judge接受已完成的有限CPU准备/补丁与阻塞证据；**没有新的GPU或下载任务授权**。原Worker已收到收口消息，等待具体新任务，不自动重复实验。

## 已完成

- EXP-025全层C1/C2普通FM训练已accepted，视觉/动作PARTIAL，新增收益未证实；停止该配置自动扩训。
- EXP-026/v1通用config解析通过，但SP8被H3 backend拒绝，不能称完整配置通过。
- EXP-026/v2保留官方SP4/DP2、2样本；CPU加载后使用已有8物理卡参数分片的独立patch和小模型测试完成。真实FSDP/CUDA、reader与视频NOT_TESTED。
- [Judge最终审核](submission/experiments/EXP-026_solar_stage2_reference/judge/FINAL_REVIEW.md) · [CPU交付](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v2/WORKER_REPORT.md)。

## 恢复条件与下一决策

1. 官方HF401/ModelScope403数据权限：用户在qma账号完成官方访问/login，或提供本地副本；不在聊天发送token。之后再授权有限下载和真实reader验证。
2. Worker朴素八副本容量估算524.75GiB，不是实测峰；观察可用546.53GiB且GPU2–5占用。650GiB是未来load-only提案的保守条件，非所有实现的硬下界。
3. 等数据/资源条件明确后，Judge审核具体load-only入口和有限预算并消息原Worker立即执行。8×600秒当前仅提案，未授权；真实加载通过后另批两条158帧视频。

不为绕过阻塞继续重写分布式代码，不训练AnyFlow/DMD，不改冻结V3，不抢占他人GPU。夜间最多8空闲卡、09:00后最多3卡仍有效。

2026-10-16 Judge更正：固定158帧官方backend允许4卡SP4/DP1单样本，独立config验证PASS。八卡仅是当前v2补丁/提案限制，不是模型最低要求；后续优先考虑显式物理四卡分片。真实加载/生成尚未测，无新GPU授权。详见submission/experiments/EXP-026_solar_stage2_reference/judge/FOUR_GPU_CORRECTION.md。
