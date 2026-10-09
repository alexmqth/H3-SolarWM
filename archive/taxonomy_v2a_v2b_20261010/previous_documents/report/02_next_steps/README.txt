# V4及之后：研究决策与验收

本页是roadmap，不是已经启动的任务。没有新训练、GPU推理、AnyFlow或DMD进程。

| 顺序 | 研究问题 | Go标准 | 当前缺口 |
|---|---|---|---|
| 1 | V3局部正结果能否接到更多自身history | 同协议第三/第四块、至少额外场景，A/D方向与人物结构共同评审 | 尚无这些视频 |
| 2 | V4能否保留Original action信息流 | 相同权重/状态/时间/可见输入；own路由和feedback/public-prefix策略明确，局部30-step通过 | 无strict causal+KV联合PASS |
| 3 | KV实现是否与同计算图重算一致 | 跨sigma逐层K/V、RoPE、velocity；区分matching-time rebuild与sigma0 clean commit | 诊断wrapper已证明受控等价；生产默认数值/重算接口仍需谨慎 |
| 4 | AnyFlow能否少步且保能力 | 4/8步接近已可信causal30；finite/diagonal与teacher endpoint双指标，完整视频复核 | 旧AnyFlow质量失败；V3未进行此阶段 |
| 5 | on-policy DMD能否减少generated-history gap | clean-history局部能力可信；真实self-rollout、frozen teacher、trainable fake score、DMD梯度 | 已有lite负结果和小模型工程准备，非完整成功 |
| 6 | 是否更快/更省 | 统一硬件/offload、warmup、多次均值；E2E、首屏实际可见延迟、每块、GPU/CPU KV | 当前只有共享主机单次耗时，不授予加速结论 |

过去审计的关键事实：cache只存video K/V；causal action prefix额外直读来自fresh prefix。统一数值路径下P0逐层误差为0，但R2无persistent KV时对Original动作差分cosine已约0.057。因此不能将动作失败全部归于缓存实现；也不能仅提高cosine就宣布画面恢复。

[研究路线](../roadmap.md) · [汇报导航](../README.md)
