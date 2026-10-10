# EXP-007 / v1 — V3-AF分阶段准备

2026-10-11，Judge。用户授权夜间连续研究至09:00。本目录当前仅批准AF0 CPU实施和验证，与EXP-006 GPU独立；GPU初始化/训练预算待Judge根据真实入口和资源成本发布，不因目录存在自动执行。

目标：在冻结V3原生条件/current-prefix/strict causal/raw KV协议上建立新的target-time finite-map student。保持Original H3+released Action LoRA、Single I0、12后5、Global、video外部1000sigma/audio1000/fixed_prefix_timesteps=False。独立入口允许可微只读cache，不更改Baseline；每次optimizer更新后用当前student重新计算sigma0/r0历史KV。不读取旧AnyFlow权重冒充V3-AF。

AF0核查：实际small-H3双层模型首12+续5、r=t初始回归、finite-map公式/时间单位、current-prefix上下文与未来动作裁剪、cache不变、target模块和QKV非零梯度、checkpoint重算不污染cache。CPU最多4线程，0GPU。不能将小模型通过当33B训练/生成能力通过。

后续拟按初始化真实模型验证→有限训练→匹配NFE视频→Judge决策推进。DMD后续独立立项，不与finite-map训练混称。
