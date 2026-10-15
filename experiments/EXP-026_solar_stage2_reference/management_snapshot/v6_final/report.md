# EXP-026 / plan v6 Worker 交付：官方单样本输入仍受权限阻塞

2026-10-16 HKT。**v6 的 CPU/输入检查已结束；没有运行 GPU、模型 forward、codec、训练或视频。** 详细结果见 [v6 Worker 报告](submission/experiments/EXP-026_solar_stage2_reference/reference_v6/WORKER_REPORT.md)、[输入状态](submission/experiments/EXP-026_solar_stage2_reference/reference_v6/INPUT_STATUS.json)和[逐提供方探测记录](submission/experiments/EXP-026_solar_stage2_reference/reference_v6/ACCESS_PROBES.json)。

项目本地没有官方 H3 的 `test-index.jsonl.gz`、silence support、encoder contract 和对应 latent shard；qma 账号下未发现 Hugging Face / ModelScope 的标准登录凭据。按本轮上限各发起一次限时小文件访问：Hugging Face 固定 revision 测试索引返回 **401**，ModelScope H3 encoder contract 返回 **403**，下载 **0 字节**，没有换 URL 重试。由于实际索引行不存在，无法固定 seed42 样本、确认 shard 大小或运行真实 CPU reader。官方 `example/` 仅用于格式/reader smoke，不能替代本次正式参考样本。

[条件性 GPU 提案](submission/experiments/EXP-026_solar_stage2_reference/reference_v6/GPU_PROPOSAL.md)已写明：输入齐全且真实 reader 通过后，才可提交一条 158 帧、4NFE、四卡 SP4/DP1、Stage2-fix EMA 的推理配置给 Judge 审核。当前提案**不可执行，也没有 GPU 启动授权**。v5 已验收的 33B+EMA 四卡 load-only 结果保持不变；本轮未重编译或重复加载。Worker 未操作 Git 或修改 `progress.md`。
