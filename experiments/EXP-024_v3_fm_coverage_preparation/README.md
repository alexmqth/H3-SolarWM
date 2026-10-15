# EXP-024/v1 · V3 普通 FM 训练覆盖与容量准备

CPU 阶段完成、待 Judge 验收。已审计 EXP-023 checkpoint0/128、实现可选择 C1/C2 的训练入口、检查34条真实编码条件与50层微型 checkpoint backward 生命周期，并提出一个受保护的 full50 rank8 / C1+C2 GPU profile。**本阶段没有运行 GPU、训练 33B 或生成视频。**

[Worker 报告](WORKER_REPORT.md) · [参数审计](PARAMETER_AUDIT.json) · [CPU 小张量测试](CPU_TEST_RESULTS.json) · [50层生命周期](FULL_LAYER_CPU_RESULTS.json) · [真实条件布局](REAL_INPUT_AUDIT.json) · [GPU profile 提案](GPU_PROFILE_PROPOSAL.md) · [任务书](taskbook_v1.md)。
