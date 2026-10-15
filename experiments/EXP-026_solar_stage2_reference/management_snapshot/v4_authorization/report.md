# EXP-026/v3 Worker 交付：四卡 Stage2-fix 真实加载尝试

2026-10-16 HKT。**本轮唯一获授权 GPU 尝试已结束，四张卡已释放；结果为环境阻塞，尚未完成 33B+EMA 分片加载。** 不重试、不改 attention backend。完整结论见 [v3 Worker 报告](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v3/WORKER_REPORT.md)。

v3 独立补丁与配置已通过 CPU 门控：官方 H3 backend 接受 world4/SP4/DP1 单样本配置；小模型验证 CPU deferred load、dtype、LoRA/EMA 数值和调用顺序；四个 Gloo rank 验证现有物理 shard4/replica1 分组，未初始化 GPU。SolarWM 官方 checkout、V3 主线及 v2 补丁未改。

真实加载使用物理 GPU0/1/6/7，torchrun PID/PGID 1082352。四 rank 成功初始化 NCCL 与分片组，base checkpoint 进度达到 14/14 shard；随后在 `set_attention_backend("flash")` 报错。当前 `flash-attn 2.8.3` 扩展与 `torch 2.6.0+cu124` 的二进制符号不兼容，CPU import 可复现同一 undefined-symbol 错误。`base_loaded` receipt 未发出，LoRA 注入、FSDP 包装和 Stage2 EMA 恢复都未达到，因此不能声称模型加载通过。未进行 reader、去噪/codec forward、训练、下载或视频生成。

GPU 尝试 26.316 秒，折合 105.265 GPU 秒；采样任务 RSS 峰 3.16 GiB，四卡物理显存占用峰各 435 MiB，退出后均归零。原始 [启动与资源账本](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v3/gpu_attempt_001/GPU_LAUNCH.json) · [失败结果](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v3/gpu_attempt_001/GPU_OUTCOME.json) · [环境诊断](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v3/ENV_DIAGNOSIS.json)。本地缓存的 flash-attn wheel 与失败二进制相同，未发现可直接替换的匹配 wheel；未下载、编译或更换环境。

CPU 准备阶段另有一次偏差：复制的 v3 Gloo 检查脚本最初将输出写到 v2 路径；修复后恢复了 v2 的八 rank 结果。恢复与 v3 四 rank 检查曾短暂并行约四秒，总计 12 个 CPU rank 进程，超过本轮 ≤8 并发限制。已在完整报告中记录。首个 watchdog 调用仅因 CPU gate 文件缺失而在 GPU 启动前退出，不计 GPU 尝试；之后的真实四卡启动是本轮唯一一次。Worker 不操作 Git 或 `progress.md`，等待 Judge 审核。
