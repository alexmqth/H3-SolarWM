# EXP-026/v4 Worker 交付：隔离 FlashAttention 修复门控

2026-10-16 HKT。**v4 在 CPU 依赖门槛处停止，没有启动新的 GPU 尝试。** 完整结果见 [v4 Worker 报告](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v4/WORKER_REPORT.md)。Judge 已独立复核该阻塞结果并要求收口，不再下载、编译或启动 GPU。

从 Dao-AILab 官方 v2.8.3 release 下载了文件名精确标注 `cu12torch2.6cxx11abiFALSE-cp310` 的 wheel，256,007,522 字节，使用 `--no-deps --target` 安装到独立 overlay。共享 `solarwm-h3` 环境、torch、官方 SolarWM checkout 与 v1–v3 证据未修改。但 overlay 原生扩展与 v3 的失败 `.so` SHA-256 完全相同：CPU import 仍报同一 `c10::Error` 未定义符号，Diffusers 的 FlashAttention backend 检查失败，CUDA 全程未初始化。`nm` 显示该扩展要求新 C++11 ABI 字符串符号，而当前 torch `libc10.so` 导出旧 ABI 符号。[环境门控证据](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v4/ENV_PREFLIGHT.json) · [wheel 来源与 SHA](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v4/WHEEL_RECEIPT.json)。

现有唯一 CUDA 编译器为 nvcc 13.0，当前 torch 2.6.0 使用 CUDA 12.4；PyTorch 自身 CPU 侧编译版本检查明确拒绝这个组合。按任务约束不能改 torch/backend 或安装系统 CUDA，因此未启动那一次允许的源码编译，也没有绕过导入错误。v4 下载约 256 MB、新增目录约 1.26 GB，均在预算内；**v4 GPU 0 秒、forward 0、codec 0、更新 0、视频 0**。v3 已用的 105.265 GPU 秒原样保留，EXP-026 v3+v4 累计仍为 105.265 GPU 秒。四卡 33B/FSDP/EMA 可行性和官方真实数据读取继续 NOT_TESTED。Worker 未操作 Git 或 `progress.md`。
