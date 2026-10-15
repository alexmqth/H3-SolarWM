# EXP-026：官方 SolarWM Stage 2-fix 参考验证

最新：**plan v6 输入检查已收口，官方视频验证仍 BLOCKED_ACCESS**。HF401 / ModelScope403，本轮0下载/0GPU；不重复请求，等待真实访问条件变化。[v6审核](judge/V6_REVIEW.md)。plan v5 环境修复与四卡真实模型加载已通过 Judge 审核，结论保持。

本实验的目的，是验证作者公开的相机控制模型，为后续训练与失败诊断提供参考。它与我们 H3-World/V3 的语言动作 LoRA 训练分开。EMA 指指数移动平均权重；FSDP 指完全分片数据并行；SP4/DP1 表示四卡序列并行、一个逻辑数据副本，本轮基础参数显式分片到四张物理卡。

## 本轮结果

固定官方 FlashAttention 2.8.3 源码及用户目录 CUDA 12.4 成功构建，CPU 实际导入通过。四张 L40 完成基础 33B → LoRA → FSDP → released Stage 2-fix EMA 加载和稀疏数值核对，157.989 秒，每卡峰 allocated 22.443 GiB / reserved 23.287 GiB，全部释放。

本轮 631.956 GPU 秒；包含此前失败累计 737.221 GPU 秒。0 forward、0 训练、0 视频，因此本目录没有生成视频。加载峰值不是生成/训练峰值，模型质量未测试。官方输入数据仍有访问缺口。

[最新 Judge 审核](judge/V5_REVIEW.md) · [Worker 报告](memory_load_v5/WORKER_REPORT.md) · [独立审计](judge/V5_AUDIT.json) · [GPU 账本](memory_load_v5/gpu_attempt_001/GPU_LEDGER.json) · [任务书](taskbook_v5.md)。

## 历史证据

[v4 依赖阻塞](judge/V4_REVIEW.md) · [v3 实际失败](memory_load_v3/WORKER_REPORT.md) · [四卡配置更正](judge/FOUR_GPU_CORRECTION.md) · [v2 CPU 补丁](memory_load_v2/WORKER_REPORT.md)。

模型、wheel、overlay 与 patched 源码副本保留本地，Git 保存来源/SHA/补丁与日志。plan v1–v5 是任务修订编号，不是研究模型版本。
