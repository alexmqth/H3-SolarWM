# 当前任务状态：EXP-026 plan v5 已收口

2026-10-16 HKT，Judge：环境修复和四卡真实模型加载通过。原 Exp Worker 完成本轮后停止新增 GPU 任务。

## 已完成

官方 FlashAttention 2.8.3 在独立 CUDA 12.4 工具链下构建并通过导入检查；GPU0/1/6/7 完成真实 33B + LoRA + FSDP + Stage 2-fix EMA 加载。157.989 秒 / 631.956 GPU 秒，每卡峰 allocated 22.443 GiB，均已释放。包含 v3 失败累计 737.221 GPU 秒。0 forward/训练/视频。

[Judge审核](submission/experiments/EXP-026_solar_stage2_reference/judge/V5_REVIEW.md) · [Worker报告](submission/experiments/EXP-026_solar_stage2_reference/memory_load_v5/WORKER_REPORT.md)。

## 下一步与范围

1. 优先补齐官方相机评测输入：可访问的 test index、H3 silence、encoder contract 和一个对应 latent 样本。既有 HF/ModelScope 下载被权限阻塞；已有访问请求待解决，不重复下载模型或重新编译。
2. 输入具备后，由 Worker 提交固定单样本的四卡官方 Stage 2 推理配置、GPU 时间预算及停止条件，Judge 审核后执行。当前未追加推理或训练授权。
3. 四卡加载通过不等于视频推理或训练可行；后续记录真正的 forward/KV/codec 峰值、耗时与视频。保持官方相机路线和 H3-World 语言动作路线分开解释。

V3 正式基线保持冻结；EXP-025 有限 FM 训练无明确新增视频收益，不自动加 rank、epochs、AnyFlow 或 DMD。遵守 ROI，下一步先完成可用的官方参考，不扩展环境研究。
