# EXP-026 plan v5 Judge 最终审核

2026-10-16 HKT。结论：**环境修复与四卡真实加载通过**。官方视频复现尚未完成；本次不升级任何模型版本，也不追加 GPU 任务。

## 已核验结果

- 官方 FlashAttention 2.8.3 源码使用独立 CUDA 12.4.131 构建，torch 2.6.0+cu124 / CXX11 ABI=0 保持。两次构建累计 2723.522 秒，峰 RSS 18.501 GiB，预算内；首次因 PATH 缺 ninja 进入串行构建，停止后修复，原因保留。
- Judge 独立核对 wheel、原生扩展、补丁及两份官方/修改源码 SHA；实际重新导入 torch/flash_attn/原生扩展并通过 Diffusers FLASH requirement，全程未初始化 CUDA。构建快照中 GPU_NOT_YET_TESTED 是构建当时状态；FlashAttention kernel 仍未执行。
- 四个 rank 阶段顺序完整：base_loaded → lora_injected → base_sharded → model_sharded → ema_loaded → spot_check_pass → load_only_pass → released。入口在 model_sharded 前校验 strict base receipt 四类错误均为零，检查代码流程通过。
- 基础参数 33,122,992,896；LoRA 624 tensors / 2,075,394,048 参数。SP4 / DP1 / 物理 shard4 / replica1。Stage 2 EMA label 为 stage2:ema:step=3900。
- 各 rank 九个 EMA 抽样数值一致；Judge 从原 EMA 文件 mmap 独立复核，逐值相等。此为稀疏检查，不是对全部权重的逐值比较。
- torchrun exit 0，157.989 秒，631.956 GPU 秒；含此前 v3 失败的 EXP-026 累计 737.221 GPU 秒（约 0.2048 GPU 小时）。
- 每 rank 峰 allocated 22.443 GiB，reserved 23.287 GiB；采样物理峰 24,283 MiB/卡；任务聚合 CPU RSS 采样峰 58.249 GiB。不同内存口径分别保留。
- 0 model forward、0 backward、0 codec、0 optimizer updates、0 视频。Judge 独立查询 GPU0/1/6/7 均为 0 MiB，其他用户 GPU 不操作。

## 能支持的结论与限制

四张 L40 能以本补丁的 CPU 延迟搬运及跨物理四卡 FSDP 方案加载实际 33B 基础模型和 Stage 2-fix EMA。此前失败是环境导入问题，不能据此声称必须八卡或四卡放不下。

本轮没有执行 attention kernel、去噪、KV rollout、VAE 或训练。加载显存不能代表推理/训练峰值；四卡完整视频推理能力、相机条件与视觉质量尚未验证。基础模型加载顺序有显式补丁，不称为未经修改的官方端到端运行。

官方样本索引、H3 silence/encoder contract 与选定 latent 样本访问仍是下一阶段缺口。先解决可访问的真实官方输入，再提出有限推理预算；不为缺数据伪造样本，也不继续扩展编译实验。V3/H3-World LoRA 训练使用独立原环境，本结果不证明其动作控制改善。

[独立审核数据](V5_AUDIT.json) · [Worker报告](../memory_load_v5/WORKER_REPORT.md) · [GPU账本](../memory_load_v5/gpu_attempt_001/GPU_LEDGER.json) · [任务书](../taskbook_v5.md)。
