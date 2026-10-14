# EXP-019/v1 · R 阶段 Worker 报告

2026-10-15 HKT。R 是固定动作布局候选在真实 33B H3 上的 **canonical A 数值回归**，不是混合动作生成或画质验收。进程在 Judge 已签发的 `judge/R_APPROVED.json` 下启动，本 Worker 接手时 PID 1283026 正在 GPU0 上运行；没有重复启动或修改运行中的 `run.py`。已设置独立 900 秒 watchdog，仅针对该 PID；进程在时限前正常退出，watchdog 未发送信号。

## 实际执行

- 命令：`CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venvs/h3world/bin/python -u submission/experiments/EXP-019_v3_mixed_action_validation/run.py --stage R --gpu 0`。
- GPU：NVIDIA L40，UUID `GPU-d77c5635-02ec-c2d2-d744-1dfcc4a14e8f`；启动空闲 45,458 MiB；`ABOT_VRAM_RESERVE_GIB=18`。Torch `2.10.0+cu128`，CUDA `12.8`，冻结 current-prefix attention 的 PyTorch SDPA 路径，`h3_fp32` 原生边界权重已恢复，Transformer 主体保持加载 dtype。原始 metadata 和精度张量哈希在 `artifacts/R/result.json`。
- 来源锁：`SOURCE_MANIFEST.json` SHA-256 `5837a43755deea41db9f0c1cdefb0a0e31afa9fb7bbbb6e44a414b153570fb3e`，与 R marker 一致；`run.py` SHA-256 `5c95118d89714b7ddb54db8b3d3fb92529862db9146288dc50e6b1805eb92d71`。入口对 manifest 内文件逐项校验后才建账本。
- 实际 **10 次 DiT forward**：C1/C2 × σ=1.0/0.5 × 旧/新入口共 8 次，另用同一冻结生成的 C1 latent 为旧/新各做一次 clean commit，共 2 次。0 text encode、0 VAE decode、0 训练。预算与结果状态均为 `complete`/`PASS`，无异常或自动重试。
- 阶段完整 wall **379.384 秒**（上限 900 秒）；模型加载/准备 **49.879 秒**。全流程 `torch.cuda.max_memory_allocated` 峰 **23.891 GiB**，reserved 峰 **27.043 GiB**。R 没有记录 CPU KV 字节峰，不能把 0 或历史任务中的缓存大小当成本次实测值。

## 数值核查

| Chunk | 固定 σ | 状态来源 | 旧/新 velocity 完全相同 | max abs | relative RMS |
|---|---:|---|---|---:|---:|
| C1 | 1.0 | 冻结初始噪声 | 是 | 0 | 0 |
| C1 | 0.5 | 同一噪声乘 0.5 | 是 | 0 | 0 |
| C2 | 1.0 | 冻结 C2 初始噪声 + 各自重建 C1 KV | 是 | 0 | 0 |
| C2 | 0.5 | 同一 C2 噪声乘 0.5 + 各自重建 C1 KV | 是 | 0 | 0 |

C1 clean commit 后，50 层各自的 `key`、`value`、`rope` 共 **150 组张量**旧/新逐值相等（全部 max abs 0）。两套 cache 都来自本轮相同生成的 C1 latent 重新计算，没有读取已清理的旧 KV。σ=0.5 是固定噪声缩放探针，**不是采样轨迹中的实际中间状态**。四组 velocity 与 150 组缓存的原始逐项数值在 `artifacts/R/comparisons.json` 和 `artifacts/R/cache_comparison.json`。

## 可支持的结论与下一门槛

本轮证明：在已测试的 canonical A、固定 C1/C2 状态、相同条件与历史下，EXP-017 固定布局候选没有改变 H3 velocity 或新建历史 KV。它不证明真实混合动作句的 embedding、future isolation、动作响应、GT-history 续写或长期 free-running 质量。按照任务书，下一阶段 E 必须先经 Judge 审核并签发与冻结来源绑定的 marker；本 Worker 没有启动 E/G。

本目录 `artifacts/R/` 是原始 `H3-World/outputs/EXP-019_v3_mixed_action_validation/R/` 与阶段 stdout 日志的逐字节副本；来源、目标、大小及 SHA-256 见 `R_ARTIFACT_MANIFEST.json`。没有保存大型 raw KV 张量。
