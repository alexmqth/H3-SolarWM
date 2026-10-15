# EXP-022/v1 Worker 报告：V3 真实普通 FM 训练通路工程通过，能力未评测

2026-10-15 HKT。按 [冻结任务书](taskbook_v1.md)和[授权](AUTHORIZATION.json)执行；未修改冻结 V3/EXP-019 代码或权重，未提交 Git。任务结论是**可微 V3 wrapper 在 checkpoint0 与冻结 V3 数值等价，并在真实 H3-33B 上完成 2 次普通 FM 更新及保存重载**；这不是画质、动作或泛化成功。新 adapter 只是 tail8 QKV rank8 的 profile 候选。

## 输入、协议及 CPU 前置

- 使用 Original H3 + released step-10000 Action LoRA、EXP-019 `h3_fp32` 精度/torch SDPA `current_prefix_feedback`、Single I0、native time、Global RoPE、own-action + action feedback、GT clean C1 raw KV、`[12,5]`。C2 同状态 noisy latent 由 EXP-016 b784d995 `full17.pt` 的 RGB39–55 对应5 latents 与 EXP-019 E fixture 的初噪构成；σ=0.5/0.75，目标均为 `epsilon-clean`。batch1、seed13、AdamW lr1e-4、无梯度累积，无 AnyFlow target time/DMD/辅助损失。
- [CPU 前置](CPU_PREFLIGHT.json)复核冻结源码和已编码张量 SHA、`prefix12` 对 `full17` 前12 latent 逐值相等、17条动作与 `[12,5]` 物理截断、C1 未来 action embedding 不可见。[来源 SHA](SOURCE_MANIFEST.json)保存实际入口和输入。冻结 `interval_cached.py` 的 guard 保留；[独立训练 wrapper](interval_train.py)仍调用相同模型函数/attention context，允许当前块读取无梯度历史 KV；仅当前块启用 gradient checkpoint 与 saved-tensor CPU offload。
- EXP-021 固定清单的 28 train +8 dev **36 条新增 56 帧 RGB**已物化；原 4 train 视频复用，没有重新编码或更换窗口。新视频均832×480/24fps、唯一递增PTS、完整56帧解码；[CPU最终审计](CPU_FINAL_AUDIT.json)独立再解码 2016 帧并核对 SHA。[数据清单](RGB_MATERIALIZATION.json)、[36条8帧contact及总览](RGB_OVERVIEW.jpg)、[质控](DATA_QA.json)可审查；没有黑帧/文件损坏导致排除，部分原场景本来较暗。新增窗口还**没有 VAE/text 编码**，不可直接作为训练 tensor。
- 直接产物文件求和：36 MP4 为 60,959,823 bytes，36 contact 为 2,007,685 bytes，合计 62,967,508 bytes。`RGB_MATERIALIZATION.json` 中 `new_disk_bytes=1,163,227,136` 是并发 GPU 期间**整文件系统 free-space 差值**，受其他写入混杂，不作为 EXP-022 实际新增磁盘量。当前 EXP 目录总量约 88 MiB，低于12GiB上限。

## GPU checkpoint0 等价门槛

GPU0 启动时空闲约45,458 MiB；单卡实际运行，外部3600秒 watchdog 和内部调用前账本。Frozen V3 先 commit GT clean C1，再对同一 C2 `zσ`、σ=0.5 求 velocity；安装零增量 tail8 QKV rank8 后，新 wrapper 另建 clean C1 cache。在50层中，K/V/RoPE 共 **150项逐值相同**（全部 max_abs=0）；C2 velocity 也逐值相同、相对 RMS=0。证据：[KV 对比](CACHE_EQUIVALENCE.json) · [前向对比](FORWARD_EQUIVALENCE.json)。新可训练参数实核 1,720,320，released Action LoRA 与 base 均冻结。只有等价门槛通过后才开始训练。

## 两次真实 FM 更新

每次 optimizer update 前用当前 adapter **重新 commit GT clean C1**，避免参数更新后读取旧 KV。只对 C2 计算普通 FM loss；第1次 σ=0.5，第2次 σ=0.75。历史 KV 不接收梯度；新 adapter 梯度 finite/非零，冻结 base/Action LoRA 的 `.grad` 均为 None，参数实际变化。两个 loss 由于 σ 不同，**不能当作下降趋势**。

| 更新 | σ | FM loss | adapter 梯度 L2 | 参数变化 L2 | clean commit / forward / backward / update |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0.5 | 0.18504572 | 7.6822e−5 | 0.07328445 | 17.051 / 9.403 / 2.619 / 0.171 秒 |
| 2 | 0.75 | 0.11402790 | 9.0628e−5 | 0.06902067 | 14.738 / 9.155 / 2.168 / 0.001 秒 |

两次每步总和约29.244/26.062秒，均不含初次加载、冷启动、等价对照、保存/重载和新数据编码。每次 C1 raw KV 为 **6,799,104,000 bytes**。Gradient checkpoint 与 saved-tensor CPU offload 在当前块训练前向均开启；backward时间包含 checkpoint 的重算/搬运，但**没有独立埋点精确记录重算层数与时间**，不能把它写成0。首块12-latent的**训练反向未测试**，本次只测首块无梯度 commit 与第二块可微训练。

[checkpoint0](checkpoint0.pt) SHA `e9df30dff44292271895eb28ab2b7c4dbdddf7aa4398848139c23ee4b80bc8ff`；[step02](step02.pt) SHA `b2a1d292d7f96d65004dadb8387b2fce074bd26969a491dab1701f564e7dbd15`，内含8个 adapter、16项 AdamW 状态、CPU/CUDA RNG、step 和协议 metadata。故意扰动一个 LoRA 参数后重载 step02，固定状态 velocity 逐值一致，CUDA RNG逐值一致；证据[重载检查](RELOAD_CHECK.json)。checkpoint 仅是本地工件，不随本报告声称具有生成质量。

## 资源账本与限制

- GPU0 进程 wall **413.200秒**；11次顶层 DiT forward（commit/等价/训练/重载探针全部包含）、2 backward、2 update、0 VAE/text/model采样调用。GPU peak `torch.cuda.max_memory_allocated` **20.293 GiB**，reserved **22.852 GiB**，低于44GiB。真实 `nvidia-smi` 占用不等于 PyTorch allocated 指标。
- [账本](GPU_BUDGET.json)事件采样记录最大 host RSS **64.704 GiB**，结束时RSS **58.848 GiB**；未持续采样，不能称64.704为绝对内存峰。最大CPU KV约6.332 GiB。首次 frozen C1 commit **147.759秒**，受冷启动/权重调入影响；后续新 wrapper commit 14–17秒，不能只挑稳态数报端到端成本。日志：[GPU_RUN.log](GPU_RUN.log) · [机器结果](GPU_RESULT.json) · [独立CPU核查](CPU_FINAL_AUDIT.json)。进程正常退出，GPU0已释放。
- 该任务只证明该单条 train episode 的 C2 训练通路，没测 checkpoint0/step02 生成视频、同状态 A/D 行为、GT/self-generated history 质量、首块训练，也没测不同训练样本吞吐。tail8 QKV rank8 容量和是否改善视觉/动作均 **NOT_TESTED**。两次更新不能支持任何模型能力结论。

## 后续预算建议

[下一阶段草案](NEXT_STAGE_BUDGET_DRAFT.md)按数据编码 pilot、8-update 第一停点、固定2开发clip匹配NFE验证、再视证据扩16/32 updates 划开预算。按照本次预编码单样本稳态，训练计算约26–30秒/更新；32 updates 的核心训练约14–16分钟，但编码、不同clip I/O、评测和失败调用另计。所有后续GPU操作均待 Judge 新任务授权。本轮 Worker 报告提交 Judge 审核；EXP-021 和正式 V3 保持冻结。
