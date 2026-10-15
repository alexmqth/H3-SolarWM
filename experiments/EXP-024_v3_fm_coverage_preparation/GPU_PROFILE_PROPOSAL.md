# EXP-024 GPU profile 执行协议（v2 已授权）

执行记录：v2单卡profile已完成，3/3顶层DiT、2/2 backward/update，76.367秒；峰值allocated 24.163 GiB、reserved 27.508 GiB。[实测结果](GPU_PROFILE_RESULT.json) · [完整Worker解释](WORKER_REPORT.md)。本文件保留执行前协议和后续预算推算口径。

唯一首选候选：**Original H3 + released Action LoRA 冻结，full50 QKV rank8，普通 FM，C1/C2 均训练**。与 EXP-023 的 tail8 rank8 / 仅 GT-history C2 相比，这是训练覆盖与 adapter 容量同时改变的**联合基线**，不能归因任一变量。选择 rank8 而非 rank32，是因为全层 rank8 的静态参数仅约10.752M，先测真实全层反向和激活峰值即可改变后续决策；rank32 目前没有收益证据。没有新 anchor、attention、AnyFlow 或 DMD。

## 冻结输入与入口

复用 EXP-023 的同一 32train+2dev 编码、Single I0、native time、Global RoPE、own-action、action feedback、current-prefix feedback、严格 causal raw KV、12+5 partition、SDPA/h3_fp32。C1 loss 用 12 latent、空历史；C2 loss 用 5 latent，先对同 clip 的 GT C1 在 σ=0、`no_grad` 下 clean commit。未来 action/video 在进入 refiner/DiT 前由 `visible_inputs` 物理裁掉。每块采用 `mean` 的逐 latent 元素 FM MSE；后续数据顺序由 `balanced_block_order` 保证每 clip 每 epoch 各贡献一次 C1 与 C2 更新，因此两块同权，而不是依 latent 长度给 C1 更大权重。

[训练覆盖实现](coverage.py)已在小张量调用真实 `interval_train`、`ChunkAttention`、`visible_inputs` 和 `verified_attention_scope`，CPU C1/C2 梯度、未来隔离、保存恢复与50层 checkpoint 重算均通过。[CPU结果](CPU_TEST_RESULTS.json) · [50层结果](FULL_LAYER_CPU_RESULTS.json) · [故障注入](GATE_FAULT_CPU.json) · [34条真实条件布局](REAL_INPUT_AUDIT.json)。KV门控对所有已提交CPU K/V/RoPE逐字节哈希，未覆盖模型临时激活。这**没有**测真实33B激活卸载、全层反向显存或输出质量。

## 已由 Judge 单独放行的两次更新 profile

入口：[profile_gpu.py](profile_gpu.py)，守卫要求新 JSON 明确包含 `task_id=EXP-024`、`gpu_authorized=true`、`candidate=full50_rank8_C1_C2`、`profile_updates_max=2` 和 `gpu_seconds_max`。旧 [CPU 授权](AUTHORIZATION.json) 的 `gpu_authorized=false` 会在调用 CUDA 前拒绝运行。当前 [v2 GPU 授权](GPU_AUTHORIZATION.json) 的 `gpu_seconds_max=1800`、单张实际空闲 L40、`CUDA_VISIBLE_DEVICES=<空闲卡>`；外层 `timeout --signal=TERM --kill-after=10s 1770s`。程序内使用可见设备 `cuda:0`，不占多卡。

已授权命令模板（启动前仍须确认卡空闲）：

```bash
CUDA_VISIBLE_DEVICES=<空闲卡> OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
timeout --signal=TERM --kill-after=10s 1770s \
.venvs/h3world/bin/python -u \
submission/experiments/EXP-024_v3_fm_coverage_preparation/profile_gpu.py \
--authorization submission/experiments/EXP-024_v3_fm_coverage_preparation/GPU_AUTHORIZATION.json
```

固定 `SELECTION.json` 首条 train clip；独立 CPU RNG 噪声、σ=0.5，先 C1 更新一次，再用更新后的 adapter 对 GT C1 clean commit，并对 C2 更新一次。profile 仅检验执行成本与梯度协议，不用两次 loss 判断模型学习。计数预期为 C1 current forward 1、C2 history clean commit 1、C2 current forward 1，共 **3 次顶层 DiT forward、2 backward、2 optimizer update**；checkpoint 内50层重算另记层数，不误算为额外顶层调用。每阶段记录同步后的耗时、GPU allocated/reserved及各自peak、host RSS；记录 C1/C2 loss、逐层grad、C2 CPU KV bytes、50层 forward/backward route coverage。每块在 optimizer 前检查完整路由、冻结参数无梯度，以及 EXP-023 结构签名和全量CPU K/V/RoPE SHA256均未变。`GPU_PROFILE_RESULT.json` 逐阶段原子落盘，含尝试/完成计数、失败route trace、precision、固定clip、源码/输入SHA和PID；失败不自动重试。

硬停：任何协议/未来泄漏/错误 checkpoint、缺失或非有限梯度、50层重算路由不一致、历史 KV 被修改、OOM、allocated ≥44 GiB、进程 RSS ≥128 GiB、总时长达到授权上限前30秒、或其它异常。若 C1 第一步失败，**不启动 C2**。在 09:00 HKT 后也仍只使用1张卡，满足项目≤3卡限制。

## 后续 32 样本拟合预算如何确定

profile 通过才由 Judge 决定是否另发训练授权。建议唯一的下一训练协议为 32 clip × C1/C2 各一次 × 1 epoch = **64 更新**；batch1、每次独立 σ~U(0,1) 与噪声、固定 seed13 数据顺序，仍按每块平均 MSE。它是小预算拟合诊断，不能预称已收敛。先以 profile 实测的 C1+C2 一对更新 wall time `P` 估算 32 对的训练成本：`32P`，加约1500 GPU 秒固定前后评测与20%缓冲。建议只有 `1.2×(32P+1500) ≤10000 GPU秒`、profile 峰值 allocated <42 GiB、RSS <120 GiB、50层 route/梯度均通过时，才考虑把 **10000 GPU秒/64 updates/单卡** 提给 Judge；否则停止该候选并复核训练覆盖或计算策略，不直接升 rank32 或追加多卡。固定评测复用 EXP-023 的4条 GT-history、2条 self-history 与8个同σ FM probe，视频能力仍需完整视觉和动作语义判断。上述是**计划上限和通过条件**，并非已授权或实测吞吐。
