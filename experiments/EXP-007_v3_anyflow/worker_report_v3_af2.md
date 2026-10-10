# EXP-007/v3 AF2 Worker 报告 — 31 次有限 AnyFlow 更新

`task_id=EXP-007` · `plan_version=3` · `phase=AF2` · `worker_status=complete_pending_judge` · `generation_quality=NOT_TESTED`。依据 [v3 任务书](taskbook_v3.md)，从已验收 AF1 step1 的配套 QKV、target-time、optimizer 与 CPU/CUDA/logical RNG 恢复；训练期间没有加载任何旧 AnyFlow 产物、没有视频解码、没有 DMD。

使用 [独立 runner](run_af2.py)、[配置](config_v3_af2.json)、[冻结来源清单](source_manifest_v3_af2.json)。CPU preflight 核查 AF1 权重元数据、20 项 optimizer state、输入形状和 A/D 端点不同后，GPU1 命令为：

`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python -u submission/experiments/EXP-007_v3_anyflow/run_af2.py --gpu 1 > H3-World/outputs/EXP-007_af2_run.log 2>&1`

每次新增更新先用 **该步当前 student 权重**对冻结 V3 generated C1 首12 latent 做一次 sigma=r=0 clean commit，创建新的 CPU raw KV；随后的 logical batch4 固定该 cache，做 2 diagonal/FM、1 endpoint、1 general finite-map 样本，每样本 3 detached + 1 gradient forward 和一次 backward。奇数累计 step 用 AA C2，偶数用 AD C2；一次 AdamW 更新后释放 KV，下一步重新建立。训练用独立 RNG，base 与 released action LoRA 始终冻结；原生 Single I0/current-prefix/Global RoPE/动作路由和 h3_fp32 边界不变。

实际完成累计 step2–32，**31 个新增 update、527 full forwards、124 backwards、0 VAE**，GPU1 占用 4287.861 秒（1.1911 GPUh），低于 2.0 GPUh 上限。每步 132.80–144.23 秒，均值 138.00 秒；allocated 峰值 26.767 GiB，reserved 峰值 27.318 GiB。31 行逐步 raw/weighted loss、time pair、梯度、cache 检查、耗时在 [原始 result](artifacts/af2/result.json)，逐调用账本在 [budget](artifacts/af2/budget.json)，stdout 在 [日志](artifacts/af2/EXP-007_af2_run.log)。步间 loss 不可作为视频画质或动作能力证据。

仅在累计 step8 和 step32 保存配套 checkpoint，包含 QKV、target-time、optimizer/RNG/协议元数据，外部位置分别为 `H3-World/outputs/EXP-007_v3_anyflow_af2/step_08/` 和 `.../step_32/`；没有复制 33B base 或 raw KV 进提交包。最终 step32 SHA-256：QKV `07c8e5e68d1c947e60217e326ca8dd4c00222a555542b286dfc03514e27e916e`，target-time `ceb7d62324834a5b2be9fd6a47bfe4b5a9d810e49d1bb33889af0fb788940f70`，trainer state `2fda88175634075b1b4feda209337ff1281e4007ce25dc540eb1992704c523b1`。step8 已由 Judge 独立核验，最终 step32 的正式验收由 Judge 执行。

AF3 的 [CPU preflight](AF3_PREP.md) 已在 AF2 完成后通过并冻结 step32 来源：完整配对 checkpoint、FM8 首12/首39 RGB、A/D 同输入噪声与原生 8-step sigma 网格均核对，**0 AF3 GPU calls**。AF3 需要单独的 Judge GPU 放行；本报告不声称 8NFE 视频质量、动作控制或相对 FM8 的收益。
