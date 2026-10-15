# 后续 GPU 阶段预算草案（尚未授权）

当前 EXP-021/v1 **批准 GPU 调用 0、训练更新 0**。用户说明 2026-10-15 夜间可使用至多 8 张空闲卡，2026-10-16 09:00 HKT 后至多 3 张；这是资源上限，不改变当前任务书。任何新 GPU 阶段须 Judge 给出编号、冻结输入、限额和停止点，并检查他人占用。

| 阶段 | 可审核产物 | 预算原则 |
| --- | --- | --- |
| 数据物化/编码 | 36 条未物化窗口的 RGB/17-latent、SHA、前缀一致性、质量检查 | 独立记账 FFmpeg、VAE、I0/text、磁盘；先抽样估时，不直接批量 36 条 |
| 训练 profile | 1 条真实 clip、checkpoint0、1–2 次完整更新、保存重载 | 先单 GPU；每次计 GT commit、当前块 forward/backward、optimizer、条件编码、失败调用和 checkpoint I/O；实测后再定上限 |
| 有限训练 | episode 隔离的数据采样与曲线 | 更新数根据 profile 推算，并设硬停；不以旧推理 30NFE 耗时替代训练吞吐 |
| 评测 | checkpoint0/训练后、GT/self-generated、constant/switch，同 seed/NFE | 编码、采样、KV commit、VAE decode 均计入；质控失败不自动追加训练 |

计划中的 tail8 QKV rank8 参数约 1.72M，但 33B backbone 权重、反向激活、optimizer 状态、CPU raw KV 和条件编码才决定实际资源。此前单卡推理 allocated 峰约 25–27 GiB、CPU raw KV 约 6.33 GiB，**不提供训练显存上界**。首个 profile 必须记录 `torch.cuda.max_memory_allocated/reserved`、进程 RSS、GPU wall 秒、调用计数、CPU KV bytes、保存与重载耗时；出现 OOM/非有限 loss/错误梯度立即停。不能未经测量承诺一夜能训练多少 updates。

夜间多卡可用于**已批准且互不干扰**的数据编码或独立 profile；不要为了占满 8 卡将 33B 模型并行/复制作为默认方案。09:00 后若还有获批任务，所有项目进程合计最多 3 卡。官方 SolarWM 8 卡推理另见 `SOLAR_STAGE2_REFERENCE.md`，不能和此 FM 预算混账。
