# EXP-008/v1 — Worker 真实 H3 DMD 单 cycle 报告

2026-10-11 HKT。状态：**GPU 执行完成，等待 Judge 独立审计与验收**。只做任务书和 [GPU_RELEASE.md](GPU_RELEASE.md) 批准的一个 cycle；没有后续追加训练或视频生成。

## 研究问题和角色

验证 33B H3 上的 V3 strict causal / persistent raw KV / AnyFlow 8-map student，能否在保留完整生成链梯度的情况下接受一次 fake-score 与 frozen teacher 差分驱动的 DMD 更新。本轮只测试当前块 AA C2，共同 C1 是 EXP-006 FM8 首12 latent；不包含完整多块 generated-history on-policy 训练。seed170008 为 student 初始噪声，score seed170108；native 8-step shift2.22，score sigma0.6。

物理 GPU0 为 frozen teacher（Original H3 + released Action LoRA，经 V3 causal 入口）；GPU2 为独立 fake velocity model（last8 rank8 QKV）；GPU5 为 AF2 step32 student（target-time gate0.25 + last8 rank8 QKV）。三份模型及 C1 KV 独立，fake 每次更新后重建自己的 KV。teacher **不是原始双向 H3 的评分器**，因此本任务也不声称已完整复现 SolarWM Stage2。

冻结入口：[taskbook_v1.md](taskbook_v1.md)、[run_pilot.py](dmd_cpu/run_pilot.py)、[config_pilot_prep.json](dmd_cpu/config_pilot_prep.json)、[source_manifest_pilot.json](dmd_cpu/source_manifest_pilot.json)，manifest SHA 为 `2bb576217bf285795190d4141af8b51c2209af9a293405b5185adc3af65180ff`。执行前 CPU preflight 导入实际依赖并核对23项来源、配套权重及 AF3 证据，返回 `CPU_PASS_NO_GPU_AUTHORIZATION`；独立 GPU marker 随后放行。

## 实际执行

在仓库根目录执行：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/h3world/bin/python -u submission/experiments/EXP-008_v3_dmd/dmd_cpu/run_pilot.py > H3-World/outputs/EXP-008_v3_dmd_pilot_run.log 2>&1
```

运行前 GPU0/2/5 均空闲，GPU3/4 的他人进程未受影响。GPU 调用账本原件为 `H3-World/outputs/EXP-008_v3_dmd_pilot/budget.json`；[小证据副本](artifacts/budget.json)及[结果副本](artifacts/result.json)已归档。进程正常退出，最终状态 `complete_pending_judge`。

| 指标 | 实际值 | 批准上限 |
| --- | ---: | ---: |
| Forward | 17：3 prefill + 8 student maps + 4 fake warmup/重建 + 2评分 | 17 |
| Backward | 3：fake 2 + student 1 | 3 |
| Optimizer update | 3：fake 2 + student 1 | 3 |
| VAE / 视频 | 0 / 无 | 0 |
| 端到端 wall time | 254.899 秒，含加载和保存 | 1800 秒 |
| 保守三卡 GPU 时间 | 0.212416 GPU 小时 | 1.5 GPU 小时 |
| allocated 峰值 | teacher 25.070、fake 25.879、student 27.965 GiB | 每卡44 GiB |

结果中的 `wall_seconds=246.332` 秒仅覆盖 `pilot()` 内部；预算账本的254.899秒还包括入口加载、退出和最终账本写入，报告资源成本采用后者。`budget.json` 的23条事件是逐调用记录的权威来源；`result.json` 里预留的 `calls/backward_roles/update_roles` 数组为空，不应把它们误读为没有调用。

## 工程验证结果

- student 连续8-map计算图保留；8个 velocity 的反向梯度范数均为有限非零，范围约 `4.98e-6–2.57e-5`。
- fake 的两次普通 FM warmup loss 为0.110606和0.108234，参数确实改变，且两次均在更新后重建自己的 C1 KV。该 loss 变化只表示本单端点上的优化步骤，不能解释为生成质量改善。
- teacher/fake 在同一个 detach+renoise 端点、相同 sigma 下评分；实际导入的 frozen `causal.dmd.py` SHA 为 `11b21eb9c7f6fa1139b007ba62e71873cdd56110fe3d97a49d09010fc4819b32`。记录的方向约定为 `fake_x0-real_x0 = sigma*(real_velocity-fake_velocity)`。
- student DMD backward 后 target-time 梯度范数0.0034630，QKV梯度范数1.7247e-5，二者参数均改变；DMD loss 为0.000147716。此 loss 只是一次方向性 surrogate，不是质量指标。
- CPU复核23条预算事件、4个 checkpoint 文件 SHA、配套任务 metadata、三张 CUDA RNG 状态和8-map非零梯度，返回 `WORKER_CPU_AUDIT_PASS`。

配套权重与 optimizer/RNG 原件保留在 `H3-World/outputs/EXP-008_v3_dmd_pilot/cycle_01/`（`student_qkv.pt`、`student_target.pt`、`fake_qkv.pt`、`trainer_state.pt`），四文件 SHA 见 [artifacts/result.json](artifacts/result.json)。checkpoint 不复制到 Git 仓库，基础33B权重也不复制。

## 范围和判断

这次证明了在当前硬件和冻结 V3 协议下，三角色 33B 的一个真实8-map DMD cycle 可以完成，显存和时间低于预定上限，student 接收到贯穿8-map链的梯度。**尚无 DMD 后视频、动作或长期画质对照，不能称 Stage2 模型能力通过，更不能把此单 cycle 等同于成熟 SolarWM Stage2。** 下一轮是否用此 checkpoint 做匹配 AF3/FM8 视频、是否允许有限追加 cycle，由 Judge 审核后另行决定；本轮不自动开展。
