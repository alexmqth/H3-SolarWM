# EXP-008 — V3 causal DMD有限训练验证

状态：**v1单cycle已完成并通过Judge独立工程审计，尚无生成质量结论**。正式范围见 [taskbook_v1.md](taskbook_v1.md)。本任务从 EXP-007 AF2 step32 的 AnyFlow student 出发，验证一条真实8-map生成链可以完成 fake-score warmup、teacher/fake同状态评分及一次 DMD student 反向。[Worker报告](worker_report_v1.md)和[Judge审核](judge/PILOT_REVIEW.md)记录完整结果与限制。

[dmd_cpu/run_pilot.py](dmd_cpu/run_pilot.py) 已迁入独立 EXP-008 目录，指向 EXP-007 的 `interval_student.py` 和 AF2/AF3 原始证据，输出隔离到 `H3-World/outputs/EXP-008_v3_dmd_pilot/`。GPU执行入口要求独立 `DMD_GPU_AUTHORIZATION.json` marker，已由Judge签发，见[GPU放行](GPU_RELEASE.md)。配置限定17 forward、3 backward、3 update、单 cycle、0 VAE、30分钟、保守1.5三卡GPU小时和09:00 HKT绝对截止。GPU0/2/5经实时空闲核对后使用，不占用GPU3/4他人进程。

CPU 核查：

- [check_protocol.py](dmd_cpu/check_protocol.py) 在随机两层 actual-H3 上测试三角色隔离、各自 KV、8 map 连续梯度、fake 更新后重建 KV、DMD 方向与参数梯度，结果 [CPU_PASS](dmd_cpu/result.json)，2.628秒、0 GPU forward。这不衡量33B吞吐或生成质量。
- `CUDA_VISIBLE_DEVICES='' ... run_pilot.py --preflight` 真正导入 pilot 内部依赖，并核对 AF2 step32 配套权重、AF3两条73帧完成状态、23项来源 SHA 与实际 `causal.dmd` 导入路径，返回 `CPU_PASS_NO_GPU_AUTHORIZATION`，0 GPU调用。冻结来源见 [source_manifest_pilot.json](dmd_cpu/source_manifest_pilot.json)。首次较早的迁移预检 manifest 留作 [历史记录](dmd_cpu/source_manifest_pilot_pre_import_audit.json)。

进程PID849858已正常退出。实际17forward/3backward/3update，wall254.899秒，保守三卡0.212416 GPU小时，三卡allocated峰值25.070/25.879/27.965GiB，8-map梯度均非零。[小证据和原日志](artifacts/)已归档；四个配套checkpoint原件留在`H3-World/outputs/EXP-008_v3_dmd_pilot/cycle_01/`。本结果不授权自动扩训练或声称视频质量改善。

v2有限延续训练已有[CPU准备报告](dmd_train/CPU_PREP.md)与独立[任务书](taskbook_v2.md)：目标从cycle1恢复，新增最多7cycle到累计8。该入口已通过29项来源和优化器/RNG恢复预检；Judge已签发[TRAIN放行](TRAIN_RELEASE_V2.md)，GPU0/2/5进程PID1017778开始运行；99forward/14backward/14update/45min/2.25GPUh上限。视频阶段仍需独立放行。
