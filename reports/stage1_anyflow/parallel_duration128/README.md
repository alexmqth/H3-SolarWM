硬件状态更正：64→68训练成功，但68→96启动被GPU3新任务触发的空闲检查拒绝；没有96训练发生。独立目录stage1_parallel_resume68_to128将从原step68转到GPU4–7接续。

最新：真实33B四卡64→68已完成，全部replica与恢复检查通过，继续96。详见[TRAIN68_RESULTS.md](TRAIN68_RESULTS.md)。下文保留启动时快照。

# Stage1四卡同batch有限续训：已启动，尚未完成

2026-10-08 13:20，控制器2337864、torchrun2338965，GPU3–6。源shift12总64完整评测未过gate，但8步A−D从0.173上升到0.623，因此只增加一次有限训练预算。先64→68检查真实恢复和replica一致性，再96评测/最多128评测；运行结果未出之前不声称改善。

- [固定协议和决策](PLAN.md)
- [输入冻结检查](preflight.json)
- [启动记录](launch.json)
- [GPU0被新任务占用后的换卡记录](gpu_reassignment.json)
- [已验证的四卡CPU/GPU候选证据](../sample_parallel_probe/RESULTS.md)
- [source64完整结果和视频](../shift12_duration64/FINAL_RESULTS.md)

原准备的GPU0/3/4/5启动被GPU0 occupancy guard拒绝，没有启动任何训练；改用空闲3–6后正常启动。GPU0不限制25GiB，独占时保留reserve6。当前是四个完整H3副本各承担同batch的一条样本，不是sequence parallel。模型、训练目标、全局batch和数据保持不变。真实33B第65次更新已完成（A/chunk2，62.97秒），四卡当前实占约40GiB且无OOM；四套adapter、Adam、teacher/update history、CPU/CUDA/logical RNG真实预更新回载与来源完全一致，固定验证8个raw loss也一致。完整64→68和最终四replica一致性仍在运行。完整runtime在源outputs/2026-10-08-13/stage1_parallel_duration128/，这些报告脚本不是可从报告目录直接启动的便携训练包。
