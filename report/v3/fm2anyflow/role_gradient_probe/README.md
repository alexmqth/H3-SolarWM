# V3角色梯度诊断：双Adapter是否有依据？

EXP-028不是新模型版本，也未训练新的Adapter。LoRA指低秩适配，token-role指动作/视频/公共前缀等token来源分组。本实验检验同一普通Flow Matching损失在这些角色上的梯度贡献，是否在V3因果协议下出现明显更强的抵消。

## 结果

两个真实ABot片段、两个噪声水平，固定原始H3+released Action LoRA，在第0/49层的rank8增量空间比较Original-visible与V3。14次DiT、8次反向，0参数更新。角色梯度求和与autograd一致；四次零增量输出完全相同。L0 cosine总体接近0；L49 V在两图均轻度负对齐（Original均值−0.1075，V3−0.1119），尚不支持因果化特有的明显冲突，因此不直接开展双Adapter训练。

## 成本与视频

GPU1单卡共享环境，总进程583.800秒，峰allocated23.753GiB/reserved26.770GiB、CPU峰RSS55.654GiB。没有采样视频：本次是固定状态前向/反向诊断，不能评价动作准确性、画质或连续性。0updates也不是训练后的模型表现。

精确role dots/跨图梯度矩阵未保存；只有两层、两片与零初始化低秩空间，不能否定所有优化冲突。Original-visible与V3联合改变历史/prefix/拓扑，不作单因素机制归因。

[完整报告](../../../../experiments/EXP-028_action_video_role_probe/report.md) · [Judge结论](../../../../experiments/EXP-028_action_video_role_probe/judge/FINAL_REVIEW.md) · [全部结果](../../../../experiments/EXP-028_action_video_role_probe/recovery01/ROLE_RESULTS.json)
